import os
import sys
import random
import re
import copy
import argparse
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

# Robust path resolution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_CURRENT_DIR)
_ROOT_DIR = os.path.dirname(_SRC_DIR)

for path in [_ROOT_DIR, _SRC_DIR, os.path.join(_SRC_DIR, "model")]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from model import ByteTokenizer, MicroMultimodalMoE
except ImportError:
    from src.model import ByteTokenizer, MicroMultimodalMoE

# -------------------------------------------------------------
# 1. Environments: Logic RL Task & Anchor Replay Buffer
# -------------------------------------------------------------
class LogicTaskEnv:
    """
    Generates strict Boolean reasoning challenges for GRPO RL exploration:
    Prompt: "Eval: True and not False -> "
    Expected completion: "True" or "False"
    """
    def __init__(self):
        self.operations = [
            ("True and True", "True"),
            ("True and False", "False"),
            ("False and True", "False"),
            ("False and False", "False"),
            ("True or True", "True"),
            ("True or False", "True"),
            ("False or True", "True"),
            ("False or False", "False"),
            ("not True", "False"),
            ("not False", "True"),
            ("True and (False or True)", "True"),
            ("False or (not False and True)", "True"),
            ("not (True or False)", "False"),
            ("not (False and True)", "True"),
        ]

    def sample_batch(self, batch_size: int = 1) -> List[Tuple[str, str]]:
        samples = []
        for _ in range(batch_size):
            expr, ans = random.choice(self.operations)
            prompt = f"Eval: {expr} -> "
            samples.append((prompt, ans))
        return samples

    @staticmethod
    def compute_reward(prompt: str, generated_text: str, ground_truth: str) -> float:
        """
        Calculates exact scalar reward:
          +1.0: Exact logical correctness
          +0.2: Outputted valid boolean token ('True' or 'False')
          -2.0: Degenerate prompt regurgitation / trailing '->'
          -0.5: Random rambling / wrong answer
          -1.0: Degenerate / empty output
        """
        cleaned = generated_text.strip().split("\n")[0].strip()

        if not cleaned:
            return -1.0

        # Severe penalty for regurgitating prompt delimiter
        if "->" in cleaned:
            return -2.0

        reward = 0.0

        # Syntax check
        if "True" in cleaned or "False" in cleaned:
            reward += 0.2

        # Exact match
        tokens = re.findall(r"\b(True|False)\b", cleaned)
        if tokens and tokens[0] == ground_truth:
            reward += 1.0
        else:
            reward -= 0.5

        return reward


class AnchorReplayEnv:
    """
    Interleaved Replay Buffer generating multimodal color grounding and arithmetic samples.
    Computes cross-entropy loss SPECIFICALLY MASKED over the target answer span
    (ignoring prompt tokens with PAD_TOKEN_ID) to reinforce color & numerical vocabulary
    directly in the top layers and output head without logit dilution.
    """
    def __init__(self, tokenizer: ByteTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.colors = ["red", "green", "blue", "yellow", "white"]

    def _generate_arithmetic(self) -> Tuple[str, str]:
        a = random.randint(10, 80)
        b = random.randint(10, 80)
        ans = a + b

        # Simple scratchpad breakdown: tens sum + units sum
        t_sum = (a // 10) * 10 + (b // 10) * 10
        u_sum = (a % 10) + (b % 10)

        # Prompt and structured step target with scratchpad
        prompt = f"Calc: {a} + {b} = "
        target = f"[{t_sum}+{u_sum}] = {ans}."
        return prompt, target

    def _generate_multimodal_sample(self) -> Tuple[str, str, torch.Tensor]:
        chosen_color = random.choice(self.colors)
        img = torch.zeros(3, 64, 64, dtype=torch.float32)

        if chosen_color == "red":
            img[0, :, :] = 1.0
        elif chosen_color == "green":
            img[1, :, :] = 1.0
        elif chosen_color == "blue":
            img[2, :, :] = 1.0
        elif chosen_color == "yellow":
            img[0, :, :] = 1.0
            img[1, :, :] = 1.0
        elif chosen_color == "white":
            img[:, :, :] = 1.0

        img_tokens = "<IMG>" * 16
        prompt = f"Visual {img_tokens} Question: Primary tint? Answer: "
        target = f"{chosen_color}."
        return prompt, target, img

    def sample_anchor(self, batch_size: int = 2) -> Dict[str, torch.Tensor]:
        """
        Generates a batch with target-span loss masking:
        labels are set to PAD_TOKEN_ID (256) across the entire prompt prefix,
        so CrossEntropyLoss ONLY calculates gradients for the target response tokens!
        """
        batch_tokens = []
        batch_labels = []
        batch_pixels = []

        for _ in range(batch_size):
            is_multimodal = random.random() < 0.6
            pixel_values = torch.zeros(3, 64, 64, dtype=torch.float32)

            if is_multimodal:
                prompt_text, target_text, pixel_values = self._generate_multimodal_sample()
            else:
                prompt_text, target_text = self._generate_arithmetic()

            # Encode prompt
            prompt_tokens = [self.tokenizer.BOS_TOKEN_ID]
            parts = prompt_text.split("<IMG>")
            for i, part in enumerate(parts):
                if part:
                    prompt_tokens.extend(list(part.encode("utf-8")))
                if i < len(parts) - 1:
                    prompt_tokens.append(self.tokenizer.IMG_TOKEN_ID)
            
            prompt_len = len(prompt_tokens)

            # Encode target
            target_tokens = list(target_text.encode("utf-8")) + [self.tokenizer.EOS_TOKEN_ID]

            full_tokens = prompt_tokens + target_tokens

            # Construct target-masked label:
            # All tokens before prompt_len are set to PAD_TOKEN_ID (ignore_index)
            # Only target_tokens receive loss!
            label_tokens = [self.tokenizer.PAD_TOKEN_ID] * prompt_len + target_tokens

            # Pad or truncate
            if len(full_tokens) > self.max_len:
                full_tokens = full_tokens[:self.max_len]
                label_tokens = label_tokens[:self.max_len]
            else:
                pad_len = self.max_len - len(full_tokens)
                full_tokens = full_tokens + [self.tokenizer.PAD_TOKEN_ID] * pad_len
                label_tokens = label_tokens + [self.tokenizer.PAD_TOKEN_ID] * pad_len

            batch_tokens.append(full_tokens)
            batch_labels.append(label_tokens)
            batch_pixels.append(pixel_values)

        return {
            "input_ids": torch.tensor(batch_tokens, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "pixel_values": torch.stack(batch_pixels, dim=0)
        }


# -------------------------------------------------------------
# 2. GRPO Utilities & Log Probability Computation
# -------------------------------------------------------------
def get_per_token_logps(model: MicroMultimodalMoE, input_ids: torch.Tensor) -> torch.Tensor:
    """Computes log probabilities for every token in input_ids."""
    logits = model(input_ids)["logits"][:, :-1, :]  # (B, T-1, V)
    targets = input_ids[:, 1:]                       # (B, T-1)
    
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return per_token_logps  # (B, T-1)


# -------------------------------------------------------------
# 3. Main GRPO Training Loop with Freezing & Interleaving
# -------------------------------------------------------------
def run_grpo(
    num_iterations: int = 200,
    group_size: int = 8,
    freeze_vision: bool = True,
    freeze_lower_layers: int = 6,
    interleave_anchor_prob: float = 0.20,
    calibration_steps: int = 50,
    checkpoint_dir: Optional[str] = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[GRPO Engine] Running on device: {device}")

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(_ROOT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = ByteTokenizer()
    env = LogicTaskEnv()
    anchor_env = AnchorReplayEnv(tokenizer=tokenizer, max_len=128)

    # 1. Initialize & Load Pretrained Model
    policy_model = MicroMultimodalMoE(
        vocab_size=260,
        d_model=256,
        num_layers=8,
        num_heads=8,
        num_experts=4,
        top_k=2,
        window_size=32,
        max_seq_len=128
    ).to(device)

    pretrained_ckpt = os.path.join(checkpoint_dir, "pretrained_25m.pt")
    if os.path.exists(pretrained_ckpt):
        policy_model.load_state_dict(torch.load(pretrained_ckpt, map_location=device, weights_only=True))
        print(f"[GRPO Engine] Loaded pretrained weights from: {pretrained_ckpt}")
    else:
        print(f"[GRPO Engine] Warning: {pretrained_ckpt} not found. Initializing with random weights.")

    # 2. Frozen Reference Model for KL Divergence Penalty
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 3. Freeze Vision Tower & Lower Transformer Layers
    total_params = sum(p.numel() for p in policy_model.parameters())

    if freeze_vision:
        print("[GRPO Engine] Freezing Vision Tower (ViT, 2D RoPE, Downsampler, SwiGLU Projector)...")
        for param in policy_model.vision_tower.parameters():
            param.requires_grad = False

    if freeze_lower_layers > 0:
        print(f"[GRPO Engine] Freezing lower {freeze_lower_layers} transformer blocks (layers 0 to {freeze_lower_layers-1})...")
        for layer in policy_model.layers[:freeze_lower_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = total_params - trainable_count

    print(f"[GRPO Engine] Total Parameters     : {total_params:,}")
    print(f"[GRPO Engine] Frozen Parameters    : {frozen_count:,} ({(frozen_count / total_params) * 100:.1f}%)")
    print(f"[GRPO Engine] Trainable Parameters : {trainable_count:,} ({(trainable_count / total_params) * 100:.1f}%)")

    # Optimizer over trainable parameters only
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

    beta_kl = 0.04        # KL penalty weight
    clip_eps = 0.2

    # Pre-RL Baseline Accuracy (Greedy decoding, max 4 tokens)
    print("\n[GRPO Engine] Evaluating Pre-RL Baseline Accuracy (Greedy T=0.0)...")
    test_samples = env.sample_batch(batch_size=20)
    baseline_correct = 0
    for prompt, expected in test_samples:
        prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
        gen_tokens = policy_model.generate(prompt_ids, max_new_tokens=4, temperature=0.0)
        gen_text = tokenizer.decode(gen_tokens[0][prompt_ids.shape[1]:])
        tokens = re.findall(r"\b(True|False)\b", gen_text.strip())
        if tokens and tokens[0] == expected:
            baseline_correct += 1
    baseline_acc = (baseline_correct / len(test_samples)) * 100
    print(f"[GRPO Engine] Baseline Accuracy: {baseline_acc:.1f}%\n")

    print("=" * 65)
    print(f"  STARTING GRPO REINFORCEMENT LEARNING (Steps={num_iterations}, Group={group_size})")
    print(f"  Task Mix: ~{int((1 - interleave_anchor_prob) * 100)}% Boolean Logic RL | ~{int(interleave_anchor_prob * 100)}% Replay Anchor CE")
    print("  Penalties: Delimiter regurgitation ('->') gets -2.0 severe penalty")
    print("=" * 65)

    policy_steps = 0
    anchor_steps = 0

    for step in range(1, num_iterations + 1):
        # Decide between Anchor Replay Step vs GRPO RL Rollout Step
        is_anchor = (random.random() < interleave_anchor_prob)

        if is_anchor:
            # --- ANCHOR REPLAY STEP: Target-Span Supervised Cross-Entropy ---
            anchor_steps += 1
            policy_model.train()
            optimizer.zero_grad()

            anchor_batch = anchor_env.sample_anchor(batch_size=2)
            a_input_ids = anchor_batch["input_ids"].to(device)
            a_labels = anchor_batch["labels"].to(device)
            a_pixels = anchor_batch["pixel_values"].to(device)

            anchor_out = policy_model(a_input_ids, pixel_values=a_pixels, labels=a_labels)
            anchor_loss = anchor_out["loss"]
            anchor_loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            optimizer.step()

            if step % 10 == 0 or step == 1:
                print(
                    f"Step [{step:03d}/{num_iterations}] [ANCHOR REPLAY] | "
                    f"Target-Masked CE Loss: {anchor_loss.item():.4f} (Vision & Calc Focused)"
                )

        else:
            # --- GRPO RL ROLLOUT STEP: Group Relative Policy Optimization ---
            policy_steps += 1

            # 1. Sample prompt
            prompt, ground_truth = env.sample_batch(batch_size=1)[0]
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            prompt_len = len(prompt_ids)
            prompt_tensor = torch.tensor([prompt_ids] * group_size, device=device)

            # 2. Rollout G responses with max 4 tokens (prevents trailing rambling)
            policy_model.eval()
            with torch.no_grad():
                rollout_tokens = policy_model.generate(
                    prompt_tensor,
                    max_new_tokens=4,
                    temperature=0.7,
                    top_k=10
                )

            # 3. Compute rewards for the group
            rewards = []
            for g_idx in range(group_size):
                response_text = tokenizer.decode(rollout_tokens[g_idx][prompt_len:])
                r = env.compute_reward(prompt, response_text, ground_truth)
                rewards.append(r)

            rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

            # 4. Group Advantage Normalization (A_i = (R_i - mu) / (sigma + eps))
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-6
            advantages = (rewards_tensor - mean_r) / std_r  # (G,)

            # 5. Policy Optimization Step
            policy_model.train()
            optimizer.zero_grad()

            curr_logps = get_per_token_logps(policy_model, rollout_tokens)
            with torch.no_grad():
                ref_logps = get_per_token_logps(ref_model, rollout_tokens)

            # Response mask for generated tokens only
            response_mask = torch.zeros_like(curr_logps, dtype=torch.bool)
            response_mask[:, prompt_len - 1:] = True

            # Policy Ratio & KL divergence
            ratio = torch.exp(curr_logps - curr_logps.detach())
            kl_div = torch.exp(ref_logps) * (ref_logps - curr_logps)

            # GRPO Surrogate Loss
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2) + beta_kl * kl_div

            loss = (policy_loss * response_mask).sum() / response_mask.sum()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            optimizer.step()

            if step % 10 == 0 or step == 1:
                sample_resp = tokenizer.decode(rollout_tokens[0][prompt_len:]).strip()
                print(
                    f"Step [{step:03d}/{num_iterations}] [GRPO RL G={group_size}]   | "
                    f"Mean Reward: {mean_r.item():+.2f} | "
                    f"GRPO Loss: {loss.item():.4f} | "
                    f"Sample: '{sample_resp}'"
                )

    # 4. Optional Joint Calibration Phase (50 Steps Curriculum Balancing)
    if calibration_steps > 0:
        print("\n" + "=" * 65)
        print(f"  RUNNING JOINT CALIBRATION PHASE ({calibration_steps} Steps)")
        print("  Balancing output head logit distribution across Vision, Calc & Logic")
        print("=" * 65)
        policy_model.train()
        for c_step in range(1, calibration_steps + 1):
            optimizer.zero_grad()
            anchor_batch = anchor_env.sample_anchor(batch_size=4)
            c_input_ids = anchor_batch["input_ids"].to(device)
            c_labels = anchor_batch["labels"].to(device)
            c_pixels = anchor_batch["pixel_values"].to(device)

            c_out = policy_model(c_input_ids, pixel_values=c_pixels, labels=c_labels)
            c_loss = c_out["loss"]
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            optimizer.step()

            if c_step % 10 == 0 or c_step == 1:
                print(f"Calibration [{c_step:02d}/{calibration_steps}] | Target-Masked CE Loss: {c_loss.item():.4f}")

    # Post-RL Evaluation (Greedy decoding, max 4 tokens)
    print("\n" + "=" * 65)
    print("  FINAL EVALUATION & ACCURACY COMPARISON (Greedy T=0.0)")
    print("=" * 65)
    print(f"Total Completed Steps : {num_iterations} (GRPO RL: {policy_steps}, Anchor Replay: {anchor_steps})")
    if calibration_steps > 0:
        print(f"Joint Calibration     : {calibration_steps} supervised steps")

    post_correct = 0
    for prompt, expected in test_samples:
        prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
        gen_tokens = policy_model.generate(prompt_ids, max_new_tokens=4, temperature=0.0)
        gen_text = tokenizer.decode(gen_tokens[0][prompt_ids.shape[1]:])
        tokens = re.findall(r"\b(True|False)\b", gen_text.strip())
        if tokens and tokens[0] == expected:
            post_correct += 1

    final_acc = (post_correct / len(test_samples)) * 100
    print(f"Pre-RL Baseline Accuracy  : {baseline_acc:.1f}%")
    print(f"Post-RL Aligned Accuracy  : {final_acc:.1f}%")
    print(f"Net Accuracy Gain         : {final_acc - baseline_acc:+.1f}%")

    # Save aligned policy checkpoint
    save_path = os.path.join(checkpoint_dir, "grpo_aligned_25m.pt")
    torch.save(policy_model.state_dict(), save_path)
    print(f"\n[GRPO Engine] Saved calibrated policy checkpoint -> {save_path}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run GRPO Reinforcement Learning on Micro-VLM.")
    parser.add_argument("--iterations", type=int, default=200, help="Total training steps (default: 200)")
    parser.add_argument("--group_size", type=int, default=8, help="Rollouts per prompt G (default: 8)")
    parser.add_argument("--freeze_lower_layers", type=int, default=6, help="Number of lower blocks to freeze (default: 6)")
    parser.add_argument("--anchor_prob", type=float, default=0.20, help="Probability of interleaved anchor sample (default: 0.20)")
    parser.add_argument("--calibration_steps", type=int, default=50, help="Supervised joint calibration steps at the end (default: 50)")
    args = parser.parse_args()

    run_grpo(
        num_iterations=args.iterations,
        group_size=args.group_size,
        freeze_vision=True,
        freeze_lower_layers=args.freeze_lower_layers,
        interleave_anchor_prob=args.anchor_prob,
        calibration_steps=args.calibration_steps
    )


if __name__ == "__main__":
    main()
