import os
import sys
import random
import re
import copy
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

# Robust path resolution to allow running directly or as module
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
# 1. Logic Environment & Deterministic Reward Evaluator
# -------------------------------------------------------------
class LogicTaskEnv:
    """
    Generates strict Boolean reasoning challenges:
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

    def sample_batch(self, batch_size: int = 4) -> List[Tuple[str, str]]:
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
          -0.5: Random rambling / wrong answer
          -1.0: Degenerate / empty output
        """
        cleaned = generated_text.strip().split("\n")[0].strip()

        if not cleaned:
            return -1.0

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

# -------------------------------------------------------------
# 2. GRPO Rollout & Loss Calculation
# -------------------------------------------------------------
def get_per_token_logps(model: MicroMultimodalMoE, input_ids: torch.Tensor) -> torch.Tensor:
    """Computes log probabilities for every token in input_ids."""
    logits = model(input_ids)["logits"][:, :-1, :]  # (B, T-1, V)
    targets = input_ids[:, 1:]                       # (B, T-1)
    
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return per_token_logps  # (B, T-1)

def run_grpo(num_iterations: int = 40, checkpoint_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GRPO Training on device: {device}")

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(_ROOT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = ByteTokenizer()
    env = LogicTaskEnv()

    # 1. Load Pretrained Policy
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
        print(f"Loaded pretrained weights from {pretrained_ckpt}")
    else:
        print(f"Warning: {pretrained_ckpt} not found. Initializing from scratch.")

    # 2. Reference Model (frozen copy to compute KL divergence penalty)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Low learning rate for RL stability
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    group_size = 4        # G = 4 rollouts per prompt
    beta_kl = 0.04        # KL penalty weight
    clip_eps = 0.2

    print("\nEvaluating Pre-RL Baseline Accuracy...")
    test_samples = env.sample_batch(batch_size=20)
    baseline_correct = 0
    for prompt, expected in test_samples:
        prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
        gen_tokens = policy_model.generate(prompt_ids, max_new_tokens=16, temperature=0.7)
        gen_text = tokenizer.decode(gen_tokens[0][prompt_ids.shape[1]:])
        tokens = re.findall(r"\b(True|False)\b", gen_text.strip())
        if tokens and tokens[0] == expected:
            baseline_correct += 1
    print(f"Baseline Accuracy: {baseline_correct / len(test_samples) * 100:.1f}%\n")

    print(f"Starting GRPO Policy Optimization ({num_iterations} iterations)...")
    for step in range(1, num_iterations + 1):
        # 1. Sample prompt
        prompt, ground_truth = env.sample_batch(batch_size=1)[0]
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        prompt_len = len(prompt_ids)
        prompt_tensor = torch.tensor([prompt_ids] * group_size, device=device)

        # 2. Rollout G responses
        policy_model.eval()
        with torch.no_grad():
            rollout_tokens = policy_model.generate(
                prompt_tensor,
                max_new_tokens=12,
                temperature=0.8,
                top_k=20
            )

        # 3. Compute rewards for the group
        rewards = []
        for g_idx in range(group_size):
            response_text = tokenizer.decode(rollout_tokens[g_idx][prompt_len:])
            r = env.compute_reward(prompt, response_text, ground_truth)
            rewards.append(r)

        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

        # 4. Group Advantage Normalization
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

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.5)
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(
                f"Step [{step:02d}/{num_iterations}] | "
                f"Mean Reward: {mean_r.item():+.2f} | "
                f"GRPO Loss: {loss.item():.4f} | "
                f"Sample: '{tokenizer.decode(rollout_tokens[0][prompt_len:]).strip()}'"
            )

    # Final Evaluation
    print("\nEvaluating Post-RL Accuracy...")
    post_correct = 0
    for prompt, expected in test_samples:
        prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
        gen_tokens = policy_model.generate(prompt_ids, max_new_tokens=16, temperature=0.2)
        gen_text = tokenizer.decode(gen_tokens[0][prompt_ids.shape[1]:])
        tokens = re.findall(r"\b(True|False)\b", gen_text.strip())
        if tokens and tokens[0] == expected:
            post_correct += 1

    final_acc = (post_correct / len(test_samples)) * 100
    print(f"Final Accuracy: {final_acc:.1f}% (Baseline was {baseline_correct / len(test_samples) * 100:.1f}%)")

    # Save aligned policy checkpoint
    save_path = os.path.join(checkpoint_dir, "grpo_aligned_25m.pt")
    torch.save(policy_model.state_dict(), save_path)
    print(f"Aligned policy checkpoint saved to {save_path}")

if __name__ == "__main__":
    run_grpo()
