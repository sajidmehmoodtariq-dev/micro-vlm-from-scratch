import os
import sys
import json
import time
import re
import torch

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

class JarvisIntentEngine:
    """
    Ultra-low latency local engine for Jarvis intent classification and OS dispatch.
    Runs entirely on CPU with zero external API calls.
    """
    def __init__(self, checkpoint_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = ByteTokenizer()
        
        # Load architecture configuration
        self.model = MicroMultimodalMoE(
            vocab_size=260,
            d_model=256,
            num_layers=8,
            num_heads=8,
            num_experts=4,
            top_k=2,
            window_size=32,
            max_seq_len=128
        ).to(self.device)

        if checkpoint_path is None:
            checkpoint_dir = os.path.join(_ROOT_DIR, "checkpoints")
            for candidate in ["grpo_aligned_25m.pt", "pretrained_25m.pt"]:
                cand_path = os.path.join(checkpoint_dir, candidate)
                if os.path.exists(cand_path):
                    checkpoint_path = cand_path
                    break

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
            print(f"[JarvisEngine] Loaded model weights from {checkpoint_path}")
        else:
            print(f"[JarvisEngine] Warning: {checkpoint_path} not found. Running initialized weights.")

        self.model.eval()

        # Known local OS action intents for deterministic fallback routing
        self.intent_rules = {
            "lock": {"action": "system_lock", "target": "workstation"},
            "sleep": {"action": "system_sleep", "target": "workstation"},
            "mute": {"action": "audio_mute", "target": "system_audio"},
            "unmute": {"action": "audio_unmute", "target": "system_audio"},
            "terminal": {"action": "launch_app", "target": "powershell"},
            "code": {"action": "launch_app", "target": "vscode"},
            "browser": {"action": "launch_app", "target": "chrome"},
            "whatsapp": {"action": "launch_app", "target": "whatsapp"},
        }

    @torch.no_grad()
    def route_intent(self, user_transcript: str) -> dict:
        """
        Takes raw spoken text from Whisper and produces structured execution JSON.
        """
        start_time = time.perf_counter()
        cleaned_input = user_transcript.strip().lower()

        # 1. Fast Heuristic Check (<0.1ms) for high-frequency direct commands
        for keyword, payload in self.intent_rules.items():
            if re.search(rf"\b{keyword}\b", cleaned_input):
                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "source": "heuristic_fast_path",
                    "intent": payload["action"],
                    "target": payload["target"],
                    "raw_input": user_transcript,
                    "latency_ms": round(latency_ms, 2),
                    "forward_to_cloud": False
                }

        # 2. Local Neural MoE Forward Pass (~15-25ms on CPU)
        prompt = f"Route: {cleaned_input} -> "
        prompt_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_bos=True)], 
            device=self.device
        )

        gen_tokens = self.model.generate(
            prompt_ids,
            max_new_tokens=16,
            temperature=0.1,  # Near-deterministic
            top_k=5
        )

        raw_output = self.tokenizer.decode(gen_tokens[0][prompt_ids.shape[1]:])
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 3. Decision Boundary: Local command vs Complex/Open QA Cloud escalation
        is_local_command = any(k in cleaned_input for k in ["open", "close", "set", "turn", "run", "stop"])
        
        return {
            "source": "neural_moe_edge",
            "model_output": raw_output.strip(),
            "raw_input": user_transcript,
            "latency_ms": round(latency_ms, 2),
            "forward_to_cloud": not is_local_command
        }
