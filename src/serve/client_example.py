import os
import sys
import time
import torch
from typing import Optional

# Path resolution
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

class MicroVLMClient:
    """
    Minimal standalone inference client for Micro-VLM.
    Runs locally on CPU or CUDA with zero external dependencies.
    """
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.tokenizer = ByteTokenizer()

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
                candidate_path = os.path.join(checkpoint_dir, candidate)
                if os.path.exists(candidate_path):
                    checkpoint_path = candidate_path
                    break

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
            print(f"[Client] Loaded weights from: {checkpoint_path}")
        else:
            print("[Client] Warning: No checkpoint found. Using randomly initialized weights.")

        self.model.eval()

    def generate(
        self, 
        prompt: str, 
        pixel_values: Optional[torch.Tensor] = None, 
        max_new_tokens: int = 24, 
        temperature: float = 0.3
    ) -> str:
        """
        Runs autoregressive inference on prompt + optional image.
        Returns generated string response.
        """
        start_t = time.perf_counter()
        
        # Tokenize text with BOS
        tokens = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([tokens], device=self.device)

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=10
            )

        # Slice out only the generated tokens
        prompt_len = input_ids.shape[1]
        generated_tokens = output_ids[0][prompt_len:]
        completion = self.tokenizer.decode(generated_tokens)
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        return completion.strip(), round(elapsed_ms, 2)


def main():
    print("=" * 60)
    print("      MICRO-VLM STANDALONE INFERENCE CLIENT")
    print("=" * 60)

    client = MicroVLMClient()

    # 1. Text Reasoning Example
    test_prompts = [
        "Eval: True and not False -> ",
        "Eval: not (False or False) -> ",
        "Calc: 15 + 27 = ",
    ]

    print("\n--- 1. Text Reasoning Demos ---")
    for prompt in test_prompts:
        completion, latency_ms = client.generate(prompt, max_new_tokens=16, temperature=0.1)
        print(f"Prompt   : {prompt}")
        print(f"Response : {completion}")
        print(f"Latency  : {latency_ms} ms\n")

    # 2. Multimodal Example (Synthetic Color Image)
    print("--- 2. Multimodal Vision Demo ---")
    # Synthetic 64x64 green canvas
    green_image = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    green_image[0, 1, :, :] = 1.0  # Green channel

    # 16 <IMG> placeholder tokens matching downsampler: (64/8)^2 / 4 = 16
    img_slots = "<IMG>" * 16
    mm_prompt = f"Visual {img_slots} Question: Primary tint? Answer: "
    
    completion, latency_ms = client.generate(
        mm_prompt, 
        pixel_values=green_image, 
        max_new_tokens=10, 
        temperature=0.1
    )
    print(f"Prompt   : Visual [64x64 Green Canvas] Question: Primary tint? Answer:")
    print(f"Response : {completion}")
    print(f"Latency  : {latency_ms} ms\n")

    print("=" * 60)

if __name__ == "__main__":
    main()
