import os
import sys
import argparse
import torch
import torch.nn as nn

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

class TextOnlyExportWrapper(nn.Module):
    """
    Wrapper exposing a static-shape forward pass for standard ONNX/TorchScript runtimes.
    Input: input_ids (B, T)
    Output: logits (B, T, vocab_size)
    """
    def __init__(self, model: MicroMultimodalMoE):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids)
        return out["logits"]

def export_quantized(model: MicroMultimodalMoE, output_path: str):
    """Dynamically quantizes linear layers to INT8 for ultra-fast CPU inference."""
    print(f"\n[Export] Quantizing model weights to INT8...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), output_path)
    
    orig_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    quant_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Saved INT8 dynamic quantized model -> {output_path}")
    print(f"[Export] Size reduction: ~{orig_size_mb:.1f} MB -> ~{quant_size_mb:.1f} MB")
    return quantized_model

def export_torchscript(model: MicroMultimodalMoE, output_path: str):
    """Exports model to TorchScript via tracing for zero-Python runtime deployment."""
    print(f"\n[Export] Tracing model with TorchScript...")
    wrapper = TextOnlyExportWrapper(model)
    wrapper.eval()
    dummy_input = torch.randint(0, 255, (1, 32), dtype=torch.long)
    with torch.no_grad():
        traced_cell = torch.jit.trace(wrapper, dummy_input)
    traced_cell.save(output_path)
    print(f"[Export] Saved TorchScript model -> {output_path}")

def export_onnx(model: MicroMultimodalMoE, output_path: str):
    """Exports text forward trunk to ONNX format with dynamic batch and sequence axes."""
    print(f"\n[Export] Exporting to ONNX format...")
    wrapper = TextOnlyExportWrapper(model)
    wrapper.eval()
    dummy_input = torch.randint(0, 255, (1, 16), dtype=torch.long)
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        )
        print(f"[Export] Saved ONNX model -> {output_path}")
    except Exception as e:
        print(f"[Export] Note: ONNX export encountered: {e}")
        print("[Export] Skipping ONNX export; dynamic quantization and TorchScript are recommended for CPU edge execution.")

def main():
    parser = argparse.ArgumentParser(description="Export Micro-VLM weights to optimized edge formats.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save exported models")
    parser.add_argument("--format", type=str, default="quantized", choices=["quantized", "torchscript", "onnx", "all"])
    args = parser.parse_args()

    checkpoint_dir = os.path.join(_ROOT_DIR, "checkpoints")
    if args.checkpoint is None:
        default_ckpt = os.path.join(checkpoint_dir, "grpo_aligned_25m.pt")
        if not os.path.exists(default_ckpt):
            default_ckpt = os.path.join(checkpoint_dir, "pretrained_25m.pt")
        args.checkpoint = default_ckpt

    if args.output_dir is None:
        args.output_dir = os.path.join(_ROOT_DIR, "export")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("           MICRO-VLM MODEL EXPORT UTILITY")
    print("=" * 60)
    print(f"Source Checkpoint : {args.checkpoint}")
    print(f"Target Directory  : {args.output_dir}")

    # Load model
    model = MicroMultimodalMoE(
        vocab_size=260,
        d_model=256,
        num_layers=8,
        num_heads=8,
        num_experts=4,
        top_k=2,
        window_size=32,
        max_seq_len=128
    )

    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[Export] Successfully loaded weights from {args.checkpoint}")
    else:
        print(f"[Export] Warning: {args.checkpoint} not found. Exporting randomly initialized architecture.")

    model.eval()

    if args.format in ["quantized", "all"]:
        q_path = os.path.join(args.output_dir, "micro_vlm_25m_int8.pt")
        export_quantized(model, q_path)

    if args.format in ["torchscript", "all"]:
        ts_path = os.path.join(args.output_dir, "micro_vlm_25m.ptc")
        export_torchscript(model, ts_path)

    if args.format in ["onnx", "all"]:
        onnx_path = os.path.join(args.output_dir, "micro_vlm_25m.onnx")
        export_onnx(model, onnx_path)

    print("\n" + "=" * 60)
    print("Export process complete.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
