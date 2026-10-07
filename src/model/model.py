import json
import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union

try:
    from .tokenizer import ByteTokenizer
    from .vision import UnifiedVisionTower, splice_multimodal_embeddings
    from .transformer import (
        HyperconnectedMoEBlock, 
        RMSNorm, 
        precompute_1d_rotary_emb
    )
except ImportError:
    from tokenizer import ByteTokenizer
    from vision import UnifiedVisionTower, splice_multimodal_embeddings
    from transformer import (
        HyperconnectedMoEBlock, 
        RMSNorm, 
        precompute_1d_rotary_emb
    )

class MicroMultimodalMoE(nn.Module):
    """
    Unified 25M Parameter Multimodal Language Model
    - Byte-level vocabulary: 260 tokens
    - Bidirectional ViT with 2D RoPE + 2x2 Spatial Downsampler + SwiGLU Projector
    - Alternating Local / Global Causal Attention with 1D RoPE
    - Top-2 Routing MoE feed-forward blocks with load balancing
    - Hyperconnection residual scaling
    """
    def __init__(
        self,
        vocab_size: int = 260,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        window_size: int = 32,
        max_seq_len: int = 512,
        # Vision hyperparams
        patch_size: int = 8,
        vit_dim: int = 192,
        vit_depth: int = 4,
        vit_heads: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_heads

        # Store config dict for easy serialization
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_experts": num_experts,
            "top_k": top_k,
            "window_size": window_size,
            "max_seq_len": max_seq_len,
            "patch_size": patch_size,
            "vit_dim": vit_dim,
            "vit_depth": vit_depth,
            "vit_heads": vit_heads,
        }

        # 1. Byte Text Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # 2. Vision Tower
        self.vision_tower = UnifiedVisionTower(
            patch_size=patch_size,
            vit_dim=vit_dim,
            vit_depth=vit_depth,
            num_heads=vit_heads,
            llm_dim=d_model
        )

        # 3. Transformer Trunk (Alternating Local / Global Attention)
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Even layers: Local windowed attention; Odd layers: Full global attention
            w_size = window_size if (layer_idx % 2 == 0) else None
            self.layers.append(
                HyperconnectedMoEBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_experts=num_experts,
                    top_k=top_k,
                    window_size=w_size
                )
            )

        # 4. Final Norm & Projection Head
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share input embedding weights with output head
        self.lm_head.weight = self.token_embeddings.weight

        # Precompute RoPE cache for fast evaluation
        cos, sin = precompute_1d_rotary_emb(max_seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @classmethod
    def from_config(cls, config_source: Union[str, Dict[str, Any]]) -> "MicroMultimodalMoE":
        """Instantiates MicroMultimodalMoE from a JSON file path or a dictionary."""
        if isinstance(config_source, str):
            with open(config_source, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = config_source
        return cls(**cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids:     (B, T) LongTensor
        pixel_values:  (B, 3, H, W) FloatTensor (optional)
        labels:        (B, T) LongTensor (optional for loss computation)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # 1. Base Text Embeddings
        x = self.token_embeddings(input_ids)

        # 2. Vision Encoding & Multimodal Splicing
        if pixel_values is not None:
            vision_tokens = self.vision_tower(pixel_values)
            x = splice_multimodal_embeddings(
                input_ids=input_ids,
                text_embeddings=x,
                vision_features=vision_tokens,
                img_token_id=ByteTokenizer.IMG_TOKEN_ID
            )

        # 3. Forward through Transformer Layers
        total_aux_loss = 0.0
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            x, aux_loss = layer(x, cos, sin, mask=padding_mask)
            total_aux_loss += aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        output = {
            "logits": logits,
            "aux_loss": total_aux_loss
        }

        # 4. Optional CrossEntropy Loss Calculation
        if labels is not None:
            # Shift tokens for next-byte prediction: logits[:-1] vs labels[1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=ByteTokenizer.PAD_TOKEN_ID)
            lm_loss = loss_fn(
                shift_logits.view(-1, self.vocab_size), 
                shift_labels.view(-1)
            )
            # Combine next-token loss with router balancing loss (scale factor: 0.01)
            output["loss"] = lm_loss + 0.01 * total_aux_loss
            output["lm_loss"] = lm_loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 10,
        eos_token_id: int = ByteTokenizer.EOS_TOKEN_ID
    ) -> torch.Tensor:
        """Autoregressive generation loop."""
        self.eval()
        curr_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            if curr_ids.shape[1] >= self.max_seq_len:
                break

            out = self.forward(curr_ids, pixel_values=pixel_values)
            next_logits = out["logits"][:, -1, :]  # (B, vocab_size)

            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Check if all batches reached EOS
            if (next_token == eos_token_id).all():
                break

        return curr_ids


if __name__ == "__main__":
    model = MicroMultimodalMoE(
        vocab_size=260,
        d_model=256,
        num_layers=8,
        num_heads=8,
        num_experts=4,
        top_k=2,
        window_size=32,
        max_seq_len=512,
        patch_size=8,
        vit_dim=192,
        vit_depth=4,
        vit_heads=3
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:     {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Forward pass smoke test with dummy multimodal data
    dummy_text = torch.randint(0, 255, (2, 32))
    dummy_text[:, 5:21] = ByteTokenizer.IMG_TOKEN_ID
    dummy_pixels = torch.randn(2, 3, 64, 64)

    out = model(dummy_text, pixel_values=dummy_pixels, labels=dummy_text)
    print(f"Total Loss: {out['loss'].item():.4f} | LM Loss: {out['lm_loss'].item():.4f}")
