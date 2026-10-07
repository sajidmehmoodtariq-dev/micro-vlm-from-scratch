import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -------------------------------------------------------------
# 1. 1D Rotary Position Embeddings (RoPE) for Text
# -------------------------------------------------------------
def precompute_1d_rotary_emb(seq_len: int, dim: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes cos and sin frequencies for sequential text positions."""
    assert dim % 2 == 0, "Dimension must be even for 1D RoPE."
    # dim is head_dim (e.g. 32). freqs has dim // 2 elements (e.g. 16).
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    angles = torch.outer(t, freqs)  # (seq_len, dim // 2)
    angles = torch.repeat_interleave(angles, 2, dim=-1)  # (seq_len, dim)
    return angles.cos(), angles.sin()

def apply_1d_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies 1D RoPE to Q or K tensors:
    x: (Batch, Num_Heads, Seq_Len, Head_Dim)
    cos, sin: (Seq_Len, Head_Dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, Seq_Len, Head_Dim)
    sin = sin.unsqueeze(0).unsqueeze(1)
    x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return (x * cos) + (x_rot * sin)

# -------------------------------------------------------------
# 2. Hybrid Causal Attention (Windowed / Global Hybrid)
# -------------------------------------------------------------
class HybridCausalAttention(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, window_size: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size  # None means full causal attention

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)

        # Apply 1D RoPE
        q = apply_1d_rotary_emb(q, cos[:T], sin[:T])
        k = apply_1d_rotary_emb(k, cos[:T], sin[:T])

        # Causal mask with optional sliding window
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        if self.window_size is not None:
            band_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-self.window_size)
            causal_mask = causal_mask & band_mask

        # Score computation via PyTorch scaled_dot_product_attention
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, T, T)
        if mask is not None:
            attn_mask = attn_mask & mask.view(B, 1, 1, T).bool()

        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# -------------------------------------------------------------
# 3. Mixture of Experts (MoE) with Top-k Gating
# -------------------------------------------------------------
class ExpertMLP(nn.Module):
    """Individual Feed-Forward Expert using SwiGLU."""
    def __init__(self, d_model: int = 256, intermediate_dim: int = 512):
        super().__init__()
        self.w_gate = nn.Linear(d_model, intermediate_dim, bias=False)
        self.w_up = nn.Linear(d_model, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts layer:
    Routes each token to the top_k best matching experts.
    """
    def __init__(
        self, 
        d_model: int = 256, 
        num_experts: int = 4, 
        top_k: int = 2, 
        intermediate_dim: int = 512
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertMLP(d_model, intermediate_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)

        # Compute router logits and normalized weights
        router_logits = self.router(x_flat)  # (B*T, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select Top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Renormalize

        final_output = torch.zeros_like(x_flat)

        # Execute expert paths
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (topk_indices == expert_idx)
            if expert_mask.any():
                token_indices, k_pos = torch.where(expert_mask)
                expert_in = x_flat[token_indices]
                expert_out = expert(expert_in)
                weight = topk_weights[token_indices, k_pos].unsqueeze(-1)
                final_output.index_add_(0, token_indices, expert_out * weight)

        # Compute router load balancing loss
        token_prob = routing_weights.mean(dim=0)
        tokens_per_expert = (topk_indices.flatten()[:, None] == torch.arange(self.num_experts, device=x.device)).float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(token_prob * tokens_per_expert)

        return final_output.view(B, T, C), aux_loss

# -------------------------------------------------------------
# 4. Transformer Block with Hyperconnections & RMSNorm
# -------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class HyperconnectedMoEBlock(nn.Module):
    """
    Core transformer block combining Hybrid Attention, MoE, and residual scaling.
    """
    def __init__(
        self, 
        d_model: int = 256, 
        num_heads: int = 8, 
        num_experts: int = 4, 
        top_k: int = 2, 
        window_size: Optional[int] = None
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = HybridCausalAttention(d_model, num_heads, window_size=window_size)
        self.norm2 = RMSNorm(d_model)
        self.moe = MoEFeedForward(d_model, num_experts=num_experts, top_k=top_k)

        # Hyperconnection / residual scaling parameters
        self.res_weight_attn = nn.Parameter(torch.ones(1) * 0.5)
        self.res_weight_moe = nn.Parameter(torch.ones(1) * 0.5)

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm + residual with learned scale
        attn_out = self.attn(self.norm1(x), cos, sin, mask=mask)
        x = x + self.res_weight_attn * attn_out

        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.res_weight_moe * moe_out

        return x, aux_loss
