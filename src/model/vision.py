import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -------------------------------------------------------------
# 1. 2D Rotary Position Embeddings (RoPE)
# -------------------------------------------------------------
def precompute_2d_rotary_emb(
    grid_h: int, 
    grid_w: int, 
    dim: int, 
    base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes cos and sin frequencies for 2D spatial patches."""
    assert dim % 4 == 0, "Per-head dimension must be divisible by 4 for 2D RoPE."
    half_dim = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))

    pos_y = torch.arange(grid_h).float()
    pos_x = torch.arange(grid_w).float()

    angles_y = torch.outer(pos_y, freqs)
    angles_x = torch.outer(pos_x, freqs)

    angles_y = torch.repeat_interleave(angles_y, 2, dim=-1)
    angles_x = torch.repeat_interleave(angles_x, 2, dim=-1)

    angles_y_grid = angles_y.unsqueeze(1).expand(grid_h, grid_w, half_dim)
    angles_x_grid = angles_x.unsqueeze(0).expand(grid_h, grid_w, half_dim)

    angles_2d = torch.cat([angles_y_grid, angles_x_grid], dim=-1)  # (H, W, dim)
    angles_2d = angles_2d.view(grid_h * grid_w, dim)

    return angles_2d.cos(), angles_2d.sin()

def apply_2d_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies 2D RoPE to Q or K tensors:
    x: (Batch, Num_Heads, Num_Patches, Head_Dim)
    cos, sin: (Num_Patches, Head_Dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, Num_Patches, Head_Dim)
    sin = sin.unsqueeze(0).unsqueeze(1)
    x_rot = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1)
    return (x * cos) + (x_rot * sin)

# -------------------------------------------------------------
# 2. ViT Attention Block with 2D RoPE
# -------------------------------------------------------------
class ViTAttention(nn.Module):
    def __init__(self, vit_dim: int = 192, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = vit_dim // num_heads
        self.qkv = nn.Linear(vit_dim, vit_dim * 3, bias=False)
        self.proj = nn.Linear(vit_dim, vit_dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Apply 2D RoPE to spatial patch queries and keys
        q = apply_2d_rotary_emb(q, cos, sin)
        k = apply_2d_rotary_emb(k, cos, sin)

        # Standard Bidirectional Self-Attention (ViT attends across all patches)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class ViTBlock(nn.Module):
    def __init__(self, vit_dim: int = 192, num_heads: int = 3, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(vit_dim)
        self.attn = ViTAttention(vit_dim, num_heads)
        self.norm2 = nn.LayerNorm(vit_dim)
        hidden_dim = int(vit_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vit_dim)
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------------------------------------------
# 3. Spatial Downsampler (2x2 Pixel-Merge) & SwiGLU Projector
# -------------------------------------------------------------
class SpatialDownsampler(nn.Module):
    """Merges adjacent 2x2 patches to cut vision sequence length by 4x."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> Tuple[torch.Tensor, int, int]:
        B, N, C = x.shape
        assert grid_h % 2 == 0 and grid_w % 2 == 0, "Grid dimensions must be even for 2x2 downsampling."
        x = x.view(B, grid_h, grid_w, C)
        # Reshape to merge 2x2 blocks: (B, H//2, 2, W//2, 2, C) -> (B, H//2, W//2, 4*C)
        x = x.view(B, grid_h // 2, 2, grid_w // 2, 2, C).permute(0, 1, 3, 2, 4, 5)
        new_h, new_w = grid_h // 2, grid_w // 2
        x = x.contiguous().view(B, new_h * new_w, 4 * C)
        return x, new_h, new_w

class SwiGLUProjector(nn.Module):
    """Projects merged vision embeddings to the LLM hidden dimension (d_model)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        hidden_dim = out_features * 2
        self.w_gate = nn.Linear(in_features, hidden_dim, bias=False)
        self.w_up = nn.Linear(in_features, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

# -------------------------------------------------------------
# 4. Full Unified Vision Tower
# -------------------------------------------------------------
class UnifiedVisionTower(nn.Module):
    def __init__(
        self, 
        patch_size: int = 8, 
        vit_dim: int = 192, 
        vit_depth: int = 4, 
        num_heads: int = 3, 
        llm_dim: int = 256
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, vit_dim, kernel_size=patch_size, stride=patch_size)
        self.head_dim = vit_dim // num_heads

        self.blocks = nn.ModuleList([
            ViTBlock(vit_dim, num_heads) for _ in range(vit_depth)
        ])
        self.downsampler = SpatialDownsampler(in_dim=vit_dim)
        self.projector = SwiGLUProjector(in_features=vit_dim * 4, out_features=llm_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 3, H, W)
        Output: (B, num_vision_tokens, llm_dim)
        """
        B, C, H, W = pixel_values.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        # 1. Patch projection
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)  # (B, grid_h*grid_w, vit_dim)

        # 2. Precompute 2D RoPE frequencies for current grid size
        cos, sin = precompute_2d_rotary_emb(grid_h, grid_w, self.head_dim)
        cos, sin = cos.to(x.device), sin.to(x.device)

        # 3. ViT layers with 2D RoPE
        for blk in self.blocks:
            x = blk(x, cos, sin)

        # 4. 2x2 Spatial Downsampling: reduces token count by 4x
        x, _, _ = self.downsampler(x, grid_h, grid_w)

        # 5. SwiGLU projection into language space
        vision_tokens = self.projector(x)
        return vision_tokens

def splice_multimodal_embeddings(
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    vision_features: torch.Tensor,
    img_token_id: int = 259,
) -> torch.Tensor:
    """Replace image-token embeddings with the corresponding vision features."""
    img_mask = input_ids == img_token_id
    merged_embeddings = text_embeddings.clone()

    if img_mask.any():
        for batch_index in range(input_ids.shape[0]):
            batch_mask = img_mask[batch_index]
            num_slots = batch_mask.sum().item()
            if num_slots:
                batch_features = vision_features[batch_index]
                assert num_slots == batch_features.shape[0], (
                    f"Mismatch: Prompt has {num_slots} <IMG> tokens, "
                    f"but vision encoder generated {batch_features.shape[0]} tokens."
                )
                merged_embeddings[batch_index, batch_mask] = batch_features

    return merged_embeddings
