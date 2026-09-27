"""Tiny DiT-style encoder/decoder velocity model, without conditional modulation.

The decoder is a compact Transformer: trajectory tokens self-attend, then
cross-attend to cached scene tokens.  It deliberately does *not* use AdaLN,
FiLM, or any condition-dependent normalization/scale/shift.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SceneEncoder(nn.Module):
    """Encodes a one-hot toy scene into two context tokens, once per sample."""

    def __init__(self, condition_dim: int, hidden_dim: int, num_tokens: int = 2):
        super().__init__()
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.token_offsets = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * .02)
        self.mlp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim))

    def forward(self, condition: Tensor) -> Tensor:
        tokens = self.condition_proj(condition).unsqueeze(1) + self.token_offsets
        return tokens + self.mlp(tokens)


class DiTBlockNoModulation(nn.Module):
    """Pre-LN self-attention + cross-attention + MLP; no adaptive conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_ratio * hidden_dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * hidden_dim, hidden_dim))

    def forward(self, x_tokens: Tensor, scene_tokens: Tensor) -> Tensor:
        normalized = self.self_norm(x_tokens)
        x_tokens = x_tokens + self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        query = self.cross_norm(x_tokens)
        x_tokens = x_tokens + self.cross_attn(query, scene_tokens, scene_tokens, need_weights=False)[0]
        return x_tokens + self.mlp(self.mlp_norm(x_tokens))


class TinyDiTDecoder(nn.Module):
    """Small DiT decoder with a concatenated time token and scene cross-attention."""

    def __init__(self, hidden_dim: int = 64, depth: int = 2, num_heads: int = 4):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.x_proj = nn.Linear(2, hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([DiTBlockNoModulation(hidden_dim, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: Tensor, t: Tensor, scene_tokens: Tensor) -> Tensor:
        if t.numel() == 1:
            t_feature = t.reshape(1, 1).expand(x.shape[0], 1)
        else:
            t_feature = t.reshape(x.shape[0], 1)
        # Keep time as its own token: [trajectory_token, time_token], never add it to x.
        trajectory_token = self.x_proj(x).unsqueeze(1)
        time_token = self.time_mlp(t_feature).unsqueeze(1)
        x_tokens = torch.cat((trajectory_token, time_token), dim=1)
        for block in self.blocks:
            x_tokens = block(x_tokens, scene_tokens)
        # Position 0 remains the trajectory token; position 1 is time only.
        return self.velocity_head(self.final_norm(x_tokens[:, 0]))


class ToyEncoderDecoderFlow(nn.Module):
    """Tiny DiT flow model with FlowPlanner-like cached encoder inference API."""

    def __init__(self, condition_dim: int = 4, hidden_dim: int = 64, depth: int = 2, num_heads: int = 4):
        super().__init__()
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.encoder = SceneEncoder(condition_dim, hidden_dim)
        self.decoder = TinyDiTDecoder(hidden_dim, depth, num_heads)
        self.reset_inference_counters()

    def reset_inference_counters(self) -> None:
        self.encoder_calls, self.decoder_calls = 0, 0

    def encode(self, condition: Tensor) -> Tensor:
        self.encoder_calls += 1
        return self.encoder(condition)

    def decode(self, x: Tensor, t: Tensor, encoded_scene: Tensor) -> Tensor:
        self.decoder_calls += 1
        return self.decoder(x, t, encoded_scene)

    def forward(self, x: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        return self.decode(x, t, self.encode(condition))
