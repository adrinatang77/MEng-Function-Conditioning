import torch.nn as nn
import torch.nn.functional as F
from . import rope


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, freqs_cis=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        assert self.dim % heads == 0

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

        if freqs_cis is not None:
            self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        query = self.w_q(x).view(B, L, self.heads, -1).transpose(1, 2)
        key = self.w_k(x).view(B, L, self.heads, -1).transpose(1, 2)
        value = self.w_v(x).view(B, L, self.heads, -1).transpose(1, 2)

        freqs_cis = rope.compute_freqs_cis(D // self.heads, L, device=x.device)
        query, key = rope.apply_rotary_emb(query, key, freqs_cis)

        if mask is not None:
            mask = mask.view(B, 1, 1, -1)

        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        return self.w_o(attn)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.lin_in = nn.Linear(dim, ff_dim)
        self.act = nn.ReLU()
        self.lin_out = nn.Linear(ff_dim, dim)

    def forward(self, x):
        x = self.lin_in(x)
        x = self.act(x)
        x = self.lin_out(x)
        return x


class OpenProtTransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_expand):
        super().__init__()
        self.mha = MultiHeadAttention(dim, heads)
        self.mha_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_expand * dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.mha(self.mha_norm(x), mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.in_norm:
            self.in_norm = nn.LayerNorm(cfg.dim)

        self.blocks = nn.ModuleList()
        for _ in range(cfg.blocks):
            self.blocks.append(
                OpenProtTransformerBlock(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    ff_expand=cfg.ff_expand,
                )
            )

    def forward(self, x, mask=None):
        if self.cfg.in_norm:
            x = self.in_norm(x)
        for block in self.blocks:
            x = block(x, mask)
        return x
