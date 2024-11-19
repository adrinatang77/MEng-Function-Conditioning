import torch
import torch.nn as nn
import numpy as np


def sinusoidal_embedding(pos, n_freqs, max_period):
    freqs = (2 * np.pi / max_period) * (torch.arange(n_freqs, device=pos.device) + 1)
    return torch.cat(
        [torch.cos(pos[..., None] * freqs), torch.sin(pos[..., None] * freqs)], -1
    )


class RegressionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, in_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(in_dim)
        self.lin_out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin_out(self.norm(self.act(self.lin_in(x))))


class PairwiseProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, out_dim)
        self.proj2 = nn.Linear(in_dim, out_dim)
        self.proj3 = nn.Linear(2 * out_dim, out_dim)
        self.regression_head = RegressionHead(out_dim, out_dim)

    def forward(self, x):
        q = self.proj1(x)
        k = self.proj2(x)  # B L D

        prod = q[:, :, None, :] * k[:, None, :, :]
        diff = q[:, :, None, :] - k[:, None, :, :]

        return self.regression_head(self.proj3(torch.cat([prod, diff], -1)))


class PositionDecoder(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.cfg = cfg
        self.dim = dim

        if cfg.type == "linear":
            self.decoder = nn.Linear(dim, 3)
        elif cfg.type == "sinusoidal":
            self.decoder = nn.Linear(dim, 6)

        else:
            raise Exception(f"PositionDecoder type {cfg.type} not recognized")

    def forward(self, x):
        if self.cfg.type == "linear":
            return self.decoder(x) * self.cfg.scale
        elif self.cfg.type == "sinusoidal":
            return self.decoder(x) * self.cfg.scale
        else:
            raise Exception(f"PositionDecoder type {self.cfg.type} not recognized")


class PositionEmbedder(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.cfg = cfg
        self.dim = dim

        if cfg.type == "linear":
            self.embedder = nn.Linear(3, dim)

        elif cfg.type == "sinusoidal":
            self.embedder = nn.Linear(6 * cfg.num_freqs, dim)

        else:
            raise Exception(f"PositionEmbedder type {cfg.type} not recognized")

    def forward(self, pos):
        if self.cfg.type == "linear":
            return self.embedder(pos)
        elif self.cfg.type == "sinusoidal":
            emb = sinusoidal_embedding(pos, self.cfg.num_freqs, self.cfg.max_period)
            return self.embedder(emb.reshape(*emb.shape[:-2], -1))
        else:
            raise Exception(f"PositionEmbedder type {self.cfg.type} not recognized")
