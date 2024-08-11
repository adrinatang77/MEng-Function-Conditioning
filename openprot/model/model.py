import torch.nn as nn


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, x):
        return self.linear(x)
