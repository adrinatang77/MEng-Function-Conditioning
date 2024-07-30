import torch
import torch.nn as nn

class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lin_out = nn.Linear(3, 5)

    def forward(self, batch):
        
        return self.lin_out(batch['a'])