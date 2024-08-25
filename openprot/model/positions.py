import torch
import torch.nn as nn
import numpy as np

def sinusoidal_embedding(pos, n_freqs, max_period):
    freqs = (2*np.pi / max_period) * (torch.arange(n_freqs, device=pos.device) + 1)
    return torch.cat([torch.cos(pos[...,None] * freqs), torch.sin(pos[...,None] * freqs)], -1)


class PositionDecoder(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        
        if cfg.type == 'linear':
            self.decoder = nn.Linear(dim, 3)

        else:
            raise RuntimeException(f"PositionDecoder type {cfg.type} not recognized")

    def forward(self, x):
        if self.cfg.type == 'linear':
            return self.decoder(x)
        else:
            raise RuntimeException(f"PositionDecoder type {self.cfg.type} not recognized")
        

class PositionEmbedder(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        
        if cfg.type == 'linear':
            self.embedder = nn.Linear(3, dim)

        elif cfg.type == 'sinusoidal':
            self.embedder = nn.Linear(6 * cfg.num_freqs, dim)
            
        else:
            raise RuntimeException(f"PositionEmbedder type {cfg.type} not recognized")

    
    
    def forward(self, pos):
        if self.cfg.type == 'linear':
            return self.embedder(pos)
        elif self.cfg.type == 'sinusoidal':
            emb = sinusoidal_embedding(pos, self.cfg.num_freqs, self.cfg.max_period)
            return self.embedder(emb.reshape(*emb.shape[:-2], -1))
        else:
            raise RuntimeException(f"PositionEmbedder type {self.cfg.type} not recognized")
