import torch
import numpy as np
from abc import abstractmethod
from ..utils.geometry import rmsdalign


class Diffusion:

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def add_noise(self, pos, t, mask=None):
        # return noisy, target
        NotImplemented

    def precondition(self, inp, t):
        return inp

    def postcondition(self, inp, out, t):
        return out
        
    @abstractmethod
    def compute_loss(self, pred, target, t, mask):
        NotImplemented

def t_to_sigma(cfg, t):
    p = cfg.sched_p
    sigma = (
        cfg.sigma_min ** (1 / p)
        + t * (cfg.sigma_max ** (1 / p) - cfg.sigma_min ** (1 / p))
    ) ** p
    return sigma
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def masked_center(x, mask=None, eps=1e-5):
    if mask is None:
        return x - x.mean(-2, keepdims=True)
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return x - com

class EDMDiffusion(Diffusion):

    def get_sigma(self, t, eps=1e-4):
        return t

    def add_noise(self, pos, t, mask=None):

        sigma = self.get_sigma(t)[..., None]
        noise = torch.randn_like(pos) * sigma
        pos = masked_center(pos, mask)

        noisy = pos + noise
        target = pos

        return noisy, target

    def precondition(self, inp, t):
        sigma = self.get_sigma(t)[..., None]
        return inp / (self.cfg.data_sigma**2 + sigma**2) ** 0.5

    def postcondition(self, inp, out, t):
        
        sigma = self.get_sigma(t)[..., None]
        cskip = self.cfg.data_sigma**2 / (self.cfg.data_sigma**2 + sigma**2)
        cout = sigma * self.cfg.data_sigma / (self.cfg.data_sigma**2 + sigma**2) ** 0.5

        return cskip * inp + cout * out

    def compute_loss(self, pred, target, t, mask, aligned=False, eps=1e-9):

        if aligned:
            target = rmsdalign(pred.detach(), target, mask)
            target = torch.where(mask[..., None].bool(), target, 0.0)
        
        
        sigma = self.get_sigma(t)
        num = sigma**2 + self.cfg.data_sigma**2
        denom = (sigma * self.cfg.data_sigma) ** 2
        weight = num / (denom + eps)
        return weight * torch.square(pred - target).sum(-1)
