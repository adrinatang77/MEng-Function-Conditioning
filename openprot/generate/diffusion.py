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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def masked_center(x, mask=None, eps=1e-5):
    if mask is None:
        return x - x.mean(-2, keepdims=True)
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return torch.where(mask, x - com, x)


class GaussianFM(Diffusion):
    def add_noise(self, pos, t, mask=None):
        noise = torch.randn_like(pos) * self.cfg.data_sigma
        pos = masked_center(pos, mask)
        if self.cfg.train_align:
            pos = rmsdalign(noise, pos, weights=mask)

        t = t[..., None]
        noisy = t * noise + (1 - t) * pos
        target = pos
        
        return noisy, target

    def precondition(self, inp, t):
        return inp / self.cfg.data_sigma

    def postcondition(self, inp, out, t):
        return inp + out * t.unsqueeze(-1) * self.cfg.data_sigma

    def compute_loss(self, pred, target, t, mask, eps=1e-6):
        return torch.square(pred - target).sum(-1) / (t**2+eps) / self.cfg.data_sigma**2

class EDMDiffusion(Diffusion):

    def get_sigma(self, t, eps=1e-4):
        return t / (1-t+eps) * self.cfg.data_sigma

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

    def compute_loss(self, pred, target, t, mask, eps=1e-9):

        if self.cfg.aligned_loss:
            target = rmsdalign(pred.detach(), target, mask)
            target = torch.where(mask[..., None].bool(), target, 0.0)
        
        
        sigma = self.get_sigma(t)
        num = sigma**2 + self.cfg.data_sigma**2
        denom = (sigma * self.cfg.data_sigma) ** 2
        weight = num / (denom + eps)
        return weight * torch.square(pred - target).sum(-1)
