import torch
import numpy as np
from abc import abstractmethod


class Diffusion:

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def add_noise(self, pos, t, mask=None):
        # return noisy, target
        NotImplemented

    @abstractmethod
    def inference(self, model, seed=None, mask=None):
        NotImplemented


def masked_center(x, mask=None):
    if mask is None:
        return x
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return torch.where(mask, x - com, x)


class GaussianFM(Diffusion):
    def add_noise(self, pos, t, mask=None):
        noise = torch.randn_like(pos) * self.cfg.prior_sigma
        if self.cfg.center_noise:
            noise = masked_center(noise, mask)

        if self.cfg.center_pos:
            pos = masked_center(pos, mask)

        if self.cfg.train_align:
            pos = rmsdalign(noise, pos, weights=mask)

        t = t[..., None]
        noisy = t * noise + (1 - t) * pos
        if self.cfg.prediction == "velocity":
            target = pos - noise
        elif self.cfg.prediction == "target":
            target = pos
        return noisy, target

    def inference(self, model, cfg=None, seed=None, mask=None, shape=None, device=None):

        if seed is not None:
            x = seed
        else:
            if mask is not None:
                shape = list(mask.shape) + [3]
                device = mask.device
            x = torch.randn(shape, device=device) * self.cfg.prior_sigma

        sched = np.linspace(1, 0, cfg.nsteps + 1)

        for t2, t1 in zip(sched[:-1], sched[1:]):
            dt = t2 - t1

            v = model(x, t2)

            s = ((1 - t2) * v - x) / t2

            noise = torch.randn_like(x)

            g = cfg.sde_weight / (1 - t2 + 0.01)
            gamma = cfg.temp_factor
            dx = v * dt + g * s * dt + np.sqrt(2 * g * gamma * dt) * noise
            
            x = x + dx
            
        return x


class EDMDiffusion(Diffusion):
    NotImplemented
