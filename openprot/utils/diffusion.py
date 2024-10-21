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

    @abstractmethod
    def inference(self, model, seed=None, mask=None):
        NotImplemented

    @abstractmethod
    def compute_loss(self, pred, gt, t, mask=None):
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

        # print((t * noise).square().sum(-1) / (t.squeeze(-1)**2 * 15**2))
        return noisy, target

    def inference(
        self,
        model,
        cfg=None,
        seed=None,
        mask=None,
        shape=None,
        device=None,
        return_traj=False,
    ):

        if seed is not None:
            x = seed
        else:
            if mask is not None:
                shape = list(mask.shape) + [3]
                device = mask.device
            x = torch.randn(shape, device=device) * self.cfg.prior_sigma

        out = [x]
        if cfg.sched_type == "linear":
            sched = np.linspace(1, 0, cfg.nsteps + 1)
        elif cfg.sched_type == "log":
            sched = np.logspace(0, -2, cfg.nsteps + 1)
            sched = (sched - sched.min()) / (sched.max() - sched.min())

        for t2, t1 in zip(sched[:-1], sched[1:]):
            dt = t2 - t1

            if self.cfg.prediction == "velocity":
                v = model(x, t2)
            elif self.cfg.prediction == "target":
                x0 = model(x, t2)

                if self.cfg.inf_align:
                    x0 = rmsdalign(x, x0, mask)
                    
                v = (x0 - x) / t2

            # score = ((1-t)x0 - x_t) / t^2 * sigma^2

            # x0 = xt + t*v
            # score = (xt + t*v - t*xt - t^2*v - xt) / t^2 * sigma^2
            #       = v - xt - t*v / t * sigma^2
            #       = = (1-t)*v - xt / t * sigma^2

            s = ((1 - t2) * v - x) / t2 / self.cfg.prior_sigma**2

            noise = torch.randn_like(x)

            g = cfg.sde_weight * t2 * self.cfg.prior_sigma**2

            gamma = cfg.temp_factor
            dx = v * dt + g * s * dt + np.sqrt(2 * g * gamma * dt) * noise

            x = x + dx
            out.append(x)

        if return_traj:
            return torch.stack(out)
        else:
            return x


class EDMDiffusion(Diffusion):
    NotImplemented
