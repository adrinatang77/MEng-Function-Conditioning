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

    def precondition(self, inp, t):
        return inp

    def postcondition(self, inp, out, t):
        return out
    
    @abstractmethod
    def compute_loss(self, pred, gt, t, mask=None):
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

    def precondition(self, inp, t):
        return inp # / self.cfg.prior_sigma
        
    def postcondition(self, inp, out, t):
        return out # * self.cfg.prior_sigma
        
    def compute_loss(self, pred, target, t):
        return torch.square(pred - target).sum(-1) # / self.cfg.prior_sigma**2

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

        preds = []
        for t2, t1 in zip(sched[:-1], sched[1:]):
            dt = t2 - t1

            if self.cfg.prediction == "velocity":
                v = model(x, t2)
            elif self.cfg.prediction == "target":
                x0 = model(x, t2)

                if self.cfg.inf_align:
                    x0 = rmsdalign(x, x0, mask)

                v = (x0 - x) / t2

            preds.append(x + v * t2)

            # score = ((1-t)x0 - x_t) / t^2 * sigma^2

            # x0 = xt + t*v
            # score = (xt + t*v - t*xt - t^2*v - xt) / t^2 * sigma^2
            #       = v - xt - t*v / t * sigma^2
            #       = = (1-t)*v - xt / t * sigma^2

            s = ((1 - t2) * v - x) / t2 / self.cfg.prior_sigma**2

            noise = torch.randn_like(x)

            g = cfg.sde_weight * t2 * self.cfg.prior_sigma**2

            g *= sigmoid((t2 - cfg.sde_cutoff_time) / cfg.sde_cutoff_width)

            gamma = cfg.temp_factor
            dx = v * dt + g * s * dt + np.sqrt(2 * g * gamma * dt) * noise

            x = x + dx
            out.append(x)

        if return_traj:
            return torch.stack(out), torch.stack(preds)
        else:
            return x


class EDMDiffusion(Diffusion):

    def get_sigma(self, t):
        p = self.cfg.sched_p
        return (
            self.cfg.sigma_min ** (1/p) +
            t * (self.cfg.sigma_max ** (1/p) - self.cfg.sigma_min ** (1/p))
        ) ** p

    def sigma_to_t(self, sigma):
        p = self.cfg.sched_p
        num = sigma ** (1/p) - self.cfg.sigma_min ** (1/p)
        denom = self.cfg.sigma_max ** (1/p) - self.cfg.sigma_min ** (1/p)
        return num / denom
        
    def add_noise(self, pos, t, mask=None):

        sigma = self.get_sigma(t)[...,None]
        noise = torch.randn_like(pos) * sigma
        pos = masked_center(pos, mask)
        
        noisy = pos + noise
        target = pos

        return noisy, target

    def precondition(self, inp, t):
        sigma = self.get_sigma(t)[...,None]
        return inp / (self.cfg.data_sigma**2 + sigma**2) ** 0.5
        
    def postcondition(self, inp, out, t):
        sigma = self.get_sigma(t)[...,None]
        cskip = self.cfg.data_sigma ** 2 / (self.cfg.data_sigma**2 + sigma**2)
        cout = sigma * self.cfg.data_sigma / (self.cfg.data_sigma**2 + sigma**2)**0.5

        return cskip * inp + cout * out
        
    def compute_loss(self, pred, target, t, eps=1e-12):
        sigma = self.get_sigma(t)
        num = (sigma**2 + self.cfg.data_sigma**2)
        denom = (sigma * self.cfg.data_sigma) ** 2
        weight = num / (denom + eps)
        return weight * torch.square(pred - target).sum(-1)

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
            x = torch.randn(shape, device=device) * self.cfg.sigma_max

        out = [x]
        sched = self.get_sigma(np.linspace(1, 0, cfg.nsteps + 1))

        # dx = g(t) dw with g(t) = \sqrt{ (d/dt) sigma^2 } = \sqrt{ 2\dot\sigma \sigma}
        # with t = \sigma this is just \sqrt{2t}
        # hence the backward coefficient on the score should be g^2 = 2t
        preds = []
        
        for t2, t1 in zip(sched[:-1], sched[1:]):
            dt = t2 - t1
            g = np.sqrt(2 * t2)
            
            x0 = model(x, self.sigma_to_t(t2))

            preds.append(x0)
            s = (x0 - x) / t2**2 # score
            
            noise = torch.randn_like(x)
            gamma = cfg.temp_factor
            
            dx = g**2 * s * dt + g * gamma * np.sqrt(dt) * noise
            x = x + dx
            out.append(x)

        if return_traj:
            return torch.stack(out), torch.stack(preds)
        else:
            return x

