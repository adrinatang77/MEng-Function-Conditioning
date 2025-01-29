import torch
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class EDMDiffusionStepper:
    def __init__(self, cfg=None):
        self.cfg = cfg
        
    def set_step(self, batch, sched, extra={}):
        t, _ = sched['structure']
        batch["struct_noise"] = torch.ones_like(batch["struct_noise"]) * t
        
    def advance(self, batch, sched, out, extra={}):

        if 'traj' not in extra: extra['traj'] = []
        if 'preds' not in extra: extra['preds'] = []
            
        x = batch['struct']
        t2, t1 = sched['structure']
        
        dt = t2 - t1
        g = np.sqrt(2 * t2)

        x0 = out['trans'][-1]
        extra['preds'].append(x0)
        
        s = (x0 - x) / t2**2  # score
        noise = torch.randn_like(x)
        gamma = self.cfg.temp_factor

        dx = g**2 * s * dt + g * gamma * np.sqrt(dt) * noise
        
        batch['struct'] = x + dx
        extra['traj'].append(batch['struct'])

class GaussianFMStepper:
    def __init__(self, cfg=None):
        self.cfg = cfg
        
    def set_step(self, batch, sched, extra={}):
        t, _ = sched['structure']
        batch["struct_noise"] = torch.ones_like(batch["struct_noise"]) * t
        
    def advance(self, batch, sched, out, extra={}):

        if 'traj' not in extra: extra['traj'] = []
        if 'preds' not in extra: extra['preds'] = []

        x = batch['struct']
        t2, t1 = sched['structure']
        dt = t2 - t1
        
        x0 = out['trans'][-1]
        extra['preds'].append(x0)

        v = (x0 - x) / t2
        s = ((1 - t2) * v - x) / t2 / self.cfg.data_sigma**2

        noise = torch.randn_like(x)
        g = self.cfg.sde_weight * t2 * self.cfg.data_sigma**2  
        g *= sigmoid(t2 / 0.001) # temporary
        gamma = self.cfg.temp_factor
        dx = v * dt + g * s * dt + np.sqrt(2 * g * gamma * dt) * noise

        batch['struct'] = x + dx
        extra['traj'].append(batch['struct'])