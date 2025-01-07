import torch
import numpy as np

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
