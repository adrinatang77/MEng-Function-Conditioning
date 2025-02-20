from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R

class InverseFolding(OpenProtTask):

    def register_loss_masks(self):
        return ["/inv_fold"]
        
    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        if np.random.rand() < self.cfg.uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.beta)

        
        L = len(data["seqres"])
        data["seq_noise"] = (np.random.rand(L) < noise_level).astype(np.float32)

        t_inv = min(100, 1/noise_level)
        t = noise_level

        ones = np.ones(L, dtype=np.float32)
        if self.cfg.reweight == 'linear':
            data["seq_weight"] = ones * (1-t) 
        elif self.cfg.reweight == 'inverse':
            data["seq_weight"] = ones * t_inv 
        else:
            assert not self.cfg.reweight, 'reweight type not recognized'
            data["seq_weight"] = ones 
        # center the structures
        pos = data["struct"]
        mask = data["struct_mask"][..., None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["struct"] -= com

        if self.cfg.random_rot:
            randrot = R.random().as_matrix()
            data["struct"] @= randrot.T

        data["/inv_fold"] = np.ones((), dtype=np.float32)
        
        return data