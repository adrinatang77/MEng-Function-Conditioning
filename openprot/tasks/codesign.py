from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R


class Codesign(OpenProtTask):

    def register_loss_masks(self):
        return ["/codesign"]

    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)

        self.add_sequence_noise(data)
        self.add_structure_noise(data)
        data["/codesign"] = np.ones((), dtype=np.float32)
        
        return data
        

        

    def add_sequence_noise(self, data, eps=1e-6):
        rand = np.random.rand()
        if rand < self.cfg.seq_max_noise_prob:
            noise_level = 1.0
        elif rand < self.cfg.seq_max_noise_prob + self.cfg.seq_uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.seq_beta)

        mask = data["seq_mask"] * (data["mol_type"] == 0).astype(np.float32)
        
        L = len(data["seqres"])
        data["seq_noise"] = (np.random.rand(L) < noise_level).astype(np.float32)

        t_inv = min(100, 1/noise_level)
        t = noise_level

        ones = np.ones(L, dtype=np.float32)
        if self.cfg.seq_reweight == 'linear':
            data["seq_weight"] = ones * (1-t) * self.cfg.seq_weight
        elif self.cfg.seq_reweight == 'inverse':
            data["seq_weight"] = ones * t_inv * self.cfg.seq_weight
        else:
            assert not self.cfg.seq_reweight, 'reweight type not recognized'
            data["seq_weight"] = ones *  self.cfg.seq_weight
        
        
    def add_structure_noise(self, data, eps=1e-6):
        rand = np.random.rand()
        if rand < self.cfg.struct_max_noise_prob:
            noise_level = 1.0
        elif rand < self.cfg.struct_max_noise_prob + self.cfg.struct_uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.struct_beta)

        L = len(data["seqres"])

        #####
        if self.cfg.rescale_time:
            p = self.cfg.sched_p
            noise_level = (
                self.cfg.sigma_min ** (1 / p)
                + noise_level * (self.cfg.sigma_max ** (1 / p) - self.cfg.sigma_min ** (1 / p))
            ) ** p
        #####
        
        data["struct_noise"] = np.ones(L, dtype=np.float32) * noise_level
        data["struct_weight"] = np.ones(L, dtype=np.float32) * self.cfg.struct_weight

        # center the structures
        pos = data["struct"]
        mask = data["struct_mask"][..., None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["struct"] -= com

        if self.cfg.random_rot:
            randrot = R.random().as_matrix()
            data["struct"] @= randrot.T

    