from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R


class CodesignTask(OpenProtTask):

        
    def add_sequence_noise(self, data, eps=1e-6, noise_level=None, sup=False):
        if noise_level is None:
            rand = np.random.rand()
            if rand < self.cfg.seq.max_noise_prob:
                noise_level = 1.0
            elif rand < self.cfg.seq.max_noise_prob + self.cfg.seq.uniform_prob:
                noise_level = np.random.rand()
            else:
                noise_level = np.random.beta(*self.cfg.seq.beta)

        L = len(data["seqres"])
        data["seq_noise"] = np.ones(L, dtype=np.float32) * noise_level
        if sup:
            data["seq_weight"] = np.ones(L, dtype=np.float32)

        return noise_level
        
    def add_structure_noise(self, data, eps=1e-6, noise_level=None, sup=False):
        if noise_level is None:
            
            rand = np.random.rand()
            if rand < self.cfg.struct.max_noise_prob:
                noise_level = 1.0
            elif rand < self.cfg.struct.max_noise_prob + self.cfg.struct.uniform_prob:
                noise_level = np.random.rand()
            else:
                noise_level = np.random.beta(*self.cfg.struct.beta)

        L = len(data["seqres"])

        #####
        p = self.cfg.edm.sched_p
        sigma = (
            self.cfg.edm.sigma_min ** (1 / p)
            + noise_level * (self.cfg.edm.sigma_max ** (1 / p) - self.cfg.edm.sigma_min ** (1 / p))
        ) ** p
        #####
        
        data["struct_noise"] = np.ones(L, dtype=np.float32) * sigma
        if sup:
            data["struct_weight"] = np.ones(L, dtype=np.float32)

        # center the structures
        pos = data["struct"]
        mask = data["struct_mask"][..., None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["struct"] -= com

        randrot = R.random().as_matrix()
        data["struct"] @= randrot.T

        return noise_level

class Codesign(CodesignTask):
    def register_loss_masks(self):
        return ["/codesign"]

    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)

        self.add_sequence_noise(data, sup=True)
        self.add_structure_noise(data, sup=True)
        data["/codesign"] = np.ones((), dtype=np.float32)
        
        return data
    