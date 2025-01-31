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

        
        L = len(data["seqres"])
        data["seq_noise"] = (np.random.rand(L) < noise_level).astype(np.float32)
        t_inv = data["seq_mask"].sum() / (eps + (data["seq_mask"] * data["seq_noise"]).sum()) 
        t = (data["seq_mask"] * data["seq_noise"]).sum()  / (eps + data["seq_mask"].sum()) 

        if self.cfg.seq_reweight:
            data["seq_weight"] = np.ones(L, dtype=np.float32) * (1-noise_level) * self.cfg.seq_weight
        else:
            data["seq_weight"] = np.ones(L, dtype=np.float32) *  self.cfg.seq_weight
        
        
    def add_structure_noise(self, data, eps=1e-6):
        rand = np.random.rand()
        if rand < self.cfg.struct_max_noise_prob:
            noise_level = 1.0
        elif rand < self.cfg.struct_max_noise_prob + self.cfg.struct_uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.struct_beta)

        L = len(data["seqres"])
        
        data["struct_noise"] = np.ones(L, dtype=np.float32) * noise_level
        data["struct_weight"] = np.ones(L, dtype=np.float32) * self.cfg.struct_weight

        # center the structures
        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["atom37_mask"][..., rc.atom_order["CA"], None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["atom37"] -= com

        if self.cfg.random_rot:
            randrot = R.random().as_matrix()
            data["atom37"] @= randrot.T

        if self.cfg.rots:
            data["rots_noise"] = data['struct_noise']

    