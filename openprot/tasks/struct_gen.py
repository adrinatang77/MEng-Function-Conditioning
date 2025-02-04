from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R


class StructureGeneration(OpenProtTask):

    def register_loss_masks(self):
        return ["/struct_gen"]

    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        

        rand = np.random.rand()
        if rand < self.cfg.struct_max_noise_prob:
            noise_level = 1.0
        elif rand < self.cfg.struct_max_noise_prob + self.cfg.struct_uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.struct_beta)

        L = len(data["seqres"])
        data["seq_noise"] = np.ones(L, dtype=np.float32)
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
        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["atom37_mask"][..., rc.atom_order["CA"], None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["atom37"] -= com

        if self.cfg.random_rot:
            randrot = R.random().as_matrix()
            data["atom37"] @= randrot.T

        if self.cfg.rots:
            data["rots_noise"] = data['struct_noise']

        data["/struct_gen"] = np.ones((), dtype=np.float32)

        return data
