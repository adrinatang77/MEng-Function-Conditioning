from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R


class StructurePrediction(OpenProtTask):

    def register_loss_masks(self):
        return ["/struct_pred", "/struct_pred/t1"]

    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        rand = np.random.rand()
        if rand < self.cfg.max_noise_prob:
            noise_level = self.cfg.sigma_max + 1 # to ensure masking
        else:
            if rand < self.cfg.max_noise_prob + self.cfg.uniform_prob:
                t = np.random.rand()
            else:
                t = np.random.beta(*self.cfg.beta)

            ####
            p = self.cfg.sched_p
            noise_level = (
                self.cfg.sigma_min ** (1 / p)
                + t * (self.cfg.sigma_max ** (1 / p) - self.cfg.sigma_min ** (1 / p))
            ) ** p
            ####

        L = len(data["seqres"])
        data["struct_noise"] = np.ones(L, dtype=np.float32) * noise_level
        data["struct_weight"] = np.ones(L, dtype=np.float32) * self.cfg.weight
        
        # center the structures
        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["atom37_mask"][..., rc.atom_order["CA"], None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["atom37"] -= com

        if self.cfg.random_rot:
            randrot = R.random().as_matrix()
            data["atom37"] @= randrot.T

        data["/struct_pred"] = np.ones((), dtype=np.float32)
        if noise_level == self.cfg.sigma_max:
            data["/struct_pred/t1"] = np.ones((), dtype=np.float32)
        else:
            data["/struct_pred/t1"] = np.zeros((), dtype=np.float32)

        return data
