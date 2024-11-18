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
            noise_level = 1.0
        elif rand < self.cfg.max_noise_prob + self.cfg.uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.beta)

        L = len(data["seqres"])
        data["trans_noise"] = np.ones(L, dtype=np.float32) * noise_level
        data["rots_noise"] = np.ones(L, dtype=np.float32) * noise_level

        # data["torsion_noise"] = np.ones(len(data["seqres"]))

        # center the structures
        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["atom37_mask"][..., rc.atom_order["CA"], None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["atom37"] -= com

        if self.cfg.random_rot:
            randrot = R.random().as_matrix()
            data["atom37"] @= randrot.T

        data["/struct_pred"] = np.ones((), dtype=np.float32)
        if noise_level == 1.0:
            data["/struct_pred/t1"] = np.ones((), dtype=np.float32)
        else:
            data["/struct_pred/t1"] = np.zeros((), dtype=np.float32)

        return data
