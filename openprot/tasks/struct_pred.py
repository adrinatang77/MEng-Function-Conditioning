from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc


class StructurePrediction(OpenProtTask):

    def register_loss_masks(self):
        return ['/struct_pred']
        
    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        ## noise EVERYTHING

        if np.random.rand() < self.cfg.max_noise_prob:
            noise_level = 1.0
        elif np.random.rand() < self.cfg.uniform_prob:
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

        data['/struct_pred'] = np.ones((), dtype=np.float32)
        
        return data
