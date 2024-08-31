from .task import Task
import numpy as np
from ..utils import residue_constants as rc


class StructurePrediction(Task):
    def prep_data(self, data, eps=1e-6):

        data["struct_mask"] = data["atom37_mask"][..., rc.atom_order["CA"]]

        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["struct_mask"][..., None] 

        # center the structures
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data['atom37'] -= com
        
        data["struct_noise"] = np.where(
            (pos[...,0] > 0) & data["struct_mask"].astype(bool),
            1.0,
            0.0
        ).astype(np.float32)
        
        return data
