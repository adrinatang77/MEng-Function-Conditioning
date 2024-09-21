from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc


class StructurePrediction(OpenProtTask):
    def prep_data(self, data, eps=1e-6):

        data["struct_mask"] = data["atom37_mask"][..., rc.atom_order["CA"]]

        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["struct_mask"][..., None]

        # center the structures
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["atom37"] -= com

        ## noise EVERYTHING
        data["struct_noise"] = np.copy(data["struct_mask"])

        return data
