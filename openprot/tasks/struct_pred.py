from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc


class StructurePrediction(OpenProtTask):
    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        # TODO: distinguish between frames and torsions

        ## noise EVERYTHING
        data["struct_mask"] = data["atom37_mask"][..., rc.atom_order["CA"]]
        data["struct_noise"] = np.copy(data["struct_mask"])

        # center the structures
        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["struct_mask"][..., None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["atom37"] -= com

        return data
