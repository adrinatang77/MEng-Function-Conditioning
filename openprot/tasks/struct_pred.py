from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc


class StructurePrediction(OpenProtTask):
    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        ## noise EVERYTHING
        data["trans_noise"] = np.ones(len(data["seqres"]), dtype=np.float32)
        data["rots_noise"] = np.ones(len(data["seqres"]), dtype=np.float32)
        # data["torsion_noise"] = np.ones(len(data["seqres"]))

        # center the structures
        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["atom37_mask"][..., rc.atom_order["CA"], None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["atom37"] -= com

        return data
