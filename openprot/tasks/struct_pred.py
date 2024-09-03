from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc


class StructurePrediction(OpenProtTask):
    def prep_data(self, data, eps=1e-6):

        data["struct_mask"] = data["atom37_mask"][..., rc.atom_order["CA"]]

        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["struct_mask"][..., None]

        if data["struct_mask"].sum() < 3:
            return data  # don't noise or supervise anything

        sel = np.random.choice(data["struct_mask"].nonzero()[0], size=3, replace=False)
        # center the structures
        # com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        com = pos[sel].mean(0)
        data["atom37"] -= com

        """
        data["struct_noise"] = np.where(
            (pos[..., 0] > 0) & data["struct_mask"].astype(bool), 1.0, 0.0
        ).astype(np.float32)
        """
        data["struct_noise"] = np.copy(data["struct_mask"])
        data["struct_noise"][sel] = 0.0

        return data
