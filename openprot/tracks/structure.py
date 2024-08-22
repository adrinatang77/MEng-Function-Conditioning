import torch
import torch.nn as nn
from .track import Track
from ..utils.geometry import atom37_to_frames, atom37_to_torsions
from ..utils import residue_constants as rc
import numpy as np


class StructureTrack(Track):

    def tokenize(self, data, data_tok):

        frames = atom37_to_frames(data["atom37"])
        data_tok["trans"] = frames._trans
        data_tok["rots"] = frames._rots._rot_mats
        aatype = np.array([rc.restype_order_with_x[c] for c in data["seqres"]])
        torsions, torsion_mask = atom37_to_torsions(data["atom37"], aatype)
        data_tok["torsions"] = torsions
        data_tok["torsion_mask"] = torsion_mask

    def add_modules(self, model):
        model.dummy_embed = nn.Linear(3, model.cfg.dim)
        model.dummy_out = nn.Linear(model.cfg.dim, 3)

    def corrupt(self, batch, noisy_batch, target):
        noisy_batch["trans"] = batch["trans"]
        target["dummy"] = batch["trans"]

    def embed(self, model, batch):
        x = model.dummy_embed(batch["trans"])
        return x

    def predict(self, model, out, readout):
        readout["dummy"] = model.dummy_out(out)

    def compute_loss(self, readout, target):
        return torch.square(readout["dummy"] - target["dummy"]).mean()
