import torch
import torch.nn as nn
import numpy as np

from .track import Track
from ..utils import residue_constants as rc


class SequenceTrack(Track):

    def tokenize(self, data, data_tok):
        data_tok["aatype"] = np.array(
            [rc.restype_order_with_x[c] for c in data["seqres"]]
        )

    def add_modules(self, model):
        model.seq_embed = nn.Embedding(21, model.cfg.dim)
        # model.seq_out = nn.Linear(model.cfg.dim, 21)

    def corrupt(self, batch, noisy_batch, target):
        noisy_batch["aatype"] = batch["aatype"]
        # target["aatype"] = batch["aatype"]

    def embed(self, model, batch):
        x = model.seq_embed(batch["aatype"])
        return x

    def predict(self, model, out, readout):
        pass

    def compute_loss(self, readout, target):
        return 0.0
