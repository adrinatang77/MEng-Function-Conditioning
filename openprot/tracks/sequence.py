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
        model.seq_out = nn.Linear(model.cfg.dim, 21)
        model.seq_mask = nn.Parameter(torch.zeros(model.cfg.dim))

    def corrupt(self, batch, noisy_batch, target):
        if self.cfg.corrupt == "mask":
            rand = torch.rand(batch["aatype"].shape, device=batch["aatype"].device)
            mask = rand < self.cfg.mask_rate

            noisy_batch["aatype"] = torch.where(mask, 20, batch["aatype"])
            noisy_batch["aatype_mask"] = ~mask

        else:
            raise Exception(
                f"SequenceTrack corrupt type {self.cfg.corrupt} not implemented"
            )

        target["aatype"] = batch["aatype"]

    def embed(self, model, batch):
        x = model.seq_embed(batch["aatype"])
        x = torch.where(batch["aatype_mask"][..., None], x, model.seq_mask)
        return x

    def predict(self, model, out, readout):
        readout["aatype"] = model.seq_out(out)

    def compute_loss(self, readout, target):
        loss = torch.nn.functional.cross_entropy(
            readout["aatype"].transpose(1, 2), target["aatype"], reduction="none"
        )

        loss = (loss * target["pad_mask"]).sum(-1) / target["pad_mask"].sum(-1)
        self.logger.log("aatype_loss", loss)
        return loss
