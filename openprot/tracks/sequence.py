import torch
import torch.nn as nn
import numpy as np

from .track import Track
from ..utils import residue_constants as rc

MASK_IDX = 21


class SequenceTrack(Track):

    def tokenize(self, data, data_tok):
        data_tok["aatype"] = np.array(
            [rc.restype_order.get(c, rc.unk_restype_index) for c in data["seqres"]]
        )

    def add_modules(self, model):
        model.seq_embed = nn.Embedding(22, model.cfg.dim)
        model.seq_out = nn.Linear(model.cfg.dim, 21)
        # model.seq_mask = nn.Parameter(torch.zeros(model.cfg.dim))

    def corrupt(self, batch, noisy_batch, target):
        if self.cfg.corrupt == "mask":
            tokens = batch["aatype"]

            mask = torch.rand(tokens.shape, device=tokens.device) < self.cfg.mask_rate
            rand = torch.rand(tokens.shape, device=tokens.device)
            randaa = torch.randint(0, 21, tokens.shape, device=tokens.device)

            inp = tokens
            inp = torch.where((rand < 0.8) & mask, MASK_IDX, inp)
            inp = torch.where((rand > 0.9) & mask, randaa, inp)

            noisy_batch["aatype"] = inp
            target["aatype_mask"] = mask

        else:
            raise Exception(
                f"SequenceTrack corrupt type {self.cfg.corrupt} not implemented"
            )

        target["aatype"] = batch["aatype"]

    def embed(self, model, batch):
        x = model.seq_embed(batch["aatype"])
        return x

    def predict(self, model, out, readout):
        readout["aatype"] = model.seq_out(out)

    def compute_loss(self, readout, target, eps=1e-6):
        loss = torch.nn.functional.cross_entropy(
            readout["aatype"].transpose(1, 2), target["aatype"], reduction="none"
        )

        loss_mask = target["pad_mask"] * target["aatype_mask"].float()

        loss = (loss * loss_mask).sum(-1) / (loss_mask.sum(-1) + eps)

        self.logger.log("aatype_loss", loss)
        self.logger.log("aatype_perplexity", torch.exp(loss))
        return loss.mean()
