import torch
import torch.nn as nn
import numpy as np

from .track import Track
from ..utils import residue_constants as rc

MASK_IDX = 21


class SequenceTrack(Track):

    def tokenize(self, data):
        data["aatype"] = np.array(
            [rc.restype_order.get(c, rc.unk_restype_index) for c in data["seqres"]]
        )

    def add_modules(self, model):
        model.seq_embed = nn.Embedding(22, model.cfg.dim)
        model.seq_out = nn.Linear(model.cfg.dim, 21)
        # model.seq_mask = nn.Parameter(torch.zeros(model.cfg.dim))

    def corrupt(self, batch, noisy_batch, target):
        if self.cfg.corrupt == "mask":
            tokens = batch["aatype"]

            mask = batch["seq_noise"].bool()
            
            rand = torch.rand(tokens.shape, device=tokens.device)
            randaa = torch.randint(0, 21, tokens.shape, device=tokens.device)

            inp = tokens
            inp = torch.where((rand < 0.8) & mask, MASK_IDX, inp)
            inp = torch.where((rand > 0.9) & mask, randaa, inp)

            noisy_batch["aatype"] = inp
            target["seq_supervise"] = (batch["seq_noise"] > 0) * batch["seq_mask"]

        else:
            raise Exception(
                f"SequenceTrack corrupt type {self.cfg.corrupt} not implemented"
            )

        target["aatype"] = batch["aatype"]
        self.logger.log("seq/toks", batch["seq_mask"].sum())

    def embed(self, model, batch):
        x = model.seq_embed(batch["aatype"])
        return x

    def predict(self, model, out, readout):
        readout["aatype"] = model.seq_out(out)

    def compute_loss(self, readout, target, pad_mask, eps=1e-6):
        loss = torch.nn.functional.cross_entropy(
            readout["aatype"].transpose(1, 2), target["aatype"], reduction="none"
        )

        mask = target["seq_supervise"]
        self.logger.log("seq/loss", loss, mask=mask)
        self.logger.log("seq/perplexity", loss, mask=mask, post=np.exp)
        self.logger.log("seq/toks_sup", mask.sum().item())
        return (loss * mask).sum() / pad_mask.sum()
        