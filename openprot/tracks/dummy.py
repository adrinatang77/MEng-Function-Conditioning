import torch
import torch.nn as nn

from .track import Track


class DummyTrack(Track):
    def __init__(self, cfg):
        self.cfg = cfg

    def tokenize(self, data, data_tok):
        # the dummy tokenizer just copies everything
        for key in data:
            data_tok[key] = data[key]

    def add_modules(self, model):
        model.dummy_embed = nn.Linear(3, model.cfg.dim)
        model.dummy_out = nn.Linear(model.cfg.dim, 3)

    def corrupt(self, batch, noisy_batch, target, logger):
        noisy_batch["trans"] = batch["trans"]
        target["dummy"] = batch["trans"]

    def embed(self, model, batch):
        x = model.dummy_embed(batch["trans"])
        c = torch.ones_like(x)
        return x, c

    def predict(self, model, out, readout):
        readout["dummy"] = model.dummy_out(out)

    def compute_loss(self, readout, target):
        return torch.square(readout["dummy"] - target["dummy"]).mean()
