from ..model.model import OpenProtModel
import torch
from ..utils.misc_utils import autoimport


class OpenProtTrackManager(dict):
    def __init__(self, cfg):
        self.cfg = cfg
        for name in cfg:
            track = autoimport(f"openprot.tracks.{name}")(cfg[name])
            self[name] = track

    def add_modules(self, model: OpenProtModel):
        for track in self.values():
            track.add_modules(model)

    def tokenize(self, data: dict):
        for track in self.values():
            track.tokenize(data)

    def embed(self, model: OpenProtModel, batch: dict):
        inp = batch.copy('name', 'pad_mask')
        inp['x'] = 0
        for track in self.values():
            track.embed(model, batch, inp)
        return inp

    def readout(self, model: OpenProtModel, out: torch.Tensor):
        readout = {}
        for track in self.values():
            track.predict(model, out, readout)
        return readout

    def corrupt(self, batch: dict, logger=None):
        noisy_batch = batch.copy('name', 'pad_mask')
        target = batch.copy('name', 'pad_mask')
        
        for track in self.values():
            track.corrupt(batch, noisy_batch, target, logger=logger)

        return noisy_batch, target

    def compute_loss(self, readout: dict, target: dict, logger=None):
        loss = 0
        for track in self.values():
            loss_ = track.compute_loss(readout, target, logger=logger)
            loss = loss + track.cfg.loss_weight * loss_
        return loss
