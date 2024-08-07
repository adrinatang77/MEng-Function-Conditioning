import pytorch_lightning as pl
import torch

from .. import tracks
from ..utils.logging import Logger
from .model import OpenProtModel


class Wrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self._logger = Logger(cfg.logger)

    def training_step(self, batch, batch_idx):
        self._logger.prefix = "train"
        out = self.general_step(batch)
        self._logger.step(self.trainer, "train")
        return out

    def validation_step(self, batch, batch_idx):
        self._logger.prefix = "val"
        self.general_step(batch)
        self.validation_step_extra(batch, batch_idx)
        self._logger.step(self.trainer, "val")

    def general_step(self, batch):
        pass

    def validation_step_extra(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="train")

    def on_validation_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="val")

    def configure_optimizers(self):
        cls = getattr(torch.optim, self.cfg.optimizer.type)
        optimizer = cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.optimizer.lr,
        )
        return optimizer


class OpenProtWrapper(Wrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = OpenProtModel(cfg.model)
        self.tracks = []
        for name in cfg.tracks:
            track = getattr(tracks, name)(cfg.tracks[name])
            track.add_modules(self.model)
            self.tracks.append(track)

    def general_step(self, batch):
        ## corrupt all the tracks
        noisy_batch, target = {}, {}
        target = {}
        for track in self.tracks:
            track.corrupt(batch, noisy_batch, target, self._logger)

        ## embed the tracks into an input and conditioning vector
        inp, cond = 0, 0
        for track in self.tracks:
            x, c = track.embed(self.model, noisy_batch)
            inp = inp + x
            cond = cond + c

        ## run it thorugh the model
        out = self.model(inp, cond)

        ## place the readouts in a dict
        readout = {}
        for track in self.tracks:
            track.predict(self.model, out, readout)

        ## compute the loss
        loss = 0
        for track in self.tracks:
            loss_ = track.compute_loss(readout, target)
            loss = loss + loss_
        self._logger.log("loss", loss)
        return loss
