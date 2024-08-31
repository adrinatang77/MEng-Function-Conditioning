import pytorch_lightning as pl
from ..utils.logger import Logger
import torch
from .model import OpenProtModel
from ..utils.misc_utils import autoimport


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

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-12,
            end_factor=1.0,
            total_iters=self.cfg.optimizer.scheduler.warmup_steps,
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.cfg.optimizer.scheduler.end_factor,
            total_iters=int(0.9 * self.cfg.optimizer.scheduler.total_steps),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[self.cfg.optimizer.scheduler.start_decay],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

        return optimizer


class OpenProtWrapper(Wrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = OpenProtModel(cfg.model)
        self.tracks = []
        for name in cfg.tracks:
            track = autoimport(name)(cfg.tracks[name], self._logger)
            track.add_modules(self.model)
            self.tracks.append(track)

    def get_lr(self):
        for param_group in self.optimizers().param_groups:
            return param_group["lr"]

    def general_step(self, batch):

        self._logger.log("toks", batch["pad_mask"].sum())

        ## corrupt all the tracks
        noisy_batch, target = {}, {}
        for track in self.tracks:
            track.corrupt(batch, noisy_batch, target)

        ## embed the tracks into an input and conditioning vector
        inp, cond = 0, 0
        for track in self.tracks:
            x = track.embed(self.model, noisy_batch)
            inp = inp + x

        ## run it thorugh the model
        out = self.model(inp, batch["pad_mask"])

        ## place the readouts in a dict
        readout = {}
        for track in self.tracks:
            track.predict(self.model, out, readout)

        ## compute the loss
        loss = 0
        for track in self.tracks:
            # pass in the batch because of the metadata
            loss_ = track.compute_loss(readout, target, batch["pad_mask"])
            loss = loss + track.cfg.loss_weight * loss_

        ## log some metrics
        self._logger.log("loss", loss)
        self._logger.log("lr", self.get_lr())
        self._logger.log("act_norm", torch.square(out).mean(-1), batch["pad_mask"])

        return loss
