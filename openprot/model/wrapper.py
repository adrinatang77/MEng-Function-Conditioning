import pytorch_lightning as pl
from ..utils.logger import Logger
import torch
from .model import OpenProtModel
from abc import abstractmethod
from ..utils.misc_utils import autoimport
import os
from ..tracks.manager import OpenProtTrackManager

# from ..evals.manager import OpenProtEvalManager


class Wrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg
        self._logger = Logger(cfg.logger)

    def training_step(self, batch, batch_idx):
        self._logger.prefix = "train"
        out = self.general_step(batch)
        self._logger.step(self.trainer, "train")
        return out

    def validation_step(self, batch, batch_idx):
        self._logger.prefix = "val"
        # self.general_step(batch)
        self.validation_step_extra(batch, batch_idx)
        self._logger.step(self.trainer, "val")

    @abstractmethod
    def general_step(self, batch):
        NotImplemented

    def validation_step_extra(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="train")

    def on_validation_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="val")

    # uncomment this to debug
    def on_before_optimizer_step(self, optimizer):
        quit = False
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is None:
                print(name, "has no grad")
                quit = True

        if quit:
            exit()

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
    def __init__(self, cfg, tracks: OpenProtTrackManager, evals: dict):
        super().__init__(cfg)
        self.model = OpenProtModel(cfg.model)
        self.tracks = tracks

        tracks.add_modules(self.model)

        self.evals = evals

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def get_lr(self):
        for param_group in self.optimizers().param_groups:
            return param_group["lr"]

    def forward(self, noisy_batch):

        ## embed the tracks into an input dict
        inp = self.tracks.embed(self.model, noisy_batch)

        ## run it thorugh the model
        out = self.model(inp)

        ## place the readouts in a dict
        readout = self.tracks.readout(self.model, inp, out)

        return out, readout

    def general_step(self, batch):

        self._logger.register_masks(batch)
        self._logger.masked_log("toks", batch["pad_mask"], sum=True)

        ## corrupt all the tracks
        noisy_batch, target = self.tracks.corrupt(batch, logger=self._logger)

        out, readout = self.forward(noisy_batch)

        ## compute the loss
        loss = self.tracks.compute_loss(readout, target, logger=self._logger)

        ## log some metrics
        self._logger.masked_log("loss", loss, batch["pad_mask"])
        self._logger.log("lr", self.get_lr())

        self._logger.clear_masks()
        
        return (loss * batch["pad_mask"]).sum() / batch['pad_mask'].sum()

    def on_validation_epoch_end(self):
        savedir = f'{os.environ["MODEL_DIR"]}/eval_step{self.trainer.global_step}'
        for name, eval_ in self.evals.items():
            eval_.compute_metrics(
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                device=self.device,
                savedir=f"{savedir}/{name}",
                logger=self._logger,
            )
        super().on_validation_epoch_end()

    def validation_step_extra(self, batch, batch_idx):
        name = batch["eval"][0]
        savedir = (
            f'{os.environ["MODEL_DIR"]}/eval_step{self.trainer.global_step}/{name}'
        )
        self.evals[name].run_batch(self, batch, savedir=savedir, logger=self._logger)
