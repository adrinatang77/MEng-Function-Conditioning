import torch, time
import pytorch_lightning as pl
from collections import defaultdict
from ..utils.logging import Logger

from .model import OpenProtModel

class Wrapper(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self._logger = Logger(cfg.logger)
        
    def training_step(self, batch, batch_idx):
        self._logger.prefix = 'train'
        out = self.general_step(batch)
        self._logger.step(self.trainer)
        return out

    def validation_step(self, batch, batch_idx):
        self._logger.prefix = 'val'
        self.general_step(batch)
        self.validation_step_extra(batch, batch_idx)
        self._logger.step(self.trainer)
        
    def general_step(self, batch):
        pass

    def validation_step_extra(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self._logger.epoch_end(self.trainer)

    def on_validation_epoch_end(self):
        self._logger.epoch_end(self.trainer)
    
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

    def general_step(self, batch):
        out = self.model(batch)
        loss = torch.square(out - batch['b']).mean()
        self._logger.log('loss', loss)
        return loss
                

        
        
    
    