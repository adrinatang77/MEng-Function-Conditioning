from abc import abstractmethod
import torch
from ..model.model import OpenProtModel


class OpenProtTrack:
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def tokenize(self, data: dict):
        NotImplemented

    @abstractmethod
    def add_modules(self, model: OpenProtModel):
        NotImplemented

    @abstractmethod
    def corrupt(self, batch: dict, noisy_batch: dict, target: dict, logger=None):
        NotImplemented

    @abstractmethod
    def embed(self, model: OpenProtModel, batch: dict):
        NotImplemented

    @abstractmethod
    def predict(self, model: OpenProtModel, out: torch.Tensor, readout: dict):
        NotImplemented

    @abstractmethod
    def compute_loss(self, readout: dict, target: dict, logger=None):
        NotImplemented
