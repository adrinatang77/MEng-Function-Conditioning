import torch
import torch.nn as nn
from .track import OpenProtTrack

class FunctionTrack(OpenProtTrack):

    def tokenize(self, data):
        NotImplemented

    def add_modules(self, model):
        NotImplemented

    def corrupt(self, batch, noisy_batch, target, logger=None):
        NotImplemented

    def embed(self, model, batch):
        NotImplemented

    def predict(self, model, out, readout):
        NotImplemented

    def compute_loss(self, readout, target, logger=None):
        NotImplemented
