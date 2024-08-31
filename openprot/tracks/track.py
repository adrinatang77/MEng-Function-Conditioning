from abc import abstractmethod

class OpenProtTrack:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    @abstractmethod
    def tokenize(self, data):
        NotImplemented

    @abstractmethod
    def add_modules(self, model):
        NotImplemented

    @abstractmethod
    def corrupt(self, batch, noisy_batch, target):
        NotImplemented

    @abstractmethod
    def embed(self, model, batch):
        NotImplemented

    @abstractmethod
    def predict(self, model, out, readout):
        NotImplemented

    @abstractmethod
    def compute_loss(self, readout, target, pad_mask):
        NotImplemented
