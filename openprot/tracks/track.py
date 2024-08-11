class Track:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    def tokenize(self, batch):
        NotImplemented

    def add_modules(self, model):
        NotImplemented

    def corrupt(self, batch):
        NotImplemented

    def embed(self, batch):
        NotImplemented

    def predict(self, batch):
        NotImplemented

    def compute_loss(self, cfg, logger):
        pass
