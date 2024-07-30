class Track:
    def __init__(self, cfg):
        self.cfg = cfg

    def tokenize(self, batch):
        ## this should be called in each DATASET object
        ## at the end of all the tokenize calls, the batch
        ## must be of type Dict[str, np.ndarray]

    def add_modules(self, model):
        pass
    
    def corrupt(self, batch, logger):
        pass
    
    def embed(self, batch):
        pass

    def predict(self, batch):
        pass
        
    def compute_loss(self, cfg, logger):
        pass
        