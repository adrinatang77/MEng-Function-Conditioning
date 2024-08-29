import torch
import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class OpenProtDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def shuffle(self):
        self.shuffled_idx = np.arange(len(self))
        with temp_seed(self.cfg.seed):
            np.random.shuffle(self.shuffled_idx)

    def get_shuffled(self, idx):
        return self[self.shuffled_idx[idx]]