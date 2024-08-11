import numpy as np
import torch


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        L = 256
        return {
            "seq": np.zeros((256, 21), np.int32),
            "rots": np.zeros((256, 3, 3), np.float32),
            "trans": np.zeros((256, 3), np.float32),
        }
