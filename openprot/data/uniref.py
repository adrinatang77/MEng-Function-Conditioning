import torch
import numpy as np
import pandas as pd
import foldcomp
from ..utils import protein
from ..utils import residue_constants as rc

class UnirefDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.db = open(cfg.path)
        self.index = np.load(cfg.index)

    def __len__(self):
        return len(self.index) - 1  # unfortunately we have to skip the last one

    def __getitem__(self, idx):
        start = self.index[idx]
        end = self.index[idx + 1]
        self.db.seek(start)
        item = self.db.read(end - start)
        lines = item.split("\n")
        header, lines = lines[0], lines[1:]
        seqres = "".join(lines)
        return {
            "seqres": seqres,
            "atom37": np.zeros((len(seqres), 37, 3), np.float32),
            "atom37_mask": np.zeros((len(seqres), 37), np.float32),
        }