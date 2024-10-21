import torch
import numpy as np
import pandas as pd
import foldcomp
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset


class UnirefDataset(OpenProtDataset):
    def setup(self):
        self.db = open(self.cfg.path)
        self.index = np.load(self.cfg.index)

    def __len__(self):
        return len(self.index) - 1  # unfortunately we have to skip the last one

    def __getitem__(self, idx: int):
        start = self.index[idx]
        end = self.index[idx + 1]
        self.db.seek(start)
        item = self.db.read(end - start)
        lines = item.split("\n")
        header, lines = lines[0], lines[1:]
        seqres = "".join(lines)
        seq_mask = np.ones(len(seqres))
        seq_mask[seqres == 'X'] = 0
        return self.make_data(
            name=header.split()[0], seqres=seqres, seq_mask=seq_mask
        )
