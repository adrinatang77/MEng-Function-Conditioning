import torch
import numpy as np
import pandas as pd
# import foldcomp
import os
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset

import threading

lock = threading.Lock()
"""From Kevin
class UniRefDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_len=2048,
    ):
        self.data_dir = data_dir
        self.split = split
        metadata = np.load(os.path.join(self.data_dir, "lengths_and_offsets.npz"))
        self.offsets = metadata["seq_offsets"]
        with open(os.path.join(data_dir, "splits.json"), "r") as f:
            self.indices = json.load(f)[self.split]
        log.info(f"Dataset size: {len(self.indices)}")
        self.metadata_lens = metadata["ells"][self.indices]
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def get_metadata_lens(self):
        return self.metadata_lens

    def __getitem__(self, idx):
        idx = self.indices[idx]
        offset = self.offsets[idx]
        with open(os.path.join(self.data_dir, "consensus.fasta")) as f:
            f.seek(offset)
            consensus = f.readline()[:-1]
        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        consensus = consensus[start:stop]
        return consensus
"""

class UnirefDataset(OpenProtDataset):
    def setup(self):
        self.db = open(self.cfg.path)
        self.index = np.load(self.cfg.index)
        self.need_setup = True

    def actual_setup(self):
        self.db = open(self.cfg.path)
        self.index = np.load(self.cfg.index)
        self.need_setup = False

    def __len__(self):
        return len(self.index) - 1  # unfortunately we have to skip the last one

    def __getitem__(self, idx: int):
        if self.need_setup:
            self.actual_setup()

        start = self.index[idx]
        end = self.index[idx + 1]
        with lock:
            self.db.seek(start)
            item = self.db.read(end - start)
        lines = item.split("\n")
        header, lines = lines[0], lines[1:]
        seqres = "".join(lines)
        seqres = "[" + seqres + "]"
        
        seq_mask = np.ones(len(seqres), dtype=np.float32)
    
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        name = header if len(header.split()) == 0 else header.split()[0]
        
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=seq_mask,
        )


class KevinUnirefDataset(OpenProtDataset):
    def setup(self):
        self.path = self.cfg.path
        metadata = np.load(self.cfg.index)
        self.offsets = metadata["seq_offsets"]
        
    def __len__(self):
        return len(self.offsets) - 1  # unfortunately we have to skip the last one

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        with open(self.cfg.path) as f:
            f.seek(offset)
            consensus = f.readline()[:-1]
        if len(consensus) - 1024 > 0:
            start = np.random.choice(len(consensus) - 1024)
            stop = start + 1024
        else:
            start = 0
            stop = len(consensus)
        seqres = consensus[start:stop]
        seqres = "[" + seqres + "]"
        
        seq_mask = np.ones(len(seqres), dtype=np.float32)
    
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        name = "" # header if len(header.split()) == 0 else header.split()[0]
        
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=seq_mask,
        )
