import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset
from ..utils import residue_constants as rc
from ..utils.prot_utils import aatype_to_seqres
import pickle

class MultiflowDataset(OpenProtDataset):

    def setup(self):
        self.df = pd.read_csv(f"{self.cfg.path}/metadata.csv", index_col='pdb_name')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.index[idx]
        with open(f"{self.cfg.path}/{name}.pkl", "rb") as f:
            data = pickle.load(f)

        seqres = aatype_to_seqres(data['aatype'])
        seq_mask = np.ones(len(seqres), dtype=np.float32)
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=seq_mask,
            atom37=data["atom_positions"].astype(np.float32),
            atom37_mask=data["atom_mask"].astype(np.float32),
        )
