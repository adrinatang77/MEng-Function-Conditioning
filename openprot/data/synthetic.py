import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset
from ..utils import residue_constants as rc
from ..utils.prot_utils import aatype_to_seqres
import pickle
from omegaconf import OmegaConf
from multiflow.datasets import PdbDataset

class SyntheticDataset(OpenProtDataset):

    def setup(self):
        self.df = pd.read_csv(f"{self.cfg.path}/metadata.csv", index_col='pdb_name')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.index[idx]
        with open(f"{self.cfg.path}/{name}.pkl", "rb") as f:
            data = pickle.load(f)

        seqres = aatype_to_seqres(data['aatype'])
        residx = np.arange(len(seqres), dtype=np.float32)
        seq_mask = np.ones(len(seqres), dtype=np.float32)
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0

        atom37 = data["atom_positions"].astype(np.float32)
        atom37_mask = data["atom_mask"].astype(np.float32)        
        
        return self.make_data(
            name=name,
            seqres=seqres,
            residx=residx,
            seq_mask=seq_mask,
            struct=atom37[:,1],
            struct_mask=atom37_mask[:,1],
        )
