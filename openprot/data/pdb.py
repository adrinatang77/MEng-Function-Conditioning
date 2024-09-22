import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset


class PDBDataset(OpenProtDataset):

    def setup(self):
        self.df = pd.read_csv(f"{self.cfg.path}/pdb_chains.csv", index_col="name")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        name = self.df.index[idx]
        prot = dict(
            np.load(f"{self.cfg.path}/{name[1:3]}/{name}.npz", allow_pickle=True)
        )
        seqres = self.df.seqres[name]
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=np.ones(len(seqres)),
            atom37=prot["atom37"],
            atom37_mask=prot["atom37_mask"],
        )
