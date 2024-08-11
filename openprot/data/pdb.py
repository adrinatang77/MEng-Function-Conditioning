import torch
import numpy as np
import pandas as pd


class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.df = pd.read_csv(f"{cfg.path}/pdb_chains.csv", index_col="name")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.index[idx]
        prot = dict(
            np.load(f"{self.cfg.path}/{name[1:3]}/{name}.npz", allow_pickle=True)
        )
        return {
            "seqres": self.df.seqres[name],
            "atom37": prot["all_atom_positions"],
            "atom37_mask": prot["all_atom_mask"],
        }
