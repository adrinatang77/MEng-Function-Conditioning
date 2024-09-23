import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset

class PDBDataset(OpenProtDataset):

    def setup(self):
        self.df = pd.read_csv(f"{self.cfg.path}/pdb_chains.csv", index_col="name")

        if self.cfg.cutoff is not None:
            self.df = self.df[self.df.release_date < self.cfg.cutoff]
        
        if self.cfg.clusters is not None:
            self.clusters = []
            with open(self.cfg.clusters) as f:
                for line in f:
                    clus = []
                    names = line.split()
                    for name in names:
                        if name in self.df.index: clus.append(name)
                    if len(clus) > 0:
                        self.clusters.append(clus)

    def __len__(self):
        if self.cfg.clusters:
            return len(self.clusters)
        else:
            return len(self.df)

    def __getitem__(self, idx: int):
        if self.cfg.clusters:
            clus = self.clusters[idx]
            name = np.random.choice(clus)
        else:
            name = self.df.index[idx]
        prot = dict(
            np.load(f"{self.cfg.path}/{name[1:3]}/{name}.npz", allow_pickle=True)
        )
        seqres = self.df.seqres[name]
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=np.ones(len(seqres)),
            atom37=prot["all_atom_positions"],
            atom37_mask=prot["all_atom_mask"],
        )
