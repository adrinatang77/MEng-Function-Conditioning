import torch
import numpy as np
import pandas as pd
import foldcomp
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset

class AFDBDataset(OpenProtDataset):

    def setup(self):
        self.db = foldcomp.open(self.cfg.path)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        name, pdb = self.db[idx]
        prot = protein.from_pdb_string(pdb)
        seqres = "".join([rc.restypes_with_x[c] for c in prot.aatype])
        return self.make_data(
            name=name[:-4],
            seqres=seqres,
            seq_mask=np.ones(len(seqres)),
            atom37=prot.atom_positions.astype(np.float32),
            atom37_mask=prot.atom_mask.astype(np.float32),
        )
