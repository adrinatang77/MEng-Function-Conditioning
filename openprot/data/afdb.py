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
        if self.cfg.blacklist is not None:
            blacklist = pd.read_csv(
                self.cfg.blacklist,
                names=["query", "target", "qcov", "tcov", "evalue"],
                sep="\t",
            )
            self.blacklist = set(blacklist["target"])
        self.annotations = dict(np.load(self.cfg.annotations))
        self.idx = np.arange(len(self.db))
        self.idx = self.idx[self.annotations["lddt"] > self.cfg.lddt_thresh]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        name, pdb = self.db[self.idx[idx]]
        name = name.split(".")[0]
        if self.cfg.blacklist is not None and name in self.blacklist:
            return self[(idx + 1) % len(self)]

        prot = protein.from_pdb_string(pdb)
        seqres = "".join([rc.restypes_with_x[c] for c in prot.aatype])
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=np.ones(len(seqres)),
            atom37=prot.atom_positions.astype(np.float32),
            atom37_mask=prot.atom_mask.astype(np.float32),
        )
