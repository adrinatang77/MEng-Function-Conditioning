import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset
from ..utils import residue_constants as rc
from ..utils.prot_utils import aatype_to_seqres
import pickle
from omegaconf import OmegaConf
from multiflow.datasets import PdbDataset

class MultiflowDataset(OpenProtDataset):

    def setup(self):
        # self.df = pd.read_csv(f"{self.cfg.path}/metadata.csv", index_col='pdb_name')
        with open(self.cfg.config_path) as f:
            cfg = OmegaConf.load(f)
        self.dataset = PdbDataset(dataset_cfg=cfg.pdb_dataset, is_training=True, task='hallucination')
        self.clusters = list(self.dataset.csv.groupby('cluster'))

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        _, clu = self.clusters[idx]
        idx = np.random.choice(clu.index)
        item = self.dataset[idx]
        seqres = aatype_to_seqres(item['aatypes_1'])
        breakpoint()
        print(idx, item['pdb_name'])
        return self.make_data(
            name=item['pdb_name'],
            seqres=aatype_to_seqres(item['aatypes_1']),
            residx=item['res_idx'],
            seq_mask=item['res_mask'].numpy(),
            atom37=item['all_atom_positions'].float().numpy(),
            atom37_mask=item['all_atom_mask'].float().numpy(),
        )
        # name = self.df.index[idx]
        # with open(f"{self.cfg.path}/{name}.pkl", "rb") as f:
        #     data = pickle.load(f)

        # seqres = aatype_to_seqres(data['aatype'])
        # residx = np.arange(len(seqres), dtype=np.float32)
        # seq_mask = np.ones(len(seqres), dtype=np.float32)
        # seq_mask[[c not in rc.restype_order for c in seqres]] = 0

        # atom37 = data["atom_positions"].astype(np.float32)
        # atom37_mask = data["atom_mask"].astype(np.float32)

        # if self.cfg.struct_mask:
        #     mask = atom37_mask[:,1].astype(bool)
        #     seqres = "".join([
        #         seqres[i] for i, c in enumerate(mask) if c
        #     ])
        #     residx = residx[mask]
        #     seq_mask = seq_mask[mask]
        #     atom37 = atom37[mask]
        #     atom37_mask = atom37_mask[mask]

        # if len(seqres) == 0:
        #     return self[(idx+1) % len(self)]
        
        
        # return self.make_data(
        #     name=name,
        #     seqres=seqres,
        #     residx=residx,
        #     seq_mask=seq_mask,
        #     atom37=atom37,
        #     atom37_mask=atom37_mask,
        # )
