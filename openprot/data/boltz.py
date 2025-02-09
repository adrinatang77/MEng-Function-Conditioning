import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset
from ..utils import residue_constants as rc
from collections import defaultdict
from boltz.data.const import prot_token_to_letter
from openprot.utils.prot_utils import seqres_to_aatype

class BoltzDataset(OpenProtDataset):

    def setup(self):
        print('Loading dataframe...')
        self.df = pd.read_csv(f"{self.cfg.path}/manifest.csv", index_col="name")

        print('Filtering dataframe...')
        if self.cfg.cutoff is not None:
            self.df = self.df[self.df.release_date < self.cfg.cutoff]

        self.df = self.df[self.df.n_prot > 0]
        if self.cfg.mol_type == 'prot':
            self.df = self.df[(self.df.n_prot == 1) & (self.df.n_nuc == 0) & (self.df.n_ligand == 0)]
        if self.cfg.mol_type == 'ppi':
            self.df = self.df[self.df.n_prot == 2]
        if self.cfg.mol_type == 'nuc':
            self.df = self.df[self.df.n_nuc == 1]
        if self.cfg.mol_type == 'lig':
            self.df = self.df[self.df.n_ligand == 1]
            
        print('Grouping dataframe...')
        self.clusters = []
        
        for _, sub_df in self.df.groupby('cluster_id'):
            self.clusters.append(sub_df)
            
    def __len__(self):
        return len(self.clusters)
        
    def __getitem__(self, idx: int):
        
        clus = self.clusters[idx]
        name = np.random.choice(clus.index)
        if self.cfg.mol_type == 'prot':
            pdb_id, chain1 = self.df.loc[name].alt_name.split('_')
            chain2 = None
        else:    
            pdb_id, chain1, chain2 = self.df.loc[name].alt_name.split('_')
        

        npz = np.load(f"{self.cfg.path}/structures/{pdb_id[1:3]}/{pdb_id}.npz")

        def unpack_chain(chain, chain_idx=0):
            start = chain['res_idx']
            end = start + chain['res_num']
            resis = npz['residues'][start:end]
            coords = npz['atoms'][resis['atom_center']]['coords']
            mask = npz['atoms'][resis['atom_center']]['is_present']
            residx = resis['res_idx']
            
            L = chain['res_num']
            ones = np.ones(L, dtype=np.float32)
            if chain['mol_type'] == 0: # prot
                seqres = ''.join([prot_token_to_letter.get(c, 'X') for c in resis['name']])
                seq_mask = np.ones(len(seqres), dtype=np.float32)
                seq_mask[[c not in rc.restype_order for c in seqres]] = 0
            
            return {
                'seqres': seqres,
                'mol_type': ones * chain['mol_type'],
                'seq_mask': seq_mask,
                'struct': coords,
                'struct_mask': mask,
                # 'ref_conf': None,
                'residx': residx,
                'chain': ones * chain_idx,
            }

        
        data = unpack_chain(npz['chains'][int(chain1)])
        if chain2 is not None:
            data2 = unpack_chain(npz['chains'][int(chain2)], 1)
            for key in data2:
                if key == 'seqres':
                    data['seqres'] += data2['seqres']
                else:
                    data[key] = np.concatenate([data[key], data2[key]])

        return self.make_data(name=name, **data)
