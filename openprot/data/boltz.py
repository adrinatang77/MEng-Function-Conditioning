import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset, OpenProtData
from ..utils import residue_constants as rc
from collections import defaultdict
from boltz.data.const import prot_token_to_letter, dna_token_to_letter, rna_token_to_letter
from ..utils.prot_utils import seqres_to_aatype
import pickle
from ..utils.structure import Structure

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

        assert self.cfg.mol_type == 'lig'
        print('Grouping dataframe...')
        self.clusters = []
        
        for _, sub_df in self.df.groupby('cluster_id'):
            self.clusters.append(sub_df)

            
    def __len__(self):
        return len(self.clusters)

    def unpack_chain(self, chain):

        if chain.mol_type == 3:
            L = len(chain.atoms)
        else:
            L = len(chain.residues)
            
        ones = np.ones(L, dtype=np.float32)

        if chain.mol_type == 3:
            return self.make_data(
                name='',
                seqres='*'*L,
                atom_num=chain.atoms['element'],
                mol_type=ones * chain.mol_type,
                seq_mask=ones,
                struct=chain.atoms['coords'],
                struct_mask=chain.atoms['is_present'],
                residx=chain.get_atom_residx(),
                chain=ones * chain.idx,
            )
        else:
            return self.make_data(
                name='',
                seqres=chain.get_seqres(),
                seq_mask=chain.get_seqres_mask(),
                struct=chain.get_central_atoms()['coords'],
                struct_mask=chain.get_central_atoms()['is_present'],
                residx=chain.residues['res_idx'],
                chain=ones*chain.idx,
            )
                

    def __getitem__(self, idx: int):
        
        clus = self.clusters[idx]
        name = np.random.choice(clus.index)
        if self.cfg.mol_type == 'prot':
            pdb_id, chain1 = self.df.loc[name].alt_name.split('_')
            chain2 = None
        else:    
            pdb_id, chain1, chain2 = self.df.loc[name].alt_name.split('_')
        
        struct = Structure.from_npz(f"{self.cfg.path}/structures/{pdb_id[1:3]}/{pdb_id}.npz")

        data = self.unpack_chain(struct.get_chain(int(chain1)))
        
        if chain2 is not None:
            data2 = self.unpack_chain(struct.get_chain(int(chain2)))
            data = OpenProtData.concat([data, data2])
            data['name'] = name
        return data
