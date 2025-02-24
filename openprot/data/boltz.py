import torch
import numpy as np
import pandas as pd
from .data import OpenProtDataset
from ..utils import residue_constants as rc
from collections import defaultdict
from boltz.data.const import prot_token_to_letter, dna_token_to_letter, rna_token_to_letter
from ..utils.prot_utils import seqres_to_aatype
import pickle

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

        # print('Loading CCD')
        # with open('/data/cb/scratch/datasets/boltz/ccd.pkl', 'rb') as f:
        #     self.ccd = pickle.load(f)
            
    def __len__(self):
        return len(self.clusters)

    def unpack_chain(self, npz, idx):
        chain = npz['chains'][idx]
        start = chain['res_idx']
        end = start + chain['res_num']
        resis = npz['residues'][start:end]

        astart = chain['atom_idx']
        aend = astart + chain['atom_num']
        atoms = npz['atoms'][astart:aend]
        
        coords = npz['atoms'][resis['atom_center']]['coords']
        mask = npz['atoms'][resis['atom_center']]['is_present']
        residx = resis['res_idx']

        if chain['mol_type'] == 3:
            L = chain['atom_num']
        else:
            L = chain['res_num']
            
        ones = np.ones(L, dtype=np.float32)
        # defaults
        atom_num = ones * 0.0
        seq_mask = np.copy(ones)
        ref_conf = np.zeros((L, 3), dtype=np.float32)
        
        if chain['mol_type'] == 0: # prot
            seqres = ''.join([prot_token_to_letter.get(c, 'X') for c in resis['name']])
            seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        
        elif chain['mol_type'] == 1: # DNA
            seqres = ''.join([dna_token_to_letter.get(c, 'N') for c in resis['name']])
            seq_mask[[c == 'N' for c in seqres]] = 0
        elif chain['mol_type'] == 2: # RNA
            seqres = ''.join([rna_token_to_letter.get(c, 'N') for c in resis['name']])
            seq_mask[[c == 'N' for c in seqres]] = 0
            
        elif chain['mol_type'] == 3: # ligand
            atom_num = atoms['element']
            seqres = '*'*len(atoms)
            coords = atoms['coords']
            mask = atoms['is_present']
            if np.unique(resis['res_idx']).size > 1:
                residx = resis['atom_idx'] - astart
                residx = (np.arange(len(atoms)) >= residx[:,None]).sum(0)
                residx = resis['res_idx'][residx-1]
            else:
                residx = ones * 0.0
            ref_conf = atoms['conformer']
            
        return {
            'seqres': seqres,
            'atom_num': atom_num,
            'mol_type': ones * chain['mol_type'],
            'seq_mask': seq_mask,
            'struct': coords,
            'struct_mask': mask,
            'ref_conf': ref_conf,
            'residx': residx,
            'chain': ones * idx,
        }
        
    def __getitem__(self, idx: int):
        
        clus = self.clusters[idx]
        name = np.random.choice(clus.index)
        if self.cfg.mol_type == 'prot':
            pdb_id, chain1 = self.df.loc[name].alt_name.split('_')
            chain2 = None
        else:    
            pdb_id, chain1, chain2 = self.df.loc[name].alt_name.split('_')
        

        npz = np.load(f"{self.cfg.path}/structures/{pdb_id[1:3]}/{pdb_id}.npz")

        data = self.unpack_chain(npz, int(chain1))
        if chain2 is not None:
            data2 = self.unpack_chain(npz, int(chain2))
            for key in data2:
                if key == 'seqres':
                    data['seqres'] += data2['seqres']
                else:
                    data[key] = np.concatenate([data[key], data2[key]])

        return self.make_data(name=name, **data)
