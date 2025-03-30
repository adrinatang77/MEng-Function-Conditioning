from .task import OpenProtTask
import numpy as np
import math
from ..utils import residue_constants as rc
from ..generate.diffusion import t_to_sigma
from scipy.spatial.transform import Rotation as R
import torch, scipy
import os
from ..data.data import OpenProtData

class CodesignTask(OpenProtTask):

    # def sample_ppi(self, data):
    #     L = len(data['seqres'])
    #     cfg = self.cfg.ppi

    #     dmat = np.square(data['struct'][None] - data['struct'][:,None]).sum(-1)**0.5 
    #     cmap = dmat < cfg.contact
    #     cmap &= data['struct_mask'][None].astype(bool) & data['struct_mask'][:,None].astype(bool)
    #     ii, jj = np.meshgrid(np.arange(L), np.arange(L))
    #     cmap &= ii - jj >= cfg.long_range
    #     if np.sum(cmap) == 0:
    #         return False

    #     i, j = np.argwhere(cmap)[np.random.choice(range(cmap.sum()))]

        
    #     gap_len = np.random.choice(range(
    #         cfg.gap_min,
    #         min(j-i-2*cfg.gap_pad, cfg.gap_max)
    #     ))
    #     gap_start = np.random.choice(range(i+cfg.gap_pad, j-cfg.gap_pad-gap_len))
        
    #     mask = np.ones(L, dtype=bool)
    #     mask[gap_start:gap_start+gap_len] = 0
    #     data.crop(idx=np.arange(L)[mask])

    #     if np.random.rand() > 0.5:
    #         data['ligand_mask'][:gap_start] = 1
    #     else:
    #         data['ligand_mask'][gap_start:] = 1
    #     data['chain'][gap_start:] = 1
        
    #     # label hotspots
    #     dmat = dmat[mask][:,mask]
    #     hotspot = dmat < self.cfg.ppi.hotspot
    #     hotspot &= (
    #         data['struct_mask'][None].astype(bool)
    #         & data['struct_mask'][:,None].astype(bool)
    #     )
    #     hotspot &= data['chain'][None] != data['chain'][:,None]
    #     hotspot = np.any(hotspot, -1)

        
    #     data['hotspot_mask'] = hotspot * data['ligand_mask']
        
            
    #     return True
            
        
    def sample_motifs(self, data):
        N = len(data['seqres'])
        Ns = np.random.randint(1, self.cfg.motif.nmax)
        Nr = np.random.randint(
            int(math.floor(N * self.cfg.motif.ymin)),
            int(math.ceil(N * self.cfg.motif.ymax)) + 1,
        )
        if Ns > Nr: return # segments cannot exceed residues
        
        B = [
            0,
            *sorted(np.random.choice(np.arange(1, Nr), size=Ns-1, replace=False)),
            Nr
        ]
        L = np.diff(B)
        class Motif(list):
            def __init__(self, l):
                super().__init__(l)
                self.rand = np.random.rand()
            
        M = [[False] for _ in range(N-Nr)] + [Motif([True]*l) for l in L]
        np.random.shuffle(M)
        
        is_motif = np.array([i for a in M for i in a])
        rand = np.array([getattr(a, 'rand', 0) for a in M for i in a])
        
        data['motif'][is_motif] = data['struct'][is_motif]
        data['motif_mask'][is_motif] = data['struct_mask'][is_motif]
        
        if np.random.rand() < self.cfg.motif.multi_prob:
            data['motif_idx'][is_motif] = (rand < 0.5).astype(np.float32)[is_motif]

        if np.random.rand() < self.cfg.motif.nma_prob:
            self.compute_motif_nma(data)

    def compute_motif_nma(self, data):

        return # for now
        if np.any(data['struct_mask'] == 0): return # missing calpha
        
        
        
        myhess = compute_hessian(data['struct'])
        eigval, eigvec = scipy.linalg.eigh(myhess, subset_by_index=[6,6])
        mode = eigvec.reshape(-1, 3) / eigval**0.5
        if np.random.rand() < 0.5:
            mode = -mode

        idx = np.unique(data['motif_idx'])
        for i in idx: 
            mask = (data['motif_mask'] == 1) & (data['motif_idx'] == i)
            nma = mode[mask]
            nma = nma - nma.mean(0)
            data['motif_nma'][mask] = nma
            data['motif_nma_mask'][mask] = 1
            
    def center_random_rot(self, data, eps=1e-6):
        # center the structures
        pos = data["struct"]
        mask = data["struct_mask"][..., None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["struct"] -= com

        randrot = R.random().as_matrix()
        data["struct"] @= randrot.T

        idx = np.unique(data['motif_idx'])
        for i in idx: 
            # note that multi-residue ligands will be wrong
            # this also handles motifs, so far correct but brittle
            motif = data["motif"][data['motif_idx'] == i]
            motif_mask = data['motif_mask'][data['motif_idx'] == i][...,None]
            motif -= (motif * motif_mask).sum(-2) / (motif_mask.sum(-2) + eps)
            randrot = R.random().as_matrix()
            motif @= randrot.T
            data["motif"][data['motif_idx'] == i] = motif
            data["motif_nma"][data['motif_idx'] == i] @= randrot.T
            
        
    def add_sequence_noise(self, data, noise_level=None):
        
        def sample_noise_level():
            rand = np.random.rand()
            probs = [
                self.cfg.seq.get('zero_prob', 0),
                self.cfg.seq.get('max_prob', 0),
                self.cfg.seq.get('uniform_prob', 0),
            ]
            probs = np.cumsum(probs)
            
            if rand < probs[0]:
                noise_level = 0.0
            elif rand < probs[1]:
                noise_level = 1.0
            elif rand < probs[2]:
                noise_level = np.random.rand()
            else:
                noise_level = np.random.beta(*self.cfg.seq.beta)
            return noise_level
        
        L = len(data["seqres"])

        if noise_level is None:
            noise_level = sample_noise_level()
        
        data["seq_noise"] = np.where( # not ligand or motif
            (data['motif_mask'] == 0) & (data['ligand_mask'] == 0),
            noise_level,
            0,
        ).astype(np.float32)

    def add_structure_noise(self, data, eps=1e-6, noise_level=None):
        

        def sample_noise_level():
            rand = np.random.rand()
            
            probs = [
                self.cfg.struct.get('zero_prob', 0),
                self.cfg.struct.get('max_prob', 0),
                self.cfg.struct.get('uniform_prob', 0),
            ]
            probs = np.cumsum(probs)
            if rand < probs[0]:
                noise_level = 0.0
            elif rand < probs[1]:
                noise_level = 1.0
            elif rand < probs[2]:
                noise_level = np.random.rand()
            else:
                noise_level = np.random.beta(*self.cfg.struct.beta)
            
            return noise_level
        
        L = len(data["seqres"])

        if noise_level is None:
            noise_level = sample_noise_level()
            
        data["struct_noise"] = np.where( # not ligand
            data['ligand_mask'] == 0,
            t_to_sigma(self.cfg.edm, noise_level),
            t_to_sigma(self.cfg.edm, 0),
        ).astype(np.float32)

        return noise_level

class BinderCodesign(CodesignTask):
    def register_loss_masks(self):
        return ["/binder", "/binder/lig", "/binder/ppi"]

    def sample_ppi(self, data):
        L = len(data['seqres'])
        cfg = self.cfg.ppi

        dmat = np.square(data['struct'][None] - data['struct'][:,None]).sum(-1)**0.5 
        cmap = dmat < cfg.contact
        cmap &= data['struct_mask'][None].astype(bool) & data['struct_mask'][:,None].astype(bool)
        ii, jj = np.meshgrid(np.arange(L), np.arange(L))
        cmap &= ii - jj >= cfg.long_range
        if np.sum(cmap) == 0:
            return False

        i, j = np.argwhere(cmap)[np.random.choice(range(cmap.sum()))]
        
        gap_len = np.random.choice(range(
            cfg.gap_min,
            min(j-i-2*cfg.gap_pad, cfg.gap_max)
        ))
        gap_start = np.random.choice(range(i+cfg.gap_pad, j-cfg.gap_pad-gap_len))
        
        mask = np.ones(L, dtype=bool)
        mask[gap_start:gap_start+gap_len] = 0
        data.crop(idx=np.arange(L)[mask])

        if np.random.rand() > 0.5:
            data['ligand_mask'][:gap_start] = 1
        else:
            data['ligand_mask'][gap_start:] = 1
        data['chain'][gap_start:] = 1

    def prep_interface(self, data, crop=None):

        if len(np.unique(data['chain'])) == 1: # all the same chain, i.e., no PPI
            ppi = self.sample_ppi(data)
            if not ppi:  # synthetic PPI failed
                data.crop(crop)
                return data
        
        L = len(data['seqres'])
        if data['mol_type'].sum() > 1e-6: # is small molecule ligand
            data['ligand_mask'] = (data['mol_type'] != 0).astype(float)
            
        else: # is PPI
            lig = np.random.choice(np.unique(data['chain']))
            data['ligand_mask'] = (data['chain'] == lig).astype(float)
    
            # label hotspots
            dmat = np.square(data['struct'][None] - data['struct'][:,None]).sum(-1)**0.5 
            hotspot = dmat < self.cfg.ppi.hotspot
            hotspot &= (
                data['struct_mask'][None].astype(bool)
                & data['struct_mask'][:,None].astype(bool)
            )
            hotspot &= data['chain'][None] != data['chain'][:,None]
            hotspot = np.any(hotspot, -1)
    
            
            data['hotspot_mask'] = hotspot * data['ligand_mask']

        datas = []
        # special crop
        if crop is not None:
            for i in np.unique(data['chain']):
                datas.append(data.crop(idx=np.arange(L)[data['chain'] == i], inplace=False))
                
            if datas[0]['ligand_mask'].sum():
                datas = datas[::-1] # move ligand to end

            
            if crop <= len(datas[1]['seqres']): # ligand is bigger than the crop
                datas[0].crop(crop//2)
                datas[1].crop(crop//2)
                
            else:
                datas[0].crop(crop-len(datas[1]['seqres']))
                
            data = OpenProtData.concat(datas)
            data['name'] = datas[0]['name']
        
        return data
        
    def prep_data(self, data, crop=None):
        data = self.prep_interface(data, crop=crop)

        self.add_sequence_noise(data)
        self.add_structure_noise(data)
        self.center_random_rot(data)
            
        data["/binder"] = np.ones((), dtype=np.float32)
        
        return data
    

class Codesign(CodesignTask):
    def register_loss_masks(self):
        return ["/codesign", "/codesign/lig", "/codesign/ppi"]

    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)

        rand = np.random.rand()
        
        if rand < self.cfg.motif_prob:
            self.sample_motifs(data)
                
        self.add_sequence_noise(data)
        self.add_structure_noise(data)

        # this must happen after motifs have been assigned
        self.center_random_rot(data)

        data["_cath_noise"] = np.array(np.random.rand(), dtype=np.float32)
            
        if data["dataset"] == "boltz_lig":
            data["/codesign/lig"] = np.ones((), dtype=np.float32)
        else:
            data["/codesign"] = np.ones((), dtype=np.float32)
        
        return data
    