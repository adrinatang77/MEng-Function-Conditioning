from .task import OpenProtTask
import numpy as np
import math
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R
import torch

class CodesignTask(OpenProtTask):

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
        
        data['seq_noise'][is_motif] = 0
        data['seq_weight'][is_motif] = 0
        data['ref_conf'][is_motif] = data['struct'][is_motif]
        data['ref_conf_mask'][is_motif] = data['struct_mask'][is_motif]
        data['struct_weight'][is_motif] *= self.cfg.motif.weight
        if np.random.rand() < self.cfg.motif.multi_prob:
            data['ref_conf_idx'][is_motif] = (rand < 0.5).astype(np.float32)[is_motif]
        

    def center_random_rot(self, data, eps=1e-6):
        # center the structures
        pos = data["struct"]
        mask = data["struct_mask"][..., None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["struct"] -= com

        randrot = R.random().as_matrix()
        data["struct"] @= randrot.T

        idx = np.unique(data['ref_conf_idx'])
        for i in idx: 
            # note that multi-residue ligands will be wrong
            # this also handles motifs, so far correct but brittle
            conf = data["ref_conf"][data['ref_conf_idx'] == i]
            conf_mask = data['ref_conf_mask'][data['ref_conf_idx'] == i][...,None]
            conf -= (conf * conf_mask).sum(-2) / (conf_mask.sum(-2) + eps)
            randrot = R.random().as_matrix()
            conf @= randrot.T
            data["ref_conf"][data['ref_conf_idx'] == i] = conf
            
        
    def add_sequence_noise(self, data, noise_level=None, sup=False):
        
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
            
        # different noise level per chain
        L = len(data["seqres"])
        
        if noise_level is None:
            data["seq_noise"] = np.zeros(L, dtype=np.float32)
            idx = np.unique(data['chain'])
            for i in idx:
                data["seq_noise"][data['chain'] == i] = sample_noise_level()
        else:
            data["seq_noise"] = np.ones(L, dtype=np.float32) * noise_level

        
        if sup:
            data["seq_weight"] = np.ones(L, dtype=np.float32)

        
    def add_structure_noise(self, data, eps=1e-6, noise_level=None, sup=False):
        

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

        def t_to_sigma(t):
            p = self.cfg.edm.sched_p
            sigma = (
                self.cfg.edm.sigma_min ** (1 / p)
                + t * (self.cfg.edm.sigma_max ** (1 / p) - self.cfg.edm.sigma_min ** (1 / p))
            ) ** p
            return sigma
        
            
        
        L = len(data["seqres"])
        if noise_level is None:
            if self.cfg.struct.get('async', False):
                data["struct_noise"] = np.zeros(L, dtype=np.float32)
                idx = np.unique(data['chain'])
                for i in idx:
                    data["struct_noise"][data['chain'] == i] = t_to_sigma(sample_noise_level())
            else:
                noise_level = sample_noise_level()
                data["struct_noise"] = np.ones(L, dtype=np.float32) * t_to_sigma(noise_level)
        else:
            data["struct_noise"] = np.ones(L, dtype=np.float32) * t_to_sigma(noise_level)
        

        if sup:
            data["struct_weight"] = np.where(
                data['mol_type'] != 0, 3.0, 1.0,
            )

        data['struct_align_mask'] = data['struct_mask'] # force default

        if self.cfg.struct.get('prot_only', False):
            data["struct_noise"] = np.where(
                data['mol_type'] == 0,
                data['struct_noise'],
                self.cfg.edm.sigma_min,
            )
            data["struct_weight"] = np.where(
                data['mol_type'] == 0,
                data['struct_weight'],
                0.0,
            )
            data['struct_align_mask'] = (data['mol_type'] == 0).astype(float) * data['struct_mask']
        
        if self.cfg.struct.get('lig_only', False):
            data["struct_noise"] = np.where(
                data['mol_type'] != 0,
                data['struct_noise'],
                self.cfg.edm.sigma_min,
            )
            data["struct_weight"] = np.where(
                data['mol_type'] != 0,
                data['struct_weight'],
                0.0,
            )
            data['struct_align_mask'] = (data['mol_type'] != 0).astype(float) * data['struct_mask']

        


        return noise_level

class Codesign(CodesignTask):
    def register_loss_masks(self):
        return ["/codesign", "/codesign/lig"]

    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)

        self.add_sequence_noise(data, sup=True)
        
        self.add_structure_noise(data, sup=True)

        if np.random.rand() < self.cfg.motif_prob:
            self.sample_motifs(data)
        
        self.center_random_rot(data)

            
        if data["dataset"] == "boltz_lig":
            data["/codesign/lig"] = np.ones((), dtype=np.float32)
        else:
            data["/codesign"] = np.ones((), dtype=np.float32)
        
        return data
    