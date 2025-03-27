import torch
import numpy as np
import pandas as pd
import foldcomp
import os
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset
import json

import threading

class UnifiedDataset(OpenProtDataset):
    def setup(self):
        self.index = np.load(self.cfg.index)
        lens = np.diff(self.index[:,0])
        mask = self.index[:,1] >= self.cfg.plddt_thresh
        self.index = self.index[mask,0]
        self.lens = lens[mask[:-1]]
        self.afdb = foldcomp.open(self.cfg.afdb)
        
    def __len__(self):
        return len(self.index) - 1  # unfortunately we have to skip the last one

    def __getitem__(self, idx: int):
        
        with open(self.cfg.path) as f:
            end = self.index[idx + 1]
            f.seek(self.index[idx])
            line = f.read(self.lens[idx])
            
        try:
            js = json.loads(line)
        except:
            breakpoint()

        # filter based on plddt
        for key90 in list(js.keys()):
            js[key90] = {
                key100: js[key90][key100] for key100 in js[key90] \
                if js[key90][key100]['afdb'][1] >= self.cfg.plddt_thresh
            }
            if len(js[key90]) == 0:
                del js[key90]
            
            
        entry = np.random.choice(list(
            np.random.choice(list(js.values())).values()
        ))
        
        with open(self.cfg.uniref) as f:
            f.seek(entry['ur'][0])
            lines = f.read(entry['ur'][1]).split("\n")
        
        header, lines = lines[0], lines[1:]
        name = header if len(header.split()) == 0 else header.split()[0]
        seqres = "".join(lines)

        seq_mask = np.ones(len(seqres), dtype=np.float32)
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        residx = np.arange(len(seqres), dtype=np.float32)
        
        L = len(seqres)
        struct = None
        struct_mask = None

        if L == entry['afdb'][2]: # afdb is not a fragment
            _, pdb = self.afdb[entry['afdb'][0]]
            prot = protein.from_pdb_string(pdb)
            afdb_seqres = "".join([rc.restypes_with_x[c] for c in prot.aatype])
            if seqres == afdb_seqres:
                struct = prot.atom_positions[:,1]
                struct_mask = prot.atom_mask[:,1]
                
        return self.make_data(
            name=name,
            seqres=seqres,
            residx=residx,
            seq_mask=seq_mask,
            struct=struct,
            struct_mask=struct_mask
        )