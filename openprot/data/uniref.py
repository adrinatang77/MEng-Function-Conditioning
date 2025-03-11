import torch
import numpy as np
import pandas as pd
# import foldcomp
import os
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset
import json

import threading

lock = threading.Lock()


class UnirefDataset(OpenProtDataset):
    def setup(self):
        self.db = open(self.cfg.path)
        self.index = np.load(self.cfg.index)
        self.need_setup = True
        if self.cfg.func_cond:
            with open('/data/cb/asapp/openprot/go_vocab_small.json', 'r') as file:
                self.go_vocab = json.load(file)
            with open('/data/cb/asapp/openprot/function_labels.json', 'r') as file:
                self.go_term_map = json.load(file)
            # print('Function conditioning (setup 1)...')

    def actual_setup(self):
        self.db = open(self.cfg.path)
        self.index = np.load(self.cfg.index)
        self.need_setup = False
        # if self.cfg.func_cond:
        #     print('Function conditioning (setup 2)...')

    def __len__(self):
        return len(self.index) - 1  # unfortunately we have to skip the last one

    def __getitem__(self, idx: int):
        if self.need_setup:
            self.actual_setup()

        start = self.index[idx]
        end = self.index[idx + 1]
        with lock:
            self.db.seek(start)
            item = self.db.read(end - start)
        lines = item.split("\n")
        header, lines = lines[0], lines[1:]
        seqres = "".join(lines)
        
        seq_mask = np.ones(len(seqres), dtype=np.float32)
    
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        name = header if len(header.split()) == 0 else header.split()[0]
        residx = np.arange(len(seqres), dtype=np.float32)

        if self.cfg.func_cond:
            print('Function conditioning (get item)...')

            print(name)
            # look up seq <-> func labels
            if name in self.go_term_map:
                go_term = self.go_term_map[name]
                func_cond = np.ones(len(seqres), dtype=np.float32) * self.go_vocab[go_term]
                print('name found')
            else:
                func_cond = np.zeros(len(seqres), dtype=np.float32)
            
            return self.make_data(name=name,seqres=seqres,seq_mask=seq_mask,residx=residx,func_cond=func_cond)
        
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=seq_mask,
            residx=residx,
        )
