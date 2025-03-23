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
            with open(self.cfg.go_vocab, 'r') as file:
                self.go_vocab = json.load(file)
            self.func_db = open(self.cfg.seq_func_map)
            with open(self.cfg.func_idx, 'r') as file:
                self.func_idx = json.load(file)

    def actual_setup(self):
        self.db = open(self.cfg.path)
        self.index = np.load(self.cfg.index)
        self.need_setup = False

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

            go_term_array = np.zeros((len(seqres), self.cfg.max_depth), dtype=int) 

            # get the func labels for this seq name
            if name in self.func_idx:
                start = self.func_idx[name]['start']
                end = self.func_idx[name]['end']
                with lock:
                    self.func_db.seek(start)
                    go_item = self.func_db.read(end - start)
                go_lines = go_item.split("\n")
                go_header, go_lines = go_lines[0], go_lines[1:]
                go_terms = "".join(go_lines)

                # split string to get GO terms
                go_terms = go_terms.split(',')
                go_term_indices = list(set(np.array([go_index for go_term in go_terms if (go_index := self.go_vocab.get(go_term)) is not None], dtype=int)))
                num_go_terms = len(go_term_indices)

                go_term_array[:, :num_go_terms] = go_term_indices # protein-level GO terms

            func_cond = go_term_array

            return self.make_data(
                name=name,
                seqres=seqres,
                seq_mask=seq_mask,
                residx=residx,
                func_cond=func_cond)
        
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=seq_mask,
            residx=residx,
        )
