import torch
import numpy as np
import pandas as pd
# import foldcomp
import os
from ..utils import protein
from ..utils import residue_constants as rc
from ..scripts import go_term_utils
from .data import OpenProtDataset
import json

import threading

lock = threading.Lock()

class UnirefDataset(OpenProtDataset):
    def setup(self):
        self.db = self.cfg.res_go_terms
        self.index = np.load(self.cfg.res_func_idx)
        if self.cfg.func_cond:
            with open(self.cfg.go_vocab, 'r') as file:
                self.go_vocab = json.load(file)

    def __len__(self):
        return len(self.index) - 1  # unfortunately we have to skip the last one

    def __getitem__(self, idx: int):
        
        start = self.index[idx]
        end = self.index[idx + 1]
        with open(self.db, 'r') as db:
            with lock:
                db.seek(start)
                item = db.read(end - start)
        lines = item.split("\n")
        header, seqres = lines[0], lines[1]
        seq_mask = np.ones(len(seqres), dtype=np.float32)
    
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        name = header[1:]
        residx = np.arange(len(seqres), dtype=np.float32)

        if self.cfg.func_cond:
            go_term_array = np.zeros((len(seqres), self.cfg.max_depth), dtype=int) 

#             ### protein-level GO terms ###s
#             if start_end != None:
#                 with lock:
#                     self.db.seek(start_end[0])
#                     item = self.prot_func_db.read(start_end[1] - start_end[0])
#                 lines = item.split("\n")
#                 header, lines = lines[0], lines[1:]
#                 if lines[0] != '': # GO term annotations exist
#                     lines = lines[0] # all GO terms
#                     go_terms = lines.split(',') # split string to get GO terms
#                     go_term_indices = list(set(np.array([go_index for go_term in go_terms if (go_index := self.go_vocab.get(go_term)) is not None], dtype=int)))
#                     num_go_terms = len(go_term_indices)
#                     go_term_array[:, :num_go_terms] = go_term_indices # protein-level GO terms

            ### residue-level GO terms ###
#             start = self.res_func_idx[idx]
#             end = self.res_func_idx[idx + 1]

            try:
                go_terms_and_pos = go_term_utils.parse_res_go_data_(item)
            except:
                print('error at index', idx)
                print(item)

            for entry_id in go_terms_and_pos: # should be just one
                go_terms_list = go_terms_and_pos[entry_id] 
                for go_term in go_terms_list:
                    go_id = go_term['go_id']
                    start = go_term['start'] - 1 # 1-indexed
                    end = go_term['end'] - 1 # 1-indexed
#                         print(go_id, start, end)

                    go_index = self.go_vocab.get(go_id)
                    if not go_index:
                        continue
                    for i in range(start, end):
                        if go_index not in go_term_array[i, :]:
                            go_term_array[i, np.argmax(go_term_array[i] != 0) + 1] = go_index # add to array 

            # # get the func labels for this seq name
            # if name in self.func_idx:
            #     start = self.func_idx[name]['start']
            #     end = self.func_idx[name]['end']
            #     with lock:
            #         self.func_db.seek(start)
            #         go_item = self.func_db.read(end - start)
            #     go_lines = go_item.split("\n")
            #     go_header, go_lines = go_lines[0], go_lines[1:]
            #     go_terms = "".join(go_lines)

            #     # split string to get GO terms
            #     go_terms = go_terms.split(',')
            #     go_term_indices = list(set(np.array([go_index for go_term in go_terms if (go_index := self.go_vocab.get(go_term)) is not None], dtype=int)))
            #     num_go_terms = len(go_term_indices)

            #     go_term_array[:, :num_go_terms] = go_term_indices # protein-level GO terms

            func_cond = go_term_array

            return self.make_data(
                name=name,
                seqres=seqres,
                seq_mask=seq_mask,
                residx=residx,
                func_cond=func_cond
            )
        
        return self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=seq_mask,
            residx=residx,
        )
