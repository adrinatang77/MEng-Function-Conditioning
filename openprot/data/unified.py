import torch
import numpy as np
import pandas as pd
import foldcomp
import os
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset, OpenProtData
import json
from ..utils.structure import Structure

class UnifiedDataset(OpenProtDataset):
    def setup(self):
        arr = np.load(self.cfg.index)

        
        if self.cfg.alphafill:
            mask = arr[:,4] >= self.cfg.plddt_thresh

        else:
            mask = arr[:,2] >= self.cfg.plddt_thresh
            
        self.index = arr[mask]
        if self.cfg.struct:
            self.afdb = foldcomp.open(self.cfg.afdb)
        
    def __len__(self):
        return len(self.index) 

    def __getitem__(self, idx: int):
        import time
        start_t = time.time()
        
        with open(self.cfg.path) as f:
            f.seek(self.index[idx][0])
            line = f.read(self.index[idx][1])
            
        js = json.loads(line)
        dur1 = time.time()-start_t

        def filter_func(entry):
            if entry['afdb'][1] < self.cfg.plddt_thresh:
                return False
            if self.cfg.alphafill_only and 'alphafill' not in entry:
                return False
            return True
            
        # filter based on plddt
        for key90 in list(js.keys()):
            js[key90] = {
                key100: js[key90][key100] for key100 in js[key90] \
                if filter_func(js[key90][key100])
            }
            if len(js[key90]) == 0:
                del js[key90]
            
            
        entry = np.random.choice(list(
            np.random.choice(list(js.values())).values()
        ))
        dur2 = time.time() - start_t
        seqres = None
        name = None
        if 'ur' in entry:
            with open(self.cfg.uniref) as f:
                f.seek(entry['ur'][0])
                lines = f.read(entry['ur'][1]).split("\n")
            
            header, lines = lines[0], lines[1:]
            name = header if len(header.split()) == 0 else header.split()[0]
            seqres = "".join(lines)
            
            # afdb is not a fragment
            if self.cfg.struct and len(seqres) == entry['afdb'][2]: 
                require_struct = True
            else:
                require_struct = False
        else:
            require_struct = True

        ## The logic in this part is kind of ugly.
        
        struct = None
        struct_mask = None
        dur3 = time.time() - start_t
        if require_struct:
            afdb_name, pdb = self.afdb[entry['afdb'][0]]
            
            prot = protein.from_pdb_string(pdb)
            afdb_seqres = "".join([rc.restypes_with_x[c] for c in prot.aatype])
            if (seqres is None) or seqres == afdb_seqres:
                struct = prot.atom_positions[:,1]
                struct_mask = prot.atom_mask[:,1]

        
        name = name or afdb_name
        seqres = seqres or afdb_seqres
        seq_mask = np.ones(len(seqres), dtype=np.float32)
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        residx = np.arange(len(seqres), dtype=np.float32)

        dur4 = time.time() - start_t
        cath = np.zeros((len(seqres), 3))
        if 'ted' in entry:            
            with open(self.cfg.ted) as f:
                f.seek(entry['ted'][0])
                line = f.read(entry['ted'][1])
            for dom in line.strip().split('\n'):
                bounds = dom.split()[3]
                mask = np.zeros(len(seqres), dtype=bool)
                for interval in bounds.split('_'):
                    start, end = interval.split('-')
                    mask[int(start)-1:int(end)] = True
                label = dom.split()[13].strip()
                if label == '-': continue
                for i, c in enumerate(label.split('.')[:3]):
                    cath[mask,i] = int(c)
                
        data = self.make_data(
            name=name,
            seqres=seqres,
            residx=residx,
            seq_mask=seq_mask,
            struct=struct,
            struct_mask=struct_mask,
            cath=cath,
        )

        dur5 = time.time() - start_t

        # print(1000*dur1, 1000*dur2, 1000*dur3, 1000*dur4, 1000*dur5)
        
        
        # if 'alphafill' in entry:
        #     _, name, _, _ = entry['alphafill'][0].split('-')
        #     path = f"{self.cfg.alphafill}/{name[:2]}/{entry['alphafill'][0]}"
        #     try:
        #         struct = Structure.from_mmcif(path)
        #     except:
        #         print('AlphaFill failure', entry)
        #         return data
        #     chain = struct.get_chain(np.random.choice(range(1, len(struct.chains))))

        #     ones = np.ones(len(chain.atoms))
        #     ligand = self.make_data(
        #         name='',
        #         seqres='*'*len(chain.atoms),
        #         atom_num=chain.atoms['element'],
        #         mol_type=ones*chain.mol_type,
        #         seq_mask=ones,
        #         struct=chain.atoms['coords'],
        #         struct_mask=chain.atoms['is_present'],
        #         residx=chain.get_atom_residx(),
        #         chain=ones,
        #     )
        #     data = OpenProtData.concat([data, ligand])

            # dur6 = time.time() - start_t
            # print(1000*dur1, 1000*dur2, 1000*dur3, 1000*dur4, 1000*dur5, 1000*dur6)
        
        return data
            

        