import argparse
import os
os.environ['OMP_NUM_THREADS'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, default=None, required=True)
parser.add_argument("--out", type=str, default=None, required=True)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--min_plddt", type=float, default=75)
parser.add_argument("--worker_id", type=int, default=0)
args = parser.parse_args()

import foldcomp, tqdm, os, scipy
import numpy as np
import pandas as pd
from io import BytesIO

def compute_hessian(coords, cutoff=15., gamma=1):
    n_atoms = coords.shape[0]
    
    relpos = coords[None] - coords[:,None]
    dmat = np.square(relpos).sum(-1) ** 0.5
    norm_relpos = relpos / (dmat[:,:,None] + 1e-12)
    
    myhess = -gamma * np.einsum('ijx,ijy->ixjy', norm_relpos, norm_relpos)
    idx = np.arange(n_atoms)
    myhess = np.where((dmat < cutoff)[:,None,:,None], myhess, 0.0)
    myhess[idx,:,idx] = 0
    myhess[idx,:,idx] = -myhess[idx].sum(-2)
            
    return myhess.reshape(3*n_atoms, 3*n_atoms)
    
print('Opening the DB')
db = foldcomp.open(args.db)
print('Done opening the DB')
df = []
pos = 0
with open(args.out, "wb") as f:
    with open(args.out+'.idx', "w") as g:
        for i in tqdm.trange(args.worker_id, len(db), args.num_workers):
            name, pdb = db[i]
            lddt = []
            x, y, z = [], [], []
            for line in pdb.split('\n'):
                if line[12:16].strip() == 'CA':
                    lddt.append(float(line[60:66]))
                    x.append(float(line[30:38]))
                    y.append(float(line[38:46]))
                    z.append(float(line[46:54]))
            
            coords = np.array([x, y, z]).T
            if len(coords) > args.max_len: continue
            if np.mean(lddt) < args.min_plddt: continue
            myhess = compute_hessian(coords)
            # D, P = np.linalg.eigh(myhess)
            # mymode = (P[:,6] / D[6]**0.5).reshape(-1, 3)
            
            eigval, eigvec = scipy.linalg.eigh(myhess, subset_by_index=[6,6])
            mymode = (eigvec / eigval**0.5).reshape(-1, 3)
    
            np_bytes = BytesIO()
            np.save(np_bytes, mymode.astype(np.float32))
            out = np_bytes.getvalue()
            f.write(out)
            
            g.write(f"{name} {pos} {len(out)}\n")
            pos += len(out)
            