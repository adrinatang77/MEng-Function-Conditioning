import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, default=None)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--worker_id", type=int, default=0)

args = parser.parse_args()

import foldcomp, tqdm
import numpy as np
from openprot.utils import protein

db = foldcomp.open(args.db)

lddt = np.zeros(len(db))
lens = np.zeros(len(db))

for i in tqdm.trange(args.worker_id, len(db), args.num_workers):
    name, pdb = db[i]
    prot = protein.from_pdb_string(pdb)
    lddt[i] = prot.b_factors[:, 1].mean()
    lens[i] = len(prot.aatype)

np.savez(args.out, lddt=lddt.astype(np.float16), lens=lens.astype(np.int16))
