import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, default=None)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--worker_id", type=int, default=0)

args = parser.parse_args()

import foldcomp, tqdm
import numpy as np
import pandas as pd
from openprot.utils import protein

db = foldcomp.open(args.db)

df = []

for i in tqdm.trange(args.worker_id, len(db), args.num_workers):
    name, pdb = db[i]
    prot = protein.from_pdb_string(pdb)
    df.append({
        'index': i,
        'name': name,
        'plddt': prot.b_factors[:, 1].mean(),
        'length': len(prot.aatype)
    })

pd.DataFrame(df).set_index('index').to_pickle(args.out)