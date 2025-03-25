import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, default=None, required=True)
parser.add_argument("--out", type=str, default=None, required=True)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--worker_id", type=int, default=0)

args = parser.parse_args()

import foldcomp, tqdm
import numpy as np
import pandas as pd
from openprot.utils import protein
from openprot.utils.prot_utils import aatype_to_seqres

db = foldcomp.open(args.db)

df = []
with open(args.out, "w") as f:
    for i in tqdm.trange(args.worker_id, len(db), args.num_workers):
        name, pdb = db[i]
        prot = protein.from_pdb_string(pdb)
        f.write(f"{i} {name} {round(prot.b_factors[:, 1].mean())} {len(prot.aatype)}\n")
        
# pd.DataFrame(df).set_index("index").to_pickle(args.out)
# with open(f"{args.out}.fasta", "w") as f:

