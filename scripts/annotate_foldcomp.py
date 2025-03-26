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

print('Opening the DB')
db = foldcomp.open(args.db)
print('Done opening the DB')
df = []
with open(args.out, "w") as f:
    for i in tqdm.trange(args.worker_id, len(db), args.num_workers):
        name, pdb = db[i]
        lddt = []
        for line in pdb.split('\n'):
            if line[12:16].strip() == 'CA':
                lddt.append(float(line[60:66]))
        if '-F2-' in name:
            breakpoint()
        # prot = protein.from_pdb_string(pdb)
        # print(f"{i} {name} {round(prot.b_factors[:, 1].mean())} {len(prot.aatype)}\n")
        f.write(f"{i} {name} {round(np.mean(lddt))} {len(lddt)}\n")
        # assert len(prot.aatype) == len(lddt)
        # assert round(prot.b_factors[:, 1].mean()) == round(np.mean(lddt))
        
# pd.DataFrame(df).set_index("index").to_pickle(args.out)
# with open(f"{args.out}.fasta", "w") as f:

