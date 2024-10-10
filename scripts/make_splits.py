import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chains', type=str, default=None)
parser.add_argument('--pdb_path', type=str, default='/scratch/projects/cgai/openprot/data/pdb_npz/')
parser.add_argument('--cov-thresh', type=float, default=0.9)
parser.add_argument('--evalue-thresh', type=float, default=0.01)
parser.add_argument('--db', type=str, default="/scratch/projects/cgai/openprot/data/afdb_rep_v4/")
parser.add_argument('--blacklist_out', type=str, default=None)
parser.add_argument('--foldseek', type=str, default='/home1/10165/bjing/foldseek/bin/foldseek')
args = parser.parse_args()

import tqdm, os, subprocess
import numpy as np
import pandas as pd
from openprot.utils import protein
from openprot.utils import residue_constants as rc

df = pd.read_csv(args.chains, index_col='name')
os.makedirs('.tmp', exist_ok=True)
for name in tqdm.tqdm(df.index):
    npz = np.load(f"{args.pdb_path}/{name[1:3]}/{name}.npz")
    L = len(df.seqres[name])
    prot = protein.Protein(
        atom_positions=npz['all_atom_positions'],
        aatype=np.array([rc.restype_order_with_x[c] for c in df.seqres[name]]),
        atom_mask=npz['all_atom_mask'],
        residue_index=np.arange(L) + 1,
        b_factors=np.zeros((L, 37)),
        chain_index=np.zeros(L, dtype=int),
    )
    with open(f".tmp/{name}.pdb", "w") as f:
        f.write(protein.to_pdb(prot))

cmd = [
    args.foldseek,
    "easy-search",
    ".tmp",
    args.db,
    args.blacklist_out,
    ".tmpFolder",
    "--format-output",
    "query,target,qcov,tcov,evalue",
    "-c",
    str(args.cov_thresh),
    "-e",
    str(args.evalue_thresh),
]
print(" ".join(cmd))
subprocess.run(cmd)
os.system('rm -r .tmp .tmpFolder')

# pd.read_csv('.aln.tsv', names=["query", "target", "qcov", "tcov", "evalue"], sep='\t')
