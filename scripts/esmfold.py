import torch
import esm
import os
import sys
import argparse
import numpy as np
import tqdm
import glob
parser = argparse.ArgumentParser()
parser.add_argument('--seq', type=str, default=None)
parser.add_argument('--outpdb', type=str, default='tmp')
parser.add_argument('--outdir', type=str, default='/tmp')
parser.add_argument('--fasta', type=str, default=None)
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--print', action='store_true')
args = parser.parse_args()

print(args)

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

# Multimer prediction can be done with chains separated by ':'
res = []

if args.seq:
    seqs = [args.seq]
    names = [args.outpdb]
if args.fasta:
    seqs = list(open(args.fasta))[1::2]
    seqs = [seq.strip() for seq in seqs]
    names = list(open(args.fasta))[::2]
    names = [name.strip()[1:] for name in names]
if args.dir:
    files = glob.glob(f"{args.dir}/*.fasta")
    seqs = [list(open(f))[1].strip() for f in files]
    names = [list(open(f))[0].strip()[1:] for f in files]
# print(seqs)
with torch.no_grad():
    for seq, name in tqdm.tqdm(zip(seqs, names), total=len(seqs)):
        
        output = model.infer_pdb(seq)

        with open(f"{args.outdir}/{name}.pdb", "w") as f:
            f.write(output)
    
        from biopandas.pdb import PandasPdb
        plddt = PandasPdb().read_pdb(f"{args.outdir}/{name}.pdb").df['ATOM']['b_factor'].mean()
        res.append(plddt)
        if args.print:
            print(seq, plddt, np.mean(res), flush=True)
        
if args.print: print(np.mean(res))
