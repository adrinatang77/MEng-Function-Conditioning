import torch
import esm
import os
import sys
import argparse
import numpy as np
import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--seq', type=str, nargs="*", default=None)
parser.add_argument('--outpdb', type=str, default='/tmp/tmp.pdb')
parser.add_argument('--fasta', type=str, default=None)
args = parser.parse_args()

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

# Multimer prediction can be done with chains separated by ':'
res = []

if args.seq:
    seqs = args.seq
if args.fasta:
    seqs = list(open(args.fasta))[1::2]
    seqs = [seq.strip() for seq in seqs]
# print(seqs)
with torch.no_grad():
    for seq in tqdm.tqdm(seqs):
        output = model.infer_pdb(seq)

        with open(args.outpdb, "w") as f:
            f.write(output)
    
        from biopandas.pdb import PandasPdb
        plddt = PandasPdb().read_pdb(args.outpdb).df['ATOM']['b_factor'].mean()
        res.append(plddt)
        print(seq, plddt, np.mean(res))
        

