import torch
import esm
import os
import sys

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

sequence = sys.argv[1]
# Multimer prediction can be done with chains separated by ':'

with torch.no_grad():
    output = model.infer_pdb(sequence)

with open(sys.argv[2], "w") as f:
    f.write(output)

from biopandas.pdb import PandasPdb
print(PandasPdb().read_pdb(sys.argv[2]).df['ATOM']['b_factor'].mean())
