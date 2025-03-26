import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

import tqdm
import numpy as np


pbar = tqdm.tqdm()
name = None
pos = None
f = open(args.fasta)
with open(args.out, "w") as g:
    while True:
        line = f.readline()
        if not line:
            break
        if line[0] == ">":
            
            newpos = f.tell() - len(line)
            if name is not None:
                g.write(f"{name} {pos} {newpos - pos}\n")
        
            pos = newpos
            name = line.split()[0][1:]
            pbar.update()


# starts = []
# pbar = tqdm.tqdm()
# with open(args.fasta) as f:
#     while True:
#         line = f.readline()
#         if not line:
#             break
#         if line[0] == ">":
#             starts.append(f.tell() - len(line))
#             pbar.update()

# np.save(args.out, np.array(starts))
