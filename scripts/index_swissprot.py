import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

import tqdm
import numpy as np
import json

# index the file
# starts = []
idx = {}
pbar = tqdm.tqdm()
with open(args.fasta) as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line[0] == ">":
            name = line.split(">")[1]
            idx[name] = f.tell() - len(line)
            # starts.append(f.tell() - len(line))
            pbar.update()

with open(args.out, 'w') as f:
    json.dump(idx, f)

# np.save(args.out, np.array(starts))
