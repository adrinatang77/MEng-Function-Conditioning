import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

import tqdm
import numpy as np

starts = []
pbar = tqdm.tqdm()
with open(args.fasta) as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line[0] == ">":
            starts.append(f.tell() - len(line))
            pbar.update()

np.save(args.out, np.array(starts))
