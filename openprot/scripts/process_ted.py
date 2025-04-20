import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ted", type=str, default='data/ted_365m.domain_summary.cath.globularity.taxid.tsv')
parser.add_argument("--out", type=str, default='tmp/ted.idx')
args = parser.parse_args()

import tqdm
import numpy as np


pbar = tqdm.tqdm()
f = open(args.ted)
pos = 0
name = None
with open(args.out, "w") as g:
    while True:
        line = f.readline()
        if not line:
            break
        # name = line.split()[0]
        new_name = line.split('_')[0]

        if new_name != name:
            if name is not None:
                g.write(f"{name} {pos} {f.tell() - len(line) - pos}\n")
            pos = f.tell() - len(line)
            name = new_name
            
        pbar.update()