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

prev_name = None

with open(args.fasta) as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line[0] == ">":
            if prev_name is None: # first 
                name = line.split('\n')[0]
                start = f.tell() - len(line)
                idx[name] = {'start': start, 'end': None}
                prev_name = name
            else:
                name = line.split('\n')[0]
                start = f.tell() - len(line)
                idx[name] = {'start': start, 'end': None}

                # set the end of last entry
                idx[prev_name]['end'] = start
                prev_name = name
            pbar.update()

with open(args.out, 'w') as f:
    json.dump(idx, f)

