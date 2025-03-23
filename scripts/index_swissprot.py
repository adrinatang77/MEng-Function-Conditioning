import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

import tqdm
import numpy as np
import json

# index the file
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

# create idx numpy array 
starts = []
prev_start = 0
uniref = '/data/cb/scratch/datasets/uniref50.fasta'
with open(uniref) as f: # going in same order as uniref50
    while True:
        line = f.readline()
        if not line:
            break
        if line[0] == ">":
            name = line.split(' ')[0]
            if name in idx: 
                starts.append(idx[name]['start']) # add to func idx
                prev_start = idx[name]['start']
            else:
                starts.append(prev_start) # no func label

np.save(args.out, np.array(starts))
# np.save('/data/cb/scratch/datasets/func_data/extracted_data/swissprot_uniref50/swissprot_GO_idx.npy', np.array(starts))