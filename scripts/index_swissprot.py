import argparse
import tqdm
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

print('Processing function data...')

func_processed = {} # name -> (start, end)
with open(args.fasta, "r") as f:  
    start = None
    name = None
    while True:
        line = f.readline()
        if not line:
            if start is not None:  # Handle the last entry
                end = f.tell()
                func_processed[name] = (start, end)
            break
        if line[0] == ">":
            if start is not None:  # Process the previous entry
                end = f.tell() - len(line)
                func_processed[name] = (start, end)
            name = line.split('\n')[0]
            start = f.tell() - len(line)

print('Processing Uniref50 order...')

# get uniref50 order
uniref = '/data/cb/scratch/datasets/uniref50.fasta' # TODO: read this in as a parameter
names = []
with open(uniref, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            name = line.split(' ')[0]
            names.append(name)  

print('Creating index...')

# now use order to create function index
locs = []
for name in names: # entry for every uniref50 entry 
    if name in func_processed:
        locs.append([func_processed[name][0], func_processed[name][1]])
    else: # not in func_processed
        locs.append(None)

locs = np.array(locs, dtype=object)
np.save(args.out, locs)