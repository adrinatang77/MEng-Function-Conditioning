import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--afdb", type=str, default='tmp/afdb_uniprot_v4.idx')
parser.add_argument("--uniref", type=str, default='tmp/uniref100.idx')
parser.add_argument("--mapping", type=str, default='tmp/uniref100.mapping')
parser.add_argument("--out", type=str, default='tmp/unified.jsonl')
args = parser.parse_args()

import tqdm, glob
import numpy as np
import gzip, json
from collections import defaultdict


afdb = {}
afdb_glob = glob.glob(f"{args.afdb}*")
for i, path in enumerate(afdb_glob):
    print(f'Loading {i+1}/{len(afdb_glob)} AFDB files', flush=True)
    with open(path) as f:
        for line in tqdm.tqdm(f):
            idx, name, plddt, length = line.strip().split()
            afdb[name] = int(idx), int(plddt), int(length)
            

uniref = {}
print('Loading uniref', flush=True)
with open(args.uniref) as f:
    for line in tqdm.tqdm(f):
        name, pos, length = line.strip().split()
        uniref[name] = int(pos), int(length)
        # if len(uniref) > 3e5: break


ur100_90 = {}
ur90_50 = {}
ur100_kb = {}
print('Loading mapping', flush=True)
with open(args.mapping) as f:
    for line in tqdm.tqdm(f):
        ur100, ur90, ur50, upkb = line.strip().split()
        ur100_90[ur100] = ur90
        ur90_50[ur90] = ur50
        ur100_kb[ur100] = upkb
        # if len(ur100_90) > 1e6: break

print('Inverting mapping', flush=True)
ur50_90 = defaultdict(list)
for ur90, ur50 in tqdm.tqdm(ur90_50.items()):
    ur50_90[ur50].append(ur90)

ur90_100 = defaultdict(list)
for ur100, ur90 in tqdm.tqdm(ur100_90.items()):
    ur90_100[ur90].append(ur100)

print('Prepping final JSON', flush=True)
count = defaultdict(int)
idx = []
pos = 0
with open(args.out, 'w') as f:
    for ur50 in tqdm.tqdm(ur50_90):
        count['ur50'] += 1
        js = {}
        for ur90 in ur50_90[ur50]:
            count['ur90'] += 1
            js[ur90] = {}
            for ur100 in ur90_100[ur90]:
                count['ur100'] += 1
                
                upkb = ur100_kb[ur100]
                ur_tup = uniref.get(ur100, None)
                afdb_tup = afdb.get(f"AF-{upkb}-F1-model_v4.cif.gz", None)
    
                if ur_tup: count['ur100_match'] += 1
                if afdb_tup: count['afdb100_match'] += 1
                    
                js[ur90][ur100] = {
                    'ur': ur_tup or (-1, 0),
                    'afdb': afdb_tup or (-1, 0, 0),
                }
    
                if count['ur100'] % int(1e6) == 0:
                    print(count, flush=True)
        s = json.dumps(js) + '\n'
        f.write(s)
        idx.append(pos)
        pos += len(s)
    
print(count, flush=True)
idx = np.array(idx)
with open(args.out+'.idx', 'wb') as f:
    np.save(f, idx)
print(idx, flush=True)