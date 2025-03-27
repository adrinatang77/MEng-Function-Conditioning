import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--afdb", type=str, default='tmp/afdb_uniprot_v4.idx')
parser.add_argument("--uniref", type=str, default='tmp/uniref100.idx')
parser.add_argument("--mapping", type=str, default='tmp/uniref100.mapping.1')
parser.add_argument("--mapping2", type=str, default='/data/cb/scratch/datasets/uniprot_uniref_map/uniref_id_mapping.dat.gz')
parser.add_argument("--out", type=str, default='tmp/unified.jsonl')
args = parser.parse_args()

import tqdm, glob
import numpy as np
import gzip, json
from collections import defaultdict

ur100_90 = {}
ur90_50 = {}
upkb_100 = {}
print('Loading mapping', flush=True)
with open(args.mapping) as f:
    for line in tqdm.tqdm(f, total=453950711):
        line = line.strip().split()
        ur100, ur90, ur50 = line[:3]
        upkb = line[3:]
        
        ur100_90[ur100] = ur90
        ur90_50[ur90] = ur50

        for name in upkb:
            upkb_100[name] = ur100
        # known issue: this only matches the representative UPKB member.

# with gzip.open(args.mapping2) as f:
#     for line in tqdm.tqdm(f):
#         upkb, key, val = line.decode('utf-8').strip().split()
#         if key == 'UniRef100':
#             upkb_100[upkb] = val
        
afdb = {}
afdb_glob = glob.glob(f"{args.afdb}*")
tot = 0
for i, path in enumerate(afdb_glob):
    print(f'Loading {i+1}/{len(afdb_glob)} AFDB files', flush=True)
    with open(path) as f:
        for line in tqdm.tqdm(f):
            idx, name, plddt, length = line.strip().split()
            _, upkb, _, _ = name.split('-')

            ur100 = upkb_100.get(upkb, None)
            if ur100:
                afdb[ur100] = int(idx), int(plddt), int(length)
            else:
                breakpoint()
            tot += 1
    print('Matched', len(afdb), 'entries out of', tot)
    

exit()

uniref = {}
print('Loading uniref', flush=True)
with open(args.uniref) as f:
    for line in tqdm.tqdm(f, total=453950711):
        name, pos, length = line.strip().split()
        uniref[name] = int(pos), int(length)

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
        valid = False
        plddt = -1
        js = {}
        for ur90 in ur50_90[ur50]:
            count['ur90'] += 1
            js[ur90] = {}
            for ur100 in ur90_100[ur90]:
                count['ur100'] += 1
                
                ur_tup = uniref.get(ur100, None)
                afdb_tup = afdb.get(ur100, None)
    
                if ur_tup:
                    count['ur100_match'] += 1
                    valid = True
                if afdb_tup:
                    count['afdb_match'] += 1
                    plddt = max(afdb_tup[1], plddt)
                    
                js[ur90][ur100] = {
                    'ur': ur_tup or (-1, 0),
                    'afdb': afdb_tup or (-1, 0, 0),
                }
    
                if count['ur100'] % int(1e6) == 0:
                    print(count, flush=True)
        
        if valid: # unfortunately there is a single missing uniref seq
            s = json.dumps(js) + '\n'
            f.write(s)
            idx.append((pos, plddt))
            pos += len(s)
        
print(count, flush=True)
idx = np.array(idx)
with open(args.out+'.idx', 'wb') as f:
    np.save(f, idx)
print(idx, flush=True)