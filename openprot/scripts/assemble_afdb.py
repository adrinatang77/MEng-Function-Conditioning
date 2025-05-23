import argparse
#  {'ur50': 2302908, 'ur90': 30045247, 'ur100': 167814824, 'afdb_match': 167732948, 'ted_match': 164222486, 'alphafill_match': 715729})
parser = argparse.ArgumentParser()
parser.add_argument("--afdb", type=str, default='tmp/afdb_uniprot_v4.idx')
parser.add_argument("--ted", type=str, default='tmp/ted.idx')
parser.add_argument("--alphafill", type=str, default='tmp/alphafill.idx')
parser.add_argument("--nma", type=str, default='data/nma/afdb_uniprot_v4.out')
parser.add_argument("--fs_mapping", type=str, default='data/1-AFDBClusters-entryId_repId_taxId.tsv')
parser.add_argument("--fs_mapping2", type=str, default='data/7-AFDB50-repId_memId.tsv')
parser.add_argument("--mapping2", type=str, default='/data/cb/scratch/datasets/uniprot_uniref_map/uniref_id_mapping.dat.gz')
parser.add_argument("--out", type=str, default=None, required=True)
parser.add_argument("--err", type=str, default='tmp/error.out')
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

import tqdm, glob
import numpy as np
import gzip, json
from collections import defaultdict

print('Loading mapping', flush=True)
# "UR100" is now UPKB
# "UR90" is now AFDB representative
# "UR50" is FoldSeek representative


ur100_90 = {}
ur90_50 = {}

with open(args.fs_mapping2) as f:
    for line in tqdm.tqdm(f, total=214684311):
        ur90, ur100 = line.strip().split('\t')
        ur100_90[ur100] = ur90
        if args.debug and len(ur100_90) > 2e6: break

with open(args.fs_mapping) as f:
    for line in tqdm.tqdm(f, total=30045247):        
        ur90, ur50, _ = line.strip().split('\t') # yes, the order is flipped
        ur90_50[ur90] = ur50
        if args.debug and len(ur90_50) > 2e6: break
                
############################### AFDB STRUCTURE ####################
afdb = {}
afdb_glob = glob.glob(f"{args.afdb}*")
for i, path in enumerate(afdb_glob):
    print(f'Loading {i+1}/{len(afdb_glob)} AFDB files', flush=True)
    with open(path) as f:
        for line in tqdm.tqdm(f):
            idx, name, plddt, length = line.strip().split()
            _, upkb, _, _ = name.split('-')

            ur100 = upkb
            afdb[ur100] = int(idx), int(plddt), int(length)
            
    if args.debug: break

############################### AFDB NMA ####################
nma = {}
nma_glob = glob.glob(f"{args.nma}.*.idx")
for i, path in enumerate(nma_glob):
    print(f'Loading {i+1}/{len(nma_glob)} NMA files', flush=True)
    with open(path) as f:
        for line in tqdm.tqdm(f):
            fidx = path.split('.')[-2]
            
            name, pos, length = line.strip().split()
            _, upkb, _, _ = name.split('-')

            ur100 = upkb
            nma[ur100] = int(fidx), int(pos), int(length)
            
    if args.debug: break


############################### TED CATH ANNOTATIONS ####################
ted = {}
print('Loading TED', flush=True)
with open(args.ted) as f:
    for line in tqdm.tqdm(f, total=214683829):
        name, pos, length = line.strip().split()
        _, upkb, _, _ = name.split('-')
        ur100 = upkb 
        ted[ur100] = int(pos), int(length)
        if args.debug and len(ted) > 100: break
print('Loaded TED', len(ted), flush=True)

############################### ALPHAFILL ####################
alphafill = {}
print('Loading AlphaFill', flush=True)
with open(args.alphafill) as f:
    for line in tqdm.tqdm(f, total=856037):
        name, count = line.strip().split()
        _, upkb, _, _ = name.split('-')
        ur100 = upkb 
        alphafill[ur100] = name, int(count)
print('Loaded AlphaFill', len(alphafill), flush=True)

############################### PROCESS MAPPINGS ####################
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
        alphafill_plddt = -1
        
        js = {}
            
        for ur90 in ur50_90[ur50]:
            count['ur90'] += 1
            js[ur90] = {}
            for ur100 in ur90_100[ur90]:
                count['ur100'] += 1
                
                afdb_tup = afdb.get(ur100, None)
                if afdb_tup is None:
                    continue 
                    
                count['afdb_match'] += 1
                plddt = max(afdb_tup[1], plddt)
                valid = True
                
                js[ur90][ur100] = {
                    'afdb': afdb_tup,
                }

                ted_tup = ted.get(ur100, None)
                if ted_tup:
                    count['ted_match'] += 1
                    js[ur90][ur100]['ted'] = ted_tup

                alphafill_tup = alphafill.get(ur100, None)
                if alphafill_tup:
                    count['alphafill_match'] += 1
                    js[ur90][ur100]['alphafill'] = alphafill_tup
                    
                    alphafill_plddt = max(afdb_tup[1], alphafill_plddt)

                nma_tup = nma.get(ur100, None)
                if nma_tup:
                    count['nma_match'] += 1
                    js[ur90][ur100]['nma'] = nma_tup
                if count['ur100'] % int(1e6) == 0:
                    print(count, flush=True)
        
        if valid: # unfortunately there is a single missing uniref seq
            s = json.dumps(js) + '\n'
            f.write(s)
            idx.append((pos, len(s), plddt, alphafill_plddt))
            pos += len(s)
        
print(count, flush=True)
idx = np.array(idx)
print(idx, flush=True)
with open(args.out+'.idx', 'wb') as f:
    np.save(f, idx)
