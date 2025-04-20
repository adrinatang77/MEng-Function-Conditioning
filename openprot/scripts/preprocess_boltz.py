import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', default='/data/cb/scratch/datasets/boltz/processed_data/manifest.json')
parser.add_argument('--out', default='/data/cb/scratch/datasets/boltz/processed_data/manifest.csv')
args = parser.parse_args()

df = []

from collections import defaultdict
import json, tqdm
import pandas as pd
with open(args.manifest) as f:
    manifest = json.load(f)


hash_to_idx = {}
def get_cluster_id(key):
    if key not in hash_to_idx:
        hash_to_idx[key] = len(hash_to_idx)
    return hash_to_idx[key]

def is_prot(chain):
    return int(chain['mol_type'] == 0)
    
def is_lig(chain):
    return int(chain['mol_type'] == 3)

def is_nuc(chain):
    return int(chain['mol_type'] == 1 or chain['mol_type'] == 2)

for entry in tqdm.tqdm(manifest):
    for chain in entry['chains']:
        if not chain['valid']: continue
        key = chain['cluster_id']
        if len(key) == 64:
            key = get_cluster_id(key)
                
        df.append({
            'name': f"{entry['id']}_{chain['chain_name']}",
            'alt_name': f"{entry['id']}_{chain['chain_id']}",
            'release_date': entry['structure']['released'],
            'n_prot': is_prot(chain),
            'n_nuc': is_nuc(chain),
            'n_ligand': is_lig(chain),
            'num_residues': chain['num_residues'],
            'cluster_id': str(key)
        })
    for interface in entry['interfaces']:
        if not interface['valid']: continue
        chain1 = entry['chains'][interface['chain_1']]
        chain2 = entry['chains'][interface['chain_2']]
        key1, key2 = chain1['cluster_id'], chain2['cluster_id']
        if len(key1) == 64:
            key1 = str(get_cluster_id(key1))
        if len(key2) == 64:
            key2 = str(get_cluster_id(key2))
        key = ':'.join(sorted([key1, key2]))
        df.append({
            'name': f"{entry['id']}_{chain1['chain_name']}_{chain2['chain_name']}",
            'alt_name': f"{entry['id']}_{chain1['chain_id']}_{chain2['chain_id']}",
            'release_date': entry['structure']['released'],
            'n_prot': is_prot(chain1) + is_prot(chain2),
            'n_nuc': is_nuc(chain1) + is_nuc(chain2),
            'n_ligand': is_lig(chain1) + is_lig(chain2),
            'num_residues': chain1['num_residues'] + chain2['num_residues'],
            'cluster_id': key
        })


df = pd.DataFrame(df).set_index('name')
df.to_csv(args.out)
        
