import pandas as pd
import numpy as np
df = pd.read_csv('splits/cameo2022.csv', index_col='name')
from openprot.utils import protein, prot_utils

def longest_contig(seq):
    longest = (-1, -1)
    start = None
    for i, s in enumerate(seq):
        if (not start) and s:
            start = i # start of current contig
        elif start and not s:
            start = None # end of current contig
        if s and (i - start + 1) > longest[1] - longest[0]:
            longest = (start, i+1)
        
    return longest

for name in df.index:
    seqres = df.seqres[name]
    L = len(seqres)
    prot = dict(np.load(f"/scratch/projects/cgai/openprot/data/pdb_npz/{name[1:3]}/{name}.npz", allow_pickle=True))
    
    mask = prot['all_atom_mask'][:,1]
    start, end = longest_contig(mask)

    print(len(seqres), end-start)
    
    seqres = seqres[start:end]
    L = len(seqres)
    prot['all_atom_positions'] = prot['all_atom_positions'][start:end]
    prot['all_atom_mask'] = prot['all_atom_mask'][start:end]
    
    prot = protein.Protein(
        atom_positions=prot['all_atom_positions'],
        aatype=np.asarray(prot_utils.seqres_to_aatype(seqres)),
        atom_mask=prot['all_atom_mask'],
        residue_index=np.arange(L) + 1,
        b_factors=np.zeros((L, 37)),
        chain_index=np.zeros(L, dtype=int),
    )
    
    with open(f'cameo/{name}.pdb', 'w') as f:
        f.write(protein.to_pdb(prot))
import subprocess, os
for name in df.index:
    cmd = [
        "python",
        "../ProteinMPNN/protein_mpnn_run.py",
        "--pdb_path",
        f"cameo/{name}.pdb",
        "--ca_only",
        "--pdb_path_chains",
        "A",
        "--out_folder",
        "./cameo/",
        "--num_seq_per_target",
        str(8),
        "--sampling_temp",
        str(0.1),
        "--seed",
        str(37),
        "--batch_size",
        str(1),
    ]
    subprocess.run(cmd)