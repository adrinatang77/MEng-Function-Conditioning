import tempfile
import subprocess
from . import protein
from . import residue_constants as rc
import numpy as np

def seqres_to_aatype(seq):
    return [rc.restype_order.get(c, rc.unk_restype_index) for c in seq]

def write_ca_traj(prot, traj):
    strs = []
    for coords in traj:
        prot.atom_positions[..., 1, :] = coords
        str_ = protein.to_pdb(prot)
        str_ = "\n".join(str_.split("\n")[1:-3])
        strs.append(str_)
    return "\nENDMDL\nMODEL\n".join(strs)

def make_ca_prot(coords, aatype, mask=None):
    L = len(coords)
    if type(aatype) is str:
        aatype = seqres_to_aatype(aatype)
    prot = protein.Protein(
        atom_positions=np.zeros((L, 37, 3)),
        aatype=aatype,
        atom_mask=np.zeros((L, 37)),
        residue_index=np.arange(L) + 1,
        b_factors=np.zeros((L, 37)),
        chain_index=np.zeros(L, dtype=int),
    )
    if mask is not None:
        prot.atom_mask[..., 1] = mask
    else:
        prot.atom_mask[..., 1] = 1.0
    prot.atom_positions[..., 1, :] = coords
    return prot

def compute_tmscore(coords1, coords2, seq1, seq2, mask1=None, mask2=None):

    path1 = tempfile.NamedTemporaryFile()
    prot1 = make_ca_prot(coords1, seq1, mask=mask1)
    open(path1.name, 'w').write(protein.to_pdb(prot1))

    path2 = tempfile.NamedTemporaryFile()
    prot2 = make_ca_prot(coords2, seq2, mask=mask2)
    open(path2.name, 'w').write(protein.to_pdb(prot2))
    try:
        out = subprocess.check_output(
            ['TMscore', '-seq', path1.name, path2.name], 
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
        
    start = out.find(b'RMSD')
    end = out.find(b'rotation')
    out = out[start:end]
    
    rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split(b'\n')
    
    rmsd = float(rmsd.split(b'=')[-1])
    tm = float(tm.split(b'=')[1].split()[0])
    gdt_ts = float(gdt_ts.split(b'=')[1].split()[0])
    gdt_ha = float(gdt_ha.split(b'=')[1].split()[0])

    return {'rmsd': rmsd, 'tm': tm, 'gdt_ts': gdt_ts, 'gdt_ha': gdt_ha}