import numpy as np
from boltz.data.const import prot_token_to_letter, dna_token_to_letter, rna_token_to_letter, prot_letter_to_token
import ihm
import io
import modelcif
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem
from .mmcif import parse_mmcif

periodic_table = Chem.GetPeriodicTable()

import pickle
with open('/data/cb/scratch/datasets/boltz/ccd.pkl', 'rb') as f:
    CCD = pickle.load(f)

ATOM = [
    ("name", np.dtype("4i1")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("coords", np.dtype("3f4")),
    ("conformer", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("chirality", np.dtype("i1")),
]

RESIDUE = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

CHAIN = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
]

def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)

def unconvert_atom_name(tup):
    # name = name.strip()
    # name = [ord(c) - 32 for c in name]
    # name = name + [0] * (4 - len(name))
    # return tuple(name)
    return "".join(chr(n+32) for n in tup).strip()

class Residue:
    def __repr__(self):
        return f"Residue(name={self.name}, type={self.res_type}, idx={self.res_idx}) with {len(self.atoms)} atoms"
    def to_rdkit(self, ccd):
        NotImplemented

    def get_atom_names(self):
        return [unconvert_atom_name(tup) for tup in self.atoms['name']]
        

    def get_letter(self):
        if self.mol_type == 0:
            return prot_token_to_letter.get(self.name, 'X')
        elif self.mol_type == 1:
            return dna_token_to_letter.get(self.name, 'X')
        elif self.mol_type == 2:
            return rna_token_to_letter.get(self.name, 'X')
        else:
            return None

    def from_rdkit(mol, use_coords=True):
        self = Residue()
        atoms = []
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        for a in mol.GetAtoms():
            atoms.append((
                convert_atom_name(a.GetProp('name')),
                a.GetAtomicNum(),
                a.GetFormalCharge(),
                tuple(pos[a.GetIdx()]) if use_coords else (0, 0, 0),
                tuple(pos[a.GetIdx()]),
                False,
                0, # not used
            ))
        self.atoms = np.array(atoms, dtype=ATOM)
        return self
        

class Chain:

    def __repr__(self):
        return f"Chain(name={self.name} mol_type={self.mol_type}) with {len(self.residues)} residues, {len(self.atoms)} atoms"

    def get_residues(self):
        return [self.get_residue(i) for i in range(len(self.residues))]
        
    def get_residue(self, idx):
        resi = Residue()
        row = self.residues[idx]

        astart, acount = row['atom_idx'], row['atom_num']
        
        resi.name = row['name']
        resi.mol_type = self.mol_type
        resi.res_type = row['res_type']
        resi.res_idx = row['res_idx']
        resi.atom_center = row['atom_center'] - astart
        resi.atom_disto = row['atom_disto'] - astart
        resi.is_standard = row['is_standard']
        resi.is_present = row['is_present']

        resi.atoms = self.atoms[astart:astart+acount]
        return resi

    def get_seqres(self):
        return ''.join([r.get_letter() for r in self.get_residues()])

    def get_seqres_mask(self):
        seqres = self.get_seqres()
        seq_mask = np.ones(len(seqres))
        if self.mol_type == 0:
            UNK = 'X'
        else:
            UNK = 'N'
        seq_mask[[c == UNK for c in seqres]] = 0
        return seq_mask
        
    def get_central_atoms(self):
        return self.atoms[self.residues['atom_center']]

    def get_entity(self):
        if self.mol_type == 0:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        elif self.mol_type == 1:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif self.mol_type == 2:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        else:
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

        if self.mol_type < 3:
            seq = [
                alphabet[c] if c in alphabet else chem_comp(item)
                for c in self.get_seqres()
            ]
        else:
            seq = [chem_comp(name) for name in self.residues['name']]
        
        return Entity(seq)

    def get_atom_residx(self):
        residx = self.residues['atom_idx']
        residx = (np.arange(len(self.atoms)) >= residx[:,None]).sum(0)
        residx = self.residues['res_idx'][residx-1]
        return residx

class Polypeptide(Chain):
    def __init__(self, seqres, name='A'):
        residues = []
        
        atoms = []
        
        for i, c in enumerate(seqres):
            key = prot_letter_to_token[c]
            mol = CCD[key]
            resi = Residue.from_rdkit(mol, use_coords=False)
            
            residues.append((
                key,
                0, # not used
                i,
                len(atoms),
                len(resi.atoms),
                len(atoms) + resi.get_atom_names().index('CA'),
                len(atoms), # not used
                True,
                True,
            ))    
            atoms.extend(resi.atoms)
        self.atoms = np.array(atoms, dtype=ATOM)
        self.residues = np.array(residues, dtype=RESIDUE)
        
        self.mol_type = 0
        self.name = name
        self.idx = -1 


class Ligand(Chain):
    
        
    def __init__(self, seq, name='A'):
        residues = []
        
        atoms = []
        
        for i, key in enumerate(seq):
            mol = CCD[key]
            resi = Residue.from_rdkit(mol, use_coords=False)
            
            residues.append((
                key,
                0, # not used
                i,
                len(atoms),
                len(resi.atoms),
                len(atoms), # not used
                len(atoms), # not used
                True,
                True,
            ))    
            atoms.extend(resi.atoms)
        self.atoms = np.array(atoms, dtype=ATOM)
        self.residues = np.array(residues, dtype=RESIDUE)
        
        self.mol_type = 3
        self.name = name
        self.idx = -1 

class Structure:
    def __repr__(self):
        return f"Structure with {len(self.chains)} chains, {len(self.residues)} residues, {len(self.atoms)} atoms"
    def from_chains(chains_):
        self = Structure()
        atoms = []
        residues = []
        chains = []
        
        for i, chain in enumerate(chains_):
            chains.append((
                chain.name,
                chain.mol_type,
                0, # entity_id, unused
                0, # sym_id, unused
                i,
                len(atoms),
                len(chain.atoms),
                len(residues),
                len(chain.residues),
            ))
            
            chain_residues = np.copy(chain.residues)
            chain_residues['atom_idx'] += len(atoms)
            chain_residues['atom_center'] += len(atoms)
            chain_residues['atom_disto'] += len(atoms)
            
            residues.extend(chain_residues)
            atoms.extend(chain.atoms)
            
        self.atoms = np.array(atoms, dtype=ATOM)
        self.residues = np.array(residues, dtype=RESIDUE)
        self.chains = np.array(chains, dtype=CHAIN)
        return self

    def get_chains(self):
        return [self.get_chain(i) for i in range(len(self.chains))]
    
        
    def from_npz(path):
        self = Structure()
        npz = np.load(path)
        self.chains = npz['chains']
        self.residues = npz['residues']
        self.atoms = npz['atoms']
        return self
        
    def from_mmcif(path):
        self = Structure()
        data = parse_mmcif(path, CCD)
        self.chains = data.data.chains
        self.residues = data.data.residues
        self.atoms = data.data.atoms
        return self
        
    def to_mmcif(self):
        asyms = []
        atoms = []

        for chain in self.get_chains():
            asym = AsymUnit(chain.get_entity(), id=chain.name)
            
            for residue in chain.get_residues():
                for atom in residue.atoms:
                    symbol = periodic_table.GetElementSymbol(
                        int(atom['element'])
                    ).upper()
                    x, y, z = atom['coords']
                    name = unconvert_atom_name(atom['name'])
                    if atom['is_present']:
                        
                        atoms.append(Atom(
                            asym_unit=asym,
                            type_symbol=symbol,
                            seq_id=residue.res_idx+1,
                            atom_id=name,
                            x=x,
                            y=y,
                            z=z,
                            het=(chain.mol_type==3),
                            biso=100,
                            occupancy=1,
                        ))
            asyms.append(asym)
        
        model = AbInitioModel(Assembly(asyms))
        for at in atoms: model.add_atom(at)
        system = System()
        system.model_groups.append(ModelGroup([model]))
        fh = io.StringIO()
        dumper.write(fh, [system])
        out = fh.getvalue()
        return out

    def get_chain(self, idx, key=None):
        
        chain = Chain()
        
        row = self.chains[idx]
        chain.idx = idx
        chain.name = row['name']
        chain.mol_type = row['mol_type']
        
        rstart, rcount = row['res_idx'], row['res_num']
        astart, acount = row['atom_idx'], row['atom_num']
        
        chain.residues = np.copy(self.residues[rstart:rstart+rcount])
        chain.residues['atom_idx'] -= astart
        chain.residues['atom_center'] -= astart
        chain.residues['atom_disto'] -= astart
        
        chain.atoms = self.atoms[astart:astart+acount]
        
        return chain

# struct = Structure.from_npz(f"/data/cb/scratch/datasets/boltz/processed_data/structures/a3/1a3n.npz")

# chain = struct.get_chain(0)
# resi = chain.get_residue(0)
# print(resi)
if __name__ == '__main__':
    struct = Structure.from_chains([
        Polypeptide('MKVLTVTTVL', name='A'),
        Ligand(['HEM'], name='B'),
    ])
    struct = Structure.from_mmcif(
        'workdir/default/eval_step0/binder/sample0_M7K.mmcif'
    )
    with open('out.cif', 'w') as f:
        f.write(struct.to_mmcif())