from .eval import OpenProtEval
import pandas as pd
from ..utils.structure import Structure
import os
from ..data.data import OpenProtData
from ..generate.sampler import OpenProtSampler
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
from ..utils.structure import Structure, Polypeptide, Ligand
import numpy as np
import pickle
import torch

def masked_center(x, mask=None, eps=1e-5):
    if mask is None:
        return x - x.mean(-2, keepdims=True)
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return np.where(mask, x - com, x)
    
class DockingEval(OpenProtEval):
    def setup(self):
        df = pd.read_csv(self.cfg.split, index_col="system_id")
        df = df[df['num_protein_chains'] == 1]
        df = df[df['num_ligand_chains'] == 1]
        keys = []
        for key in df.index:
            pdb_id, _, prot, lig = key.split('__') 
            if os.path.exists(f"{self.cfg.path}/structures/{pdb_id[1:3]}/{pdb_id}.npz"):
                keys.append(key)
        
        self.df = df.loc[keys]
        with open(self.cfg.ccd, 'rb') as f:
            self.ccd = pickle.load(f)
        
    def run(self, model):
        NotImplemented

    def unpack_chain(self, chain):

        if chain.mol_type == 3:
            L = len(chain.atoms)
        else:
            L = len(chain.residues)
            
        ones = np.ones(L, dtype=np.float32)

        if chain.mol_type == 3:
            return {
                'seqres': '*'*L,
                'atom_num': chain.atoms['element'],
                'mol_type': ones * chain.mol_type,
                'seq_mask': ones,
                'struct': chain.atoms['coords'] * 0.0,
                'struct_mask': chain.atoms['is_present'],
                'ref_conf': chain.atoms['conformer'],
                'residx': chain.get_atom_residx(),
                'chain': ones * chain.idx,
            }
        else:
            return {
                'seqres': chain.get_seqres(),
                'atom_num': np.zeros(L),
                'mol_type': ones * chain.mol_type,
                'seq_mask': chain.get_seqres_mask(),
                'struct': chain.get_central_atoms()['coords'],
                'struct_mask': chain.get_central_atoms()['is_present'],
                'ref_conf': np.zeros((L, 3)),
                'residx': chain.residues['res_idx'],
                'chain': ones * chain.idx,
            }
                

    def __len__(self):
        return 100 # len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        pdb_id, _, prot, lig = self.df.index[idx].split('__')        
        prot = prot.split('.')[-1]
        lig = lig.split('.')[-1]
        struct = Structure.from_npz(
            f"{self.cfg.path}/structures/{pdb_id[1:3]}/{pdb_id}.npz"
        )

        prot_data = self.make_data(
            name=f"{pdb_id}_{prot}_",
            **self.unpack_chain(struct.get_chain(key=prot))
        )
        prot_data['struct_noise'].fill(self.cfg.struct.edm.sigma_min)
        # lig = self.make_data(
        #     name=f"_{item.ligand_ccd_code}",
        #     **self.unpack_chain(struct.get_chain(key=lig))
        # )
        prot_data['struct'] = masked_center(
            prot_data['struct'],
            prot_data['struct_mask']
        )
        mol = self.ccd[item.ligand_ccd_code]
        coords = mol.GetConformer().GetPositions()
        coords -= coords.mean(0)
        nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
        
        K = len(coords)
        lig_data = self.make_data(
            name=item.ligand_ccd_code,
            seqres="*"*K,
            seq_mask=np.ones(K),
            struct=np.zeros((K, 3)), 
            struct_mask=np.ones(K),
            struct_noise=np.ones(K) * self.cfg.struct.edm.sigma_max,
            atom_num=np.array(nums),
            mol_type=np.ones(K)*3,
            ref_conf=coords,
            chain=np.ones(K) * int(np.argwhere(struct.chains['name'] == lig))
        )
        
        data = OpenProtData.concat([prot_data, lig_data])
        data['dataset'] = prot_data['dataset']
        data['ref'] = Structure.from_chains([
            struct.get_chain(key=prot),
            struct.get_chain(key=lig),
        ])
        return data
        
    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):

        def edm_sched_fn(t):
            p = self.cfg.struct.edm.sched_p
            sigma_max = self.cfg.struct.edm.sigma_max
            sigma_min = self.cfg.struct.edm.sigma_min 
            return (
                sigma_min ** (1 / p)
                + (1-t) * (sigma_max ** (1 / p) - sigma_min ** (1 / p))
            ) ** p

        schedules = {
            'structure': edm_sched_fn
        }

        mask = batch['mol_type'] == 3
        sampler = OpenProtSampler(schedules, steppers=[
            EDMDiffusionStepper(self.cfg.struct, mask=mask),
        ])

        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)

        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        batch['struct'] = sample['struct']
        # batch['aatype'] = sample['aatype']
        
        datas = batch.unbatch()
        
        for i, data in enumerate(datas):

            L = (data['mol_type'] == 0).sum().item()
            # data.update_seqres()
            
            name = data["name"]
            struct = Structure.from_chains([
                Polypeptide(data['seqres'][:L], name='A'),
                Ligand([name.split('_')[-1]], name='B'),
            ])
            struct.chains['name'] = data['ref'].chains['name']
            
            prot = struct.get_chain(0)
            prot.atoms['coords'][prot.residues['atom_center']] = data['struct'].cpu()[:L]
            prot.atoms['is_present'][prot.residues['atom_center']] = data['struct_mask'].cpu()[:L].bool()

            lig = struct.get_chain(1)
            lig.atoms['coords'] = data['struct'].cpu()[L:]
            lig.atoms['is_present'] = True
            
            ref_str = struct.to_mmcif()
            
            _, pname, ccd = name.split('_')
            with open(f"{savedir}/{name}.mmcif", "w") as f:
                f.write(ref_str)
            
            rmsd = data['ref'].ligand_rmsd(struct)
            distortion = struct.get_chain(1).get_residue(0).get_distortion()
            print(rmsd, distortion)
            if logger is not None:
                logger.log(f"{self.cfg.name}/rmsd", rmsd)
                logger.log(f"{self.cfg.name}/distortion", distortion)
            