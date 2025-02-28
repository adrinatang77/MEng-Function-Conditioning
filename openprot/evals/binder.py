from .codesign import CodesignEval
import pandas as pd
import pickle
import torch
from ..utils.misc_utils import temp_seed
import numpy as np
from rdkit import Chem
from ..utils import residue_constants as rc
from ..data.data import OpenProtData
from ..generate.sampler import OpenProtSampler
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
from ..generate.sequence import SequenceUnmaskingStepper
from ..utils.prot_utils import write_mmcif
from ..utils.structure import Structure, Polypeptide, Ligand

class BinderEval(CodesignEval):
    def setup(self):
        super().setup()
       
        with open(self.cfg.path, 'rb') as f:
            self.ccd = pickle.load(f)
        
        self.keys = list(self.ccd.keys())
        with temp_seed(137):
            np.random.shuffle(self.keys)
        
    def __getitem__(self, idx):
        max_noise = self.cfg.struct.edm.sigma_max
        min_noise = self.cfg.struct.edm.sigma_min
        
        L = self.cfg.sample_length
        
        key = self.keys[idx]
        mol = self.ccd[self.keys[idx]]
        coords = mol.GetConformer().GetPositions()
        coords -= coords.mean(0)
        nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
        prot = self.make_data(
            name=f"sample{idx}_",
            seqres="A"*L,
            seq_mask=np.ones(L),
            seq_noise=np.ones(L),
            struct_noise=np.ones(L) * max_noise,
            struct=np.zeros((L, 3)),
            struct_mask=np.ones(L),
            residx=np.arange(L),
        )
        K = len(coords)
        target = self.make_data(
            name=key,
            seqres="*"*K,
            seq_mask=np.ones(K),
            struct=np.zeros((K, 3)), # coords + np.array([10, 0, 0]),
            struct_mask=np.ones(K),
            struct_noise=np.ones(K) * max_noise,
            atom_num=np.array(nums),
            mol_type=np.ones(K)*3,
            ref_conf=coords,
            chain=np.ones(K)
        )
        
        data = OpenProtData.concat([prot, target])
        data['dataset'] = prot['dataset']
        
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

        schedules = {
            'structure': self.struct_sched_fn,
            #'sequence': self.seq_sched_fn,
        }

        mask = batch['mol_type'] == 0
        sampler = OpenProtSampler(schedules, steppers=[
            EDMDiffusionStepper(self.cfg.struct, mask=None),
            #SequenceUnmaskingStepper(self.cfg.seq, mask=mask)
        ])

        # noisy_batch['pad_mask'] = (noisy_batch['mol_type'] == 0).float()
        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps, trunc=self.cfg.truncate)

        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        batch['struct'] = sample['struct']
        # batch['aatype'] = sample['aatype']
        
        datas = batch.unbatch()
        
        for i, data in enumerate(datas):

            L = self.cfg.sample_length
            # data.update_seqres()
            name = data["name"]
            struct = Structure.from_chains([
                Polypeptide(data['seqres'][:L], name='A'),
                Ligand([name.split('_')[-1]], name='B'),
            ])
            prot = struct.get_chain(0)
            prot.atoms['coords'][prot.residues['atom_center']] = data['struct'].cpu()[:L]
            prot.atoms['is_present'][prot.residues['atom_center']] = True

            lig = struct.get_chain(1)
            lig.atoms['coords'] = data['struct'].cpu()[L:]
            lig.atoms['is_present'] = True
            
            ref_str = struct.to_mmcif()
            
            pname, ccd = name.split('_')
            mol = self.ccd[ccd]
            smi = Chem.MolToSmiles(mol)
            with open(f"{savedir}/{name}.mmcif", "w") as f:
                f.write(ref_str)

            
            mask = datas[i]['mol_type'] == 0
            seq = prot.get_seqres()
            
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">protein|name={pname}\n")  # FASTA format header
                f.write(seq + "\n")
                f.write(f">ligand|name=LIG\n{smi}\n")

            # with Chem.SDWriter(f"{savedir}/{ccd}.sdf") as w:
            #     w.write(mol)

            # chai-lab fold sample0_M7K.fasta 
            # OST_COMPOUNDS_CHEMLIB=/data/cb/bjing/ccd.chemlib /data/cb/bjing/usr/openstructure/stage/bin/ost compare-ligand-structures 
            # -rl 
            # -m 1A3N.cif -r 1A3N.cif --fault-tolerant --min-pep-length 4 --min-nuc-length 4  --lddt --bb-lddt --qs-score --dockq --ics --ips --rigid-scores --patch-scores --tm-score

            # prot = make_ca_prot(
            #     sample['struct'][i].cpu().numpy(),
            #     sample["aatype"][i].cpu().numpy(),
            #     batch["struct_mask"][i].cpu().numpy(),
            # )
    
            # with open(f"{savedir}/{name}_traj.pdb", "w") as f:
            #     f.write(write_ca_traj(prot, samp_traj[:, i].cpu().numpy()))
    
            # with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
            #     f.write(write_ca_traj(prot, pred_traj[:, i].cpu().numpy()))



            # with open(f"{savedir}/{name}_traj.fasta", "w") as f:
            #     for seqs in extra['seq_traj']:
            #         seq = "".join([rc.restypes_with_x[aa] for aa in seqs[i]])
            #         seq = seq.replace('X', '-')
            #         f.write(seq+'\n')
                
                



        