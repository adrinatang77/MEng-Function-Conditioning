from .eval import OpenProtEval
# import foldcomp
from ..utils import protein
from ..utils.geometry import compute_lddt, rmsdalign, compute_rmsd
from ..utils import residue_constants as rc
from ..utils.prot_utils import make_ca_prot, compute_tmscore, write_ca_traj
from ..generate.sampler import OpenProtSampler
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
import numpy as np
import torch
import os, tqdm
import pandas as pd


class StructurePredictionEval(OpenProtEval):
    def setup(self):
        self.df = pd.read_csv(self.cfg.split, index_col="name")

    def run(self, model):
        NotImplemented

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.index[idx]
  
        seqres = self.df.seqres[name]
        L = len(seqres)
        data = self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=np.ones(L),
            struct=np.zeros((L,3)),
            struct_mask=np.ones(L),
            struct_noise=np.ones(L) * self.cfg.struct.edm.sigma_max,
            residx=np.arange(L),
        )

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

        
        sampler = OpenProtSampler(schedules={
            'structure': edm_sched_fn,
        }, steppers=[
            EDMDiffusionStepper(self.cfg.struct)
        ])
        
        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)
        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        for i, name in enumerate(batch['name']):
            prot = make_ca_prot(
                sample['struct'][i].cpu().numpy(),
                sample["aatype"][i].cpu().numpy(),
                sample["struct_mask"][i].cpu().numpy(),
            )
            
            with open(f"{savedir}/{name}_traj.pdb", "w") as f:
                f.write(write_ca_traj(prot, samp_traj[:, i].cpu().numpy()))
    
            with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
                f.write(write_ca_traj(prot, pred_traj[:, i].cpu().numpy()))

        batch['struct'] = sample['struct']
        
        for i, data in enumerate(batch.unbatch()):

            name = data["name"]
            
            ref = dict(np.load(f"{self.cfg.path}/{name[1:3]}/{name}.npz", allow_pickle=True))
            
            ref_pos = torch.from_numpy(ref['all_atom_positions'][:,1]).to(data['struct'])
            mask = torch.from_numpy(ref['all_atom_mask'][:,1]).to(data['struct'])

            lddt = compute_lddt(data['struct'], ref_pos, mask)
            rmsd = compute_rmsd(data['struct'], ref_pos, mask)
            
            # second is reference
            aatype = data['aatype'].cpu().numpy()
            tmscore = compute_tmscore(
                coords1=data['struct'].cpu(),
                coords2=ref_pos.cpu(),
                seq1=aatype,
                seq2=aatype,
                mask1=None,
                mask2=mask.cpu(),
                seq=True
            )
            
            if logger:
                logger.log(f"{self.cfg.name}/lddt", lddt)
                logger.log(f"{self.cfg.name}/tm", tmscore["tm"])
                logger.log(f"{self.cfg.name}/gdt_ts", tmscore["gdt_ts"])
                logger.log(f"{self.cfg.name}/gdt_ha", tmscore["gdt_ha"])
                logger.log(f"{self.cfg.name}/rmsd", rmsd)
    

            prot = make_ca_prot(
                data['struct'].cpu().numpy(),
                data["aatype"].cpu().numpy(),
                data["struct_mask"].cpu().numpy(),
            )
            pred_str = protein.to_pdb(prot)
            prot = make_ca_prot(
                ref['all_atom_positions'][:,1],
                data["aatype"].cpu().numpy(), # ref['aatype'] is one-hot...
                ref['all_atom_mask'][:,1],
            )
            ref_str = protein.to_pdb(prot)
            
            
            ref_str = "\n".join(ref_str.split("\n")[1:-3])
            pred_str = "\n".join(pred_str.split("\n")[1:-3])
    
            name = batch["name"][i]
            with open(f"{savedir}/{name}_{lddt.item():.2f}.pdb", "w") as f:
                f.write("\nENDMDL\nMODEL\n".join([ref_str, pred_str]))
