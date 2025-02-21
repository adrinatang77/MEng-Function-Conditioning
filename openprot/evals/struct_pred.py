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
        prot = dict(
            np.load(f"{self.cfg.path}/{name[1:3]}/{name}.npz", allow_pickle=True)
        )
        seqres = self.df.seqres[name]
        L = len(seqres)
        data = self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=np.ones(len(seqres)),
            struct=prot["all_atom_positions"][:,1].astype(np.float32),
            struct_mask=prot["all_atom_mask"][:,1].astype(np.float32),
            struct_noise=np.ones(L, dtype=np.float32) * 160, # temporary
            residx=np.arange(L, dtype=np.float32),
        )

        return data

    def run_diffusion(self, model, batch, noisy_batch, savedir):
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
        
        sample_batch, extra = sampler.sample(model, noisy_batch, self.cfg.steps)
        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        B = len(sample_batch['struct'])
        
        for i in range(B):
            prot = make_ca_prot(
                sample_batch['struct'][i].cpu().numpy(),
                batch["aatype"][i].cpu().numpy(),
                batch["struct_mask"][i].cpu().numpy(),
            )
    
            ref_str = protein.to_pdb(prot)
            name = batch["name"][i]
            with open(f"{savedir}/{name}.pdb", "w") as f:
                f.write(ref_str)
    
            with open(f"{savedir}/{name}_traj.pdb", "w") as f:
                f.write(write_ca_traj(prot, samp_traj[:, i].cpu().numpy()))
    
            with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
                f.write(write_ca_traj(prot, pred_traj[:, i].cpu().numpy()))

        return sample_batch['struct']

    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):

        noisy_batch["struct"] = torch.randn_like(noisy_batch["struct"]) * 160.
        if self.cfg.diffusion:
            coords = self.run_diffusion(model, batch, noisy_batch, savedir)
        else:
            _, readout = model.forward(noisy_batch)
            coords = readout["trans"][-1]

        
        coords = rmsdalign(batch["struct"], coords, batch["struct_mask"])
        B = len(coords)
        for i in range(B):
            aatype = batch["aatype"].cpu().numpy()[i]
            
            lddt = compute_lddt(
                coords[i],
                batch["struct"][i],
                batch["struct_mask"][i]
            )
            
            tmscore = compute_tmscore(  # second is reference
                coords1=coords.cpu().numpy()[i],
                coords2=batch["struct"].cpu().numpy()[i],
                seq1=aatype,
                seq2=aatype,
                mask1=None,
                mask2=batch["struct_mask"].cpu().numpy()[i],
                seq=True
            )
    
            rmsd = compute_rmsd(batch["struct"][i], coords[i], batch["struct_mask"][i])
            if logger:
                logger.log(f"{self.cfg.name}/lddt", lddt)
                logger.log(f"{self.cfg.name}/tm", tmscore["tm"])
                logger.log(f"{self.cfg.name}/gdt_ts", tmscore["gdt_ts"])
                logger.log(f"{self.cfg.name}/gdt_ha", tmscore["gdt_ha"])
                logger.log(f"{self.cfg.name}/rmsd", rmsd)
    
            prot = make_ca_prot(
                coords=batch["struct"].cpu().numpy()[i],
                aatype=aatype,
                mask=batch["struct_mask"].cpu().numpy()[i],
            )
            ref_str = protein.to_pdb(prot)
            
            prot.atom_mask[..., 1] = batch["pad_mask"].cpu().numpy()[i]
            prot.atom_positions[..., 1, :] = coords.cpu().numpy()[i]
            pred_str = protein.to_pdb(prot)
    
            ref_str = "\n".join(ref_str.split("\n")[1:-3])
            pred_str = "\n".join(pred_str.split("\n")[1:-3])
    
            name = batch["name"][i]
            with open(f"{savedir}/{name}_{lddt.item():.2f}.pdb", "w") as f:
                f.write("\nENDMDL\nMODEL\n".join([ref_str, pred_str]))
