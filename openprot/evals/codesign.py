from .eval import OpenProtEval
# import foldcomp
from ..utils import protein
from ..utils.prot_utils import make_ca_prot, write_ca_traj
from ..utils.geometry import compute_lddt, rmsdalign, compute_rmsd
from ..utils import residue_constants as rc
from ..generate.sampler import OpenProtSampler
from ..generate.structure import EDMDiffusionStepper
from ..generate.sequence import SequenceUnmaskingStepper
import numpy as np
from ..tasks import StructurePrediction
import torch
import os, tqdm, math, subprocess
import pandas as pd


class CodesignEval(OpenProtEval):
    def setup(self):
        pass

    def run(self, model):
        NotImplemented

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        L = self.cfg.sample_length
        data = self.make_data(
            name=f"sample{idx}",
            seqres="A" * L,
            seq_mask=np.ones(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
            struct_noise=np.ones(L, dtype=np.float32) * self.cfg.sigma_max,
            atom37=np.zeros((L, 37, 3), dtype=np.float32),
            atom37_mask=np.ones((L, 37), dtype=np.float32),
        )
        return data

    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):

        if self.cfg.run_designability:
            torch.cuda.empty_cache()

            idx = list(range(rank, len(self), world_size))
            os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)
            for i in idx:
                cmd = ['cp', f"{savedir}/sample{i}.fasta", f"{savedir}/rank{rank}"]
                subprocess.run(cmd)
                
            cmd = [
                "bash",
                "scripts/switch_conda_env.sh",
                "python",
                "-m",
                "scripts.esmfold",
                "--outdir",
                f"{savedir}/rank{rank}",
                "--dir",
                f"{savedir}/rank{rank}",
                # "--print",
            ]
            
            out = subprocess.run(cmd)  
            
            for i in idx:
                
                with open(f"{savedir}/sample{i}.pdb") as f:
                    prot = protein.from_pdb_string(f.read())
                with open(f"{savedir}/rank{rank}/sample{i}.pdb") as f:
                    pred = protein.from_pdb_string(f.read())
                lddt = compute_lddt(
                    torch.from_numpy(pred.atom_positions[:,1]), 
                    torch.from_numpy(prot.atom_positions[:,1]), 
                    torch.from_numpy(prot.atom_mask[:,1])
                )
                rmsd = compute_rmsd(
                    torch.from_numpy(pred.atom_positions[:,1]),  
                    torch.from_numpy(prot.atom_positions[:,1])
                )
                if logger is not None:
                    logger.log(f"{self.cfg.name}/sclddt", lddt)
                    logger.log(f"{self.cfg.name}/scrmsd", rmsd)
                    logger.log(f"{self.cfg.name}/scrmsd<2", (rmsd < 2).float())
                

    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):

        def sched_fn(t):
            p = self.cfg.sched_p
            return (
                self.cfg.sigma_min ** (1 / p)
                + (1-t) * (self.cfg.sigma_max ** (1 / p) - self.cfg.sigma_min ** (1 / p))
            ) ** p

        sampler = OpenProtSampler(schedules={
            'structure': sched_fn,
            'sequence': lambda t: 1-t
        }, steppers=[
            EDMDiffusionStepper(self.cfg.struct),
            SequenceUnmaskingStepper(self.cfg.seq)
        ])
        
        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)

        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        B = len(sample['struct'])
        
        for i in range(B):
            prot = make_ca_prot(
                sample['struct'][i].cpu().numpy(),
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

            seq = "".join([rc.restypes_with_x[aa] for aa in sample["aatype"][i]])
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")
