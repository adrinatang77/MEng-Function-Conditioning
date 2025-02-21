from .eval import OpenProtEval
# import foldcomp
from ..utils import protein
from ..utils.prot_utils import make_ca_prot, write_ca_traj
from ..utils.geometry import compute_lddt, rmsdalign
from ..utils import residue_constants as rc
from ..generate.sampler import OpenProtSampler
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
import numpy as np
from ..utils.prot_utils import compute_tmscore
from ..tasks import StructurePrediction
import torch
import os, tqdm, math, subprocess
import pandas as pd


class StructureGenerationEval(OpenProtEval):
    def setup(self):
        pass

    def run(self, model):
        NotImplemented

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        L = self.cfg.sample_length
        if self.cfg.struct.rescale_time:
            max_noise = self.cfg.struct.sigma_max
        else:
            max_noise = self.cfg.struct.max_t
        data = self.make_data(
            name=f"sample{idx}",
            seqres="A" * L,
            seq_mask=np.ones(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
            struct_noise=np.ones(L, dtype=np.float32) * max_noise,
            atom37=np.zeros((L, 37, 3), dtype=np.float32),
            atom37_mask=np.ones((L, 37), dtype=np.float32),
            residx=np.arange(L, dtype=np.float32),
        )
        return data


    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):

        if self.cfg.run_designability:
            torch.cuda.empty_cache()

            ## distributed designability
            count = math.ceil(self.cfg.num_samples / world_size)
            start = rank * count
            end = min((rank + 1) * count, self.cfg.num_samples)

            cmd = [
                "bash",
                "scripts/run_genie_pipeline.sh",
                savedir,
                str(start),
                str(end - 1),
            ]
            cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cvd:
                dev = cvd.split(',')[torch.cuda.current_device()]
            else:
                dev = torch.cuda.current_device()
            subprocess.run(cmd, env=os.environ | {
                'CUDA_VISIBLE_DEVICES': str(dev)
            })  

            df = pd.read_csv(
                f"{savedir}/eval{start}_{end-1}/info.csv", index_col="domain"
            )
            df["designable"] = df["scRMSD"] < 2
            if logger is not None:
                for col in df.columns:
                    for val in df[col].tolist():
                        logger.log(f"{self.cfg.name}/{col}", val)

            # os.makedirs(f"{savedir}/designable", exist_ok=True)
            # for name in df[df.scRMSD < 100].index:
            #     subprocess.run([
            #         'cp',
            #         f"{savedir}/eval{start}_{end-1}/designs/{name}.pdb",
            #         f"{savedir}/designable"
            #     ])

        
        if self.cfg.run_diversity:
            idx = list(range(rank, len(self), world_size))
            tm_arr = np.zeros((len(self), len(self)))
            for i in idx:
                with open(f"{savedir}/sample{i}.pdb") as f:
                    prot1 = protein.from_pdb_string(f.read())
                for j in range(len(self)):
                    with open(f"{savedir}/sample{j}.pdb") as f:
                        prot2 = protein.from_pdb_string(f.read())
                    tm_arr[i,j] = compute_tmscore(  # second is reference
                        coords1=prot1.atom_positions[:,1],
                        coords2=prot2.atom_positions[:,1],
                    )['tm']
            np.save(f"{savedir}/tmscores{rank}.npy", tm_arr)
            if world_size > 1:
                torch.distributed.barrier()
                
            # every rank has to compute it, otherwise error
            tm_arr = np.zeros((len(self), len(self)))
            for r in range(world_size):
                tm_arr += np.load(f"{savedir}/tmscores{r}.npy")
            eigvals = np.linalg.eigvals(tm_arr / len(self))
            vendi = np.e**np.nansum(-eigvals * np.log(eigvals))
            if logger is not None:
                logger.log(f"{self.cfg.name}/vendi", vendi)


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
            p = self.cfg.struct.sched_p
            sigma_max = self.cfg.struct.sigma_max
            sigma_min = self.cfg.struct.sigma_min 
            return (
                sigma_min ** (1 / p)
                + (1-t) * (sigma_max ** (1 / p) - sigma_min ** (1 / p))
            ) ** p

        
        StructureStepper = {
            'EDMDiffusion': EDMDiffusionStepper,
            'GaussianFM': GaussianFMStepper,
        }[self.cfg.struct.type]
        max_t = self.cfg.struct.max_t
        def log_sched_fn(t):
            exp = 10 ** (-2*t)
            return max_t * (exp - 1e-2) / (1 - 1e-2)

        if self.cfg.struct.sched == 'linear':
            sched_fn = lambda t: max_t * (1-t)
        elif self.cfg.struct.sched == 'edm':
            sched_fn = edm_sched_fn
        elif self.cfg.struct.sched == 'log':
            sched_fn = log_sched_fn
        else:
            raise Exception("unrecognized schedule")

        sampler = OpenProtSampler(schedules={
            'structure': sched_fn,
        }, steppers=[
            StructureStepper(self.cfg.struct)
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
