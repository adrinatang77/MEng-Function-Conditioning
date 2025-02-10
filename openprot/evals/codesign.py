from .eval import OpenProtEval
# import foldcomp
from ..utils import protein
from ..utils.prot_utils import make_ca_prot, write_ca_traj, compute_tmscore
from ..utils.geometry import compute_lddt, rmsdalign, compute_rmsd
from ..utils import residue_constants as rc
from ..generate.sampler import OpenProtSampler
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
from ..generate.sequence import SequenceUnmaskingStepper
import numpy as np
from ..tasks import StructurePrediction
import torch
import os, tqdm, math, subprocess
import pandas as pd
from biopandas.pdb import PandasPdb
from ..utils.secondary import assign_secondary_structures
from collections import defaultdict

class CodesignEval(OpenProtEval):
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
            seqres="A"*L,
            seq_mask=np.ones(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
            struct_noise=np.ones(L, dtype=np.float32) * max_noise,
            struct=np.zeros((L, 3), dtype=np.float32),
            struct_mask=np.ones(L, dtype=np.float32),
            residx=np.arange(L, dtype=np.float32),
        )
        return data

    def run_designability(self, idx, rank, world_size, savedir, logger, df):
        for i in idx:
            cmd = ['cp', f"{savedir}/sample{i}.fasta", f"{savedir}/rank{rank}"]
            subprocess.run(cmd)
                
        cmd = [
            "bash",
            "scripts/switch_conda_env.sh",
            "eval",
            "python",
            "-m",
            "scripts.esmfold",
            "--outdir",
            f"{savedir}/rank{rank}",
            "--dir",
            f"{savedir}/rank{rank}",
            # "--print",
            "--device",
            str(torch.cuda.current_device())
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
            tmscore = compute_tmscore(  # second is reference
                coords1=pred.atom_positions[:,1],
                coords2=prot.atom_positions[:,1],
            )['tm']

            plddt = PandasPdb().read_pdb(f"{savedir}/rank{rank}/sample{i}.pdb").df['ATOM']['b_factor'].mean()
            
            if logger is not None:
                logger.log(f"{self.cfg.name}/sclddt", lddt)
                logger.log(f"{self.cfg.name}/scrmsd", rmsd)
                logger.log(f"{self.cfg.name}/scrmsd<2", (rmsd < 2).float())
                logger.log(f"{self.cfg.name}/scTM", tmscore)
                logger.log(f"{self.cfg.name}/sclddt>80", (lddt > 0.8).float())
                logger.log(f"{self.cfg.name}/scTM>80", tmscore > 0.8)
                logger.log(f"{self.cfg.name}/plddt", plddt)

            df[f"sample{i}"]["plddt"] = plddt
            df[f"sample{i}"]["scrmsd"] = float(rmsd)
            df[f"sample{i}"]["sctm"] = tmscore
            df[f"sample{i}"]["sclddt"] = float(lddt)
    

    def run_diversity(self, idx, rank, world_size, savedir, logger, df):
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
        np.save(f"{savedir}/rank{rank}/tmscores.npy", tm_arr)
        if world_size > 1:
            torch.distributed.barrier()
            
        # every rank has to compute it, otherwise error
        tm_arr = np.zeros((len(self), len(self)))
        for r in range(world_size):
            tm_arr += np.load(f"{savedir}/rank{r}/tmscores.npy")
        eigvals = np.linalg.eigvals(tm_arr / len(self))
        vendi = np.e**np.nansum(-eigvals * np.log(eigvals))
        if logger is not None:
            logger.log(f"{self.cfg.name}/vendi", vendi)

    def run_secondary(self, idx, rank, world_size, savedir, logger, df):
        for i in idx:
            with open(f"{savedir}/sample{i}.pdb") as f:
                prot = protein.from_pdb_string(f.read())
            ss = assign_secondary_structures(
                torch.from_numpy(prot.atom_positions[None,:,1]), 
                full=False,
                return_encodings=False
            )[0]
            
            pos = prot.atom_positions[:,1]
            L = len(pos)
            pos = pos - pos.mean(0, keepdims=True)
            ii, jj = np.meshgrid(np.arange(L), np.arange(L))
            dmat = np.square(pos[None] - pos[:,None]).sum(-1) ** 0.5
            lrc = (np.abs(ii-jj) > 12) & (dmat < 8)
            lrc = lrc.sum() / L
            
            if logger is not None:
                logger.log(f"{self.cfg.name}/helix", ss.count('h') / len(ss))
                logger.log(f"{self.cfg.name}/sheet", ss.count('s') / len(ss))
                logger.log(f"{self.cfg.name}/loop", ss.count('-') / len(ss))
                logger.log(f"{self.cfg.name}/lrc", lrc)
            df[f"sample{i}"]["helix"] = ss.count('h') / len(ss)
            df[f"sample{i}"]["sheet"] = ss.count('s') / len(ss)
            df[f"sample{i}"]["loop"] = ss.count('-') / len(ss)
            df[f"sample{i}"]["lrc"] = lrc


    def make_plot(self, idx, rank, world_size, savedir, logger, df):
        cmd = [
            "bash",
            "scripts/switch_conda_env.sh",
            "pymol",
            "python",
            "-m",
            "scripts.visualize",
            "--dir",
            savedir,
            "--out",
            f"{savedir}/out.png",
            "--annotate",
        ]
        out = subprocess.run(cmd) 

    def run_pmpnn_designability(self, idx, rank, world_size, savedir, logger, df):
        # count = math.ceil(self.cfg.num_samples / world_size)
        # start = rank * count
        # end = min((rank + 1) * count, self.cfg.num_samples)

        os.makedirs(f"{savedir}/rank{rank}/pmpnn/pdbs", exist_ok=True)
        for i in idx:
            cmd = [
                'cp',
                f"{savedir}/sample{i}.pdb",
                f"{savedir}/rank{rank}/pmpnn/pdbs"
            ]
            subprocess.run(cmd)
        cmd = [
            "bash",
            "scripts/run_genie_pipeline.sh",
            f"{savedir}/rank{rank}/pmpnn",
        ]
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cvd:
            dev = cvd.split(',')[torch.cuda.current_device()]
        else:
            dev = torch.cuda.current_device()
        subprocess.run(cmd, env=os.environ | {
            'CUDA_VISIBLE_DEVICES': str(dev)
        })  

        pmpnn_df = pd.read_csv(
            f"{savedir}/rank{rank}/pmpnn/info.csv", index_col="domain"
        )
        pmpnn_df["designable"] = pmpnn_df["scRMSD"] < 2
        if logger is not None:
            for col in pmpnn_df.columns:
                for val in pmpnn_df[col].tolist():
                    logger.log(f"{self.cfg.name}/pmpnn_{col}", val)
        for i in idx:
            df[f"sample{i}"]["pmpnn_scrmsd"] = pmpnn_df.loc[f"sample{i}"].scRMSD
            df[f"sample{i}"]["pmpnn_scTM"] = pmpnn_df.loc[f"sample{i}"].scTM
            df[f"sample{i}"]["pmpnn_pLDDT"] = pmpnn_df.loc[f"sample{i}"].pLDDT
        
    def save_df(self, idx, rank, world_size, savedir, logger, df):
        df = pd.DataFrame(df).T.astype(float)
        df.to_csv(f"{savedir}/rank{rank}.csv")
    
        if world_size > 1:
            torch.distributed.barrier()
        if rank == 0:
            dfs = []
            for r in range(world_size):
                dfs.append(pd.read_csv(f"{savedir}/rank{r}.csv", index_col=0))
                subprocess.run(["rm", f"{savedir}/rank{r}.csv"])
            df = pd.concat(dfs).sort_index()
            df.to_csv(f"{savedir}/info.csv")
        
    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):

        idx = list(range(rank, len(self), world_size))
        os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)

        df = defaultdict(dict) 
        
        if self.cfg.run_designability:
            torch.cuda.empty_cache()
            self.run_designability(idx, rank, world_size, savedir, logger, df)
                   
        if self.cfg.run_secondary:
            self.run_secondary(idx, rank, world_size, savedir, logger, df)

        if self.cfg.run_diversity:
            self.run_diversity(idx, rank, world_size, savedir, logger, df)

        if self.cfg.run_pmpnn_designability:
            torch.cuda.empty_cache()
            self.run_pmpnn_designability(idx, rank, world_size, savedir, logger, df)
        
        # this has to be last
        self.save_df(idx, rank, world_size, savedir, logger, df)
        if self.cfg.run_plot and rank == 0:
            self.make_plot(idx, rank, world_size, savedir, logger, df)
            
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

        def t_skew_func(t, skew):
            midpoint_y = 0.5 + skew / 2# [0, 1]
            midpoint_x = 0.5 
            if t < midpoint_x:
                return midpoint_y / midpoint_x * t
            else:
                return midpoint_y + (1 - midpoint_y) / (1 - midpoint_x) * (t - midpoint_x)

        
        sampler = OpenProtSampler(schedules={
            'structure': lambda t: sched_fn(t_skew_func(t, self.cfg.skew)),
            'sequence': lambda t: 1-t_skew_func(t, -self.cfg.skew)
        }, steppers=[
            StructureStepper(self.cfg.struct),
            SequenceUnmaskingStepper(self.cfg.seq)
        ])
        
        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)

        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        B = len(sample['struct'])
        
        for i in range(B):
            prot = make_ca_prot(
                sample['struct'][i].cpu().numpy(),
                sample["aatype"][i].cpu().numpy(),
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
