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


class MultiflowEval(OpenProtEval):
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
                "--print",
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
                subprocess.run(cmd)  # env=os.environ | {"CUDA_VISIBLE_DEVICES")
    
                df = pd.read_csv(
                    f"{savedir}/eval{start}_{end-1}/info.csv", index_col="domain"
                )
                df["designable"] = df["scRMSD"] < 2
                if logger is not None:
                    for col in df.columns:
                        for val in df[col].tolist():
                            logger.log(f"{self.cfg.name}/genie_{col}", val)

    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):

        mf_batch = {'res_mask': batch['pad_mask']}
        bb, aa = model.model.inference(mf_batch)
        
        B = len(bb)
        
        for i in range(B):
            prot = make_ca_prot(
                bb[i,:,1], aa[i], batch["struct_mask"][i].cpu().numpy(),
            )
    
            ref_str = protein.to_pdb(prot)
            name = batch["name"][i]
            with open(f"{savedir}/{name}.pdb", "w") as f:
                f.write(ref_str)
    
            seq = "".join([rc.restypes_with_x[aaa] for aaa in aa[i]])
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")
