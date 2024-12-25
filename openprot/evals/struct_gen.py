from .eval import OpenProtEval
import foldcomp
from ..utils import protein
from ..utils.prot_utils import make_ca_prot, write_ca_traj
from ..utils.geometry import compute_lddt, rmsdalign
from ..utils import residue_constants as rc
import numpy as np
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
        data = self.make_data(
            name=f"sample{idx}",
            seqres="A" * L,
            seq_mask=np.ones(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
            atom37=np.zeros((L, 37, 3), dtype=np.float32),
            atom37_mask=np.ones((L, 37), dtype=np.float32),
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
            subprocess.run(cmd)  # env=os.environ | {"CUDA_VISIBLE_DEVICES")

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

    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):

        os.makedirs(savedir, exist_ok=True)

        noisy_batch = batch.copy("name", "pad_mask")
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        diffusion = self.tracks["StructureTrack"].diffusion

        def model_func(pos, t):
            noisy_batch["trans_noise"] = torch.ones_like(noisy_batch["trans_noise"]) * t
            noisy_batch["frame_trans"] = pos
            _, readout = model.forward(noisy_batch)
            
            return readout["trans"][-1]

        samp_traj, pred_traj = diffusion.inference(
            model_func, cfg=self.cfg, mask=noisy_batch["pad_mask"], return_traj=True
        )

        prot = make_ca_prot(
            samp_traj[-1, -1].cpu().numpy(),
            batch["aatype"].cpu().numpy()[0],
            batch["frame_mask"].cpu().numpy()[0],
        )

        ref_str = protein.to_pdb(prot)
        name = batch["name"][0]
        with open(f"{savedir}/{name}.pdb", "w") as f:
            f.write(ref_str)

        with open(f"{savedir}/{name}_traj.pdb", "w") as f:
            f.write(write_ca_traj(prot, samp_traj[:, 0].cpu().numpy()))

        with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
            f.write(write_ca_traj(prot, pred_traj[:, 0].cpu().numpy()))
