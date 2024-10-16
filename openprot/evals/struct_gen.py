from .eval import OpenProtEval
import foldcomp
from ..utils import protein
from ..utils.geometry import compute_lddt, rmsdalign
from ..utils import residue_constants as rc
import numpy as np
from ..tasks import StructurePrediction
import torch
import os, tqdm
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
            name=f'sample{idx}',
            seqres='X'*L,
            seq_mask=np.zeros(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
            atom37=np.random.randn(L, 37, 3).astype(np.float32),
            atom37_mask=np.ones((L, 37), dtype=np.float32),
        )
        return data

    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):

        os.makedirs(savedir, exist_ok=True)

        
        noisy_batch = batch.copy("name", "pad_mask")
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        diffusion = self.tracks['StructureTrack'].diffusion

        def model_func(pos, t):
            noisy_batch["trans_noise"] = torch.ones_like(noisy_batch["trans_noise"]) * t
            _, readout = model.forward(noisy_batch)
            return readout["trans"][-1]

        samps = diffusion.inference(model_func, cfg=self.cfg, mask=noisy_batch["pad_mask"])
            
        coords = samps[-1]
        L = batch["frame_trans"].shape[1]
        atom37 = np.zeros((L, 37, 3))
        prot = protein.Protein(
            atom_positions=np.zeros((L, 37, 3)),
            aatype=batch["aatype"].cpu().numpy()[0],
            atom_mask=np.zeros((L, 37)),
            residue_index=np.arange(L) + 1,
            b_factors=np.zeros((L, 37)),
            chain_index=np.zeros(L, dtype=int),
        )

        prot.atom_mask[..., 1] = batch["frame_mask"].cpu().numpy()
        prot.atom_positions[..., 1, :] = coords.cpu().numpy()
        ref_str = protein.to_pdb(prot)

        name = batch["name"][0]
        with open(f"{savedir}/{name}.pdb", "w") as f:
            f.write(ref_str)
