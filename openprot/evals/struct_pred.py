from .eval import OpenProtEval
# import foldcomp
from ..utils import protein
from ..utils.geometry import compute_lddt, rmsdalign, compute_rmsd
from ..utils import residue_constants as rc
from ..utils.prot_utils import make_ca_prot, compute_tmscore, write_ca_traj
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
        data = self.make_data(
            name=name,
            seqres=seqres,
            seq_mask=np.ones(len(seqres)),
            atom37=prot["all_atom_positions"].astype(np.float32),
            atom37_mask=prot["all_atom_mask"].astype(np.float32),
        )

        L = len(seqres)
        data["struct_noise"] = np.ones(L, dtype=np.float32) * 160.

        return data

    def run_diffusion(self, model, batch, noisy_batch, savedir):
        diffusion = self.tracks["StructureTrack"].diffusion

        # noisy_batch["frame_trans"] = self
        def model_func(pos, t):
            noisy_batch["struct_noise"] = torch.ones_like(noisy_batch["struct_noise"]) * t
            noisy_batch["struct"] = pos
            _, readout = model.forward(noisy_batch)
            return readout["trans"]

        samp_traj, pred_traj = diffusion.inference(
            model_func, cfg=self.cfg, mask=noisy_batch["pad_mask"], return_traj=True
        )

        prot = make_ca_prot(
            samp_traj[-1, -1].cpu().numpy(),
            batch["aatype"].cpu().numpy()[0],
            batch["frame_mask"].cpu().numpy()[0],
        )
        name = batch["name"][0]
        with open(f"{savedir}/{name}_traj.pdb", "w") as f:
            f.write(write_ca_traj(prot, samp_traj[:, 0].cpu().numpy()))

        with open(f"{savedir}/{name}_pred_traj.pdb", "w") as f:
            f.write(write_ca_traj(prot, pred_traj[:, 0].cpu().numpy()))

        return samp_traj[-1]

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

        L = batch["struct"].shape[1]
        aatype = batch["aatype"].cpu().numpy()[0]

        coords = rmsdalign(batch["struct"], coords, batch["struct_mask"])

        lddt = compute_lddt(coords, batch["struct"], batch["struct_mask"])

        tmscore = compute_tmscore(  # second is reference
            coords1=coords.cpu().numpy()[0],
            coords2=batch["struct"].cpu().numpy()[0],
            seq1=aatype,
            seq2=aatype,
            mask1=None,
            mask2=batch["struct_mask"].cpu().numpy()[0],
            seq=True
        )

        rmsd = compute_rmsd(batch["struct"], coords, batch["struct_mask"])
        if logger:
            logger.log("struct/lddt", lddt)
            logger.log("struct/tm", tmscore["tm"])
            logger.log("struct/gdt_ts", tmscore["gdt_ts"])
            logger.log("struct/gdt_ha", tmscore["gdt_ha"])
            logger.log("struct/rmsd", rmsd)

        prot = make_ca_prot(
            coords=batch["struct"].cpu().numpy()[0],
            aatype=aatype,
            mask=batch["struct_mask"].cpu().numpy()[0],
        )
        ref_str = protein.to_pdb(prot)

        prot.atom_mask[..., 1] = 1.0
        prot.atom_positions[..., 1, :] = coords.cpu().numpy()
        pred_str = protein.to_pdb(prot)

        ref_str = "\n".join(ref_str.split("\n")[1:-3])
        pred_str = "\n".join(pred_str.split("\n")[1:-3])

        name = batch["name"][0]
        with open(f"{savedir}/{name}_{lddt.item():.2f}.pdb", "w") as f:
            f.write("\nENDMDL\nMODEL\n".join([ref_str, pred_str]))
