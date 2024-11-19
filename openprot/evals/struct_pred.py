from .eval import OpenProtEval
import foldcomp
from ..utils import protein
from ..utils.geometry import compute_lddt, rmsdalign, compute_rmsd
from ..utils import residue_constants as rc
from ..utils.prot_utils import make_ca_prot, compute_tmscore
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
        data["trans_noise"] = np.ones(L, dtype=np.float32) * 1.0
        data["rots_noise"] = np.ones(L, dtype=np.float32) * 1.0

        return data

    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):
        os.makedirs(savedir, exist_ok=True)

        noisy_batch = batch.copy("name", "pad_mask")
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        _, readout = model.forward(noisy_batch)

        L = batch["frame_trans"].shape[1]
        aatype = batch["aatype"].cpu().numpy()[0]
        coords = readout["pos"][-1]
        coords = rmsdalign(batch["frame_trans"], coords, batch["frame_mask"])

        lddt = compute_lddt(coords, batch["frame_trans"], batch["frame_mask"])

        tmscore = compute_tmscore(  # second is reference
            coords1=coords.cpu().numpy()[0],
            coords2=batch["frame_trans"].cpu().numpy()[0],
            seq1=aatype,
            seq2=aatype,
            mask1=None,
            mask2=batch["frame_mask"].cpu().numpy()[0],
        )

        rmsd = compute_rmsd(batch["frame_trans"], coords, batch["frame_mask"])
        if logger:
            logger.log("struct/lddt", lddt)
            logger.log("struct/tm", tmscore["tm"])
            logger.log("struct/rmsd", tmscore["rmsd"])
            logger.log("struct/gdt_ts", tmscore["gdt_ts"])
            logger.log("struct/gdt_ha", tmscore["gdt_ha"])
            logger.log("struct/rmsd2", rmsd)

        prot = make_ca_prot(
            coords=batch["frame_trans"].cpu().numpy()[0],
            aatype=aatype,
            mask=batch["frame_mask"].cpu().numpy()[0],
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
