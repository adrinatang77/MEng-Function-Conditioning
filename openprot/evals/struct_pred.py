from .eval import OpenProtEval
import foldcomp
from ..utils import protein
from ..utils.geometry import compute_lddt
from ..utils import residue_constants as rc
import numpy as np
from ..tasks import StructurePrediction
import torch
import os
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
        return StructurePrediction.prep_data(self, data)

    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):
        os.makedirs(savedir, exist_ok=True)

        noisy_batch = batch.copy("name", "pad_mask")
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        _, readout = model.forward(noisy_batch)

        lddt = compute_lddt(
            readout["trans"][-1], batch["frame_trans"], batch["frame_mask"]
        )

        if logger:
            # logger.log("struct/rmsd", msd_error, post=np.sqrt)
            logger.log("struct/lddt", lddt)

        coords = readout["trans"][-1]
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
        prot.atom_positions[..., 1, :] = batch["frame_trans"].cpu().numpy()
        ref_str = protein.to_pdb(prot)

        prot.atom_positions[..., 1, :] = coords.cpu().numpy()
        pred_str = protein.to_pdb(prot)

        ref_str = "\n".join(ref_str.split("\n")[1:-3])
        pred_str = "\n".join(pred_str.split("\n")[1:-3])

        name = batch["name"][0]
        with open(f"{savedir}/{name}_{lddt:.2f}.pdb", "w") as f:
            f.write("\nENDMDL\nMODEL\n".join([ref_str, pred_str]))
