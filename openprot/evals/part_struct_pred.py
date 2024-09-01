from .eval import OpenProtEval
import foldcomp
from ..utils import protein
from ..utils import residue_constants as rc
import numpy as np
from ..tasks import StructurePrediction
import torch
import os


class PartialStructurePrediction(OpenProtEval):
    def setup(self):
        self.db = foldcomp.open(self.cfg.path)

    def run(self, model):
        NotImplemented

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        name, pdb = self.db[idx]
        prot = protein.from_pdb_string(pdb)
        seqres = "".join([rc.restypes_with_x[c] for c in prot.aatype])
        data = self.make_data(
            name=name[:-4],
            seqres=seqres,
            seq_mask=np.ones(len(seqres)),
            atom37=prot.atom_positions.astype(np.float32),
            atom37_mask=prot.atom_mask.astype(np.float32),
        ).crop(512)
        return StructurePrediction.prep_data(None, data)

    def run_batch(self, model, batch, savedir=".", device=None, logger=None):
        os.makedirs(savedir, exist_ok=True)

        noisy_batch = {"pad_mask": batch["pad_mask"]}
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        _, readout = model.forward(noisy_batch)

        track = model.tracks["StructureTrack"]

        logits = readout["trans"]
        ang = torch.atan2(logits[..., 1::2], logits[..., ::2])
        gt_ang = 2 * np.pi * batch["frame_trans"] / track.cfg.decoder.max_period
        ang_error = ((gt_ang - ang) + np.pi) % (2 * np.pi) - np.pi
        msd_error = ang_error * track.cfg.decoder.max_period / (2 * np.pi)

        if logger:
            logger.log("struct/rmsd", torch.square(msd_error).sum(-1), post=np.sqrt)

        L = logits.shape[-2]
        atom37 = np.zeros((L, 37, 3))
        prot = protein.Protein(
            atom_positions=np.zeros((L, 37, 3)),
            aatype=batch["aatype"].cpu().numpy()[0],
            atom_mask=np.zeros((L, 37)),
            residue_index=np.arange(L) + 1,
            b_factors=np.zeros((L, 37)),
            chain_index=np.zeros(L, dtype=int),
        )

        prot.atom_mask[..., 1] = 1
        prot.atom_positions[..., 1, :] = batch["frame_trans"].cpu().numpy()

        ref_str = protein.to_pdb(prot)

        prot.atom_mask[..., 1] = (batch["struct_noise"] == 0.0).cpu().float()

        fixed_str = protein.to_pdb(prot)

        prot.atom_positions[..., 1, :] = (
            ang.cpu() * track.cfg.decoder.max_period / (2 * np.pi)
        )
        prot.atom_mask[..., 1] = (batch["struct_noise"] == 1.0).cpu().float()

        pred_str = protein.to_pdb(prot)

        ref_str = "\n".join(ref_str.split("\n")[1:-3])
        fixed_str = "\n".join(fixed_str.split("\n")[1:-3])
        pred_str = "\n".join(pred_str.split("\n")[1:-3])

        name = batch["name"][0]
        with open(f"{savedir}/{name}.pdb", "w") as f:
            f.write("\nENDMDL\nMODEL\n".join([ref_str, fixed_str, pred_str]))
