import torch
import torch.nn as nn
from .track import OpenProtTrack
from ..utils.geometry import (
    atom37_to_frames,
    atom37_to_torsions,
    compute_fape,
    gram_schmidt,
    compute_pade,
    compute_lddt,
    rmsdalign,
)
from ..utils.rotation_conversions import axis_angle_to_matrix, random_rotations
from ..utils.rigid_utils import Rigid, Rotation
from ..utils import residue_constants as rc
from ..model.positions import PositionEmbedder, PositionDecoder, PairwiseProjectionHead
from ..generate import diffusion
from functools import partial
import numpy as np
import math


def sinusoidal_embedding(pos, n_freqs, max_period, min_period):
    periods = torch.exp(
        torch.linspace(
            math.log(min_period), math.log(max_period), n_freqs, device=pos.device
        )
    )
    freqs = 2 * np.pi / periods
    return torch.cat(
        [torch.cos(pos[..., None] * freqs), torch.sin(pos[..., None] * freqs)], -1
    )


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    is_gly = aatype == rc.restype_order["G"]
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


class StructureTrack(OpenProtTrack):

    def setup(self):
        self.diffusion = getattr(diffusion, self.cfg.diffusion.type)(self.cfg.diffusion)

    def tokenize(self, data):

        frames, frame_mask = atom37_to_frames(data["atom37"], data["atom37_mask"])
        data["frame_trans"] = frames._trans
        data["frame_rots"] = frames._rots._rot_mats
        data["frame_mask"] = frame_mask
        """
        aatype = np.array(
            [rc.restype_order.get(c, rc.unk_restype_index) for c in data["seqres"]]
        )
        torsions, torsion_mask = atom37_to_torsions(data["atom37"], aatype)
        data["torsions"] = torsions
        data["torsion_mask"] = torsion_mask
        """

    def add_modules(self, model):
        model.frame_mask = nn.Parameter(torch.zeros(model.cfg.dim))
        if self.cfg.readout_pairwise:
            model.pairwise_out = nn.Linear(model.cfg.pairwise_dim, 64)
        if self.cfg.embed_trans == "sinusoidal":
            model.trans_in = nn.Linear(6 * (model.cfg.dim // 6), model.cfg.dim)
            torch.nn.init.zeros_(model.trans_in.weight)
            torch.nn.init.zeros_(model.trans_in.bias)
        elif self.cfg.embed_trans == "edm":
            model.trans_in = nn.Linear(3, model.cfg.dim)
        if self.cfg.embed_pairwise:
            model.pairwise_in = nn.Linear(
                model.cfg.pairwise_dim, model.cfg.pairwise_dim
            )
            torch.nn.init.zeros_(model.pairwise_in.weight)
            torch.nn.init.zeros_(model.pairwise_in.bias)
        if self.cfg.readout_trans in ["trunk", "sm"]:
            model.trans_out = nn.Sequential(
                nn.LayerNorm(model.cfg.dim), nn.Linear(model.cfg.dim, 3)
            )

    def corrupt(self, batch, noisy_batch, target, logger=None):

        for key in [
            "trans_noise",
            "rots_noise",
            "frame_mask",
        ]:
            noisy_batch[key] = batch[key]

        target["trans_noise"] = batch["trans_noise"]
        target["struct_supervise"] = batch["frame_mask"]

        # add noise
        noisy, target_tensor = self.diffusion.add_noise(
            batch["frame_trans"], batch["trans_noise"], batch["frame_mask"].bool()
        )
        noisy_batch["frame_trans"] = noisy

        # training targets
        target["frame_trans"] = target_tensor  # diffusion target
        target["frame_pos"] = batch["frame_trans"]  # sm target

        dev = batch["frame_trans"].device
        B, L, _ = batch["frame_trans"].shape

        # placeholder
        noisy_batch["frame_rots"] = torch.eye(3, device=dev).expand(B, L, 3, 3)

        # needed for computing distogram loss
        target["beta"], target["beta_mask"] = pseudo_beta_fn(
            batch["aatype"], batch["atom37"], batch["atom37_mask"]
        )

        if logger:
            logger.masked_log("struct/toks", batch["frame_mask"], sum=True)

    def embed(self, model, batch, inp):

        B, L = batch["trans_noise"].shape
        dev = batch["frame_trans"].device

        # currently zerod out before they are used
        inp["trans"] = batch["frame_trans"]
        inp["rots"] = batch["frame_rots"]

        if self.cfg.embed_trans == "sinusoidal":
            emb = sinusoidal_embedding(inp["trans"], model.cfg.dim // 6, 100, 1)
            inp["x"] += model.trans_in(emb.view(*emb.shape[:-2], -1))
        elif self.cfg.embed_trans == "edm":
            precond = self.diffusion.precondition(inp["trans"], batch["trans_noise"])
            inp["x"] += torch.where(
                batch["frame_mask"].bool()[..., None],  
                model.trans_in(precond),
                model.frame_mask[None, None],
            ) # this stuff is not being consistently handled atm

        # tell the model which frames were not present
        inp["x"] += torch.where(
            batch["frame_mask"].bool()[..., None],
            0.0,
            model.frame_mask[None, None],
        )
        
        inp["relpos_mask"] = batch["frame_mask"]
            
        t_emb = batch['trans_noise'] ** (1/self.cfg.t_emb_p)
        t_emb = t_emb / self.cfg.t_emb_max ** (1/self.cfg.t_emb_p)
        t_emb = sinusoidal_embedding(t_emb, model.cfg.dim // 2, 1, 0.01).float()
        t_emb = torch.where(
            batch["frame_mask"][...,None].bool(), 
            t_emb,
            model.frame_mask[None, None]
        )
        inp["x_cond"] += t_emb

        if self.cfg.embed_pairwise:
            dmat = inp["trans"][..., None, :, :] - inp["trans"][..., None, :]
            dmat = torch.square(dmat).sum(-1).sqrt()
            inp["z"] = inp.get("z", 0) + model.pairwise_in(
                sinusoidal_embedding(dmat, model.cfg.pairwise_dim // 2, 100, 1)
            )
        
        inp["postcond_fn"] = lambda trans: self.diffusion.postcondition(
            inp["trans"],
            trans,
            batch["trans_noise"]
        )

    def predict(self, model, inp, out, readout):
        if self.cfg.readout_trans == "trunk":
            readout["trans"] = model.trans_out(out["x"])[None]
        elif self.cfg.readout_trans == "sm":
            if model.cfg.trunk_adaLN:
                readout["trans"] = model.trans_out(out["sm"]["x"], inp["x_cond"])
            else:
                readout["trans"] = model.trans_out(out["sm"]["x"])

        else:
            readout["trans"] = out["sm"]["trans"]
        if self.cfg.postcondition:
            readout["trans"] = inp["postcond_fn"](readout["trans"])
        if self.cfg.readout_pairwise:
            readout["pairwise"] = model.pairwise_out(out["z"])

    def compute_loss(self, readout, target, logger=None, **kwargs):

        loss = 0

        if "mse" in self.cfg.losses:
            loss = loss + self.cfg.losses["mse"] * self.compute_mse_loss(
                readout, target, logger=logger
            )

        if "lddt" in self.cfg.losses:
            loss = loss + self.cfg.losses["lddt"] * self.compute_lddt_loss(
                readout, target, logger=logger
            )

        if "distogram" in self.cfg.losses:
            loss = loss + self.cfg.losses["distogram"] * self.compute_distogram_loss(
                readout, target, logger=logger
            )

        lddt = compute_lddt(
            readout["trans"][-1], target["frame_pos"], target["struct_supervise"]
        )
        if logger:
            logger.masked_log("struct/lddt", lddt, dims=0)

        # compute dmat lddt
        if self.cfg.readout_pairwise:
            bins = torch.linspace(2.3125, 22, 64, device=readout["pairwise"].device)
            idx = readout["pairwise"].argmax(-1)

            distmat = bins[idx] - 0.3125

            dmat_lddt = compute_lddt(
                distmat,
                target["frame_pos"],
                target["struct_supervise"],
                pred_is_dmat=True,
            )
            if logger:
                logger.masked_log("struct/dmat_lddt", dmat_lddt, dims=0)

        mask = target["struct_supervise"]
        if logger:
            logger.masked_log("struct/toks_sup", mask, sum=True)
            logger.masked_log("struct/loss", loss, mask=mask)

        loss = loss * mask

        return loss

    def compute_lddt_loss(self, readout, target, logger=None):

        pred = readout["trans"]
        gt = target["frame_pos"]
        mask = target["struct_supervise"]
        soft_lddt = compute_lddt(pred, gt, mask, soft=True, reduce=(-1,))

        soft_lddt_loss = 1 - soft_lddt  # [9, B, L]

        if logger:
            logger.masked_log("struct/lddt_loss", soft_lddt_loss[-1], mask=mask)
            logger.masked_log("struct/lddt_loss_aux", soft_lddt_loss.mean(0), mask=mask)

        w = self.cfg.int_loss_weight
        return w * soft_lddt_loss.mean(0) + (1 - w) * soft_lddt_loss[-1]

    def compute_mse_loss(self, readout, target, logger=None, eps=1e-5):

        pred = readout["trans"]
        gt = target["frame_pos"]
        mask = target["struct_supervise"]
        t = target["trans_noise"]

        mse = self.diffusion.compute_loss(
            pred=pred,
            target=gt,
            t=t,
            mask=mask,  # not exactly right
        )

        if logger:
            logger.masked_log("struct/mse_loss", mse[-1], mask=mask)
            logger.masked_log("struct/mse_loss_aux", mse.mean(0), mask=mask)

        w = self.cfg.int_loss_weight
        return w * mse.mean(0) + (1 - w) * mse[-1]

    def compute_distogram_loss(self, readout, target, logger=None, eps=1e-6):

        dev = readout["pairwise"].device
        bins = torch.linspace(2.3125, 21.6875, 63, device=dev)

        distmat = target["beta"][:, None] - target["beta"][:, :, None]
        distmat = torch.sqrt(torch.square(distmat).sum(-1))
        label = (distmat[..., None] > bins).sum(-1)

        loss = torch.nn.functional.cross_entropy(
            readout["pairwise"].permute(0, 3, 1, 2), label, reduction="none"
        )

        # some temporary stuff
        mask = target["struct_supervise"] * target["beta_mask"]
        mask = mask[:, None] * mask[:, :, None]
        if logger:
            logger.masked_log("struct/distogram", loss, mask, dims=2)

        return (loss * mask).sum(-1) / (eps + mask.sum(-1))

    def compute_diffusion_loss(self, readout, target, logger=None, eps=1e-6):

        loss = self.diffusion.compute_loss(
            pred=readout["trans"],
            target=target["frame_trans"],
            t=target["trans_noise"],
            mask=target["struct_supervise"],  # not exactly right
        )
        if logger:
            logger.masked_log(
                "struct/diffusion_loss", loss, mask=target["struct_supervise"]
            )

        return loss

   