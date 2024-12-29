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

        # frames, frame_mask = atom37_to_frames(data["atom37"], data["atom37_mask"])
        data["struct"] = data["atom37"][:,1] # frames._trans
        data["struct_mask"] = data["atom37_mask"][:,1] # frame_mask
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
        model.frame_mask_cond = nn.Parameter(torch.zeros(model.cfg.dim))
        if self.cfg.embed_trans:
            model.trans_in = nn.Linear(3, model.cfg.dim)
        if self.cfg.embed_pairwise:
            model.pairwise_in = nn.Linear(
                model.cfg.pairwise_dim, model.cfg.pairwise_dim
            )
            torch.nn.init.zeros_(model.pairwise_in.weight)
            torch.nn.init.zeros_(model.pairwise_in.bias)
        if self.cfg.readout_pairwise:
            model.pairwise_out = nn.Linear(model.cfg.pairwise_dim, 64)
        if self.cfg.readout_trans:
            model.trans_out = nn.Sequential(
                nn.LayerNorm(model.cfg.dim), nn.Linear(model.cfg.dim, 3)
            )

    def corrupt(self, batch, noisy_batch, target, logger=None):

        for key in [
            "struct_noise",
            "struct_mask",
        ]:
            noisy_batch[key] = batch[key]

        # add noise
        noisy, target_tensor = self.diffusion.add_noise(
            batch["struct"], batch["struct_noise"], batch["struct_mask"].bool() 
        ) # the mask is used to center the coords
        noisy_batch["struct"] = noisy

        # training targets
        target["struct_noise"] = batch["struct_noise"] # used to compute diffusion loss
        target["struct_supervise"] = torch.where(
            batch["struct_mask"].bool(), batch["struct_weight"], 0.0
        )
        target["struct"] = target_tensor 

        # needed for computing distogram loss
        target["beta"], target["beta_mask"] = pseudo_beta_fn(
            batch["aatype"], batch["atom37"], batch["atom37_mask"]
        )

        if logger:
            logger.masked_log("struct/toks", batch["struct_mask"], sum=True)

    def embed(self, model, batch, inp):

        tmax = self.cfg.t_emb_max
        embed_as_mask = (batch["struct_noise"] >= tmax) | ~batch["struct_mask"].bool()

        coords = torch.where(
            embed_as_mask[...,None],
            tmax * torch.randn_like(batch["struct"]),
            batch["struct"]
        )
        noise = torch.where(
            embed_as_mask,
            tmax,
            batch["struct_noise"]
        )
        # linear embed coords if specified
        if self.cfg.embed_trans:
            precond = self.diffusion.precondition(coords, noise)
            inp["x"] += torch.where(
                embed_as_mask[...,None],
                0.0,
                model.trans_in(precond)
            )
        inp["x"] += torch.where(
            embed_as_mask[...,None],
            model.frame_mask[None,None],
            0.0,
        )

        # embed sigma
        def sigma_transform(noise_level):
            t_emb = noise_level ** (1/self.cfg.t_emb_p)
            t_emb = t_emb / self.cfg.t_emb_max ** (1/self.cfg.t_emb_p)
            return t_emb
            
        t_emb = sigma_transform(noise) 
        t_emb = sinusoidal_embedding(t_emb, model.cfg.dim // 2, 1, 0.01).float()
        inp["x_cond"] += torch.where(
            embed_as_mask[...,None],
            model.frame_mask_cond[None,None],
            t_emb,
        )
            
        if self.cfg.embed_pairwise:
            raise Exception("This hasn't been touched in a while, should fix")
            dmat = inp["struct"][..., None, :, :] - inp["struct"][..., None, :]
            dmat = torch.square(dmat).sum(-1).sqrt()
            inp["z"] = inp.get("z", 0) + model.pairwise_in(
                sinusoidal_embedding(dmat, model.cfg.pairwise_dim // 2, 100, 1)
            )

        # finally provide the raw struct for relpos
        inp["struct"] = coords
        inp["struct_mask"] = ~embed_as_mask
        inp["postcond_fn"] = lambda x: self.diffusion.postcondition(
            coords, x, noise,
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
            readout["trans"] = inp["postcond_fn"](out["sm"]["trans"])
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
            readout["trans"][-1], target["struct"], target["struct_supervise"]
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

    def compute_lddt_loss(self, readout, target, logger=None, eps=1e-5):

        pred = readout["trans"]
        gt = target["struct"]
        mask = (target["struct_supervise"] > eps).float() # lddt loss mask is binary
        soft_lddt = compute_lddt(pred, gt, mask, soft=True, reduce=(-1,))

        soft_lddt_loss = 1 - soft_lddt  # [9, B, L]

        if logger:
            logger.masked_log("struct/lddt_loss", soft_lddt_loss[-1], mask=mask)
            logger.masked_log("struct/lddt_loss_aux", soft_lddt_loss.mean(0), mask=mask)

        w = self.cfg.int_loss_weight
        return w * soft_lddt_loss.mean(0) + (1 - w) * soft_lddt_loss[-1]

    def compute_mse_loss(self, readout, target, logger=None, eps=1e-5):

        pred = readout["trans"]
        gt = target["struct"]
        mask = target["struct_supervise"] # weighted mse!
        t = target["struct_noise"]

        
        mse = self.diffusion.compute_loss(
            pred=pred,
            target=gt,
            t=t,
            mask=mask, # used for alignment
        )

        if logger:
            logger.masked_log("struct/mse_loss", mse[-1], mask=mask)
            logger.masked_log("struct/mse_loss_aux", mse.mean(0), mask=mask)

        w = self.cfg.int_loss_weight
        return w * mse.mean(0) + (1 - w) * mse[-1]

    def compute_distogram_loss(self, readout, target, logger=None, eps=1e-6):
        raise Exception("This hasn't been used in a while")
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
   