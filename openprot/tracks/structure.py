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
from ..utils import diffusion
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


"""
def log_I0(x):  # this is unsatisfactory
    k_star = torch.round(x / 2)
    logx = torch.log(x)
    log_A_k_star = 2 * (k_star * (logx - np.log(2)) - torch.lgamma(k_star + 1))
    ks = torch.arange(-30, 31, device=x.device) + k_star[..., None]

    log_A_k_ratio = 2 * (ks - k_star[..., None]) * (logx[..., None] - np.log(2)) + 2 * (
        torch.lgamma(k_star[..., None] + 1) - torch.lgamma(ks + 1)
    )
    log_sum_A_k_ratio = torch.logsumexp(log_A_k_ratio, -1)
    return log_sum_A_k_ratio + log_A_k_star


def torus_logZ(x):  # x = 1 / sigma^2
    return torch.where(
        x < 100,
        np.log(2 * np.pi) + log_I0(x) - x,
        0.5 * np.log(2 * np.pi) - 0.5 * torch.log(x),
    )
"""


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
        if self.cfg.embed_trans:
            model.trans_in = nn.Linear(3, model.cfg.dim)
        if self.cfg.embed_pairwise:
            model.pairwise_in = nn.Linear(
                model.cfg.pairwise_dim, model.cfg.pairwise_dim
            )
            torch.nn.init.zeros_(model.pairwise_in.weight)
            torch.nn.init.zeros_(model.pairwise_in.bias)
        if self.cfg.readout_trans:
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
            batch["frame_trans"], batch["trans_noise"]
        )
        noisy_batch["frame_trans"] = noisy

        # training targets
        target["frame_trans"] = target_tensor  # diffusion target
        target["frame_pos"] = batch["frame_trans"]  # sm target

        dev = batch["frame_trans"].device
        B, L, _ = batch["frame_trans"].shape

        # if self.cfg.rot_augment > 1:
        #     randrots = random_rotations(self.cfg.rot_augment, device=dev)
        #     noisy_batch["frame_trans"] = torch.einsum(
        #         "rij,blj->rbli", randrots, noisy_batch["frame_trans"]
        #     )
        #     target["frame_trans"] = torch.einsum(
        #         "rij,blj->rbli", randrots, target["frame_trans"]
        #     )

        # placeholder
        noisy_batch["frame_rots"] = torch.eye(3, device=dev).expand(B, L, 3, 3)

        # needed for computing distogram loss
        target["beta"], target["beta_mask"] = pseudo_beta_fn(
            batch["aatype"], batch["atom37"], batch["atom37_mask"]
        )

        if logger:
            logger.log("struct/toks", batch["frame_mask"].sum())

    def embed(self, model, batch, inp):

        B, L = batch["trans_noise"].shape
        dev = batch["frame_trans"].device

        inp["trans"] = batch["frame_trans"]
        inp["rots"] = batch["frame_rots"]

        if self.cfg.embed_trans:
            inp["x"] += model.trans_in(inp["trans"])

        # tell the model which frames were not present
        inp["x"] += torch.where(
            batch["frame_mask"].bool()[..., None],
            0.0,
            model.frame_mask[None, None],
        )

        inp["x_cond"] += sinusoidal_embedding(
            batch["trans_noise"], model.cfg.dim // 2, 1, 0.01
        ).float()

        if self.cfg.embed_pairwise:
            dmat = inp["trans"][..., None, :, :] - inp["trans"][..., None, :]
            dmat = torch.square(dmat).sum(-1).sqrt()
            inp["z"] = inp.get("z", 0) + model.pairwise_in(
                sinusoidal_embedding(dmat, model.cfg.pairwise_dim // 2, 100, 1)
            )

    def predict(self, model, out, readout):
        if self.cfg.readout_trans:
            readout["trans"] = model.trans_out(out["x"])
        readout["pos"] = out["sm"]["trans"]

        if self.cfg.readout_pairwise:
            readout["pairwise"] = model.pairwise_out(out["z"])

    def compute_sm_loss(self, readout, target, logger=None):

        loss = 0

        if "mse" in self.cfg.losses:
            loss = loss + self.cfg.losses["mse"] * self.compute_mse_loss(
                readout, target, logger=logger
            )

        if "lddt" in self.cfg.losses:
            loss = loss + self.cfg.losses["lddt"] * self.compute_lddt_loss(
                readout, target, logger=logger
            )

        if "pade" in self.cfg.losses:
            loss = loss + self.cfg.losses["pade"] * self.compute_pade_loss(
                readout, target, logger=logger
            )

        if "nape" in self.cfg.losses:
            loss = loss + self.cfg.losses["nape"] * self.compute_nape_loss(
                readout, target, logger=logger
            )

        return loss

    def compute_loss(self, readout, target, logger=None):

        loss = self.compute_sm_loss(readout, target, logger)

        if "distogram" in self.cfg.losses:
            loss = loss + self.cfg.losses["distogram"] * self.compute_distogram_loss(
                readout, target, logger=logger
            )

        if "diffusion" in self.cfg.losses:
            loss = loss + self.cfg.losses["diffusion"] * self.compute_diffusion_loss(
                readout, target, logger=logger
            )

        lddt = compute_lddt(
            readout["pos"][-1], target["frame_pos"], target["struct_supervise"]
        )

        # compute dmat lddt
        if self.cfg.readout_pairwise:
            bins = torch.linspace(2.3125, 22, 64, device=readout["pos"].device)
            idx = readout["pairwise"].argmax(-1)

            distmat = bins[idx] - 0.3125

            dmat_lddt = compute_lddt(
                distmat,
                target["frame_trans"],
                target["struct_supervise"],
                pred_is_dmat=True,
            )
            if logger:
                logger.log("struct/dmat_lddt", dmat_lddt)

        mask = target["struct_supervise"]
        if logger:
            logger.log("struct/toks_sup", mask.sum())
            logger.log("struct/loss", loss, mask=mask)
            logger.log("struct/lddt", lddt)

        loss = (loss * mask).sum() / target["pad_mask"].sum()

        return loss

    def compute_lddt_loss(self, readout, target, logger=None):

        pred = readout["pos"]
        gt = target["frame_pos"]
        mask = target["struct_supervise"]
        soft_lddt = compute_lddt(pred, gt, mask, soft=True, reduce=(-1,))

        soft_lddt_loss = 1 - soft_lddt  # [9, B, L]

        if logger:
            logger.log("struct/lddt_loss", soft_lddt_loss[-1], mask=mask)
            logger.log("struct/lddt_loss_aux", soft_lddt_loss.mean(0), mask=mask)

        w = self.cfg.int_loss_weight
        return w * soft_lddt_loss.mean(0) + (1 - w) * soft_lddt_loss[-1]

    def compute_mse_loss(self, readout, target, logger=None, eps=1e-5):

        pred = readout["pos"]
        gt = target["frame_pos"]
        mask = target["struct_supervise"]
        t = target["trans_noise"]

        def compute_mse(pred, gt, mask, clamp=None):
            gt = rmsdalign(pred.detach(), gt, mask)
            gt = torch.nan_to_num(gt, 0.0)
            mse = torch.square(pred - gt).sum(-1)

            if clamp is not None:
                mse = torch.clamp(mse, max=clamp)
            return mse

        if self.cfg.clamp_mse:
            mse = 0.9 * compute_mse(pred, gt, mask, clamp=100) + 0.1 * compute_mse(
                pred, gt, mask
            )
        else:
            mse = compute_mse(pred, gt, mask)

        if logger:
            logger.log("struct/mse_loss", mse[-1], mask=mask)
            logger.log("struct/mse_loss_aux", mse.mean(0), mask=mask)

        w = self.cfg.int_loss_weight
        return w * mse.mean(0) + (1 - w) * mse[-1]

    def compute_pade_loss(self, readout, target, logger=None):
        pred = readout["pos"]
        gt = target["frame_pos"]
        mask = target["struct_supervise"]  # this does not do exactly what we want

        if self.cfg.clamp_pade:
            pade = 0.9 * compute_pade(pred, gt, mask, clamp=10) + 0.1 * compute_pade(
                pred, gt, mask
            )
        else:
            pade = compute_pade(pred, gt, mask)

        if logger:
            logger.log("struct/pade_loss", pade[-1], mask=mask)
            logger.log("struct/pade_loss_aux", pade.mean(0), mask=mask)
        w = self.cfg.int_loss_weight
        return w * pade.mean(0) + (1 - w) * pade[-1]

    def compute_nape_loss(self, readout, target, eps=1e-5, logger=None):
        pred = readout["pos"]
        gt = target["frame_pos"]
        mask = target["struct_supervise"]

        def compute_nape(pred, gt, mask, cutoff=15, scale=10, clamp=None):
            gt_dmat = (gt[..., None, :] - gt[..., None, :, :]).square().sum(-1).sqrt()
            dists_to_score = (
                (gt_dmat < cutoff) * mask.unsqueeze(-1) * mask.unsqueeze(-2)
            )

            aligned_pos = rmsdalign(
                pred.unsqueeze(-3).detach(), gt.unsqueeze(-3), dists_to_score
            )
            aligned_pos = torch.nan_to_num(aligned_pos)

            pred_pos = pred.unsqueeze(-3)

            nape = (eps + (aligned_pos - pred_pos).square().sum(-1)).sqrt()

            if clamp is not None:
                nape = torch.clamp(nape, max=clamp)
            nape = nape / scale
            nape = (nape * dists_to_score).sum(-1) / (dists_to_score.sum(-1) + eps)

            return nape

        if self.cfg.clamp_nape:
            nape = 0.1 * compute_nape(
                pred, gt, mask, cutoff=self.cfg.nape_cutoff
            ) + 0.9 * compute_nape(
                pred, gt, mask, cutoff=self.cfg.nape_cutoff, clamp=10
            )
        else:
            nape = compute_nape(pred, gt, mask, cutoff=self.cfg.nape_cutoff)

        nape = nape.mean(0)
        if self.cfg.rot_augment > 1:
            nape = nape.mean(0)

        if logger:
            logger.log("struct/nape_loss", nape, mask=mask)

        return nape

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
            logger.log("struct/distogram", loss, mask)
            # logger.log("struct/distogram2", (loss * mask).sum((-1,-2)) / mask.sum((-1,-2)))

        return (loss * mask).sum(-1) / (eps + mask.sum(-1))

    def compute_diffusion_loss(self, readout, target, logger=None, eps=1e-6):

        loss = self.diffusion.compute_loss(
            pred=readout["trans"],
            target=target["frame_trans"],
            t=target["trans_noise"],
        )
        if logger:
            logger.log("struct/diffusion_loss", loss, mask=target["struct_supervise"])

        return loss
