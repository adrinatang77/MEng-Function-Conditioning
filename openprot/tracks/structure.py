import torch
import torch.nn as nn
from .track import OpenProtTrack
from ..utils.geometry import (
    atom37_to_frames,
    atom37_to_torsions,
    compute_fape,
    gram_schmidt,
    compute_pade,
)
from ..utils.rotation_conversions import axis_angle_to_matrix
from ..utils.rigid_utils import Rigid, Rotation
from ..utils import residue_constants as rc
from ..model.positions import PositionEmbedder, PositionDecoder
from functools import partial
import numpy as np


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


class StructureTrack(OpenProtTrack):

    def tokenize(self, data):

        frames, frame_mask = atom37_to_frames(data["atom37"], data["atom37_mask"])
        data["frame_trans"] = frames._trans
        data["frame_rots"] = frames._rots._rot_mats
        data["frame_mask"] = frame_mask
        aatype = np.array(
            [rc.restype_order.get(c, rc.unk_restype_index) for c in data["seqres"]]
        )
        torsions, torsion_mask = atom37_to_torsions(data["atom37"], aatype)
        data["torsions"] = torsions
        data["torsion_mask"] = torsion_mask

    def add_modules(self, model):
        if self.cfg.embedder.type == "frames":
            return  # temporary
        model.trans_mask = nn.Parameter(torch.zeros(model.cfg.dim))
        model.trans_null = nn.Parameter(torch.zeros(model.cfg.dim))
        model.trans_embed = PositionEmbedder(self.cfg.embedder, model.cfg.dim)
        model.trans_out = PositionDecoder(self.cfg.decoder, model.cfg.dim)
        # model.rots_embed = nn.Linear(9, model.cfg.dim)
        # model.trans_out = nn.Linear(model.cfg.dim, 3)
        model.rots_out = nn.Linear(model.cfg.dim, 3)

    def corrupt(self, batch, noisy_batch, target, logger=None):

        ## right now our corruption is just masking EVERYTHING
        noisy_batch["struct_mask"] = batch["struct_mask"]
        noisy_batch["struct_noise"] = batch["struct_noise"]

        dev = batch["frame_trans"].device
        B, L, _ = batch["frame_trans"].shape

        target["frame_trans"] = batch["frame_trans"]
        target["frame_rots"] = batch["frame_rots"]
        target["struct_supervise"] = batch["frame_mask"]

        noisy_batch["frame_trans"] = torch.zeros_like(batch["frame_trans"])
        noisy_batch["frame_rots"] = torch.eye(3, device=dev).expand(B, L, 3, 3)

        # batch["struct_mask"] * (batch["struct_noise"] > 0)

        if logger:
            logger.log("struct/toks", batch["struct_mask"].sum())

    def embed(self, model, batch, inp):

        if self.cfg.embedder.type == "frames":
            inp["rots"] = batch["frame_rots"]
            inp["trans"] = batch["frame_trans"]
        else:
            pos_embed = model.trans_embed(batch["frame_trans"])
            # NULL embed the non-exisistent = unsupervised tokens
            pos_embed = torch.where(
                batch["struct_mask"][..., None].bool(),
                pos_embed,
                model.trans_null[None, None],
            )
            # MASK embed the fully noised tokens
            pos_embed = torch.where(
                batch["struct_noise"][..., None] == 1.0,
                model.trans_mask[None, None],
                pos_embed,
            )
            inp["x"] += pos_embed

    def predict(self, model, out, readout):
        if self.cfg.decoder.type == "frames":
            readout["trans"] = out["trans"]
            readout["rots"] = out["rots"]

        elif self.cfg.decoder.type == "linear":
            readout["trans"] = model.trans_out(out["x"])
            rotvec = model.rots_out(out["x"])
            readout["rots"] = axis_angle_to_matrix(rotvec)

    def compute_loss(self, readout, target, logger=None):

        loss = 0

        if "rmsd" in self.cfg.losses:
            loss = loss + self.cfg.losses["rmsd"] * self.compute_rmsd_loss(
                readout, target, logger=logger
            )

        if "sinusoidal" in self.cfg.losses and self.cfg.decoder.type == "sinusoidal":
            raise Exception("sinusoidal structural loss no longer supported")
            loss = loss + self.cfg.losses["sinusoidal"] * self.compute_sinusoidal_loss(
                readout, target, logger=logger
            )

        if "pade" in self.cfg.losses:
            loss = loss + self.cfg.losses["pade"] * self.compute_pade_loss(
                readout, target, logger=logger
            )

        if "fape" in self.cfg.losses:
            loss = loss + self.cfg.losses["fape"] * self.compute_fape_loss(
                readout, target, logger=logger
            )

        mask = target["struct_supervise"]

        if logger:
            logger.log("struct/toks_sup", mask.sum())
            logger.log("struct/loss", loss, mask=mask)

        loss = (loss * mask).sum() / target["pad_mask"].sum()

        return loss

    def compute_pade_loss(self, readout, target, logger=None):
        pred = readout["trans"]
        gt = target["frame_trans"]
        mask = target["struct_supervise"]  # this does not do exactly what we want

        pade = compute_pade(pred, gt, mask)
        if logger:
            logger.log("struct/pade_loss", pade, mask=mask)
        return pade

    def compute_sinusoidal_loss(self, readout, target, logger=None, eps=1e-6):

        logits = readout["trans"]
        logits = logits.reshape(*logits.shape[:2], 3, 2)

        sqnorm = torch.square(logits).sum(-1)
        norm = torch.sqrt(eps + sqnorm)

        if self.cfg.logit_max_norm:
            denom = torch.sqrt(1 + (sqnorm / self.cfg.logit_max_norm) ** 2)
            logits = logits / denom[..., None]

        mask = target["struct_supervise"]
        ang = torch.atan2(logits[..., 1], logits[..., 0])
        gt_ang = 2 * np.pi * target["frame_trans"] / self.cfg.decoder.max_period

        logp = torch.cos(gt_ang) * logits[..., 0] + torch.sin(gt_ang) * logits[..., 1]
        logz = np.log(2 * np.pi) + log_I0(norm)
        nll = logz - logp

        if logger:
            logger.log("struct/logit_norm", norm, mask=mask[..., None])
            logger.log("struct/sinusoidal_loss", nll.sum(-1), mask=mask)

        return nll.sum(-1)  # * mask).sum() / target["pad_mask"].sum()

    def compute_fape_loss(self, readout, target, logger=None):

        pred_frames = Rigid(
            trans=readout["trans"], rots=Rotation(rot_mats=readout["rots"])
        )
        target_frames = Rigid(
            trans=target["frame_trans"], rots=Rotation(rot_mats=target["frame_rots"])
        )

        mask = target["struct_supervise"]

        fape_fn = partial(
            compute_fape,
            pred_frames=pred_frames,
            target_frames=target_frames,
            frames_mask=mask,
            pred_positions=pred_frames.get_trans(),
            target_positions=target_frames.get_trans(),
            positions_mask=mask,
            length_scale=10,
        )
        fape_loss = (
            0.1 * fape_fn()
            + 0.45 * fape_fn(thresh=15)
            + 0.45 * fape_fn(l1_clamp_distance=10)
        )

        if logger:
            logger.log("struct/fape_loss", fape_loss, mask=mask)
        return fape_loss

    def compute_rmsd_loss(self, readout, target, logger=None, eps=1e-5):
        pred = readout["trans"]
        gt = target["frame_trans"]
        mask = target["struct_supervise"]

        err = torch.square(pred - gt).sum(-1)
        if logger:
            logger.log("struct/rmsd", err, mask, post=np.sqrt)
            logger.log("struct/rmsd_loss", err, mask=mask)

        return err  # (err * mask).sum() / target["pad_mask"].sum()
