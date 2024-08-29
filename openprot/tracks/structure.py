import torch
import torch.nn as nn
from .track import Track
from ..utils.geometry import (
    atom37_to_frames,
    atom37_to_torsions,
    compute_fape,
    gram_schmidt,
)
from ..utils.rigid_utils import Rigid, Rotation
from ..utils import residue_constants as rc
from ..model.positions import PositionEmbedder, PositionDecoder
from functools import partial
import numpy as np


def log_I0(x):
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


class StructureTrack(Track):

    def tokenize(self, data, data_tok):

        frames, frame_mask = atom37_to_frames(data["atom37"], data["atom37_mask"])
        data_tok["trans"] = frames._trans
        data_tok["rots"] = frames._rots._rot_mats
        data_tok["frame_mask"] = frame_mask
        data_tok["ca_mask"] = data["atom37_mask"][..., rc.atom_order["CA"]]
        aatype = np.array(
            [rc.restype_order.get(c, rc.unk_restype_index) for c in data["seqres"]]
        )
        torsions, torsion_mask = atom37_to_torsions(data["atom37"], aatype)
        data_tok["torsions"] = torsions
        data_tok["torsion_mask"] = torsion_mask

    def add_modules(self, model):
        model.trans_mask = nn.Parameter(torch.zeros(model.cfg.dim))
        model.trans_embed = PositionEmbedder(self.cfg.embedder, model.cfg.dim)
        model.trans_out = PositionDecoder(self.cfg.decoder, model.cfg.dim)
        # model.rots_embed = nn.Linear(9, model.cfg.dim)
        # model.trans_out = nn.Linear(model.cfg.dim, 3)
        # model.rots_out = nn.Linear(model.cfg.dim, 9)

    def corrupt(self, batch, noisy_batch, target):

        com = (batch["trans"] * batch["ca_mask"][..., None]).sum(-2) / batch[
            "ca_mask"
        ].sum(-1)[..., None]
        trans = torch.where(
            batch["ca_mask"][..., None].bool(), batch["trans"] - com[..., None, :], 0.0
        )

        # for now delete everything with +x
        noisy_batch["trans_noise"] = trans[..., 0] > 0
        noisy_batch["trans"] = torch.where(
            noisy_batch["trans_noise"][..., None], 0.0, trans
        )

        target["trans"] = trans
        noisy_batch["ca_mask"] = target["ca_mask"] = batch["ca_mask"]

        # noisy_batch["trans"] = batch["trans"]
        # target["trans"] = batch["trans"]
        # target["rots"] = batch["rots"]
        # target["frame_mask"] = batch["frame_mask"]
        # target["ca_mask"] = batch["ca_mask"]

    def embed(self, model, batch):
        pos_embed = model.trans_embed(batch["trans"])
        x = torch.where(
            batch["trans_noise"][..., None], model.trans_mask[None, None], pos_embed
        )
        x = torch.where(
            batch["ca_mask"][..., None].bool(),
            x,
            model.trans_mask[None, None],
        )
        return x

    def predict(self, model, out, readout):
        readout["trans"] = model.trans_out(out)
        # readout["rots"] = model.rots_out(out)

    def compute_loss(self, readout, target):
        if self.cfg.decoder.type == "linear":
            rmsd_loss = self.compute_rmsd_loss(
                readout["trans"], target["trans"], target["ca_mask"]
            )
            return rmsd_loss.mean()

        elif self.cfg.decoder.type == "sinusoidal":
            sinusoidal_loss = self.compute_sinusoidal_loss(readout, target)
            return sinusoidal_loss.mean()

        else:
            raise Exception(
                f"PositionDecoder type {self.cfg.decoder.type} not recognized"
            )

        # fape_loss = self.compute_fape_loss(readout, target)

    def compute_sinusoidal_loss(self, readout, target, eps=1e-6):
        logits = readout["trans"]
        ang = torch.atan2(logits[..., 1::2], logits[..., ::2])
        gt_ang = 2 * np.pi * target["trans"] / self.cfg.decoder.max_period
        ang_error = ((gt_ang - ang) + np.pi) % (2 * np.pi) - np.pi
        self.logger.log(
            "trans_rmsd",
            torch.sqrt(
                (torch.square(ang_error).sum(-1) * target["ca_mask"]).sum(-1)
                / target["ca_mask"].sum(-1)
            )
            * self.cfg.decoder.max_period
            / (2 * np.pi),
        )
        norm = torch.sqrt(logits[..., 1::2] ** 2 + logits[..., ::2] ** 2 + 1e-5)
        logp = (
            torch.cos(gt_ang) * logits[..., ::2] + torch.sin(gt_ang) * logits[..., 1::2]
        )
        logz = np.log(2 * np.pi) + log_I0(norm)
        nll = logz - logp

        # prec = norm**2
        # logp_ = torch.where(
        #     prec < 100,
        #     prec * (torch.cos(ang_error) - 1),
        #     - ang_error**2 * prec / 2
        # )
        # logz_ = torus_logZ(prec)
        # nll_ = logz_ - logp_

        self.logger.log("trans_logit_norm", norm, mask=target["ca_mask"][..., None])
        loss = (nll.sum(-1) * target["ca_mask"]).sum(-1) / (
            eps + target["ca_mask"].sum()
        )
        self.logger.log("trans_loss", loss)
        return loss

    def compute_fape_loss(self, readout, target):
        shape = readout["rots"].shape[:-1] + (3, 3)
        pred_rots = readout["rots"].reshape(*shape)
        pred_rots = gram_schmidt(
            torch.zeros_like(pred_rots[..., 0]), pred_rots[..., 0], pred_rots[..., 1]
        )

        pred_frames = Rigid(trans=readout["trans"], rots=Rotation(rot_mats=pred_rots))
        target_frames = Rigid(
            trans=target["trans"], rots=Rotation(rot_mats=target["rots"])
        )

        fape_fn = partial(
            compute_fape,
            pred_frames=pred_frames,
            target_frames=target_frames,
            frames_mask=target["frame_mask"],
            pred_positions=pred_frames.get_trans(),
            target_positions=target_frames.get_trans(),
            positions_mask=target["ca_mask"],
            length_scale=10,
        )
        fape_loss = (
            0.1 * fape_fn()
            + 0.45 * fape_fn(thresh=15)
            + 0.45 * fape_fn(l1_clamp_distance=10)
        )
        return fape_loss

    def compute_rmsd_loss(self, pred, target, mask, eps=1e-5):
        err = torch.square(pred - target).sum(-1)
        err = (err * mask).sum(-1) / mask.sum(-1)
        return torch.sqrt(err + eps)
