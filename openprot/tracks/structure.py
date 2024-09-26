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
)
from ..utils.rotation_conversions import axis_angle_to_matrix
from ..utils.rigid_utils import Rigid, Rotation
from ..utils import residue_constants as rc
from ..model.positions import PositionEmbedder, PositionDecoder, PairwiseProjectionHead
from functools import partial
import numpy as np

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
        if not self.cfg.frames:
            model.trans_embed = nn.Linear(3, model.cfg.dim)
            model.rots_embed = nn.Linear(9, model.cfg.dim)

            model.trans_out = nn.Linear(model.cfg.dim, 3)
            model.rots_out = nn.Linear(model.cfg.dim, 3)
        
        model.frame_mask = nn.Parameter(torch.zeros(model.cfg.dim))
        model.frame_null = nn.Parameter(torch.zeros(model.cfg.dim))
        if self.cfg.pairwise:
            model.pairwise_out = PairwiseProjectionHead(model.cfg.dim, 64)
        elif self.cfg.trunk:
            model.pairwise_out = nn.Linear(model.cfg.trunk.pairwise_state_dim, 64)

    def corrupt(self, batch, noisy_batch, target, logger=None):

        ## right now our corruption is just masking EVERYTHING
        for key in [
            "trans_noise",
            "rots_noise",
            "frame_mask",
        ]:
            noisy_batch[key] = batch[key]

        for key in [
            "frame_trans",
            "frame_rots",
        ]:
            target[key] = batch[key]

        target["struct_supervise"] = batch["frame_mask"]

        dev = batch["frame_trans"].device
        B, L, _ = batch["frame_trans"].shape

        # add noise # placeholder
        noisy_batch["frame_trans"] = torch.zeros_like(batch["frame_trans"])

        # add noise # placeholder
        noisy_batch["frame_rots"] = torch.eye(3, device=dev).expand(B, L, 3, 3)

        target['beta'], target['beta_mask'] = pseudo_beta_fn(
            batch['aatype'], 
            batch['atom37'], 
            batch['atom37_mask']
        )

        if logger:
            logger.log("struct/toks", batch["frame_mask"].sum())

    def embed(self, model, batch, inp):

        # currently not embedding noise levels

        B, L, _ = batch["frame_trans"].shape
        dev = batch["frame_trans"].device

        # these masks add up to 1
        full_noise_mask = (
            (batch["rots_noise"] == 1.0)
            & (batch["trans_noise"] == 1.0)
            & batch["frame_mask"].bool()
        )
        null_mask = ~batch["frame_mask"].bool()
        embed_mask = ~full_noise_mask & ~null_mask

        if self.cfg.frames:
            empty_rots = torch.eye(3, device=dev).expand(B, L, 3, 3)
            inp["trans"] = torch.where(embed_mask[...,None], batch["frame_trans"], 0.0)
            inp["rots"] = torch.where(embed_mask[...,None,None], batch["frame_rots"], empty_rots)
        else:
            inp["x"] += torch.where(
                embed_mask[..., None], model.trans_embed(batch["frame_trans"]), 0.0
            )
            inp["x"] += torch.where(
                embed_mask[..., None],
                model.rots_embed(batch["frame_rots"].view(B, L, 9)),
                0.0,
            )

        # tell the model which frames were not present
        inp["x"] += torch.where(
            null_mask[..., None],
            model.frame_null[None, None],
            0.0,
        )
        # tell the model which frames were fully noised
        inp["x"] += torch.where(
            full_noise_mask[..., None],
            model.frame_mask[None, None],
            0.0,
        )

    def predict(self, model, out, readout):
        if self.cfg.frames:
            readout["trans"] = out["trans"]
            readout["rots"] = out["rots"]

        else:
            readout["trans"] = model.trans_out(out["x"])
            rotvec = model.rots_out(out["x"])
            readout["rots"] = axis_angle_to_matrix(rotvec)
        if self.cfg.pairwise:
            readout["pairwise"] = model.pairwise_out(out["x"])
        elif self.cfg.trunk:
            readout["pairwise"] = model.pairwise_out(out["z"])

    def compute_loss(self, readout, target, logger=None):

        loss = 0

        """
        if "rmsd" in self.cfg.losses:
            loss = loss + self.cfg.losses["rmsd"] * self.compute_rmsd_loss(
                readout, target, logger=logger
            )

        if "sinusoidal" in self.cfg.losses and self.cfg.decoder.type == "sinusoidal":
            raise Exception("sinusoidal structural loss no longer supported")
            loss = loss + self.cfg.losses["sinusoidal"] * self.compute_sinusoidal_loss(
                readout, target, logger=logger
            )
        """
        if "pade" in self.cfg.losses:
            loss = loss + self.cfg.losses["pade"] * self.compute_pade_loss(
                readout, target, logger=logger
            )

        if "fape" in self.cfg.losses:
            loss = loss + self.cfg.losses["fape"] * self.compute_fape_loss(
                readout, target, logger=logger
            )

        if "distogram" in self.cfg.losses:
            loss = loss + self.cfg.losses["distogram"] * self.compute_distogram_loss(
                readout, target, logger=logger
            )

        lddt = compute_lddt( # temporary
            readout["trans"][-1], target["frame_trans"], target["struct_supervise"]
        )
            
        mask = target["struct_supervise"]

        if logger:
            logger.log("struct/toks_sup", mask.sum())
            logger.log("struct/loss", loss, mask=mask)
            logger.log("struct/lddt", lddt)

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

    """
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
    """

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
            # + 0.45 * fape_fn(thresh=15)
            + 0.9 * fape_fn(l1_clamp_distance=10)
        )
        
        fape_loss = fape_loss.mean(0) # loss weighted and reduced correctly! 

        if logger:
            logger.log("struct/fape_loss", fape_loss, mask=mask)
        return fape_loss

    """
    def compute_rmsd_loss(self, readout, target, logger=None, eps=1e-5):
        pred = readout["trans"]
        gt = target["frame_trans"]
        mask = target["struct_supervise"]

        err = torch.square(pred - gt).sum(-1)
        if logger:
            logger.log("struct/rmsd", err, mask, post=np.sqrt)
            logger.log("struct/rmsd_loss", err, mask=mask)

        return err  # (err * mask).sum() / target["pad_mask"].sum()
    """

    def compute_distogram_loss(self, readout, target, logger=None, eps=1e-5):

        dev = readout["pairwise"].device
        bins = torch.linspace(0, 22, 64, device=dev)[:63]

        distmat = target["beta"][:, None] - target["beta"][:, :, None]
        distmat = torch.sqrt(torch.square(distmat).sum(-1))
        label = (distmat[..., None] > bins).sum(-1)

        loss = torch.nn.functional.cross_entropy(
            readout["pairwise"].permute(0, 3, 1, 2), label, reduction="none"
        )

        # some temporary stuff
        mask = target["struct_supervise"] * target['beta_mask']
        mask = mask[:, None] * mask[:, :, None]
        if logger:
            logger.log("struct/distogram", loss, mask)

        return (loss * mask).sum(-1) / (eps + mask.sum(-1))
