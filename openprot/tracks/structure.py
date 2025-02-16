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
from multiflow.data import so3_utils
import numpy as np
import math

def modulate(x, shift, scale):
    if shift is not None:
        return x * (1 + scale) + shift
    else:
        return x


def gate(x, gate_):
    if gate_ is not None:
        return x * gate_
    else:
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, dim, out):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
        
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
        self._igso3 = None
    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def tokenize(self, data):
        pass 

    def add_modules(self, model):
        if self.cfg.embed_mask:
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
            if self.cfg.readout_adaLN:
                model.trans_out = FinalLayer(model.cfg.dim, 3)
            else:
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
            
        if logger:
            logger.masked_log("struct/toks", batch["struct_mask"], sum=True)

    def embed(self, model, batch, inp):

        if self.cfg.rescale_time:
            tmax = self.cfg.t_emb_max
        else:
            tmax = 1
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
            inp["x"] += model.trans_in(precond)

        # embed sigma
        def sigma_transform(noise_level):
            t_emb = noise_level ** (1/self.cfg.t_emb_p)
            t_emb = t_emb / self.cfg.t_emb_max ** (1/self.cfg.t_emb_p)
            return t_emb
            
        if self.cfg.rescale_time:
            t_emb = sigma_transform(noise) 
            t_emb = sinusoidal_embedding(t_emb, model.cfg.dim // 2, 1, 0.01).float()
        else:
            t_emb = sinusoidal_embedding(noise, model.cfg.dim // 2, 1, 0.01).float()
        inp["x_cond"] += t_emb
        
        if self.cfg.embed_sigma:
            inp["x"] += t_emb
            
        if self.cfg.embed_pairwise:
            sq_mask = (~embed_as_mask[...,None]) & (~embed_as_mask[...,None,:])
            dmat = coords[...,None,:,:] - coords[...,None,:]
            dmat = torch.square(dmat).sum(-1).sqrt()
            inp["z"] = inp.get("z", 0) + model.pairwise_in(
                sinusoidal_embedding(dmat, model.cfg.pairwise_dim // 2, 10.0, 0.1)
            ) * sq_mask.float()[...,None]
            

        # finally provide the raw struct for relpos
        inp["struct"] = coords
        if self.cfg.postcondition:
            inp["postcond_fn"] = lambda x: self.diffusion.postcondition(
                coords, x, noise,
            )
        else:
            inp['postcond_fn'] = lambda x: x

    def predict(self, model, inp, out, readout):
        if self.cfg.readout_trans == "trunk":
            if self.cfg.readout_adaLN:
                readout["trans"] = inp["postcond_fn"](model.trans_out(out["x"], inp["x_cond"]))[None]
            else:
                readout["trans"] = inp["postcond_fn"](model.trans_out(out["x"]))[None]
        elif self.cfg.readout_trans == "sm":
            raise Exception("check")
        elif self.cfg.copy_trans == 'sm':
            readout["trans"] = inp["postcond_fn"](out["sm"]["trans"])
        elif self.cfg.copy_trans == 'trunk':
            readout['trans'] = inp["postcond_fn"](out['trans'])[None]
        if self.cfg.readout_pairwise:
            readout["pairwise"] = model.pairwise_out(out["z"])


    def compute_loss(self, readout, target, logger=None, **kwargs):


        mse = self.compute_mse_loss(
            readout, target, aligned=False, logger=logger
        )
        aligned_mse = self.compute_mse_loss(
            readout, target, aligned=True, logger=logger
        )
        soft_lddt = self.compute_lddt_loss(
            readout, target, logger=logger
        )

        if self.cfg.aligned_loss:
            w = target['seq_occupancy'].unsqueeze(-1)
            loss = w * self.cfg.loss['aligned_mse'] * aligned_mse 
            loss += w * self.cfg.loss['lddt'] * soft_lddt
            loss += (1-w) * self.cfg.loss['mse'] * mse

        else:
            loss = mse
            
        lddt = compute_lddt(
            readout["trans"][-1], target["struct"], target["struct_supervise"]
        )
        if logger:
            logger.masked_log("struct/lddt", lddt, dims=0)

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

    def compute_mse_loss(self, readout, target, logger=None, aligned=False, eps=1e-5):

        pred = readout["trans"]
        gt = target["struct"]
        
        mask = target["struct_supervise"] # weighted mse!
        t = target["struct_noise"]
        
        mse = self.diffusion.compute_loss(
            pred=pred,
            target=gt,
            t=t,
            mask=mask, # used for alignment
            aligned=aligned,
        ) 
        # mse = torch.clamp(2 * mse, max=5)

        if logger:
            logger.masked_log("struct/mse_loss", mse[-1], mask=mask)
            logger.masked_log("struct/mse_loss_aux", mse.mean(0), mask=mask)

        w = self.cfg.int_loss_weight
        return w * mse.mean(0) + (1 - w) * mse[-1]   