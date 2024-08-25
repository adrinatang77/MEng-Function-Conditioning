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


class StructureTrack(Track):

    def tokenize(self, data, data_tok):

        frames, frame_mask = atom37_to_frames(data["atom37"], data["atom37_mask"])
        data_tok["trans"] = frames._trans
        data_tok["rots"] = frames._rots._rot_mats
        data_tok["frame_mask"] = frame_mask
        data_tok["ca_mask"] = data["atom37_mask"][..., rc.atom_order["CA"]]
        aatype = np.array([rc.restype_order_with_x[c] for c in data["seqres"]])
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

        com = (batch['trans'] * batch['ca_mask'][...,None]).sum(-2) / batch['ca_mask'].sum(-1)[...,None]
        trans = batch['trans'] - com[...,None,:] * batch['ca_mask'][...,None]

        # for now delete everything with +x
        noisy_batch['trans_noise'] = trans[...,0] > 0
        noisy_batch['trans'] = torch.where(noisy_batch['trans_noise'][...,None], 0.0, trans)
        
        target["trans"] = trans
        target["ca_mask"] = batch["ca_mask"]
        
        # noisy_batch["trans"] = batch["trans"]
        # target["trans"] = batch["trans"]
        # target["rots"] = batch["rots"]
        # target["frame_mask"] = batch["frame_mask"]
        # target["ca_mask"] = batch["ca_mask"]

    def embed(self, model, batch):
        pos_embed = model.trans_embed(batch['trans'])
        x = torch.where(
            batch['trans_noise'][...,None], 
            model.trans_mask[None,None],
            pos_embed
        )
        return x

    def predict(self, model, out, readout):
        readout["trans"] = model.trans_out(out)
        # readout["rots"] = model.rots_out(out)

    def compute_loss(self, readout, target):
        rmsd_loss = self.compute_rmsd_loss(
            readout["trans"], target["trans"], target['ca_mask'])
        return rmsd_loss
        # fape_loss = self.compute_fape_loss(readout, target)
        
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
