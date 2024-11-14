import torch
import torch.nn as nn
import numpy as np
from typing import override
from .track import OpenProtTrack
from ..utils import residue_constants as rc
from ..utils.prot_utils import seqres_to_aatype
from functools import partial


import esm
from esm import Alphabet

load_fn = esm.pretrained.load_model_and_alphabet
esm_registry = {
    "esm2_8M": partial(load_fn, "esm2_t6_8M_UR50D_500K"),
    "esm2_8M_270K": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_35M": partial(load_fn, "esm2_t12_35M_UR50D_500K"),
    "esm2_35M_270K": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_150M": partial(load_fn, "esm2_t30_150M_UR50D_500K"),
    "esm2_150M_270K": partial(load_fn, "esm2_t30_150M_UR50D_270K"),
    "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_650M_270K": partial(load_fn, "esm2_t33_650M_270K_UR50D"),
    "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_3B_270K": partial(load_fn, "esm2_t36_3B_UR50D_500K"),
    "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
}


class SequenceTrack(OpenProtTrack):

    def tokenize(self, data):
        data["aatype"] = np.array(seqres_to_aatype(data["seqres"]))

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in rc.restypes_with_x]
        return torch.tensor(esm_reorder)

    def add_modules(self, model):
        model.seq_embed = nn.Embedding(22, model.cfg.dim)
        model.seq_out = nn.Linear(model.cfg.dim, 21)
        if self.cfg.esm is not None:

            model.esm, self.esm_dict = esm_registry.get(self.cfg.esm)()
            model.esm.requires_grad_(False)
            model.esm.half()
            esm_dim = model.esm.layers[0].final_layer_norm.bias.shape[0]
            # model.esm_in = nn.Linear(esm_dim, model.cfg.dim)
            model.esm_s_combine = nn.Parameter(torch.zeros(model.esm.num_layers + 1))
            model.esm_s_mlp = nn.Sequential(
                nn.LayerNorm(esm_dim),
                nn.Linear(esm_dim, model.cfg.dim),
                nn.ReLU(),
                nn.Linear(model.cfg.dim, model.cfg.dim),
            )

            model.register_buffer("af2_to_esm", self._af2_to_esm(self.esm_dict))

        # model.seq_mask = nn.Parameter(torch.zeros(model.cfg.dim))

    def _compute_language_model_representations(self, esmaa):
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=self.cfg.use_esm_attn_map,
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        esm_z = (
            res["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
            if self.cfg.use_esm_attn_map
            else None
        )
        return esm_s, esm_z

    def corrupt(self, batch, noisy_batch, target, logger=None):
        MASK_IDX = 20  # not ideal but whatever

        if self.cfg.corrupt == "mask":
            tokens = batch["aatype"]

            mask = batch["seq_noise"].bool()

            rand = torch.rand(tokens.shape, device=tokens.device)
            randaa = torch.randint(0, 20, tokens.shape, device=tokens.device)

            inp = tokens
            inp = torch.where((rand < 0.8) & mask, MASK_IDX, inp)
            inp = torch.where((rand > 0.9) & mask, randaa, inp)

            noisy_batch["aatype"] = inp
            target["seq_supervise"] = (batch["seq_noise"] > 0) * batch["seq_mask"]

        else:
            raise Exception(
                f"SequenceTrack corrupt type {self.cfg.corrupt} not implemented"
            )

        target["aatype"] = batch["aatype"]

        if logger:
            logger.masked_log("seq/toks", batch["seq_mask"], sum=True)

    def embed(self, model, batch, inp):

        def _af2_idx_to_esm_idx(aa, mask):
            aa = (aa + 1).masked_fill(mask != 1, 0)
            return model.af2_to_esm[aa]

        if self.cfg.esm is not None:

            esmaa = _af2_idx_to_esm_idx(batch["aatype"], batch["pad_mask"])
            res = model.esm(esmaa, repr_layers=range(model.esm.num_layers + 1))
            esm_s = torch.stack(
                [v for _, v in sorted(res["representations"].items())], dim=2
            )
            # rep = res['representations'][model.esm.num_layers]
            esm_s = esm_s.detach().float()

            # === preprocessing ===
            esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
            inp["x"] += model.esm_s_mlp(esm_s)

        inp["x"] += model.seq_embed(batch["aatype"])

    def predict(self, model, inp, out, readout):
        readout["aatype"] = model.seq_out(out["x"])

    def compute_loss(self, readout, target, logger=None, eps=1e-6, **kwargs):
        loss = torch.nn.functional.cross_entropy(
            readout["aatype"].transpose(1, 2), target["aatype"], reduction="none"
        )

        mask = target["seq_supervise"]
        if logger:
            logger.masked_log("seq/loss", loss, mask=mask)
            logger.masked_log("seq/perplexity", loss, mask=mask, post=np.exp)
            logger.masked_log("seq/toks_sup", mask, sum=True)
        return loss * mask
