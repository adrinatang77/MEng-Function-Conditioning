import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from typing import override
from .track import OpenProtTrack
from ..utils import residue_constants as rc
from functools import partial

import pandas as pd
import math

MASK_IDX = 20
NUM_TOKENS = 21


# processes a sequence (integers) into its class distribution representation (one-hot encoding)
# returns tensor of size (seqlen, NUM_TOKENS)
def seq2prob(seq):
    seq = seq.to(torch.int64)
    return F.one_hot(seq, num_classes=NUM_TOKENS)


# given a probability distribution over classes, returns single sample
# pt is probability distribution of shape (seqlen, NUM_TOKENS)
def sample_p(pt):
    pt = pt.to(torch.float32)
    return torch.multinomial(pt, 1, replacement=True)


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

    def setup(self):
        # rate_matrix = pd.read_csv(self.cfg.rate_matrix_path, index_col=0)
        # flow = torch.from_numpy(rate_matrix.values).float()
        # self.flow = flow[:-1, :-1]
        pass

    def tokenize(self, data):

        data["aatype"] = np.array(
            [rc.restype_order.get(c, rc.unk_restype_index) for c in data["seqres"]]
        )

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in rc.restypes_with_x]
        return torch.tensor(esm_reorder)

    """
    def noise_transform(self, t, tokens, start = 0):
        '''
        takes tokens and noise_levels t and returns new noised distribution of tokens, offset by starting noise
        '''
        batch_size, seq_len = tokens.shape[0], tokens.shape[1]
        start = torch.full(t.shape, start, device=t.device)
        device = t.device
        tokens = tokens.to(device)
        self.steady_state = self.steady_state.to(device)
        noisy_state = self.steady_state.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(*tokens.shape)

        if self.cfg.noise_schedule == "tan":
            t[t==1] = 1 - 1e-5
            start = torch.tan(start * (torch.pi/2))
            end = torch.tan(t * (torch.pi/2))
            converted_noise = end - start
            flows = self.flow.unsqueeze(0).repeat(batch_size, seq_len, 1, 1).to(device)
            transition = torch.matrix_exp(converted_noise.repeat(1, 1, NUM_TOKENS, NUM_TOKENS) * flows)
            return torch.matmul(transition, tokens)
        
        elif self.cfg.noise_schedule == "exp":
            start = self.cfg.noise_min + (self.cfg.noise_max - self.cfg.noise_min) * (1- torch.exp(-self.cfg.noise_k * start))
            end = self.cfg.noise_min + (self.cfg.noise_max - self.cfg.noise_min) * (1- torch.exp(-self.cfg.noise_k * t))
            converted_noise = end - start
            flows = self.flow.unsqueeze(0).repeat(batch_size, seq_len, 1, 1).to(t.device)
            transition = torch.matrix_exp(converted_noise.repeat(1, 1, NUM_TOKENS, NUM_TOKENS) * flows)
            return torch.matmul(transition, tokens)
        
        raise ValueError("Invalid noise schedule")

    """

    def apply_flow(self, tokens, t):

        def t_transform(t):
            return t  # temporary

        tokens_ = torch.where(tokens == MASK_IDX, 0, tokens)  # temporary
        B, L = tokens.shape

        probs = torch.matrix_exp(self.flow.to(t) * t_transform(t)[..., None, None])
        toks_oh = F.one_hot(tokens_, num_classes=20)
        probs = torch.einsum("...ij,...j->...i", probs, toks_oh.float())

        new_toks = torch.distributions.categorical.Categorical(probs).sample()

        return torch.where(tokens == MASK_IDX, MASK_IDX, new_toks)

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

    def add_modules(self, model):
        model.seq_embed = nn.Embedding(21, model.cfg.dim)
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
        # keep this separate to avoid confusion

    def corrupt(self, batch, noisy_batch, target, logger=None):
        eps = 1e-6
        
        tokens = batch["aatype"]
        
        mask = batch["seq_noise"].bool() & batch["seq_mask"].bool()

        noisy_batch["aatype"] = torch.where(mask, MASK_IDX, tokens)
        noisy_batch["seq_noise"] = batch["seq_noise"]

        target["aatype"] = tokens
        
        if self.cfg.loss_reweight == 'var':
            t = mask.sum(-1) / (eps + batch['seq_mask'].sum(-1))
            target["seq_supervise"] = mask / (eps + t[...,None])
            
        if self.cfg.loss_reweight == 'norm':
            t = mask.sum(-1)
            target["seq_supervise"] = mask / (eps + t[...,None])
            
        elif self.cfg.loss_reweight == 'linear':
            t = mask.sum(-1) / (eps + batch['seq_mask'].sum(-1))
            target["seq_supervise"] = mask * (1 - t[...,None])

        else:
            target["seq_supervise"] = mask    
        
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
        if self.cfg.embed_t:
            t_emb = sinusoidal_embedding(
                batch["seq_noise"], model.cfg.dim // 2, 1, 0.01
            )
            inp["x"] += torch.where(batch["aatype"] != MASK_IDX, noise_embed, 0.0)

    def predict(self, model, inp, out, readout, inf=1e6):
        readout["aatype"] = model.seq_out(out["x"])
        readout["aatype"][...,-1] = -inf # ban X

    def compute_loss(self, readout, target, logger=None, eps=1e-6, **kwargs):
        loss = torch.nn.functional.cross_entropy(
            readout["aatype"].transpose(1, 2), target["aatype"], reduction="none"
        )

        mask = target["seq_supervise"]
        # denoise_mask = target["denoise_seq_supervise"]
        
        if logger:
            logger.masked_log("seq/loss", loss, mask=mask)
            logger.masked_log("seq/perplexity", loss, mask=mask, post=np.exp)
            # logger.masked_log("seq/denoise_loss", loss, mask=denoise_mask)
            # logger.masked_log("seq/denoise_perplexity", loss, mask=denoise_mask, post=np.exp)
            
            # logger.masked_log("seq/toks_sup", unmask_mask, sum=True)
            # logger.masked_log("seq/denoise_toks_sup", denoise_mask, sum=True)

        return loss * mask # self.cfg.unmask_weight * loss * unmask_mask + self.cfg.denoise_weight * loss * denoise_mask
