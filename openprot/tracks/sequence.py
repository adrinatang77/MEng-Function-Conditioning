import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from typing import override
from .track import OpenProtTrack
from ..utils import residue_constants as rc
from ..utils.prot_utils import seqres_to_aatype
from functools import partial

import pandas as pd
import math
from transformers import EsmTokenizer

MASK_IDX = 20
NUM_TOKENS = MASK_IDX+1


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

RNA_LETTERS = {"A": 0, "G": 1, "C": 2, "U": 3}
DNA_LETTERS = {"A": 0, "G": 1, "C": 2, "T": 3}

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

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class EsmLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(in_dim)
        self.dense = nn.Linear(in_dim, in_dim)
        self.layer_norm2 = nn.LayerNorm(in_dim)

        self.decoder = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.dense(x)
        x = gelu(x)
        x = self.layer_norm2(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x
        
class SequenceTrack(OpenProtTrack):

    def setup(self):
        self.alphabet = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')

        if self.cfg.all_atom:
            self.ntoks = 21 + 5 + 5 + 128
        else:
            self.ntoks = NUM_TOKENS # 21
    def tokenize(self, data):
        
        prot_aatype = np.array(seqres_to_aatype(data["seqres"]))
        rna_aatype = np.array([RNA_LETTERS.get(c, 4) for c in data["seqres"]])
        dna_aatype = np.array([DNA_LETTERS.get(c, 4) for c in data["seqres"]])
        lig_aatype = data["atom_num"].astype(int)
        data['aatype'] = (data['mol_type'] == 0) * prot_aatype
        data['aatype'] += (data['mol_type'] == 1) * (dna_aatype + 21)
        data['aatype'] += (data['mol_type'] == 2) * (rna_aatype + 26)
        data['aatype'] += (data['mol_type'] == 3) * (lig_aatype + 31)
        data['aatype'] = data['aatype'].astype(int)    
        
    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in rc.restypes_with_x]
        return torch.tensor(esm_reorder)

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
        model.seq_embed = nn.Embedding(self.ntoks, model.cfg.dim)
        if self.cfg.init:
            torch.nn.init.normal_(model.seq_embed.weight, std=self.cfg.init)
        if self.cfg.esm_lm_head:
            model.seq_out = EsmLMHead(model.cfg.dim, self.ntoks)
            if self.cfg.tied_weights:
                model.seq_out.decoder.weight = model.seq_embed.weight
        else:
            model.seq_out = nn.Linear(model.cfg.dim, self.ntoks)

        if self.cfg.all_atom:
            model.mol_type_cond = nn.Embedding(4, model.cfg.dim)
        
        
        if self.cfg.esm is not None:

            model.esm, self.esm_dict = esm_registry.get(self.cfg.esm)()
            model.esm.requires_grad_(False)
            model.esm.half()
            model.esm.eval()
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
        rand_mask = torch.rand_like(batch['seq_noise']) < batch['seq_noise']
        rand_mask &= batch['mol_type'] == 0 # corrupt protein sequences only

        mask = rand_mask | ~batch["seq_mask"].bool() # these will be input as MASK
        sup = rand_mask & batch["seq_mask"].bool() # these will be actually supervised

        #                                   !!!!!!!
        noisy_batch["aatype"] = torch.where(rand_mask, MASK_IDX, tokens)

        
        # present = ~batch["seq_noise"].bool() & batch["seq_mask"].bool()
        # target["seq_occupancy"] = present.sum(-1) / (1+batch['seq_mask'].sum(-1))        
        
        target["seq_supervise"] = torch.where(sup, batch["seq_weight"], 0.0)


        if self.cfg.reweight == 'linear':
            target["seq_supervise"] *= (1-batch['seq_noise'])
        elif self.cfg.reweight == 'inverse':
            target["seq_supervise"] *= 1/(batch['seq_noise'] + self.cfg.reweight_eps)
        target["aatype"] = tokens

        
        mlm_mask = (torch.rand_like(batch['seq_noise']) < self.cfg.mlm_prob) & (batch['mol_type'] == 0) & ~mask

        noisy_batch['aatype'] = torch.where(
            mlm_mask & (torch.rand_like(batch['seq_noise']) < self.cfg.mlm_ratio),
            torch.randint(0, 20, mlm_mask.shape, device=mlm_mask.device),
            noisy_batch['aatype']
        )
        target["seq_supervise"].masked_fill_(mlm_mask, self.cfg.mlm_weight)

        
        
        
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

        if self.cfg.all_atom:
            inp["x_cond"] += model.mol_type_cond(batch['mol_type'].int())
        
        
    def predict(self, model, inp, out, readout):
        readout["aatype"] = model.seq_out(out["x"])
        if not model.training:
            readout["aatype"][...,MASK_IDX:] = -np.inf # only predict proteins
        
    def compute_loss(self, readout, target, logger=None, eps=1e-6, **kwargs):
        
        loss = torch.nn.functional.cross_entropy(
            readout["aatype"].transpose(1, 2), target["aatype"], reduction="none"
        )

        mask = target["seq_supervise"]
        
        if logger:
            logger.masked_log("seq/loss", loss, mask=mask)
            logger.masked_log("seq/perplexity", loss, mask=mask, post=np.exp)

        # logits = readout["aatype"]
        # logits[...,-1] -= 1e5
        ## extract the pseudo-mask likelihoods
        # probs = logits.softmax(-1)
        # oh = torch.nn.functional.one_hot(target['noisy_aatype'], num_classes=self.ntoks)
        # denom = 0.5 * oh + 0.05
        # new_probs = probs / denom
        # new_probs /= new_probs.sum(-1, keepdims=True)
        # is_mask_prob = ((probs - oh) / (new_probs - oh))[...,0]
        # is_unmask = (target['noisy_aatype'] != MASK_IDX)
        # # print((is_mask_prob * is_unmask).sum(-1) / is_unmask.sum(-1))
        # # print((is_mask_prob * is_unmask).sum() / is_unmask.sum())
        # if logger:
        #     logger.masked_log("seq/is_mask_prob", is_mask_prob, mask=is_unmask)
        
        return loss * mask
