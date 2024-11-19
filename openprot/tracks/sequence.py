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

MASK_IDX = 21
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

    def tokenize(self, data):

        data["aatype"] = np.array(
            [rc.restype_order.get(c, rc.unk_restype_index) for c in data["seqres"]]
        )
    
    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in rc.restypes_with_x]
        return torch.tensor(esm_reorder)

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
        
        elif self.cfg.noise_schedule == "linear":
            a = t - start
            return ((1-a) * tokens) + (a * noisy_state)

        elif self.cfg.noise_schedule == "cos":
            start_a = torch.pow(torch.cos((start + 1e-2)/(1 + 1e-2) * torch.pi/2), 2)
            end_a = torch.pow(torch.cos((t + 1e-2)/(1 + 1e-2) * torch.pi/2), 2)
            a = start_a - end_a
            return ((1-a) * tokens) + (a * noisy_state)

        raise ValueError("Invalid noise schedule")

        
        rate_matrix = pd.read_csv(self.cfg.rate_matrix_path, index_col=0)
        self.flow = torch.tensor(rate_matrix.values, dtype=torch.float32)
        
        flow = self.flow[:-1, :-1]

        Q = torch.zeros((NUM_TOKENS, NUM_TOKENS-1))
        Q[:-1, :] = flow
        Q[-1, :] = 1

        b = torch.zeros(NUM_TOKENS)
        b[-1] = 1

        self.steady_state = torch.linalg.lstsq(Q, b, rcond=None).solution
        self.steady_state = torch.cat((self.steady_state, torch.tensor([0])))

        # model.seq_mask = nn.Parameter(torch.zeros(model.cfg.dim))

    def apply_flow(self, tokens, flow, noise_levels):
        num_tokens, seq_len = tokens.shape

        # compute transition probability for each noise level
        # new_token sampled from probability for that token i arising from ith column in transition probability
        # compute transition probabilities for each token
        flow = flow.to(tokens.device)
        noise_levels = noise_levels.view(num_tokens, seq_len, 1, 1)

        tokens_one_hot = F.one_hot(tokens, num_classes = NUM_TOKENS).unsqueeze(-1).float()
        new_tokens_distribution = self.noise_transform(noise_levels, tokens_one_hot)
        new_tokens_distribution = new_tokens_distribution.squeeze(-1)

        new_tokens = torch.multinomial(new_tokens_distribution.view(-1, 21), num_samples=1)
        new_tokens = new_tokens.view(num_tokens, seq_len).squeeze(-1)

        return new_tokens

        
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

    def add_modules(self, model):
        model.seq_embed = nn.Embedding(21, model.cfg.dim)
        model.seq_out = nn.Linear(model.cfg.dim, 21)
        
        rate_matrix = pd.read_csv(self.cfg.rate_matrix_path, index_col=0)
        self.flow = torch.tensor(rate_matrix.values, dtype=torch.float32)
        
        flow = self.flow[:-1, :-1]

        Q = torch.zeros((NUM_TOKENS, NUM_TOKENS-1))
        Q[:-1, :] = flow
        Q[-1, :] = 1

        b = torch.zeros(NUM_TOKENS)
        b[-1] = 1

        self.steady_state = torch.linalg.lstsq(Q, b, rcond=None).solution
        self.steady_state = torch.cat((self.steady_state, torch.tensor([0])))

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


    def corrupt(self, batch, noisy_batch, target, logger=None):
        if self.cfg.corrupt == "mask":
            tokens = batch["aatype"]

            mask = batch["seq_noise"].bool()

            rand = torch.rand(tokens.shape, device=tokens.device)
            randaa = torch.randint(0, 21, tokens.shape, device=tokens.device)

            inp = tokens
            inp = torch.where((rand < 0.8) & mask, MASK_IDX, inp)
            inp = torch.where((rand > 0.9) & mask, randaa, inp)

            noisy_batch["aatype"] = inp
            noisy_batch['seq_noise'] = batch['seq_noise']    
            target["seq_supervise"] = (batch["seq_noise"] > 0) * batch["seq_mask"]

        elif self.cfg.corrupt == "discrete_fm":
            tokens = batch["aatype"]
            
            # apply flow matrix
            flowed_tokens = self.apply_flow(tokens, self.flow, noise_levels=batch["seq_noise"])
            
            # update noisy batch
            noisy_batch["aatype"] = flowed_tokens
            noisy_batch["seq_noise"] = batch["seq_noise"]
            target["seq_supervise"] = torch.ones_like(tokens).float() * batch["seq_mask"] 

            target["aatype"] = batch["aatype"]  # original tokens as target

            if logger:
                logger.log("seq/toks", batch["seq_mask"].sum())

        # elif self.cfg.corrupt == "discrete_uniform": 

        #     tokens = batch["aatype"]

        #     # compute noise level for current timestep
        #     noise_level = scheduler('linear', timestep)

        #     # apply noise; probability of corruption increases with the timestep
        #     rand = torch.rand(tokens.shape, device=tokens.device)
        #     rand_perturbation = torch.randint(0, 21, tokens.shape, device=tokens.device)

        #     # mask indices for noisy tokens 
        #     mask_tokens = torch.full(tokens.shape, MASK_IDX, device=tokens.device)

        #     # apply corruption using a threshold based on the noise level
        #     inp = torch.where(rand < noise_level, rand_perturbation, tokens)
        #     # inp = torch.where(rand < noise_level * 0.8, mask_tokens, inp) # further mask some tokens

        #     # update noisy batch and supervision target
        #     noisy_batch["aatype"] = inp
        #     target["seq_supervise"] = (rand < noise_level).float() * batch["seq_mask"]

        #     target["aatype"] = batch["aatype"]

        #     if logger:
        #         logger.log("seq/timestep", timestep)
        #         logger.log("seq/noise_level", noise_level)
        #         logger.log("seq/toks", batch["seq_mask"].sum())

        # # corrupt according to rate matrix transition probs
        # elif self.cfg.corrupt == "discrete_rate_matrix": 

        #     rate_matrix = np.ones((21, 21)) / 21 # uniform as a placeholder
            
        #     tokens = batch["aatype"]

        #     # compute noise level for the current timestep
        #     noise_level = scheduler(timestep)
            
        #     # determine which tokens to corrupt
        #     rand = torch.rand(tokens.shape, device=tokens.device)
        #     mask = rand < noise_level # tokens to be corrupted

        #     # sample tokens based on rate matrix
        #     num_tokens = rate_matrix.shape[0]

        else:
            raise Exception(
                f"SequenceTrack corrupt type {self.cfg.corrupt} not implemented"
            )

        target["aatype"] = batch["aatype"]

        if logger:
            logger.log("seq/toks", batch["seq_mask"].sum())
            
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
            noise_embed = sinusoidal_embedding(batch["seq_noise"], model.cfg.dim//2, 1, .01)
            inp["x"] += noise_embed

      
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
