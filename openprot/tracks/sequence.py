import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import override
from .track import OpenProtTrack
from ..utils import residue_constants as rc

MASK_IDX = 21

class SequenceTrack(OpenProtTrack):

    def tokenize(self, data):
        data["aatype"] = np.array(
            [rc.restype_order.get(c, rc.unk_restype_index) for c in data["seqres"]]
        )

    def add_modules(self, model):
        model.seq_embed = nn.Embedding(22, model.cfg.dim)
        model.seq_out = nn.Linear(model.cfg.dim, 21)
        model.flow = nn.Parameter(torch.randn(21, 21)) # random flow matrix 
        # model.seq_mask = nn.Parameter(torch.zeros(model.cfg.dim))

    def apply_flow(self, tokens, flow, timestep, total_steps):
        num_tokens = flow.shape[0]
        
        # flow weight
        flow_weight = timestep / total_steps
        
        # compute transition probabilities for each token 
        transition_probs = F.softmax(flow * flow_weight, dim=-1) 
        
        # for each token, sample the next token based on the matrix
        new_tokens = tokens.clone()
        for i in range(num_tokens):
            token_mask = (tokens == i)
            if token_mask.any():
                new_token_distribution = transition_probs[i]  # get transition probabilities for token `i`
                sampled_tokens = torch.multinomial(new_token_distribution, token_mask.sum(), replacement=True).to(tokens.device)
                new_tokens[token_mask] = sampled_tokens

        return new_tokens

    def corrupt(self, batch, noisy_batch, target, timestep=None, total_steps=None, logger=None):
        if self.cfg.corrupt == "mask":
            tokens = batch["aatype"]

            mask = batch["seq_noise"].bool()

            rand = torch.rand(tokens.shape, device=tokens.device)
            randaa = torch.randint(0, 21, tokens.shape, device=tokens.device)

            inp = tokens
            inp = torch.where((rand < 0.8) & mask, MASK_IDX, inp)
            inp = torch.where((rand > 0.9) & mask, randaa, inp)

            noisy_batch["aatype"] = inp
            target["seq_supervise"] = (batch["seq_noise"] > 0) * batch["seq_mask"]

        elif self.cfg.corrupt == "discrete_fm":
            tokens = batch["aatype"]
            
            # apply flow matrix
            flowed_tokens = self.apply_flow(tokens, self.flow, timestep, total_steps)
            
            # update noisy batch
            noisy_batch["aatype"] = flowed_tokens
            target["seq_supervise"] = torch.ones_like(tokens).float() * batch["seq_mask"] 

            target["aatype"] = batch["aatype"]  # original tokens as target

            if logger:
                logger.log("seq/timestep", timestep)
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

    def embed(self, model, batch):
        x = model.seq_embed(batch["aatype"])
        return x

    def predict(self, model, out, readout):
        readout["aatype"] = model.seq_out(out)

    def compute_loss(self, readout, target, logger=None, eps=1e-6):
        loss = torch.nn.functional.cross_entropy(
            readout["aatype"].transpose(1, 2), target["aatype"], reduction="none"
        )

        mask = target["seq_supervise"]
        if logger:
            logger.log("seq/loss", loss, mask=mask)
            logger.log("seq/perplexity", loss, mask=mask, post=np.exp)
            logger.log("seq/toks_sup", mask.sum().item())
        return (loss * mask).sum() / target["pad_mask"].sum()
