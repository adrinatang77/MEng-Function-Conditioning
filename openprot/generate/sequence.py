from ..tracks.sequence import MASK_IDX, NUM_TOKENS
from torch.distributions.categorical import Categorical
import torch
import numpy as np
from ..utils import residue_constants as rc

def topk_masking(scores, k, mask=None, temp=1.0):
    gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
    scores = scores + temp * gumbel
    if mask is not None:
        scores = torch.where(mask, scores, -np.inf) 
    new = torch.zeros_like(scores).bool()
    idx = torch.topk(scores, k, dim=-1).indices
    new.scatter_(-1, idx, True)
    return new

class SequenceUnmaskingStepper:
    def __init__(self, cfg=None):
        self.cfg = cfg
        
    def set_step(self, batch, sched, extra={}):
        pass
        
    def advance(self, batch, sched, out, extra={}):

        
        
        t, s = sched['sequence']

        L = batch['aatype'].shape[1]
        
        num_mask = int(round(t * L))    
        num_unmask = int(L - num_mask)
        new_num_mask = int(round(s * L))
        
        logits = out['aatype'] 
        
        
        is_unmask = (batch['aatype'] != MASK_IDX)
        is_mask = (batch['aatype'] == MASK_IDX)

        probs = logits.softmax(-1)
        oh = torch.nn.functional.one_hot(batch['aatype'], num_classes=NUM_TOKENS)
        denom = 0.5 * oh + 0.05
        new_probs = probs / denom
        new_probs /= new_probs.sum(-1, keepdims=True)
        if self.cfg.logits == 'standard':
            curr_tok_logits = Categorical(
                logits=new_probs.log() / self.cfg.temp
            ).log_prob(batch['aatype'])
        elif self.cfg.logits == 'gumbel':
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            curr_tok_logits = Categorical(
                logits = (new_probs.log() + gumbel_noise) / self.cfg.temp
            ).log_prob(batch['aatype'])
        
        # is_mask_prob = ((probs - oh) / (new_probs - oh))[...,0]

        if self.cfg.adjust_unmasked:
            logits = torch.where(is_unmask[...,None], new_probs.log(), logits)
        
        if self.cfg.logits == 'standard':
            cat = Categorical(logits=logits / self.cfg.temp)
            sample_ = cat.sample()
            scores_ = cat.log_prob(sample_)
            
        elif self.cfg.logits == 'gumbel':
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = logits + gumbel_noise 
            scores_, sample_ = (logits / self.cfg.temp).log_softmax(dim=-1).max(dim=-1) 

        if self.cfg.strategy == 'one_stage':            
            remask_prob = self.cfg.remask * max(0, 0.1 - 0.2*s)
            num_remask = int(torch.distributions.Binomial(
                is_unmask.sum().item(), remask_prob
            ).sample().item())
           
            topk = topk_masking(
                -curr_tok_logits,
                num_remask, 
                is_unmask,
                temp=self.cfg.topk_temp
            )
            
            batch['aatype'] = torch.where(topk, MASK_IDX, batch['aatype'])
            
            topk = topk_masking(
                scores_, 
                num_mask - new_num_mask,
                is_mask,
                temp=self.cfg.topk_temp
            )
            batch['aatype'] = torch.where(topk, sample_, batch['aatype'])

        elif self.cfg.strategy == 'dplm':

            if self.cfg.replace_unmasked:
                scores_ = torch.where(is_unmask, curr_tok_logits, scores_)
                sample_ = torch.where(is_unmask, batch['aatype'], sample_)
            
            topk = topk_masking(
                scores_,
                L - new_num_mask, 
                temp=self.cfg.topk_temp
            )
            
            if self.cfg.allow_mutation:
                sample = sample_
            else:
                sample = torch.where(is_mask, sample_, batch['aatype'])
                
            batch['aatype'] = torch.where(topk, sample, MASK_IDX)

        elif self.cfg.strategy == 'random':

            topk = topk_masking(
                torch.rand_like(scores_),
                num_mask - new_num_mask,
                is_mask
            )
            batch['aatype'] = torch.where(topk, sample_, batch['aatype'])

        elif self.cfg.strategy == 'purity':
            
            logits_1_wo_mask = logits[:, :, 0:-1] # (B, D, S-1)
            pt_x1_probs = torch.softmax(logits_1_wo_mask / self.cfg.temp, dim=-1) # (B, D, S-1)
            # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
            max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
            # bias so that only currently masked positions get chosen to be unmasked

            # max_logprob = scores_ 
            max_logprob = max_logprob - is_unmask.float() * 1e9
            sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)


            # d_t: t - s
            unmask_probs = min(1.0, (t-s) * ( (1 + self.cfg.noise * (1-t)) / (1-(1-t))))

            
            number_to_unmask = torch.distributions.binomial.Binomial(total_count=is_mask.long().sum(-1), probs=torch.tensor(unmask_probs).to(is_mask.device)).sample()
            unmasked_samples = sample_
        
            # Vectorized version of:
            # for b in range(B):
            #     for d in range(D):
            #         if d < number_to_unmask[b]:
            #             aatypes_t[b, sorted_max_logprobs_idcs[b, d]] = unmasked_samples[b, sorted_max_logprobs_idcs[b, d]]

            B, L, _ = logits.shape
            D_grid = torch.arange(L, device=logits.device).view(1, -1).repeat(B, 1)
            mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
            inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, L)
            masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
            mask2 = torch.zeros((B, L), device=logits.device)
            mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((B, L), device=logits.device))
            unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, L).float()
            mask2 = mask2 * (1 - unmask_zero_row)
            batch['aatype'] = batch['aatype'] * (1 - mask2) + unmasked_samples * mask2
    
            # re-mask
            if s > 0:
                u = torch.rand(B, L, device=logits.device)
                re_mask_mask = (u < (t-s) * self.cfg.noise).float()
                batch['aatype'] = batch['aatype'] * (1 - re_mask_mask) + MASK_IDX * re_mask_mask

        batch['aatype'] = batch['aatype'].long()

        seq = "".join([rc.restypes_with_x[aa] for aa in batch["aatype"][0]])
        seq = seq.replace("X", "-")
        # print(seq)
        return batch
        
