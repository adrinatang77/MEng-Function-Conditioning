from ..tracks.sequence import MASK_IDX, NUM_TOKENS
from torch.distributions.categorical import Categorical
import torch
import numpy as np

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
                  
        return batch
        # seq = "".join([rc.restypes_with_x[aa] for aa in noisy_batch["aatype"][0]])
        # seq = seq.replace("X", "-")
        # print(seq)    
    
