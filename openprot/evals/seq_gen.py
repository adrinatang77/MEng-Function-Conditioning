from .eval import OpenProtEval
from ..utils import protein
from ..utils.geometry import compute_lddt
from ..utils import residue_constants as rc
from ..tracks.sequence import MASK_IDX
import numpy as np
import torch
import os
import math
import tqdm
import torch.nn.functional as F
import subprocess
from torch.distributions.categorical import Categorical
from biopandas.pdb import PandasPdb

def topk_masking(scores, k, mask=None, temp=1.0):
    gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
    scores = scores + temp * gumbel
    if mask is not None:
        scores = torch.where(mask, scores, -np.inf) 
    new = torch.zeros_like(scores).bool()
    idx = torch.topk(scores, k, dim=-1).indices
    new.scatter_(-1, idx, True)
    return new
    
class SequenceGenerationEval(OpenProtEval):
    def setup(self):
        pass

    def run(self, model):
        NotImplemented

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        L = self.cfg.sample_length
        data = self.make_data(
            name=f"sample{idx}",
            seqres="A" * L,
            seq_mask=np.ones(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
        )
        return data

    def compute_sequence_entropy(self, seq):
        p = np.zeros(21)
        for s in seq:
            p[rc.restype_order_with_x[s]] += 1
        p /= p.sum()
        return np.e ** (-np.nansum(p * np.log(p)))

    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):
        torch.cuda.empty_cache()

        idx = list(range(rank, self.cfg.num_samples, world_size))
        os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)
        for i in idx:
            cmd = ['cp', f"{savedir}/sample{i}.fasta", f"{savedir}/rank{rank}"]
            subprocess.run(cmd)
            
        cmd = [
            "bash",
            "scripts/switch_conda_env.sh",
            "python",
            "-m",
            "scripts.esmfold",
            "--outdir",
            savedir,
            "--dir",
            f"{savedir}/rank{rank}",
            "--print",
        ]
        out = subprocess.run(cmd)  
        for i in idx:
            plddt = PandasPdb().read_pdb(f"{savedir}/sample{i}.pdb").df['ATOM']['b_factor'].mean()
            if logger is not None:
                logger.log(f"{self.cfg.name}/plddt", plddt)

    

    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):
        
        L = len(batch["seqres"][0])

        sched = np.linspace(1.0, 0, self.cfg.steps+1)
        is_mask_probs = []
        curr_tok_probs = []
        
        for t, s in tqdm.tqdm(zip(sched[:-1], sched[1:]), total=self.cfg.steps): # t > s
                
            num_mask = int(round(t * self.cfg.sample_length))
            
            num_unmask = int(self.cfg.sample_length - num_mask)
            remask = int(min(self.cfg.remask, int(0.5*num_mask), num_unmask))
            
            new_num_mask = int(round(s * self.cfg.sample_length))
            
            _, out = model.forward(noisy_batch)
            
            logits = out['aatype'] 
            
            # ## extract the pseudo-mask likelihoods
            # probs = logits.softmax(-1)
            # oh = torch.nn.functional.one_hot(noisy_batch['aatype'], num_classes=21)
            # denom = 0.5 * oh + 0.05
            # new_probs = probs / denom
            # new_probs /= new_probs.sum(-1, keepdims=True)

            # is_mask_prob = ((probs - oh) / (new_probs - oh))[...,0]
            curr_tok_prob = Categorical(logits=logits).log_prob(noisy_batch['aatype'])
            # is_mask_probs.append(is_mask_prob.cpu().numpy())
            # curr_tok_probs.append(curr_tok_prob.cpu().numpy())

            # # # rewrite the is mask prob
            is_unmask = (noisy_batch['aatype'] != MASK_IDX)
            is_mask = (noisy_batch['aatype'] == MASK_IDX)
            # # # target_prob = s**0.5
            # # # is_mask_prob = torch.ones_like(is_mask_prob)
            # is_mask_prob = torch.clamp(2 * is_mask_prob, max=1)
            # # # print(target_prob, agg_is_mask_prob)

            # # p(is_mask | Y_i, Y_\i) = p(Y_i | is_mask, Y_\i) * p(is_mask | Y_\i) / p(Y_i | Y_\i)
            # # # else:
            # agg_is_mask_prob = (is_mask_prob * is_unmask.float()).sum(-1) / is_unmask.sum(-1)
            # print(agg_is_mask_prob)
            # # #     target_prob = max(agg_is_mask_prob.item(), 0.3)
            # # #     is_mask_prob *= target_prob / agg_is_mask_prob
                

            # new_probs = is_mask_prob[...,None] * new_probs + (1-is_mask_prob[...,None]) * oh
            # probs = 0.5 * oh + 0.5 * probs
            # p = 0.25 * s + 0.0 # replace prob
            # new_probs = (p) * probs + (1-p) * oh
            # new_logits = new_probs.log()

            
            # logits = torch.where(
            #     is_unmask[...,None], 
            #     logits, #new_logits,  
            #     logits
            # ) / self.cfg.temp

            # (0.5 + 1.0*s) # 
            # logits = logits - 10 * oh.float().mean(1, keepdims=True) / s
            
            # sample = Categorical(logits=logits / self.cfg.temp).sample()
            # entropy = torch.nansum(logits.softmax(-1) * logits.log_softmax(-1), -1)
            
            # topk = topk_masking(entropy, num_mask - new_num_mask, noisy_batch['aatype'] == 20, temp=0.0)
            # noisy_batch['aatype'] = torch.where(topk, sample, noisy_batch['aatype'])
            
            # two stage: remask a fixed amount, predict, then unmask a fixed amount
            # one stage: predict, but unmask & remask a fixed amount
            # dplm_custom: predict, but unmask a top-k from original logits+new logits
            # dplm: predict, unmask top-k from new logits

            
            # _, out = model.forward(noisy_batch)
            if self.cfg.logits == 'standard':
                sample_ = Categorical(logits=logits).sample()
                scores_ = Categorical(logits=logits).log_prob(sample_)
            elif self.cfg.logits == 'gumbel':
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                logits = logits + gumbel_noise 
                scores_, sample_ = (logits / self.cfg.temp).log_softmax(dim=-1).max(dim=-1) # softmax AFTER gumbel!

            new_num_mask = round(s * self.cfg.sample_length)
            if self.cfg.strategy == 'one_stage':            

                # keep this, this works! 
                remask_prob = self.cfg.remask * max(0, 0.1 - 0.2*s) # roughly speaking this controls pLDDT for a given temp, topk_temp
                # num_remask = int(round(remask_prob * ))

                num_remask = int(torch.distributions.Binomial(is_unmask.sum().item(), remask_prob).sample().item())

               
                topk = topk_masking(-curr_tok_prob, num_remask, is_unmask, temp=self.cfg.topk_temp)
                
                noisy_batch['aatype'] = torch.where(topk, MASK_IDX, noisy_batch['aatype'])
                
                topk = topk_masking(scores_, num_mask - new_num_mask, is_mask, temp=self.cfg.topk_temp)
                noisy_batch['aatype'] = torch.where(topk, sample_, noisy_batch['aatype'])

            elif self.cfg.strategy == 'dplm':
                
                topk = topk_masking(
                    scores_,
                    self.cfg.sample_length - new_num_mask, 
                    temp=self.cfg.topk_temp
                )
                
                if self.cfg.allow_mutation:
                    sample = sample_
                else:
                    sample = torch.where(is_mask, sample_, noisy_batch['aatype'])
                    
                noisy_batch['aatype'] = torch.where(topk, sample, MASK_IDX)
     
            
            # seq = "".join([rc.restypes_with_x[aa] for aa in noisy_batch["aatype"][0]])
            # seq = seq.replace("X", "-")
            # print(seq)    
        
        
        for name, aatype in zip(noisy_batch['name'], noisy_batch['aatype']):
            seq = "".join(
                [rc.restypes_with_x[aa] for aa in aatype]
            ) # in case there is still mask
            if logger is not None:
                logger.log(f"{self.cfg.name}/seqent", self.compute_sequence_entropy(seq))
            
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")

        #breakpoint()

        """
        batch_size, seq_len = batch['aatype'].shape
        track = model.tracks["SequenceTrack"]

        dev = batch['aatype'].device
        track.steady_state = track.steady_state.to(dev)  
        xt = torch.multinomial(track.steady_state, batch_size * seq_len, replacement=True)
        xt = xt.view(batch_size, seq_len)
        noisy_batch = batch.copy("name", "pad_mask")
        noisy_batch['aatype'] = xt
        noisy_batch['seq_noise'] = torch.ones_like(xt)
        
        _, output = model.forward(noisy_batch)
        p_x0_g_xt = F.softmax(output['aatype'], dim=-1)
        predicted = torch.argmax(output['aatype'], dim=2)

        num_steps = self.cfg.n_steps
        timesteps = [t/num_steps for t in range(0, num_steps)]
        timesteps.reverse()
        step = 1/num_steps

        # get rate matrix from track
        flow = track.flow
        for t in timesteps:
            flows = flow.unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
            flows = flows.to(dev)

            # helper variables to isolate rows for state of xt
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, seq_len)
            n_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
            identity = torch.eye(21).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1).to(dev)

            # p(x_t-1 | x0)
            noise = torch.full([batch_size, seq_len, 1, 1], t, device=dev)
            p_xtm1_g_x0 = track.noise_transform(noise, identity)

            # p(x_t | x0)
            noise = torch.full([batch_size, seq_len, 1, 1], t + step, device=dev)
            p_xt_g_x0 = track.noise_transform(noise, identity)
            
            # isolate rows for value of xt and repeat to propogate over extra dimension
            p_xt_g_x0 = p_xt_g_x0[batch_indices, n_indices, xt]
            p_xt_g_x0 = p_xt_g_x0.unsqueeze(-1).transpose(-1, -2).repeat(1, 1, 21, 1)

            # p(x_t | x_t-1, x0)
            noise = torch.full([batch_size, seq_len, 1, 1], t + step, device=dev)
            p_xt_g_xtm1 = track.noise_transform(noise, identity, t)

            # isolate rows for value of xt and repeat to propogate over extra dimension
            p_xt_g_xtm1 = p_xt_g_xtm1[batch_indices, n_indices, xt]
            p_xt_g_xtm1_x0 = p_xt_g_xtm1.unsqueeze(-1).repeat(1, 1, 1, 21)

            p_xtm1_g_xt_x0 = p_xt_g_xtm1_x0 * p_xtm1_g_x0 / p_xt_g_x0
            p_xtm1_g_xt_x0[torch.isnan(p_xtm1_g_xt_x0)] = 0


            p_xtm1_g_xt = torch.matmul(p_xtm1_g_xt_x0, p_x0_g_xt.unsqueeze(-1))
            xt = torch.multinomial(p_xtm1_g_xt.squeeze(0).squeeze(-1),num_samples=1).view(batch_size, seq_len)
#             print(''.join(rc.restypes_with_x[aa] for aa in xt[0]))
            noisy_batch['aatype'] = xt
            noisy_batch['seq_noise'] = torch.full((batch_size, seq_len), t, device=dev)
            _, output = model.forward(noisy_batch)
            predicted = torch.argmax(output['aatype'], dim=2)
            p_x0_g_xt = F.softmax(output['aatype'], dim=-1)
        """
