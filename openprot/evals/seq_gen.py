from .eval import OpenProtEval
from ..utils import protein
from ..utils.geometry import compute_lddt
from ..utils import residue_constants as rc
import numpy as np
from ..tasks import SequenceDenoising
import torch
import os
import math
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
    new[...,torch.topk(scores, k, dim=-1).indices] = True
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
        ## WORKS FOR SINGLE RANK ONLY AT THE MOMENT
        ## distributed designability
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


    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):
        os.makedirs(savedir, exist_ok=True)

        noisy_batch = batch.copy("name", "pad_mask")
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        L = len(batch["seqres"][0])

        
        
        mask = noisy_batch['seq_noise'].bool()
        scores = torch.ones_like(noisy_batch['seq_noise']) * -np.inf
        sched = np.linspace(1.0, 0, self.cfg.steps+1)
        
        for t, s in zip(sched[:-1], sched[1:]): # t > s
                
            num_mask = round(t * self.cfg.sample_length)
            
            # num_unmask = self.cfg.sample_length - num_mask
            # # remask = min(self.cfg.remask, int(0.5*num_mask), num_unmask)
            # gumbel_factor = 1.0
            
            new_num_mask = round(s * self.cfg.sample_length)
            

            _, out = model.forward(noisy_batch)
            logits = out['aatype']
            sample = Categorical(logits=logits / self.cfg.temp).sample()
            entropy = torch.nansum(logits.softmax(-1) * logits.log_softmax(-1), -1)
            
            topk = topk_masking(entropy, num_mask - new_num_mask, noisy_batch['aatype'] == 20, temp=0.0)
            noisy_batch['aatype'] = torch.where(topk, sample, noisy_batch['aatype'])
            """
            
            if self.cfg.strategy == 'two_stage':
                topk = topk_masking(topk_factor, num_unmask - remask, noisy_batch['aatype'] != 20, temp=self.cfg.gumbel * gumbel_factor) 
                noisy_batch['aatype'] = torch.where(topk, noisy_batch['aatype'], 20)
                scores = torch.where(topk, scores, -np.inf)
            
            _, out = model.forward(noisy_batch)
            logits = out['aatype'] / temp
            if self.cfg.logits == 'standard':
                sample_ = Categorical(logits=logits).sample()
                scores_ = Categorical(logits=logits).log_prob(sample_)
            elif self.cfg.logits == 'gumbel':
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                logits = logits + gumbel_noise
                scores_, sample_ = logits.log_softmax(dim=-1).max(dim=-1) # softmax AFTER gumbel!

            new_num_mask = round(s * self.cfg.sample_length)
            if self.cfg.strategy == 'one_stage':            
                topk = (  
                    topk_masking(scores, num_unmask - remask, ~mask, temp=self.cfg.gumbel * gumbel_factor) | 
                    topk_masking(scores_, remask+(num_mask - new_num_mask), mask, temp=self.cfg.gumbel * gumbel_factor) 
                )
            elif self.cfg.strategy == 'two_stage':
                topk = topk_masking(
                    scores_,
                    remask+(num_mask - new_num_mask), 
                    noisy_batch['aatype'] == 20,
                    temp=self.cfg.gumbel * gumbel_factor
                )
            elif self.cfg.strategy == 'dplm':
                topk = topk_masking(scores_, self.cfg.sample_length - new_num_mask, temp=0.0)
            elif self.cfg.strategy == 'dplm_custom':
                gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
                topk = topk_masking(
                    torch.where(
                        mask,
                        scores_ + self.cfg.gumbel*gumbel * gumbel_factor, 
                        scores+self.cfg.gumbel*gumbel * gumbel_factor
                    ), 
                    self.cfg.sample_length - new_num_mask, temp=0.0
                )
                
            
            if self.cfg.strategy == 'two_stage':
                noisy_batch['aatype'] = torch.where(topk, sample_, noisy_batch['aatype'])
                scores = torch.where(topk, scores_, scores)
            else:
                noisy_batch['aatype'] = torch.where(
                    topk, torch.where(mask, sample_, noisy_batch['aatype']), 20
                )
                
                scores = torch.where(
                    topk, torch.where(mask, scores_, scores), -np.inf
                )
                mask = ~topk

            #### gumbel logits
            # dplm 71.95978381529582
            # +gumbel 51.759465084490145
            # -0.75 44.87779806512229 (61)
            # -0.0 47.44080854354403 (40)
            # -inf 39.08901199273137 (45)
            # -0.5 + gumbel 46.5057778487319 (44)

            #### normal logits
            #+4*gumbel, +4*gumbel 62.950148782852956 (30)
            #+2*gumbel, +10*gumbel 64.92839980669494 (27)
            
            # for a, b, c, d in zip(
            #     scores[~mask].flatten(), 
            #     noisy_batch['aatype'][~mask].flatten(),
            #     scores_[~mask].flatten(),
            #     sample_[~mask].flatten()
            # ):
            #     f.write(f"{t},{a.item()},{b.item()},{c.item()},{d.item()}\n")
            
            # print(f"currently {mask.sum()} masked, unmasking {(mask & topk).sum()}, remasking {(~mask & ~topk).sum()}")
            
                                     
            # # # # at time t p(MASK | x0) = t, p(x0 | x0) = 1 - t
            # # # # R(x0 -> MASK) = 1 / (1 - t)
            # # # # going backwards R(MASK -> x0) = 1 / t
            
            # Langevin mixing R(MASK -> x0) = \sqrt{1-t / t}
            # Langevin mixing R(x0 -> mask) = \sqrt{t / (1-t)}
            """
         
            # seq = "".join([rc.restypes_with_x[aa] for aa in noisy_batch["aatype"][0]])
            # print(seq.replace('X', '-'))
            
       
        seq = "".join([rc.restypes_with_x[aa] for aa in noisy_batch["aatype"][0]])
        print(seq)
        if logger is not None:
            logger.log(f"{self.cfg.name}/seqent", self.compute_sequence_entropy(seq))
        
        with open(f"{savedir}/{batch['name'][0]}.fasta", "w") as f:
            f.write(f">{batch['name'][0]}\n")  # FASTA format header
            f.write(seq + "\n")

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
