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

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking
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

    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):
        torch.cuda.empty_cache()
        
        ## distributed designability
        idx = list(range(rank, self.cfg.num_samples, world_size))
        
        for i in idx:
            seq = list(open(f"{savedir}/sample{i}.fasta"))[1].strip()
            cmd = [
                "bash",
                "scripts/switch_conda_env.sh",
                "python",
                "-m",
                "scripts.esmfold",
                seq,
                f"{savedir}/sample{i}.pdb",
            ]
            out = subprocess.run(
                cmd,
                capture_output = True, # Python >= 3.7 only
                text = True
            )  # env=os.environ | {"CUDA_VISIBLE_DEVICES")
            
            if logger is not None:
                logger.log(f"{self.cfg.name}/plddt", float(out.stdout.split()[-1]))


    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):
        os.makedirs(savedir, exist_ok=True)

        noisy_batch = batch.copy("name", "pad_mask")
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        L = len(batch["seqres"][0])
        
        
        mask = noisy_batch['seq_noise'].bool()
        toks = noisy_batch['aatype'].clone()
        scores = torch.zeros_like(noisy_batch['seq_noise'])
        
        sched = np.linspace(0.99, 0, self.cfg.steps+1)
        # for i in range(self.cfg.steps): 
        for t, s in zip(sched[:-1], sched[1:]): # t > s
            dt = t - s
            
            _, out = model.forward(noisy_batch)
            logits = out['aatype'] 

            # mask, toks, scores = self.dplm_sample(logits, mask, toks, scores, step=i, max_iter=self.cfg.steps)             
            # noisy_batch['aatype'] = toks
            # noisy_batch['seq_noise'] = mask.float()

            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = logits + gumbel_noise
            scores, sample = logits.log_softmax(dim=-1).max(dim=-1) # softmax AFTER gumbel!
            
            
            # # scores[~is_mask.bool()] -= np.inf
            # # unmask_prob = (1/t * dt) + self.cfg.sigma * np.sqrt((1-t) / t) * dt
            # unmask_prob = 1 - t
            # # unmask = (torch.rand_like(is_mask) < unmask_prob) & is_mask.bool()
            # unmask = torch.rand_like(is_mask) < unmask_prob
            # k = unmask.sum()
            # # unmask = unmask & False
            # # unmask[:,torch.topk(scores, k, dim=-1).indices] = True

            rate = s
            # rate = 1 - (step+1) / max_iter
            cutoff_len = (rate * torch.ones_like(sample).sum(-1, keepdims=True)).long()
            lowest_k_mask = topk_masking(scores, cutoff_len, stochastic=False) 
            # lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=1.5 * rate)
        
            noisy_batch['aatype'] = torch.where(
                ~lowest_k_mask, 
                torch.where(mask, sample, noisy_batch['aatype']),
                20
            )
            mask = lowest_k_mask

                                     
            # # # # at time t p(MASK | x0) = t, p(x0 | x0) = 1 - t
            # # # # R(x0 -> MASK) = 1 / (1 - t)
            # # # # going backwards R(MASK -> x0) = 1 / t

            # # remask_prob = self.cfg.sigma * np.sqrt(s / (1 - s)) * dt
            # # remask = (torch.rand_like(is_mask) < remask_prob) & ~is_mask.bool()

            # # # # print(unmask_prob, remask_prob)

            
            # aatype = noisy_batch['aatype']
            # # # aatype = torch.where(unmask, sample, aatype)
            # aatype = torch.where(unmask, sample, 20)
            # # # aatype = torch.where(remask, 20, aatype)
            # noisy_batch['aatype'] = aatype
            
            # noise = noisy_batch["seq_noise"] 
            # noise = torch.where(unmask, 0.0, noise)
            # noise = torch.where(remask, 1.0, noise)            
            # noisy_batch['seq_noise'] = noise

            
            # Langevin mixing R(MASK -> x0) = \sqrt{1-t / t}
            # Langevin mixing R(x0 -> mask) = \sqrt{t / (1-t)}
            
            # if self.cfg.gumbel:
            #     rand = torch.rand_like(logits)
            #     gumbel = -torch.log(-torch.log(rand))
            #     logits += gumbel * self.cfg.temp
                
            # if self.cfg.unmask_order == "random":
            #     i = np.random.choice(
            #         torch.argwhere(noisy_batch["seq_noise"][0])[:, 0].cpu()
            #     )
            # elif self.cfg.unmask_order == "purity":    
            #     i = torch.argmax(logits.max(-1)[0]).item()

            # if self.cfg.gumbel:
            #     noisy_batch["aatype"][:, i] = logits[:,i].argmax(-1)
            # else:                                   
            #     noisy_batch["aatype"][:, i] = Categorical(logits=logits[:,i] / self.cfg.temp).sample()
            # noisy_batch["seq_noise"][:, i] = 0.0
            seq = "".join([rc.restypes_with_x[aa] for aa in noisy_batch["aatype"][0]])
            print(seq.replace('X', '-'))
            

        seq = "".join([rc.restypes_with_x[aa] for aa in noisy_batch["aatype"][0]])
        print(seq.replace('X', '-'))
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
