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

def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores

def stochastic_sample_from_categorical(logits=None, temperature=1.0, noise_scale=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores
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

    def resample_conditional(self, _tokens, _scores, ratio, scale):
        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * ratio:#max(0.3/(step+1) ** 0.2, 0.1):
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio:#max(0.3/(step+1) ** 0.2, 0.1):
                        mask |= seq.eq(k)
                resample_input_mask.append(mask)
                resample_input.append(seq.masked_fill(mask, self.mask_id))
                #resample_input.append(seq.masked_scatter(mask, xt[i][mask]))

        
        if len(to_be_resample_idx) > 0:
            resample_input = torch.stack(resample_input, dim=0).type_as(_tokens)
            resample_input_scores = torch.stack(resample_input_scores, dim=0).type_as(_scores)
            resample_input_mask = torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            resample_logits = self.net(
                input_ids=resample_input,
            )['logits']
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)
            resample_logits[..., self.mask_id] = -math.inf
            resample_logits[..., self.x_id] = -math.inf
            resample_logits[..., self.pad_id] = -math.inf
            resample_logits[..., self.bos_id] = -math.inf
            resample_logits[..., self.eos_id] = -math.inf
            
            resample_logits = top_k_top_p_filtering(resample_logits, top_p=0.95)
            #noise_scale = 1.5 - 0.2 * ((step + 1) / max_step)
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            resample_tokens, resample_scores = stochastic_sample_from_categorical(resample_logits, temperature=0.0, noise_scale=noise_scale)
            resample_input.masked_scatter_(resample_input_mask, resample_tokens[resample_input_mask])
            resample_input_scores.masked_scatter_(resample_input_mask, resample_scores[resample_input_mask])
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = resample_input, resample_input_scores

    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
    ):
        """
            This function is used to perform reparameterized decoding.
        """
        # output_tokens: [B, N]
        # output_scores: [B, N]
        # cur_tokens: [B, N]
        # cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)

        # first set the denoising rate according to the schedule
        rate = 1 - t / max_step
        
        # compute the cutoff length for denoising top-k positions
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()
        # number of positions that should be kept as mask = 127
        
        # set the scores of special symbols to a large value so that they will never be selected
        # cut scores = scores of selected tokens
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
        to_be_resample = []
        for i, seq in enumerate(cur_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * 0.25:
                to_be_resample.append(i) # index of sequence needing resampling
                
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False) # 127 indices are True, i.e., they will be kept as mask
        if len(to_be_resample) > 0:
            noise_scale = 1.5
            #print(lowest_k_mask[to_be_resample[0]])
            lowest_k_mask[to_be_resample] = topk_masking(_scores_for_topk[to_be_resample], cutoff_len[to_be_resample], 
                                                         stochastic=True, temp=noise_scale * rate)
            # if need resampling, set the topk stochastically
        not_v1_t = lowest_k_mask # 127 indices are True, i.e., they will be kept as mask
         # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        # not v1_t, not_v2_t = not unmask
        last_mask_position = xt_neq_x0 # True if mask
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)): # noise = mask_id
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t # is mask & unmask
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        assert (new_xt_neq_x0 == not_v2_t).all()
        return new_xt_neq_x0, output_tokens, output_scores
         
    def dplm_sample(self, logits, mask, toks, scores, step, max_iter):

        noise_scale = 1.0
        _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=noise_scale)
        self.resample_conditional(_tokens, _scores, ratio=0.25, scale=1.0)

        non_special_sym_mask = torch.ones_like(mask).bool()

        output_masks, output_tokens, output_scores = self._reparam_decoding(
            output_tokens=toks.clone(),
            output_scores=scores.clone(),
            cur_tokens=_tokens.clone(),
            cur_scores=_scores.clone(),
            xt_neq_x0=mask,
            non_special_sym_mask=non_special_sym_mask,
            t=step + 1,
            max_step=max_iter,
            noise=20,
        )
        return output_masks, output_tokens, output_scores
            

    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):
        os.makedirs(savedir, exist_ok=True)

        noisy_batch = batch.copy("name", "pad_mask")
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        L = len(batch["seqres"][0])
        
        
        mask = noisy_batch['aatype'].bool()
        toks = noisy_batch['aatype'].clone()
        scores = torch.zeros_like(noisy_batch['seq_noise'])
        
        sched = np.linspace(0.99, 0, self.cfg.steps+1)
        for i in range(self.cfg.steps): #t, s in zip(sched[:-1], sched[1:]): # t > s
            # dt = t - s
            
            _, out = model.forward(noisy_batch)
            logits = out['aatype'] 

            mask, toks, scores = self.dplm_sample(logits, mask, toks, scores, step=i, max_iter=self.cfg.steps)    
            
            noisy_batch['aatype'] = toks
            noisy_batch['seq_noise'] = mask.float()
            # logits = torch.log_softmax(logits / self.cfg.temp, dim=-1)
            # gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
            
            # scores, sample = torch.max(logits + gumbel, -1)
            # sample = Categorical(logits = logits / self.cfg.temp).sample()
            # scores = Categorical(logits = logits / self.cfg.temp).log_prob(sample)
            # is_mask = noisy_batch['seq_noise']
            
            # # scores[~is_mask.bool()] -= np.inf
            # # # breakpoint()

            # unmask_prob = (1/t * dt) + self.cfg.sigma * np.sqrt((1-t) / t) * dt
            # # print(unmask_prob)
            # unmask = (torch.rand_like(is_mask) < unmask_prob) & is_mask.bool()
            # # k = unmask.sum()
            # # unmask = unmask & False
            # # unmask[:,torch.topk(scores, k, dim=-1).indices] = True
            # # at time t p(MASK | x0) = t, p(x0 | x0) = 1 - t
            # # R(x0 -> MASK) = 1 / (1 - t)
            # # going backwards R(MASK -> x0) = 1 / t

            # remask_prob = self.cfg.sigma * np.sqrt(s / (1 - s)) * dt
            # remask = (torch.rand_like(is_mask) < remask_prob) & ~is_mask.bool()

            # # print(unmask_prob, remask_prob)

            
            # aatype = noisy_batch['aatype']
            # aatype = torch.where(unmask, sample, aatype)
            # aatype = torch.where(remask, 20, aatype)
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
            # print(seq.replace('X', '-'))
            

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
