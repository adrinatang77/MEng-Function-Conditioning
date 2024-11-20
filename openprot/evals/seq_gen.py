from .eval import OpenProtEval
from ..utils import protein
from ..utils.geometry import compute_lddt
from ..utils import residue_constants as rc
import numpy as np
from ..tasks import SequenceDenoising
import torch
import os
import torch.nn.functional as F


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

    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):
        os.makedirs(savedir, exist_ok=True)

        noisy_batch = batch.copy("name", "pad_mask")
        for track in model.tracks.values():
            track.corrupt(batch, noisy_batch, {})

        L = len(batch["seqres"])

        for i in range(L):

            _, out = model.forward(noisy_batch)
            if self.cfg.unmask_order == "random":
                i = np.random.choice(
                    torch.argwhere(noisy_batch["seq_noise"][0])[:, 0].cpu()
                )
            elif self.cfg.unmask_order == "purity":
                i = torch.argmax(
                    torch.where(
                        noisy_batch["seq_noise"] > 0, out["aatype"].max(-1)[0], -np.inf
                    )
                ).item()

            noisy_batch["aatype"][:, i] = torch.distributions.categorical.Categorical(
                logits=out["aatype"][:, i] / self.cfg.temp
            ).sample()
            noisy_batch["seq_noise"][:, i] = 0.0

        filename = "seqs.fasta"
        filepath = os.path.join(savedir, filename)
        seq = "".join([rc.restypes_with_x[aa] for aa in noisy_batch["aatype"][0]])
        with open(filepath, "a") as f:
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
