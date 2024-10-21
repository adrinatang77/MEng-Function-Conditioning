from .eval import OpenProtEval
from ..utils import protein
from ..utils.geometry import compute_lddt
from ..utils import residue_constants as rc
import numpy as np
from ..tasks import SequenceDenoising
import torch
import os
import torch.nn.functional as F


class SequenceGeneration(OpenProtEval):
    def setup(self):
        self.db = open(self.cfg.path)
        self.index = np.load(self.cfg.index)

    def run(self, model):
        NotImplemented

    def __len__(self):
        return len(self.index) - 1

    def __getitem__(self, idx):
        start = self.index[idx]
        end = self.index[idx + 1]
        self.db.seek(start)
        item = self.db.read(end - start)
        lines = item.split("\n")
        header, lines = lines[0], lines[1:]
        seqres = "".join(lines)
        seq_mask = np.ones(len(seqres))
        seq_mask[seqres == 'X'] = 0

        data = self.make_data(
            name=header.split()[0], seqres=seqres, seq_mask=seq_mask
        )

        return SequenceDenoising.prep_data(None, data)

    def run_batch(self, model, batch: dict, savedir=".", device=None, logger=None):
        os.makedirs(savedir, exist_ok=True)
        torch.set_printoptions(edgeitems=5)

        # noisy_batch = batch.copy("name", "pad_mask")
        batch_size, seq_len = batch['aatype'].shape
        track = model.tracks["SequenceTrack"]        
        xt = torch.multinomial(track.steady_state, seq_len * batch_size, replacement=True)
        xt = xt.view(batch_size, seq_len)
        noisy_batch = batch.copy()
        noisy_batch['aatype'] = xt

        _, output = model.forward(noisy_batch)
        p_x0_g_xt = F.softmax(output['aatype'], dim=-1)
        # print('output', output['aatype'])

        num_steps = 10
        timesteps = [t/num_steps for t in range(0, num_steps)]
        timesteps.reverse()
        step = 1/num_steps

        flow = track.flow
        for t in timesteps:
            flows = flow.unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
            
            converted_noise = torch.tan(torch.tensor([step]*seq_len) * (3.14/2)).view(batch_size, seq_len, 1, 1)
            p_xt_g_xm1 = torch.linalg.matrix_exp(converted_noise * flows)
            
            converted_noise = torch.tan(torch.tensor([t]*seq_len) * (3.14/2)).view(batch_size, seq_len, 1, 1)
            p_xtm1_g_x0 = torch.linalg.matrix_exp(converted_noise * flows)
            # print(p_xtm1_g_x0)
            # print(p_xtm1_g_x0.sum(dim = -2))

            converted_noise = torch.tan(torch.tensor([t + step]*seq_len) * (3.14/2)).view(batch_size, seq_len, 1, 1)
            # print(t+step)
            p_xt_g_x0 = torch.linalg.matrix_exp(converted_noise * flows)
            # print('p_xt_g_x0', p_xt_g_x0)

            p_xtm1_g_xt_x0 = torch.matmul(p_xt_g_xm1, p_xtm1_g_x0)/ p_xt_g_x0
            p_xtm1_g_xt_x0[torch.isnan(p_xtm1_g_xt_x0)] = 0
            # print('p_xt_g_xm1', p_xt_g_xm1)
            # print('p_xtm1_g_x0', p_xtm1_g_x0)
            # print('p_xtm1_g_xt_x0', p_xtm1_g_xt_x0)
            print('p_xtm1_g_xt_x0', p_xtm1_g_xt_x0.sum(dim=-2))

            p_xtm1_g_xt = p_xtm1_g_xt_x0.sum(dim = -1) * p_x0_g_xt
            print('p_xtm1_g_xt', p_xtm1_g_xt.sum(dim = -2))
            # print('p_x0_g_xt', p_x0_g_xt)
            # print('p_xtm1_g_xt', p_xtm1_g_xt)

            xt = torch.multinomial(p_xtm1_g_xt.squeeze(0), num_samples=1).view(batch_size, seq_len)
            print(xt.shape)
            noisy_batch['aatype'] = xt
            _, output = model.forward(noisy_batch)
            p_x0_g_xt = F.softmax(output['aatype'], dim=-1)

        filename = 'generated_seq'
        filepath = os.path.join(savedir, filename)
        sequences = [''.join(rc.restypes[aa] for aa in seq) for seq in xt]
        with open(filepath, "a") as f:
            for i, seq in enumerate(sequences):
                f.write(f">Sequence_{i+1}\n")  # FASTA format header
                f.write(seq + "\n")
                print(f"Sequence saved to {filepath}")
        
        # save xt to a save_fil in directory
        # fold xt with alphafold and record plddt score


