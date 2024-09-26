import torch
import torch.nn as nn
import torch.nn.functional as F
from .gmha import GeometricMultiHeadAttention
from ..utils.rotation_conversions import axis_angle_to_matrix
from .trunk import RelativePosition
from ..utils.rigid_utils import Rigid

from .layers import (
    Dropout,
    PairToSequence,
    SequenceToPair,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from openfold.utils.checkpointing import checkpoint_blocks

class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.lin_in = nn.Linear(dim, ff_dim)
        self.act = nn.ReLU()
        self.lin_out = nn.Linear(ff_dim, dim)

    def forward(self, x):
        x = self.lin_in(x)
        x = self.act(x)
        x = self.lin_out(x)
        return x


class OpenProtTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_expand, 
        pairwise_dim=128,
        pairwise_heads=4,
        dropout=0,
        geometric_attn=False,
        frame_update=False
    ):
        super().__init__()
        self.mha = GeometricMultiHeadAttention(dim, heads, geometric=geometric_attn)
        self.mha_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_expand * dim)
        self.ff_norm = nn.LayerNorm(dim)

        ##########
        
        # self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        self.sequence_to_pair = SequenceToPair(dim, pairwise_dim // 2, pairwise_dim)
        self.pair_to_sequence = PairToSequence(pairwise_dim, heads)

        # self.seq_attention = Attention(
        #     sequence_state_dim, sequence_num_heads, sequence_head_width, gated=True
        # )
        self.tri_mul_out = TriangleMultiplicationOutgoing(pairwise_dim, pairwise_dim)
        self.tri_mul_in = TriangleMultiplicationIncoming(pairwise_dim, pairwise_dim)
            
        # self.mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=dropout)
        self.mlp_pair = FeedForward(pairwise_dim, ff_expand * pairwise_dim)

        self.drop = nn.Dropout(dropout)
        self.row_drop = Dropout(dropout * 2, 2)
        self.col_drop = Dropout(dropout * 2, 1)

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        # torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        # torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        # torch.nn.init.zeros_(self.mlp_seq.mlp[-2].weight)
        # torch.nn.init.zeros_(self.mlp_seq.mlp[-2].bias)
        # torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        # torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)
        ##########

        
        if frame_update:
            self.ff_update = nn.Linear(dim, 6)
        else:
            self.ff_update = None

    def forward(self, x, z, mask, rots, trans):
        
        bias = self.pair_to_sequence(z)
        x = x + self.mha(
            self.mha_norm(x), 
            mask.bool(),
            rots=rots, 
            trans=trans,
            bias=bias.permute(0,3,1,2)
        )
        x = x + self.ff(self.ff_norm(x))

        z = z + self.sequence_to_pair(x)
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        z = z + self.row_drop(self.tri_mul_out(z, mask=tri_mask))
        z = z + self.col_drop(self.tri_mul_in(z, mask=tri_mask))
        z = z + self.mlp_pair(z)
        
        if self.ff_update is not None:
            vec, rotvec = self.ff_update(x).split(3, dim=-1)
            trans = trans + torch.einsum("blij,blj->bli", rots, vec)
            rots = rots @ axis_angle_to_matrix(rotvec).mT
        # exit()
        return x, z, mask, rots, trans


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # if cfg.trunk:
        #     self.trunk = FoldingTrunk(cfg.trunk)
        #     return

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim
        self.pairwise_positional_embedding = RelativePosition(cfg.trunk.position_bins, c_z)
        
        if cfg.in_norm:
            self.in_norm = nn.LayerNorm(cfg.dim)

        self.blocks = nn.ModuleList()
        for i in range(cfg.blocks):
            self.blocks.append(
                OpenProtTransformerBlock(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    ff_expand=cfg.ff_expand,
                    geometric_attn=i in cfg.geometric_attn,
                    frame_update=i in cfg.frame_update,
                )
            )

    def forward(self, inp):

        x = inp['x']
        B, L, _ = x.shape
        # true_aa = torch.zeros(B, L, dtype=torch.long, device=seq_feats.device)
        residx = torch.arange(L, device=x.device)[None].expand(B, -1)
        mask = inp["pad_mask"]
        z = self.pairwise_positional_embedding(residx, mask=mask)
        
        rots = inp.get("rots", None)
        trans = inp.get("trans", None)
        if self.cfg.in_norm:
            x = self.in_norm(x)

        
        #for block in self.blocks:
        x, z, mask, rots, trans = checkpoint_blocks(
            self.blocks,
            args=(x, z, mask, rots, trans),
            blocks_per_ckpt=1,
        )
             # = torch.utils.checkpoint.checkpoint(
             #    block, x, z, mask, rots, trans)
        return {
            "x": x,
            "z": z,
            "rots": rots,
            "trans": trans,
        }
