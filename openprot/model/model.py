import torch
import torch.nn as nn
import torch.nn.functional as F
from .gmha import GeometricMultiHeadAttention
from ..utils.rotation_conversions import axis_angle_to_matrix
from ..utils.rigid_utils import Rigid, Rotation
from .layers import (
    Dropout,
    PairToSequence,
    SequenceToPair,
)
from .tri_mul import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from ..utils.checkpointing import checkpoint_blocks



class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin_in = nn.Linear(dim, ff_dim)
        self.act = nn.ReLU()
        self.lin_out = nn.Linear(ff_dim, dim)

    def forward(self, x):
        x = self.norm(x)
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
        rope=False,        # rotate scalar queries and keys
        pair_bias=False,   # use pairs to bias
        pair_updates=False,
        tri_mul=False,
        pair_values=False, # aggregate values from pair reps
        ipa=False,         # use point attention 
        ipa_frames=False,  # use frames in point attention
        relpos=False,      # instead use trans relpos
        embed_rots=False,
        no_qk_points=4,
        no_v_points=8,
        frame_update=False,
        update_rots=False,
    ):
        super().__init__()
        self.mha = GeometricMultiHeadAttention(
            dim=dim,
            heads=heads,
            pairwise_dim=pairwise_dim,
            rope=rope,
            pair_bias=pair_bias,
            pair_values=pair_values,
            ipa=ipa,
            ipa_frames=ipa_frames,
            relpos=relpos,
            embed_rots=embed_rots,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
        )
        if pair_bias or pair_values:
            self.pair_norm = nn.LayerNorm(pairwise_dim)
        if pair_updates:
            self.pair_ff_norm = nn.LayerNorm(pairwise_dim)
            
        self.pair_updates = pair_updates
        self.frame_update = frame_update
        self.update_rots = update_rots
        self.tri_mul = tri_mul

        self.mha_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_expand * dim)
        self.ff_norm = nn.LayerNorm(dim)

        if frame_update:
            self.linear_frame_update = nn.Sequential(
                nn.LayerNorm(dim),
                # nn.Linear(dim, dim),
                # nn.ReLU(),
                nn.Linear(dim, 6 if update_rots else 3),
            )
            torch.nn.init.zeros_(self.linear_frame_update[-1].weight)
            torch.nn.init.zeros_(self.linear_frame_update[-1].bias)

        if pair_updates:
            self.sequence_to_pair = SequenceToPair(dim, pairwise_dim // 2, pairwise_dim)

            if tri_mul:
                self.tri_mul_out = TriangleMultiplicationOutgoing(
                    pairwise_dim, pairwise_dim
                )
                self.tri_mul_in = TriangleMultiplicationIncoming(
                    pairwise_dim, pairwise_dim
                )
                dropout=0
                self.row_drop = Dropout(dropout * 2, 2)
                self.col_drop = Dropout(dropout * 2, 1)
                torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
                torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
                torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
                torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)

            
            self.mlp_pair = FeedForward(pairwise_dim, ff_expand * pairwise_dim)
    
            
            torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
            torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)

    def forward(self, x, z, mask, rots, trans):

        ### no pair2sequence
        
        x = x + self.mha(
            x=self.mha_norm(x),
            z=self.pair_norm(z) if hasattr(self, 'pair_norm') else None,
            mask=mask.bool(),
            trans=trans,
            rots=rots,
        )

        x = x + self.ff(self.ff_norm(x))

        if self.pair_updates:
            z = z + self.sequence_to_pair(x)
            if self.tri_mul:
                tri_mask = (
                    mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
                )
                z = z + self.row_drop(self.tri_mul_out(z, mask=tri_mask))
                z = z + self.col_drop(self.tri_mul_in(z, mask=tri_mask))
            z = z + self.mlp_pair(self.pair_ff_norm(z)) 

        if self.frame_update:
            
            if self.update_rots:
                vec, rotvec = self.linear_frame_update(x).split(3, dim=-1)
                trans = trans + torch.einsum("blij,blj->bli", rots, vec)
                rots = rots @ axis_angle_to_matrix(rotvec).mT
            else:
                vec = self.linear_frame_update(x)
                trans = trans + vec
                
        return x, z, mask, rots, trans


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.pairwise_positional_embedding = RelativePosition(
            cfg.position_bins, cfg.pairwise_dim
        )

        self.blocks = nn.ModuleList()
        for i in range(cfg.blocks):
            is_pair = i % cfg.pair_block_interval == 0
            is_ipa = False # i % cfg.ipa_block_interval == 0
            self.blocks.append(
                OpenProtTransformerBlock(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    ff_expand=cfg.ff_expand,
                    pairwise_dim=cfg.pairwise_dim,
                    pairwise_heads=cfg.pairwise_heads,
                    rope=cfg.rope,
                    ipa=is_ipa and cfg.ipa,
                    ipa_frames=is_ipa and cfg.ipa_frames,
                    relpos=is_ipa and cfg.relpos,
                    pair_bias=is_pair,
                    pair_updates=is_pair,
                    tri_mul=is_pair and cfg.tri_mul,
                )
            )

        
        block_fn = lambda: OpenProtTransformerBlock(
            dim=cfg.dim,
            heads=cfg.heads,
            ff_expand=cfg.ff_expand,
            pairwise_dim=cfg.pairwise_dim,
            pair_bias=True,
            pair_values=True,
            ipa=cfg.ipa,
            ipa_frames=cfg.ipa_frames,
            relpos=cfg.relpos,
            embed_rots=cfg.embed_rots,
            no_qk_points=cfg.no_qk_points,
            no_v_points=cfg.no_v_points,
            frame_update=True,
            update_rots=cfg.update_rots,
        )
        if cfg.separate_ipa_blocks:
            self.ipa_blocks = nn.ModuleList()
            for i in range(cfg.ipa_blocks):
                self.ipa_blocks.append(block_fn())
        else:
            self.ipa_block = block_fn()

    def forward(self, inp):

        x = inp["x"]
        B, L, _ = x.shape
        residx = torch.arange(L, device=x.device)[None].expand(B, -1)
        mask = inp["pad_mask"]
        z = self.pairwise_positional_embedding(residx, mask=mask)

        rots = inp.get("rots", None)
        trans = inp.get("trans", None)

        for block in self.blocks:
            if block.pair_updates:
                x, z, mask, rots, trans = torch.utils.checkpoint.checkpoint(
                    block, x, z, mask, rots, trans
                )
            else:
                x, z, mask, rots, trans = block(x, z, mask, rots, trans)

        all_rots = [rots]
        all_trans = [trans]
        for i in range(self.cfg.ipa_blocks):
            if self.cfg.separate_ipa_blocks:
                block = self.ipa_blocks[i]
            else:
                block = self.ipa_block
                
            x, z, mask, rots, trans = block(x, z, mask, rots, trans)
            all_rots.append(rots)
            all_trans.append(trans)
            
            if self.cfg.detach_rots:
                rots = rots.detach()
            if self.cfg.detach_trans:
                trans = trans.detach()

        return {
            "x": x,
            "z": z,
            "rots": torch.stack(all_rots) if rots is not None else None,
            "trans": torch.stack(all_trans),
        }
