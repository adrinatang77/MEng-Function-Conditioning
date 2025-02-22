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


def modulate(x, shift, scale):
    if shift is not None:
        return x * (1 + scale) + shift
    else:
        return x


def gate(x, gate_):
    if gate_ is not None:
        return x * gate_
    else:
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, dim, out):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


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
    def __init__(self, dim, ff_dim, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LayerNorm(dim))
        self.layers.append(nn.Linear(dim, ff_dim))
        for i in range(layers - 2):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(ff_dim, ff_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(ff_dim, dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OpenProtTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_expand=4,
        ff_layers=2,
        pairwise_dim=128,
        pairwise_heads=4,
        rope=False,  # rotate scalar queries and keys
        pair_bias=False,  # use pairs to bias
        pair_bias_linear=False,
        pair_updates=False,
        pair_ffn=False,
        pair_ff_expand=1,
        tri_mul=False,
        pair_values=False,  # aggregate values from pair reps
        ipa_attn=False,  # use point attention
        ipa_values=False,
        ipa_frames=False,  # use frames in point attention
        relpos_attn=False,  # instead use trans relpos
        relpos_rope=False,
        relpos_values=False,
        relpos_freqs=32,
        relpos_max=100,
        relpos_min=1,
        custom_rope=False,
        embed_rots=False,
        embed_trans=False,
        no_qk_points=4,
        no_v_points=8,
        frame_update=False,
        update_rots=False,
        readout_rots=False,
        update_x=True,
        adaLN=False,
        readout_adaLN=False,
        chain_mask=False,
        rots_type="vec",
        dropout=0.0,
        token_dropout=0.0,
    ):
        super().__init__()
        self.mha = GeometricMultiHeadAttention(
            dim=dim,
            heads=heads,
            pairwise_dim=pairwise_dim,
            rope=rope,
            pair_bias=pair_bias,
            pair_values=pair_values,
            pair_bias_linear=pair_bias_linear,
            chain_mask=chain_mask,
            ipa_attn=ipa_attn,
            ipa_values=ipa_values,
            ipa_frames=ipa_frames,
            relpos_attn=relpos_attn,
            relpos_rope=relpos_rope,
            relpos_values=relpos_values,
            relpos_freqs=relpos_freqs,
            relpos_min=relpos_min,
            relpos_max=relpos_max,
            custom_rope=custom_rope,
            embed_rots=embed_rots,
            embed_trans=embed_trans,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
            dropout=token_dropout,
        )
        self.ff = FeedForward(dim, ff_expand * dim, layers=ff_layers)

        self.pair_updates = pair_updates
        self.pair_ffn = pair_ffn
        self.frame_update = frame_update
        self.update_rots = update_rots
        self.rots_type = rots_type
        # self.readout_rots = readout_rots
        self.tri_mul = tri_mul
        self.update_x = update_x
        self.adaLN = adaLN
        self.readout_adaLN = readout_adaLN
        self.mha_norm = nn.LayerNorm(dim, elementwise_affine=not adaLN)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=not adaLN)
        self.mha_dropout = nn.Dropout(p=dropout)
        self.ff_dropout = nn.Dropout(p=dropout)

        if adaLN:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        if (pair_bias and pair_bias_linear) or pair_values:
            self.pair_norm = nn.LayerNorm(pairwise_dim)

        rot_dim = {"quat": 4, "vec": 3}[rots_type]
        if frame_update:

            if readout_adaLN:
                self.linear_frame_update = FinalLayer(
                    dim,
                    3 + rot_dim if update_rots else 3,
                )
            else:
                self.linear_frame_update = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 3 + rot_dim if update_rots else 3),
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
                dropout = 0
                self.row_drop = Dropout(dropout * 2, 2)
                self.col_drop = Dropout(dropout * 2, 1)
                torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
                torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
                torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
                torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)

            if pair_ffn:
                self.mlp_pair = FeedForward(pairwise_dim, pair_ff_expand * pairwise_dim)
                self.pair_ff_norm = nn.LayerNorm(pairwise_dim)

            torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
            torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)

    def forward(
        self,
        x,
        z,
        trans,
        mask,
        x_cond=None,
        postcond_fn=lambda x: x,
        relpos_mask=None,
        rots=None,
        idx=None,
        chain=None,
    ):

        ### no pair2sequence

        if self.adaLN:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(x_cond).chunk(6, dim=-1)
            )
        else:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = [None] * 6

        x_in = x

        

        x = x + self.mha_dropout(gate(
            self.mha(
                x=modulate(self.mha_norm(x), shift_mha, scale_mha),
                z=self.pair_norm(z) if hasattr(self, "pair_norm") else z,
                mask=mask.bool(),
                trans=postcond_fn(trans),
                rots=rots,
                relpos_mask=relpos_mask,
                idx=idx,
                chain=chain,
            ),
            gate_mha,
        ))

        x = x + self.ff_dropout(gate(self.ff(modulate(
            self.ff_norm(x), shift_mlp, scale_mlp
        )), gate_mlp))

        if self.pair_updates:
            z = z + self.sequence_to_pair(x)
            if self.tri_mul:
                tri_mask = (
                    mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
                )
                z = z + self.row_drop(self.tri_mul_out(z, mask=tri_mask))
                z = z + self.col_drop(self.tri_mul_in(z, mask=tri_mask))
            if self.pair_ffn:
                z = z + self.mlp_pair(self.pair_ff_norm(z))

        
        if self.frame_update:
            if self.readout_adaLN:
                update = self.linear_frame_update(x, x_cond)
            else:
                update = self.linear_frame_update(x)
            if self.update_rots:
                # rigids = Rigid(trans=trans, rots=Rotation(rot_mats=rots))
                # rigids = rigids.compose_q_update_vec(update)
                # trans = rigids.get_trans()
                # rots = rigids.get_rots().get_rot_mats()
                
                vec, rotvec = self.linear_frame_update(x).split(3, dim=-1)
                trans = trans + vec # torch.einsum("blij,blj->bli", rots, vec)
                rots = rots @ axis_angle_to_matrix(rotvec).mT
            else:
                trans = trans + update

        if not self.update_x:
            x = x_in
        return x, z, trans, rots


class StructureModule(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        block_fn = lambda: OpenProtTransformerBlock(
            dim=cfg.dim,
            heads=cfg.heads,
            ff_expand=cfg.ff_expand,
            pairwise_dim=cfg.pairwise_dim,
            pair_bias=cfg.ipa_pair_bias,
            pair_values=cfg.ipa_pair_values,
            adaLN=cfg.sm_adaLN,
            readout_adaLN=cfg.readout_adaLN,
            frame_update=cfg.ipa_frame_update,
            ipa_attn=cfg.ipa_nipa,
            ipa_values=cfg.ipa_nipa,
            relpos_attn=cfg.ipa_relpos,
            relpos_values=cfg.ipa_relpos,
            relpos_rope=cfg.ipa_rope,
            relpos_freqs=cfg.sm_relpos[0],
            relpos_min=cfg.sm_relpos[1],
            relpos_max=cfg.sm_relpos[2],
        )
        if cfg.separate_ipa_blocks:
            self.ipa_blocks = nn.ModuleList()
            for i in range(cfg.ipa_blocks):
                self.ipa_blocks.append(block_fn())
        elif self.cfg.ipa_blocks > 0:
            self.ipa_block = block_fn()
        if self.cfg.move_x_to_xcond:
            self.x_cond_linear = nn.Sequential(
                nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim)
            )

    def forward(self, x, z, trans, mask, x_cond, postcond_fn, relpos_mask=None, idx=None):
            
        if self.cfg.move_x_to_xcond:
            x_cond = x_cond + self.x_cond_linear(x)

        if self.cfg.zero_x_before_ipa:
            x = torch.zeros_like(x)

        
        trans = torch.zeros_like(trans) 
        all_trans = []
        all_x = []
        for i in range(self.cfg.ipa_blocks):

            if self.cfg.detach_trans:
                trans = trans.detach()
            
            if self.cfg.separate_ipa_blocks:
                block = self.ipa_blocks[i]
            else:
                block = self.ipa_block

            x, z, trans, rots = block(x, z, trans, mask, x_cond, postcond_fn, relpos_mask=relpos_mask, idx=idx)
            all_trans.append(trans)
            all_x.append(x)

        return {
            "x": torch.stack(all_x),
            "trans": torch.stack(all_trans),
        }


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        default_block_args = dict(
            dim=cfg.dim,
            ff_expand=cfg.ff_expand,
            heads=cfg.heads,
            rope=cfg.rope,
            custom_rope=cfg.custom_rope,
            adaLN=cfg.adaLN,
            chain_mask=cfg.all_atom,
            dropout=cfg.dropout,
            token_dropout=cfg.token_dropout,
        )


        self.blocks = nn.ModuleList()
        # if cfg.all_atom:
        #     self.z_emb = nn.Embedding(2, cfg.blocks * cfg.heads)
        #     torch.nn.init.zeros_(self.z_emb.weight)
        
        for i in range(cfg.blocks):
            block_args = default_block_args 
            self.blocks.append(OpenProtTransformerBlock(**block_args))

    def forward(self, inp):

        x = inp["x"]
        B, L, _ = x.shape
        
        residx = inp['residx']
        mask = inp["pad_mask"]
        z = inp.get("z", None)
        chain = inp.get("chain", None)
            
        trans = inp.get("struct", None)
        rots = inp.get("rots", None)
        x_cond = inp.get("x_cond", None)
        postcond_fn = inp.get("postcond_fn", None)
        struct_mask = inp.get("struct_mask", None)
        
        for i, block in enumerate(self.blocks):

            x = block(x, z, trans, mask, x_cond=x_cond, idx=residx, chain=chain)[0]

        
        return {"x": x, "z": z, "trans": trans, "rots": rots}
