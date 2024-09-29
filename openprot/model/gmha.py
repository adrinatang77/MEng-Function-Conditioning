from . import rope
import math
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from openfold.model.primitives import ipa_point_weights_init_, Linear
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)
from openfold.utils.rigid_utils import Rotation, Rigid
def sinusoidal_embedding(pos, n_freqs, max_period, min_period):
    periods = torch.exp(torch.linspace(
        math.log(min_period), 
        math.log(max_period), 
        n_freqs, 
        device=pos.device
    ))
    freqs = 2 * np.pi / periods
    return torch.cat(
        [torch.cos(pos[..., None] * freqs), torch.sin(pos[..., None] * freqs)], -1
    )


class GeometricMultiHeadAttention(nn.Module):
    def __init__(
        self, 
        dim,
        heads,
        pairwise_dim=128,
        pairwise_heads=4,
        rope=False,        # rotate scalar queries and keys
        pair_bias=False,   # use pairs to bias
        pair_bias_norm=False,
        pair_values=False, # aggregate values from pair reps
        ipa=False,         # use point attention 
        ipa_frames=False,  # use frames in point attention
        relpos=False,      # instead use trans relpos
        no_qk_points=4,
        no_v_points=8,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.pairwise_dim = pairwise_dim
        self.pairwise_heads = pairwise_heads
        self.rope = rope
        self.pair_bias = pair_bias
        self.pair_values = pair_values
        self.ipa = ipa
        self.ipa_frames = ipa_frames
        self.relpos = relpos
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points

        
        ## basic stuff we always need
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

        if relpos or (ipa and not ipa_frames):
            self.embed_rots = nn.Sequential(
                nn.Linear(9, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if ipa:
            hpq = heads * no_qk_points * 3
            self.linear_q_points = Linear(dim, hpq)
    
            hpkv = heads * (no_qk_points + no_v_points) * 3
            self.linear_kv_points = Linear(dim, hpkv)
    
            hpv = heads * no_v_points * 3
    
            self.head_weights = nn.Parameter(torch.zeros((heads)))
            ipa_point_weights_init_(self.head_weights)
            self.softplus = nn.Softplus()
            self.w_pt = Linear(heads * no_v_points * 4, dim)
    
        if pair_bias:
            self.linear_b = Linear(pairwise_dim, heads, bias=False)
            
        if relpos:
            self.linear_relpos_query = nn.Linear(dim, heads * 3 * 64, bias=False)
            self.linear_relpos_value = nn.Linear(heads * 3 * 64, dim, bias=False)
            
        if pair_values:
            self.w_z = Linear(heads * pairwise_dim, dim)

    def get_pt_attn(self, x, trans, rots):

        if self.ipa_frames:
            r = Rigid(trans=trans, rots=Rotation(rot_mats=rots))
            
        q_pts = self.linear_q_points(x)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)

        if self.ipa_frames:
            q_pts = r[..., None].apply(q_pts)
        else:
            q_pts = trans[:,:,None] + q_pts

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(x)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)

        if self.ipa_frames:
            kv_pts = r[..., None].apply(kv_pts)
        else:
            kv_pts = trans[:,:,None] + kv_pts

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            2.0 / (self.no_qk_points * 9.0)
        )

        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        
        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        return v_pts, pt_att

    def get_ipa_values(self, attn, v_pts, trans, rots):
        o_pt = torch.sum(
            (
                attn[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        if self.ipa_frames:
            r = Rigid(trans=trans, rots=Rotation(rot_mats=rots))
            o_pt = r[..., None, None].invert_apply(o_pt)
        
        else:
            o_pt = o_pt - trans[:,:,None,None]

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + 1e-6), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        
        return torch.cat((*torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1)
        

    def forward(self, x, z, mask, trans, rots):

        B, L, D = x.shape

        if self.relpos or (self.ipa and not self.ipa_frames):
            x = x + self.embed_rots(rots.view(B, L, -1))

        query = self.w_q(x).view(B, L, self.heads, -1).transpose(1, 2)  # B H L D
        key = self.w_k(x).view(B, L, self.heads, -1).transpose(1, 2)

        if self.rope:
            freqs_cis = rope.compute_freqs_cis(D // self.heads, L, device=x.device)
            query, key = rope.apply_rotary_emb(query, key, freqs_cis)
        
        attn = query @ key.mT / math.sqrt(D / self.heads)  # B H L L

        if mask is None:
            mask = torch.ones(B, L, D, dtype=bool, device=x.device)
        
        mask = mask.view(B, 1, 1, -1)
        attn = attn + torch.where(mask, 0, -float("inf")) 

        if self.pair_bias:
            attn = attn + self.linear_b(z).permute(0, 3, 1, 2)

        if self.ipa:
            v_pts, pt_attn = self.get_pt_attn(x, trans, rots)
            attn = attn + pt_attn
            # attn = attn * math.sqrt(1/3)
            
        elif self.relpos:
            relpos_query = self.linear_relpos_query(x)
            relpos_query = relpos_query.view(B, L, self.heads, -1).transpose(1, 2)
            
            relpos = trans[:,None] - trans[:,:,None]
            relpos_emb = sinusoidal_embedding(
                relpos, n_freqs=32, max_period=100, min_period=1
            )
            relpos_emb = relpos_emb.view(B, L, L, -1)
            
            relpos_attn = torch.einsum('bhid,bijd->bhij', relpos_query, relpos_emb) 
            relpos_attn = relpos_attn / math.sqrt(3 * 64)
            attn = attn + relpos_attn
            
        attn = torch.softmax(attn, dim=-1)

        ## scalar values
        value = self.w_v(x).view(B, L, self.heads, -1).transpose(1, 2)
        out = (attn @ value).transpose(1, 2).reshape(B, L, D)

        out = self.w_o(out)

        if self.pair_values:
            z_out = torch.einsum("bhij,bijd->bihd", attn, z)
            z_out = z_out.reshape(B, L, -1)
            out = out + self.w_z(z_out)

        if self.ipa:
            ipa_out = self.get_ipa_values(attn, v_pts, trans, rots)
            out = out + self.w_pt(ipa_out)
            
        return out
    """
    def geometric_forward(self, x, mask, rots, trans, bias=None, inf=1e6):
        B, L, D = x.shape

        ## scalar attention
        query = self.w_q(x).view(B, L, self.heads, -1).transpose(1, 2)  # B H L D
        key = self.w_k(x).view(B, L, self.heads, -1).transpose(1, 2)
        freqs_cis = rope.compute_freqs_cis(D // self.heads, L, device=x.device)
        query, key = rope.apply_rotary_emb(query, key, freqs_cis)
        scalar_attn = query @ key.mT / math.sqrt(D)  # B H L L

        ## vector attention
        query = self.w_vq(x).view(B, L, self.heads, self.vdim, 3) @ rots.mT.unsqueeze(2)
        # B L H D 3
        key = self.w_vk(x).view(B, L, self.heads, self.vdim, 3) @ rots.mT.unsqueeze(2)
        query = query.view(B, L, self.heads, 3 * self.vdim).transpose(1, 2)
        key = key.view(B, L, self.heads, 3 * self.vdim).transpose(1, 2)
        vector_attn = query @ key.mT / math.sqrt(3 * self.vdim)

        ## affine attention
        query = self.w_aq(x).view(B, L, self.heads, self.adim, 3)
        query = query @ rots.mT.unsqueeze(2) + trans[:, :, None, None]
        query = query.transpose(1, 2)

        key = self.w_ak(x).view(B, L, self.heads, self.adim, 3)
        key = key @ rots.mT.unsqueeze(2) + trans[:, :, None, None]
        key = key.transpose(1, 2)

        affine_attn = -torch.sqrt(
            torch.square(query[:, :, :, None] - key[:, :, None]).sum(-1)
        ).sum(-1)
        affine_attn = F.softplus(self.affine_weights)[None, :, None, None] * affine_attn

        attn = scalar_attn + vector_attn + affine_attn

        if mask is not None:
            mask = mask.view(B, 1, 1, -1)

        if bias is not None:
            if mask is not None:
                mask = torch.where(mask, 0, -float("inf")) + bias
            else:
                mask = bias

        if mask is not None:
            attn = attn + mask

        attn = torch.softmax(attn, dim=-1)

        ## scalar values
        value = self.w_v(x).view(B, L, self.heads, -1).transpose(1, 2)
        scalar_out = (attn @ value).transpose(1, 2).reshape(B, L, D)

        ## vector values
        value = self.w_vv(x).view(B, L, self.heads, self.vdim, 3) @ rots.mT.unsqueeze(2)
        vector_out = torch.einsum("bhij,bhjdn->bhidn", attn, value.transpose(1, 2))
        vector_out = vector_out @ rots.unsqueeze(1)
        vector_out = vector_out.transpose(1, 2).reshape(
            B, L, 3 * self.heads * self.vdim
        )

        ## affine values
        value = self.w_av(x).view(B, L, self.heads, self.adim, 3)
        value = value @ rots.mT.unsqueeze(2) + trans[:, :, None, None]
        affine_out = torch.einsum("bhij,bhjdn->bhidn", attn, value.transpose(1, 2))
        affine_out = (affine_out - trans[:, None, :, None]) @ rots.unsqueeze(1)
        affine_out = affine_out.transpose(1, 2).reshape(
            B, L, 3 * self.heads * self.adim
        )

        return self.w_o(scalar_out) + self.w_vo(vector_out) + self.w_ao(affine_out)
    
    def forward(self, x, mask=None, bias=None, **kwargs):
        if self.geometric:
            return self.geometric_forward(x, mask=mask, bias=bias, **kwargs)
        elif self.pair:
            return self.pair_forward(x, mask=mask, bias=bias, **kwargs)
        B, L, D = x.shape

        query = self.w_q(x).view(B, L, self.heads, -1).transpose(1, 2)
        key = self.w_k(x).view(B, L, self.heads, -1).transpose(1, 2)
        value = self.w_v(x).view(B, L, self.heads, -1).transpose(1, 2)

        freqs_cis = rope.compute_freqs_cis(D // self.heads, L, device=x.device)
        query, key = rope.apply_rotary_emb(query, key, freqs_cis)

        if mask is not None:
            mask = mask.view(B, 1, 1, -1)

        if bias is not None:
            if mask is not None:
                mask = torch.where(mask, 0, -float("inf")) + bias
            else:
                mask = bias

        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        return self.w_o(attn)
    """