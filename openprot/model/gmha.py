from . import rope
import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class GeometricMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, geometric=False, vdim=12, adim=1, freqs_cis=None):
        super().__init__()
        self.dim = dim
        self.vdim = vdim
        self.adim = adim
        self.heads = heads
        assert self.dim % heads == 0

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

        if freqs_cis is not None:
            self.register_buffer("freqs_cis", freqs_cis)

        if geometric:
            self.w_vq = nn.Linear(dim, 3 * heads * vdim, bias=True)
            self.w_vk = nn.Linear(dim, 3 * heads * vdim, bias=True)
            self.w_vv = nn.Linear(dim, 3 * heads * vdim, bias=True)
            self.w_vo = nn.Linear(3 * heads * vdim, dim, bias=True)

            self.w_aq = nn.Linear(dim, 3 * heads * adim, bias=True)
            self.w_ak = nn.Linear(dim, 3 * heads * adim, bias=True)
            self.w_av = nn.Linear(dim, 3 * heads * adim, bias=True)
            self.w_ao = nn.Linear(3 * heads * adim, dim, bias=True)

            self.affine_weights = nn.Parameter(torch.zeros(heads))
        self.geometric = geometric

    def geometric_forward(self, x, mask, rots, trans, inf=1e6):
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

        key = self.w_aq(x).view(B, L, self.heads, self.adim, 3)
        key = key @ rots.mT.unsqueeze(2) + trans[:, :, None, None]
        key = key.transpose(1, 2)

        affine_attn = -torch.sqrt(
            torch.square(query[:, :, :, None] - key[:, :, None]).sum(-1)
        ).sum(-1)
        affine_attn = F.softplus(self.affine_weights)[None, :, None, None] * affine_attn

        attn = scalar_attn + vector_attn + affine_attn

        if mask is not None:
            mask = mask.view(B, 1, 1, -1)
            attn = attn - inf * (1 - mask)

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

    def forward(self, x, mask=None, **kwargs):
        if self.geometric:
            return self.geometric_forward(x, mask, **kwargs)
        B, L, D = x.shape

        query = self.w_q(x).view(B, L, self.heads, -1).transpose(1, 2)
        key = self.w_k(x).view(B, L, self.heads, -1).transpose(1, 2)
        value = self.w_v(x).view(B, L, self.heads, -1).transpose(1, 2)

        freqs_cis = rope.compute_freqs_cis(D // self.heads, L, device=x.device)
        query, key = rope.apply_rotary_emb(query, key, freqs_cis)

        if mask is not None:
            mask = mask.view(B, 1, 1, -1)

        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attn = attn.transpose(1, 2).view(B, L, D)
        return self.w_o(attn)
