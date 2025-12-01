import os
import sys
import json
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union, Optional
from omegaconf import ListConfig, OmegaConf
from functools import partial
from einops import rearrange, repeat
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from sat.model.base_model import BaseModel, non_conflict
from sat.model.mixins import BaseMixin
from sat.ops.layernorm import LayerNorm
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
from sat.mpu.mappings import copy_to_model_parallel_region
from sat.mpu.layers import ColumnParallelLinear, RowParallelLinear
from sat.mpu.utils import split_tensor_along_last_dim, scaled_init_method, unscaled_init_method
from sat.helpers import print_rank0
from sat import mpu

from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)

from sgm.modules.diffusionmodules.openaimodel import (
    Timestep,
    timestep_embedding
)
from sgm.modules.diffusionmodules.util import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)

class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(
            self,
            in_channels,
            hidden_size,
            patch_size,
            bias=True,
            text_hidden_size=None,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels * patch_size * patch_size, hidden_size, bias=bias)
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        else:
            self.text_proj = None

    def word_embedding_forward(self, input_ids, **kwargs):
        # now is 3d patch
        images = kwargs["images"] # (b,tc,h,w)
        emb = self.proj(images) # ((b t),d,h/2,w/2)

        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs['encoder_outputs'])
            text_emb = text_emb.view(-1, text_emb.shape[-1])
            emb[kwargs['txt_mask']] = text_emb

        emb = emb.contiguous()
        return emb # (b,n_t+t*n_i,d)

    def reinit(self, parent_model=None):
        # w1 = self.proj_space.weight.data
        # nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # nn.init.constant_(self.proj_space.bias, 0)
        # w2 = self.proj_time.weight.data
        # nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        # nn.init.constant_(self.proj_time.bias, 0)
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)
        del self.transformer.word_embeddings

def get_3d_sincos_pos_embed(embed_dim, grid_height, grid_width, t_size, cls_token=False, space_interpolation=1.0):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_height, dtype=np.float32) / space_interpolation
    grid_w = np.arange(grid_width, dtype=np.float32) / space_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_height*grid_width, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    # pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    return pos_embed # [T, H*W, D]

def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Basic2DPositionEmbeddingMixin(BaseMixin):
    def __init__(self, height, width, compressed_num_frames, hidden_size, text_length=0):
        super().__init__()
        self.height = height
        self.width = width
        self.spatial_length = height * width
        self.pos_embedding = nn.Parameter(torch.zeros(1, int(text_length + self.spatial_length), int(hidden_size)), requires_grad=False)
        
    def position_embedding_forward(self, position_ids, **kwargs):
        return self.pos_embedding
    
    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], self.height, self.width)
        self.pos_embedding.data[:,-self.spatial_length:].copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


class Basic3DPositionEmbeddingMixin(BaseMixin):
    def __init__(self, height, width, compressed_num_frames, hidden_size, 
                 text_length=0, space_interpolation=1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.text_length = text_length
        self.compressed_num_frames = compressed_num_frames
        self.spatial_length = height * width
        self.num_patches = height * width * compressed_num_frames
        self.pos_embedding = nn.Parameter(torch.zeros(1, int(text_length + self.num_patches), int(hidden_size)),
                                          requires_grad=False)
        self.space_interpolation = space_interpolation

    def position_embedding_forward(self, position_ids, **kwargs):
        if kwargs['images'].shape[1] ==  1:
            return self.pos_embedding[:, :self.text_length + self.spatial_length]

        return self.pos_embedding[:, :self.text_length + kwargs["seq_length"]]

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_3d_sincos_pos_embed(self.pos_embedding.shape[-1], self.height, self.width, self.compressed_num_frames, 
                                            space_interpolation=self.space_interpolation)
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed = rearrange(pos_embed, 't n d -> (t n) d')
        self.pos_embedding.data[:, -self.num_patches:].copy_(pos_embed)

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

class Rotary3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        hidden_size_head,
        theta = 10000,
        rot_v=False,
        pnp=False,
        space_interpolation=1.0
    ):
        super().__init__()
        self.rot_v = rot_v

        dim_t = hidden_size_head // 4
        dim_h = hidden_size_head // 8 * 3
        dim_w = hidden_size_head // 8 * 3

        # 'lang':
        freqs_t = 1. / (theta ** (torch.arange(0, dim_t, 2)[:(dim_t // 2)].float() / dim_t))
        freqs_h = 1. / (theta ** (torch.arange(0, dim_h, 2)[:(dim_h // 2)].float() / dim_h))
        freqs_w = 1. / (theta ** (torch.arange(0, dim_w, 2)[:(dim_w // 2)].float() / dim_w))

        grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        freqs_t = torch.einsum('..., f -> ... f', grid_t, freqs_t)
        freqs_h = torch.einsum('..., f -> ... f', grid_h, freqs_h)
        freqs_w = torch.einsum('..., f -> ... f', grid_w, freqs_w)

        freqs_t = repeat(freqs_t, '... n -> ... (n r)', r = 2)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_t[:,None,None,:], freqs_h[None,:,None,:], freqs_w[None,None,:,:]), dim=-1)
        # (T H W D)

        self.pnp = pnp
        if not self.pnp:
            freqs = rearrange(freqs, 't h w d -> (t h w) d')

        freqs = freqs.contiguous()
        self.register_buffer('freqs', freqs)
        # torch.register_after_fork(self, self._set_freqs)

        # freqs_cos = freqs.contiguous().cos()
        # freqs_sin = freqs.contiguous().sin()
        # self.freqs_cos = freqs_cos
        # self.freqs_sin = freqs_sin

    def rotary(self, t, **kwargs):
        if self.pnp:
            t_coords = kwargs['rope_position_ids'][:, :, 0]
            x_coords = kwargs['rope_position_ids'][:, :, 1]
            y_coords = kwargs['rope_position_ids'][:, :, 2]
            # 创建布尔掩码，标记哪些位置的 coords 值不为 -1
            mask = (x_coords != -1) & (y_coords != -1) & (t_coords != -1)
            # 仅对 mask 为 True 的位置应用频率编码
            freqs = torch.zeros([t.shape[0], t.shape[2], t.shape[3]], dtype=t.dtype, device=t.device)
            freqs[mask] = self.freqs[t_coords[mask], x_coords[mask], y_coords[mask]]
            freqs = freqs.unsqueeze(1)
            # freqs_cos = self.freqs_cos[t_coords[mask], x_coords[mask], y_coords[mask]].unsqueeze(1)
            # freqs_sin = self.freqs_sin[t_coords[mask], x_coords[mask], y_coords[mask]].unsqueeze(1)

        else:
            freqs = self.freqs

        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()

        return t * freqs_cos + rotate_half(t) * freqs_sin

    def position_embedding_forward(self, position_ids, **kwargs):
        # x = kwargs["images"]
        # return torch.zeros((1, self.num_patches+self.num_addition_tokens, self.hidden_size), dtype=x.dtype, device=x.device)
        return None

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        **kwargs
    ):

        query_layer = self.rotary(query_layer, **kwargs)
        key_layer = self.rotary(key_layer, **kwargs)
        if self.rot_v:
            value_layer = self.rotary(value_layer, **kwargs)

        batch_size, num_query_heads = query_layer.shape[:2]  # [b, np, s, hn]
        num_kv_heads = key_layer.shape[1]  # [b, np, s, hn]
        key_layer = key_layer.unsqueeze(2).expand(-1, -1, num_query_heads // num_kv_heads, -1,
                                                      -1).contiguous().view(batch_size, num_query_heads,
                                                                            *key_layer.shape[2:])
        value_layer = value_layer.unsqueeze(2).expand(-1, -1, num_query_heads // num_kv_heads, -1,
                                                          -1).contiguous().view(batch_size, num_query_heads,
                                                                                *value_layer.shape[2:])


        dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False
        )
        return attn_output

        # return attention_fn_default(query_layer, key_layer, value_layer, attention_mask,
        #                             attention_dropout=attention_dropout,
        #                             log_attention_weights=log_attention_weights,
        #                             scaling_attention_score=scaling_attention_score,
        #                             **kwargs)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unpatchify(x, c, p, w, h, rope_position_ids=None, **kwargs):
    """
    x: (N, T/2 * S, patch_size**3 * C)
    imgs: (N, T, H, W, C)
    """
    if rope_position_ids is not None:
        assert NotImplementedError
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum('nlpqc->ncplq', x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        b = x.shape[0]
        imgs = rearrange(x,'b (t h w) (c p q) -> b t c (h p) (w q)',b=b,h=h,w=w,c=c,p=p,q=p)

        # imgs = imgs[:, 1:]
    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(
            self,
            hidden_size,
            time_embed_dim,
            patch_size,
            out_channels,
            latent_width,
            latent_height,
            elementwise_affine,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)
        )
        
        self.spatial_length = latent_width * latent_height // patch_size**2
        self.latent_width = latent_width
        self.latent_height = latent_height

    def final_forward(self, logits, **kwargs):
        shift, scale = self.adaLN_modulation(kwargs['emb']).chunk(2, dim=1)
        x = modulate(self.norm_final(logits), shift, scale)
        x = self.linear(x)

        return x

    def reinit(self, parent_model=None):
        # nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class AdaLNMixin(BaseMixin):
    def __init__(
            self,
            width,
            height,
            hidden_size,
            num_layers,
            time_embed_dim,
            compressed_num_frames,
            qk_ln=True,
            hidden_size_head=None,
            params_dtype=torch.float,
            device=torch.device('cpu'),
            elementwise_affine=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.width = width
        self.height = height


        self.adaLN_modulations = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 12 * hidden_size)
            ) for _ in range(num_layers)
        ])     

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])
            self.key_layernorm_list = nn.ModuleList(
                [LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])


    def layer_forward(
            self,
            hidden_states,
            mask,
            txt_mask,
            image_mask,
            *args,
            **kwargs,
    ):
        seq_len = hidden_states.shape[-2]
        def concat_hidden_states(text_hidden_states, img_hidden_states, txt_mask, image_mask):
            hidden_shape = list(text_hidden_states.shape)
            hidden_shape[-2] = seq_len
            hidden_states = torch.empty(hidden_shape, dtype=text_hidden_states.dtype, device=text_hidden_states.device)
            hidden_states[image_mask] = img_hidden_states
            hidden_states[txt_mask] = text_hidden_states
            return hidden_states

        # hidden_states (b,(n_t+t*n_i),d)
        # text_hidden_states = hidden_states[:, :text_length] # (b,n,d)
        # img_hidden_states = hidden_states[:, text_length:] # (b,(t n),d)
        text_hidden_states = hidden_states[txt_mask]
        img_hidden_states = hidden_states[image_mask]

        layer = self.transformer.layers[kwargs['layer_id']]
        adaLN_modulation = self.adaLN_modulations[kwargs['layer_id']]
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, \
            text_shift_msa, text_scale_msa, text_gate_msa, text_shift_mlp, text_scale_mlp, text_gate_mlp \
                = adaLN_modulation(kwargs['emb']).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = \
            gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1), text_gate_msa.unsqueeze(1), text_gate_mlp.unsqueeze(1)


        # self full attention (b,(t n),d)
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = layer.input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa)

        attention_input = concat_hidden_states(text_attention_input, img_attention_input, txt_mask, image_mask)

        attention_output = layer.attention(attention_input, mask, **kwargs)
        text_attention_output = attention_output[txt_mask] # (b,n,d)
        img_attention_output = attention_output[image_mask] # (b,(t n),d)

        if self.transformer.layernorm_order == 'sandwich':
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_hidden_states = img_hidden_states + gate_msa * img_attention_output # (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output # (b,n,d)

        # mlp (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(img_hidden_states) # vision (b,(t n),d)
        text_mlp_input = layer.post_attention_layernorm(text_hidden_states) # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)

        mlp_input = concat_hidden_states(text_mlp_input, img_mlp_input, txt_mask, image_mask)

        mlp_output = layer.mlp(mlp_input, **kwargs)
        text_mlp_output = mlp_output[txt_mask] # (b,n,d)
        img_mlp_output = mlp_output[image_mask] # (b,(t n),d)

        if self.transformer.layernorm_order == 'sandwich':
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output # vision (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output # language (b,n,d)

        hidden_states = concat_hidden_states(text_hidden_states, img_hidden_states, txt_mask, image_mask)
        return hidden_states

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    @non_conflict
    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                     attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, old_impl=attention_fn_default , **kwargs):

        if self.qk_ln:
            query_layernorm = self.query_layernorm_list[kwargs['layer_id']]
            key_layernorm = self.key_layernorm_list[kwargs['layer_id']]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)
        # change to memory efficient version
        # https://facebookresearch.github.io/xformers/components/ops.html#module-xformers.ops
        # xformers.ops.fmha.attn_bias.BlockDiagonalMask
        return old_impl(query_layer, key_layer, value_layer, attention_mask, attention_dropout=attention_dropout,
                        log_attention_weights=log_attention_weights, scaling_attention_score=scaling_attention_score, **kwargs)


str_to_dtype = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}


class DiffusionTransformer(BaseModel):
    def __init__(
        self, 
        transformer_args,
        num_frames,
        time_compressed_rate,
        latent_width,
        latent_height,
        patch_size,
        in_channels,
        out_channels,
        hidden_size,
        num_layers,
        num_attention_heads,
        elementwise_affine,
        time_embed_dim=None,
        num_classes=None,
        modules={},
        input_time='adaln',
        adm_in_channels=None,
        parallel_output=True,
        space_interpolation=1.0,
        **kwargs
    ):
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.time_compressed_rate = time_compressed_rate
        self.spatial_length = latent_width * latent_height // patch_size**2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.model_channels = hidden_size
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.is_decoder = transformer_args.is_decoder
        self.elementwise_affine = elementwise_affine
        self.space_interpolation = space_interpolation
        try:
            self.dtype = str_to_dtype[kwargs.pop('dtype')]
        except:
            self.dtype = torch.float32

        if 'activation_func' not in kwargs:
            approx_gelu = nn.GELU(approximate='tanh')
            kwargs['activation_func'] = approx_gelu
        transformer_args.num_layers = num_layers
        transformer_args.hidden_size = hidden_size
        transformer_args.num_attention_heads = num_attention_heads
        transformer_args.parallel_output = parallel_output
        super().__init__(args=transformer_args, transformer=None, layernorm=partial(LayerNorm, elementwise_affine=elementwise_affine, eps=1e-6),  **kwargs)

        module_configs = modules
        self._build_modules(module_configs)


    def _build_modules(self, module_configs):
        model_channels = self.hidden_size
        # time_embed_dim = model_channels * 4
        time_embed_dim = self.time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert self.adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(self.adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()
            
        pos_embed_config = module_configs.pop('pos_embed_config')
        self.add_mixin('pos_embed', instantiate_from_config(pos_embed_config, height=self.latent_height//self.patch_size, width=self.latent_width//self.patch_size,
                                                            compressed_num_frames=(self.num_frames-1)//self.time_compressed_rate+1, hidden_size=self.hidden_size,
                                                            space_interpolation=self.space_interpolation), reinit=True)
        
        patch_embed_config = module_configs.pop('patch_embed_config')
        self.add_mixin('patch_embed', instantiate_from_config(patch_embed_config, patch_size=self.patch_size, hidden_size=self.hidden_size, in_channels=self.in_channels), reinit=True)
        if self.input_time == 'adaln':
            adaln_layer_config = module_configs.pop('adaln_layer_config')
            self.add_mixin('adaln_layer', instantiate_from_config(adaln_layer_config, height=self.latent_height//self.patch_size, width=self.latent_width//self.patch_size,
                                                                  hidden_size=self.hidden_size, num_layers=self.num_layers, compressed_num_frames=(self.num_frames-1)//self.time_compressed_rate+1,
                                                                  hidden_size_head=self.hidden_size//self.num_attention_heads, time_embed_dim=self.time_embed_dim,
                                                                  elementwise_affine=self.elementwise_affine))
        else:
            raise NotImplementedError
        final_layer_config = module_configs.pop('final_layer_config')
        self.add_mixin('final_layer', instantiate_from_config(final_layer_config, hidden_size=self.hidden_size, patch_size=self.patch_size,
                                                              out_channels=self.out_channels, time_embed_dim=self.time_embed_dim,
                                                              latent_width=self.latent_width, latent_height=self.latent_height,
                                                              elementwise_affine=self.elementwise_affine), reinit=True)

        return
    
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            # assert y.shape[0] == x.shape[0]
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        kwargs['images'] = x
        kwargs['emb'] = emb
        kwargs['encoder_outputs'] = context

        kwargs['input_ids'] = kwargs['position_ids'] = torch.ones((1, 1)).to(x.dtype)
        output = super().forward(**kwargs)[0]
        return output

