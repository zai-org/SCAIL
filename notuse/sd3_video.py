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

from sat.model.base_model import BaseModel
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
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        else:
            self.text_proj = None

    def word_embedding_forward(self, input_ids, **kwargs):
        # now is 3d patch
        images = kwargs["images"] # (b,t,c,h,w)
        B, T = images.shape[:2]
        first_frame = images[:,0].unsqueeze(1) # copy first frame
        images = torch.cat([first_frame, images], dim=1)
        emb = rearrange(images, 'b t d h w -> b d t h w')
        emb = self.proj(emb) # (b,d,t/2,h/2,w/2)
        emb = rearrange(emb, 'b d t h w -> b t d h w')
        emb = emb.flatten(3).transpose(2, 3) # (b,t/2,n,d)

        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs['encoder_outputs'])
            text_emb = torch.cat([text_emb.unsqueeze(1)]*emb.shape[1], dim=1)
            emb = torch.cat((text_emb, emb), dim=2) # (b,t/2,n_t+n_i,d)

        emb = emb.contiguous()
        return emb # (b,t/2,n_t+n_i,d)

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

def get_3d_sincos_pos_embed(embed_dim, grid_height, grid_width, t_size, cls_token=False):
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
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
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
    def __init__(self, height, width, compressed_num_frames, hidden_size, text_length=0):
        super().__init__()
        self.height = height
        self.width = width
        self.compressed_num_frames = compressed_num_frames
        self.spatial_length = height * width
        self.pos_embedding = nn.Parameter(torch.zeros(compressed_num_frames, int(text_length + self.spatial_length), int(hidden_size)),
                                          requires_grad=False)

    def position_embedding_forward(self, position_ids, **kwargs):
        if kwargs['images'].shape[1] ==  1:
            return self.pos_embedding[0].unsqueeze(0)
        return self.pos_embedding

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_3d_sincos_pos_embed(self.pos_embedding.shape[-1], self.height, self.width, self.compressed_num_frames)
        self.pos_embedding.data[:, -self.spatial_length:].copy_(torch.from_numpy(pos_embed).float())


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
        imgs = rearrange(x,'b (t h w) (r c p q) -> b (t r) c (h p) (w q)',b=b,h=h,w=w,c=c,r=p,p=p,q=p)

        imgs = imgs[:, 1:]
    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(
            self,
            hidden_size,
            time_embed_dim,
            patch_size,
            spatial_length,
            out_channels,
            num_frames,
            latent_width,
            latent_height,
            elementwise_affine,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.spatial_length = spatial_length
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames
        
        assert latent_width * latent_height // patch_size**2 == spatial_length
        self.latent_width = latent_width
        self.latent_height = latent_height

    def final_forward(self, logits, **kwargs):
        x, emb = logits[:, :, -self.spatial_length:, :], kwargs['emb'] # x:(b,t/2,n,d),只取了x中后面image的部分
        x = rearrange(x, 'b t n d -> b (t n) d')
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return unpatchify(x, c=self.out_channels, p=self.patch_size,
                          w=self.latent_width//self.patch_size, h=self.latent_height//self.patch_size,
                          rope_position_ids=kwargs.get('rope_position_ids', None), **kwargs)

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
            st_attention_dropout_prob=0,
            st_output_dropout_prob=0,
            st_window_size=8,
            elementwise_affine=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.width = width
        self.height = height
        self.st_window_size = st_window_size
        self.compressed_num_frames = compressed_num_frames

        output_layer_init_method = scaled_init_method(0.02, num_layers)
        init_method = unscaled_init_method(0.02)

        self.adaLN_modulations = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 18 * hidden_size)
            ) for _ in range(num_layers)
        ])

        # for attention forward
        self.text_input_layernorm_list = nn.ModuleList(
            [LayerNorm(hidden_size, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])

        self.text_query_key_value_list = nn.ModuleList([ColumnParallelLinear(
                hidden_size,
                3 * hidden_size,
                stride=3,
                gather_output=False,
                init_method=init_method,
                bias=True,
                params_dtype=params_dtype,
                module=self,
                name="text_query_key_value",
                skip_init=False,
                device=device
            ) for _ in range(num_layers)])
        
        self.text_dense_list = nn.ModuleList([RowParallelLinear(
                hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                bias=True,
                params_dtype=params_dtype,
                module=self,
                name="text_dense",
                skip_init=False,
                device=device,
                final_bias=True
            ) for _ in range(num_layers)])        

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])
            self.key_layernorm_list = nn.ModuleList(
                [LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])
            self.st_query_layernorm_list = nn.ModuleList(
                [LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])
            self.st_key_layernorm_list = nn.ModuleList(
                [LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])

        # for spatiotemporal attention forward
        self.st_input_layernorm_list = nn.ModuleList(
            [LayerNorm(hidden_size, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])
        self.st_text_input_layernorm_list = nn.ModuleList(
            [LayerNorm(hidden_size, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])

        self.st_query_key_value_list = nn.ModuleList([ColumnParallelLinear(
                hidden_size,
                3 * hidden_size,
                stride=3,
                gather_output=False,
                init_method=init_method,
                bias=True,
                params_dtype=params_dtype,
                module=self,
                name="st_query_key_value",
                skip_init=False,
                device=device
            ) for _ in range(num_layers)])
        
        self.st_text_query_key_value_list = nn.ModuleList([ColumnParallelLinear(
                hidden_size,
                3 * hidden_size,
                stride=3,
                gather_output=False,
                init_method=init_method,
                bias=True,
                params_dtype=params_dtype,
                module=self,
                name="st_text_query_key_value",
                skip_init=False,
                device=device
            ) for _ in range(num_layers)])

        self.st_attention_dropout_list = nn.ModuleList([torch.nn.Dropout(st_attention_dropout_prob) for _ in range(num_layers)])
        
        self.st_dense_list = nn.ModuleList([RowParallelLinear(
                hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                bias=True,
                params_dtype=params_dtype,
                module=self,
                name="st_dense",
                skip_init=False,
                device=device,
                final_bias=True
            ) for _ in range(num_layers)]) 
        
        self.st_text_dense_list = nn.ModuleList([RowParallelLinear(
                hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                bias=True,
                params_dtype=params_dtype,
                module=self,
                name="st_text_dense",
                skip_init=False,
                device=device,
                final_bias=True
            ) for _ in range(num_layers)]) 
        
        self.st_output_dropout_list = nn.ModuleList([torch.nn.Dropout(st_output_dropout_prob) for _ in range(num_layers)])

        # for mlp forward
        self.text_post_attention_layernorm_list = nn.ModuleList(
            [LayerNorm(hidden_size, eps=1e-6, elementwise_affine=elementwise_affine) for _ in range(num_layers)])
        
        self.text_dense_h_to_4h_list = nn.ModuleList([ColumnParallelLinear(
                hidden_size,
                4 * hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=True,
                params_dtype=params_dtype,
                module=self,
                name="text_dense_h_to_4h",
                skip_init=False,
                device=device
            ) for _ in range(num_layers)])

        self.text_dense_4h_to_h_list = nn.ModuleList([RowParallelLinear(
                4 * hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                bias=True,
                params_dtype=params_dtype,
                module=self,
                name="text_dense_4h_to_h",
                skip_init=False,
                device=device,
                final_bias=True
            ) for _ in range(num_layers)])


    def layer_forward(
            self,
            hidden_states,
            mask,
            *args,
            **kwargs,
    ):
        text_length = kwargs['text_length']
        text_hidden_states = hidden_states[:, :1, :text_length] # (b,1,n,d)
        img_hidden_states = hidden_states[:, :, text_length:] # (b,t,n,d)

        layer = self.transformer.layers[kwargs['layer_id']]
        adaLN_modulation = self.adaLN_modulations[kwargs['layer_id']]
        
        shift_msa, scale_msa, gate_msa, shift_msta, scale_msta, gate_msta, shift_mlp, scale_mlp, gate_mlp, \
            text_shift_msa, text_scale_msa, text_gate_msa, text_shift_msta, text_scale_msta, text_gate_msta, text_shift_mlp, text_scale_mlp, text_gate_mlp \
                = adaLN_modulation(kwargs['emb']).chunk(18, dim=1)
        gate_msa, gate_mlp, gate_msta, text_gate_msa, text_gate_mlp, text_gate_msta = \
            gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1), gate_msta.unsqueeze(1), text_gate_msa.unsqueeze(1), text_gate_mlp.unsqueeze(1), text_gate_msta.unsqueeze(1)

        B, T, N, D = hidden_states.shape # (b,t,n,d)
        # self spatial attention ((b t),n,d)
        img_hidden_states = rearrange(img_hidden_states, 'b t n d -> b (t n) d') # (b,(t n),d)
        text_hidden_states = rearrange(text_hidden_states, 'b t n d -> b (t n) d') # (b,(1 n),d)

        text_input_layernorm = self.text_input_layernorm_list[kwargs['layer_id']]
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = text_input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa)

        img_attention_input = rearrange(img_attention_input, 'b (t n) d -> (b t) n d', t=T)
        kwargs['text_attention_input'] = text_attention_input # (b,(1 n),d)
        attention_output = layer.attention(img_attention_input, mask, **kwargs)
        (text_attention_output, img_attention_output) = attention_output
        if self.transformer.layernorm_order == 'sandwich':
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_attention_output = rearrange(img_attention_output, '(b t) n d -> b (t n) d', b=B)
        img_hidden_states = img_hidden_states + gate_msa * img_attention_output # (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output # (b,n,d)

        # self spatiotemporal attention
        st_input_layernorm = self.st_input_layernorm_list[kwargs['layer_id']]
        st_text_input_layernorm = self.st_text_input_layernorm_list[kwargs['layer_id']]

        st_img_attention_input = st_input_layernorm(img_hidden_states)
        st_text_attention_input = st_text_input_layernorm(text_hidden_states)
        st_img_attention_input = modulate(st_img_attention_input, shift_msta, scale_msta) # (b,(t n),d)
        st_text_attention_input = modulate(st_text_attention_input, text_shift_msta, text_scale_msta) # (b,n,d)

        attention_fn = attention_fn_default
        if 'attention_fn' in layer.attention.hooks:
            attention_fn = layer.attention.hooks['attention_fn']
        st_query_key_value = self.st_query_key_value_list[kwargs['layer_id']]
        st_text_query_key_value = self.st_text_query_key_value_list[kwargs['layer_id']]
        st_dense = self.st_dense_list[kwargs['layer_id']]
        st_text_dense = self.st_text_dense_list[kwargs['layer_id']]
        st_output_dropout = self.st_output_dropout_list[kwargs['layer_id']]

        if kwargs['images'].shape[1] > 1:
            ### full attention (b,(t n),d)
            # st_img_attention_input = rearrange(st_img_attention_input, 'b (t h w) d -> b t h w d', h=self.height, w=self.width)
            # st_img_attention_input = rearrange(st_img_attention_input, 'b t (h p) (w q) d-> (b h w) (t p q) d', p=self.st_window_size, q=self.st_window_size)
            # ###
            
            st_img_qkv = st_query_key_value(st_img_attention_input) # vision (b,(t n),d)
            st_text_qkv = st_text_query_key_value(st_text_attention_input) # language (b,n,d)
            # st_text_qkv = torch.cat([st_text_qkv.unsqueeze(1)]*T, dim=1) # language (b,t,n,d)
            st_mixed_raw_layer = torch.cat((st_text_qkv, st_img_qkv), dim=1)
            # st_mixed_raw_layer = rearrange(st_mixed_raw_layer, 'b t n d -> (b n) t d')

            (mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer) = split_tensor_along_last_dim(st_mixed_raw_layer, 3)
        
            st_attention_dropout = self.st_attention_dropout_list[kwargs['layer_id']]
            dropout_fn = st_attention_dropout if self.training else None
            query_layer = layer.attention._transpose_for_scores(mixed_query_layer)
            key_layer = layer.attention._transpose_for_scores(mixed_key_layer)
            value_layer = layer.attention._transpose_for_scores(mixed_value_layer)

            if self.qk_ln:
                st_query_layernorm = self.st_query_layernorm_list[kwargs['layer_id']]
                st_key_layernorm = self.st_key_layernorm_list[kwargs['layer_id']]
                query_layer = st_query_layernorm(query_layer)
                key_layer = st_key_layernorm(key_layer)

            context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kwargs)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (layer.attention.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)
        
            # context_layer = rearrange(context_layer, '(b n) t d -> b n t d', b=B)
            img_output = context_layer[:,text_length:] # vision (b,(t n),d)
            text_output = context_layer[:,:text_length] # language (b,n,d)
            # text_output = text_output[:,:,0] # language (b,n,d)

            st_img_attention_output = st_dense(img_output)
            st_text_attention_output = st_text_dense(text_output)
            if self.training:
                st_img_attention_output = st_output_dropout(st_img_attention_output) # vision (b,(t n),d)
                st_text_attention_output = st_output_dropout(st_text_attention_output) # language (b,n,d)
            # st_img_attention_output = rearrange(st_img_attention_output, 'b n t d -> b (t n) d')
        else:
            st_img_qkv = st_query_key_value(st_img_attention_input) # vision ((b,(1 n),d)
            st_text_qkv = st_text_query_key_value(st_text_attention_input) # language (b,n,d)
            st_mixed_raw_layer = torch.cat((st_text_qkv, st_img_qkv), dim=1)
            (mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer) = split_tensor_along_last_dim(st_mixed_raw_layer, 3)
            
            img_value_layer = mixed_value_layer[:,text_length:] # vision ((b,(1 n),d)
            text_value_layer = mixed_value_layer[:,:text_length] # language (b,n,d)
            st_img_attention_output = st_dense(img_value_layer)
            st_text_attention_output = st_text_dense(text_value_layer)
            if self.training:
                st_img_attention_output = st_output_dropout(st_img_attention_output) # vision ((b,(1 n),d)
                st_text_attention_output = st_output_dropout(st_text_attention_output) # language (b,n,d)

        img_hidden_states = img_hidden_states + gate_msta * st_img_attention_output # vision ((b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_msta * st_text_attention_output # language (b,n,d)

        # mlp (b,(t n),d)
        text_post_attention_layernorm = self.text_post_attention_layernorm_list[kwargs['layer_id']]
        img_mlp_input = layer.post_attention_layernorm(img_hidden_states) # vision (b,(t n),d)
        text_mlp_input = text_post_attention_layernorm(text_hidden_states) # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
        kwargs['text_mlp_input'] = text_mlp_input
        kwargs['T'] = T

        mlp_output = layer.mlp(img_mlp_input, **kwargs)
        img_mlp_output = mlp_output[:,text_length:] # vision (b,(t n),d)
        text_mlp_output = mlp_output[:,:text_length] # language (b,n,d)
        if self.transformer.layernorm_order == 'sandwich':
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output # vision (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output # language (b,n,d)
        
        img_hidden_states = rearrange(img_hidden_states, 'b (t n) d -> b t n d', t=T)
        text_hidden_states = torch.cat([text_hidden_states.unsqueeze(1)]*T, dim=1)
        hidden_states = torch.cat((text_hidden_states, img_hidden_states), dim=2)

        return hidden_states

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    def attention_forward(self, hidden_states, mask, *args, **kwargs):
        mixin_self = self
        self = self.transformer.layers[kwargs['layer_id']].attention
        img_hidden_states = hidden_states # ((b t),n,d)
        text_hidden_states = kwargs['text_attention_input'] # (b,(1 n),d)
        attention_fn = attention_fn_default
        if 'attention_fn' in self.hooks:
            attention_fn = self.hooks['attention_fn']

        text_length = kwargs['text_length']
        text_query_key_value = mixin_self.text_query_key_value_list[kwargs['layer_id']]

        img_qkv = self.query_key_value(img_hidden_states)  # vision_mixed_raw_layer ((b t),n,d)
        text_qkv = text_query_key_value(text_hidden_states)  # language_mixed_raw_layer (b,(1 n),d)
        T = len(img_qkv) // len(text_qkv)
        text_qkv = torch.cat([text_qkv.unsqueeze(1)]*T, dim=1) # language_mixed_raw_layer (b,t,n,d)
        text_qkv = rearrange(text_qkv, 'b t n d -> (b t) n d')
        mixed_raw_layer = torch.cat((text_qkv, img_qkv), dim=1) 

        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        
        if mixin_self.qk_ln:
            query_layernorm = mixin_self.query_layernorm_list[kwargs['layer_id']]
            key_layernorm = mixin_self.key_layernorm_list[kwargs['layer_id']]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kwargs)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        text_dense = mixin_self.text_dense_list[kwargs['layer_id']]

        img_output = context_layer[:,text_length:] # vision ((b t),n,d)
        text_output = context_layer[:,:text_length] # language ((b t),n,d)
        text_output = rearrange(text_output, '(b t) n d -> b t n d', t=T)
        text_output = text_output[:,0] # language ((b 1),n,d)
        img_output = self.dense(img_output)  # vision ((b t),n,d)
        text_output = text_dense(text_output)  # language ((b 1),n,d)

        if self.training:
            img_output = self.output_dropout(img_output)
            text_output = self.output_dropout(text_output)

        return text_output, img_output  # text, img
    
    def mlp_forward(self, hidden_states, **kwargs):
        mixin_self = self
        self = self.transformer.layers[kwargs['layer_id']].mlp
        text_hidden_states = kwargs['text_mlp_input']
        img_hidden_states = hidden_states

        text_dense_h_to_4h = mixin_self.text_dense_h_to_4h_list[kwargs['layer_id']]
        text_dense_4h_to_h = mixin_self.text_dense_4h_to_h_list[kwargs['layer_id']]
       
        text_intermediate_parallel = text_dense_h_to_4h(text_hidden_states)
        text_intermediate_parallel = self.activation_func(text_intermediate_parallel)
        text_output = text_dense_4h_to_h(text_intermediate_parallel)  # language_output (b,n,d)

        img_intermediate_parallel = self.dense_h_to_4h(img_hidden_states)
        img_intermediate_parallel = self.activation_func(img_intermediate_parallel)
        img_output = self.dense_4h_to_h(img_intermediate_parallel)  # vision_output (b,(t n),d)

        if self.training:
            text_output = self.dropout(text_output)
            img_output = self.dropout(img_output)

        T = kwargs['T']
        mlp_output = torch.cat((text_output, img_output), dim=1) # (b,(n_t+t*n_i),d)
        return mlp_output


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
        **kwargs
    ):
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.patch_size = patch_size
        self.num_frames = num_frames
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
                                                            compressed_num_frames=(self.num_frames+1)//self.patch_size, hidden_size=self.hidden_size), reinit=True)
        
        patch_embed_config = module_configs.pop('patch_embed_config')
        self.add_mixin('patch_embed', instantiate_from_config(patch_embed_config, patch_size=self.patch_size, hidden_size=self.hidden_size, in_channels=self.in_channels), reinit=True)
        if self.input_time == 'adaln':
            adaln_layer_config = module_configs.pop('adaln_layer_config')
            self.add_mixin('adaln_layer', instantiate_from_config(adaln_layer_config, height=self.latent_height//self.patch_size, width=self.latent_width//self.patch_size,
                                                                  hidden_size=self.hidden_size, num_layers=self.num_layers, compressed_num_frames=(self.num_frames+1)//self.patch_size,
                                                                  hidden_size_head=self.hidden_size//self.num_attention_heads, time_embed_dim=self.time_embed_dim,
                                                                  elementwise_affine=self.elementwise_affine))
        else:
            raise NotImplementedError
        final_layer_config = module_configs.pop('final_layer_config')
        self.add_mixin('final_layer', instantiate_from_config(final_layer_config, hidden_size=self.hidden_size, patch_size=self.patch_size, spatial_length=self.spatial_length,
                                                              out_channels=self.out_channels, num_frames=self.num_frames+1, time_embed_dim=self.time_embed_dim,
                                                              latent_width=self.latent_width, latent_height=self.latent_height,
                                                              elementwise_affine=self.elementwise_affine), reinit=True)

        return
    
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        b,t = x.shape[:2]
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        kwargs['images'] = x
        kwargs['emb'] = emb
        kwargs['encoder_outputs'] = context
        kwargs['text_length'] = context.shape[1]

        kwargs['input_ids'] = kwargs['position_ids'] = kwargs['attention_mask'] = torch.ones((1, 1)).to(x.dtype)
        output = super().forward(**kwargs)[0]
        return output

