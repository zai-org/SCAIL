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
            append_emb=False,
            add_emb=False,
            reg_token_num=0,
            text_hidden_size=None,
            compress_frame=True,
            pad_first=False,
    ):
        super().__init__()
        self.compress_frame = compress_frame
        self.proj_space = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        if self.compress_frame:
            self.proj_time = nn.Conv3d(hidden_size, hidden_size, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=bias)
        else:
            self.proj_time = None
        self.append_emb = append_emb
        self.add_emb = add_emb

        self.reg_token_num  = reg_token_num
        if reg_token_num > 0:
            self.register_parameter('reg_token_emb', nn.Parameter(torch.zeros(reg_token_num, hidden_size)))
            nn.init.normal_(self.reg_token_emb, mean=0., std=0.02)

        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        else:
            self.text_proj = None
        self.pad_first = pad_first

    def word_embedding_forward(self, input_ids, **kwargs):
        # now is 3d patch
        images = kwargs["images"] # (b,t,c,h,w)
        B, T = images.shape[:2]
        images = images.view(-1, *images.shape[2:]) # ((b t),c,h,w)
        emb = self.proj_space(images) # ((b t),d,h/2,w/2)
        emb = rearrange(emb, '(b t) d h w -> b t d h w', b=B)
        if self.pad_first:
            first_frame = emb[:, 0].clone().unsqueeze(1)  # copy first frame
            emb = torch.cat([first_frame, emb], dim=1)
            emb = rearrange(emb, 'b t d h w -> b d t h w')
            emb = self.proj_time(emb)  # (b,d,t/2,h/2,w/2)
            emb = rearrange(emb, 'b d t h w -> b t d h w')
        elif emb.shape[1] > 1 and self.compress_frame:
            emb = rearrange(emb, 'b t d h w -> b d t h w')
            emb = self.proj_time(emb) # (b,d,t/2,h/2,w/2)
            emb = rearrange(emb, 'b d t h w -> b t d h w')
        emb = emb.flatten(3).transpose(2, 3) # (b,t/2,n,d)

        if self.append_emb:
            emb = torch.cat((kwargs["emb"][:, None, :], emb), dim=1)
        if self.reg_token_num > 0:
            emb = torch.cat((self.reg_token_emb[None, ...].repeat(emb.shape[0], 1, 1), emb), dim=1)
        if self.add_emb:
            emb = emb + kwargs["emb"][:, None, :]
        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs['encoder_outputs'])
            emb = torch.cat((text_emb, emb), dim=1)

        emb = emb.contiguous()
        return emb # (b,t/2,n,d)

    def reinit(self, parent_model=None):
        w1 = self.proj_space.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.proj_space.bias, 0)
        if self.proj_time is not None:
            w2 = self.proj_time.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.proj_time.bias, 0)
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


def unpatchify(x, c, p, num_frames, w, h, rope_position_ids=None, compress_frame=True, pad_first=False, **kwargs):
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
        if pad_first:
            b = x.shape[0]
            imgs = rearrange(x, 'b (t h w) (r c p q) -> b (t r) c (h p) (w q)', b=b, h=h, w=w, c=4, r=p, p=p, q=p)

            imgs = imgs[:, 1:]
        elif kwargs['images'].shape[1] > 1: # for videos
            compressed_num_frames = num_frames // p if compress_frame else num_frames
            tp = p if compress_frame else 1
            assert h * w * compressed_num_frames == x.shape[1]

            b = x.shape[0]
            x = x.reshape(shape=(b, compressed_num_frames, h, w, tp, p, p, c))
            imgs = rearrange(x, 'n t h w r p q c -> n (t r) c (h p) (w q)')
        else: # for images
            assert h * w == x.shape[1]

            b = x.shape[0]
            x = x.reshape(shape=(b, 1, h, w, p, p, c))
            imgs = rearrange(x, 'n t h w p q c -> n t c (h p) (w q)')

    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(
            self,
            hidden_size,
            patch_size,
            num_patches,
            out_channels,
            num_frames,
            latent_width,
            latent_height,
            compress_frame=True,
            pad_first=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        spatiotemporal_output = patch_size**3 * out_channels if compress_frame else patch_size**2 * out_channels
        self.spatiotemporal_linear = nn.Linear(hidden_size, spatiotemporal_output, bias=True)
        if compress_frame and not pad_first:
            self.time_compressed_linear = nn.Linear(patch_size * patch_size * patch_size * out_channels, patch_size * patch_size * out_channels, bias=True)

        self.compress_frame = compress_frame
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames
        self.pad_first = pad_first
        assert latent_width * latent_height * num_frames // patch_size**3 == num_patches
        self.latent_width = latent_width
        self.latent_height = latent_height


    def final_forward(self, logits, **kwargs):
        x, emb = logits[:, :, -self.num_patches:, :], kwargs['emb'] # x: (b,t/2,n,d)
        x = rearrange(x, 'b t n d -> b (t n) d')
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.spatiotemporal_linear(x)
        if kwargs['images'].shape[1] == 1 and self.compress_frame and not self.pad_first:
            x = self.time_compressed_linear(x)

        return unpatchify(x, num_frames=self.num_frames, c=self.out_channels, p=self.patch_size,
                          w=self.latent_width//self.patch_size, h=self.latent_height//self.patch_size,
                          rope_position_ids=kwargs.get('rope_position_ids', None),
                          compress_frame=self.compress_frame,
                          pad_first=self.pad_first, **kwargs)

    def reinit(self, parent_model=None):
        # nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.spatiotemporal_linear.weight)
        nn.init.constant_(self.spatiotemporal_linear.bias, 0)
        if self.compress_frame and not self.pad_first:
            nn.init.xavier_uniform_(self.time_compressed_linear.weight)
            nn.init.constant_(self.time_compressed_linear.bias, 0)


class AdaLNMixin(BaseMixin):
    def __init__(
            self,
            width,
            height,
            hidden_size,
            num_layers,
            compressed_num_frames,
            qk_ln=True,
            hidden_size_head=None,
            is_decoder=True,
            params_dtype=torch.float,
            device=torch.device('cpu'),
            st_attention_dropout_prob=0,
            st_output_dropout_prob=0,
            st_window_size=8,
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
                nn.Linear(hidden_size, 9 * hidden_size)
            ) for _ in range(num_layers)
        ])

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm = LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=False)
            self.key_layernorm = LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=False)

        self.is_decoder = is_decoder

        self.st_input_layernorm = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        # for spatiotemporal attention forward
        self.st_query_key_value_list = nn.ModuleList([ColumnParallelLinear(
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

        self.st_attention_dropout_list = nn.ModuleList([torch.nn.Dropout(st_attention_dropout_prob) for _ in range(num_layers)])
        
        self.st_dense_list = nn.ModuleList([RowParallelLinear(
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
        
        self.st_output_dropout_list = nn.ModuleList([torch.nn.Dropout(st_output_dropout_prob) for _ in range(num_layers)])

    def layer_forward(
            self,
            hidden_states,
            mask,
            *args,
            **kwargs,
    ):
        layer = self.transformer.layers[kwargs['layer_id']]
        adaLN_modulation = self.adaLN_modulations[kwargs['layer_id']]
        
        shift_msa, scale_msa, gate_msa, shift_t, scale_t, gate_t, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(
            kwargs['emb']).chunk(9, dim=1)
        gate_msa, gate_t, gate_mlp = gate_msa.unsqueeze(1), gate_t.unsqueeze(1), gate_mlp.unsqueeze(1)

        B, T, N, D = hidden_states.shape # (b,t,n,d)
        
        # self spatial attention ((b t),n,d)
        hidden_states = rearrange(hidden_states, 'b t n d -> b (t n) d')
        attention_input = layer.input_layernorm(hidden_states)
        attention_input = modulate(attention_input, shift_msa, scale_msa)
        attention_input = rearrange(attention_input, 'b (t n) d -> (b t) n d', t=T)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        if self.transformer.layernorm_order == 'sandwich':
            attention_output = layer.third_layernorm(attention_output)
        attention_output = rearrange(attention_output, '(b t) n d -> b (t n) d', b=B)
        hidden_states = hidden_states + gate_msa * attention_output # (b,(t n),d)

        # self spatiotemporal attention 
        st_attention_input = self.st_input_layernorm(hidden_states)
        st_attention_input = modulate(st_attention_input, shift_t, scale_t) # (b,(t n),d)

        attention_fn = attention_fn_default
        if 'attention_fn' in layer.attention.hooks:
            attention_fn = layer.attention.hooks['attention_fn']
        st_query_key_value = self.st_query_key_value_list[kwargs['layer_id']]
        st_dense = self.st_dense_list[kwargs['layer_id']]
        st_output_dropout = self.st_output_dropout_list[kwargs['layer_id']]

        if kwargs['images'].shape[1] > 1:
            ### st window attention ((b h' w'),(t p p),d)
            st_attention_input = rearrange(st_attention_input, 'b (t h w) d -> b t h w d', h=self.height, w=self.width)
            st_attention_input = rearrange(st_attention_input, 'b t (h p) (w q) d-> (b h w) (t p q) d', p=self.st_window_size, q=self.st_window_size)
            ###
            
            st_mixed_raw_layer = st_query_key_value(st_attention_input)
            (mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer) = split_tensor_along_last_dim(st_mixed_raw_layer, 3)
        
            st_attention_dropout = self.st_attention_dropout_list[kwargs['layer_id']]
            dropout_fn = st_attention_dropout if self.training else None
            query_layer = layer.attention._transpose_for_scores(mixed_query_layer)
            key_layer = layer.attention._transpose_for_scores(mixed_key_layer)
            value_layer = layer.attention._transpose_for_scores(mixed_value_layer)

            context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kwargs)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (layer.attention.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)
        
            st_attention_output = st_dense(context_layer)
            if self.training:
                st_attention_output = st_output_dropout(st_attention_output)
            st_attention_output = rearrange(st_attention_output, '(b h w) (t p q) d -> b (t h p w q) d', p=self.st_window_size, 
                                            q=self.st_window_size, h=self.height//self.st_window_size, w=self.width//self.st_window_size)
        else:
            st_mixed_raw_layer = st_query_key_value(st_attention_input) # ((b,(t n),d)
            (mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer) = split_tensor_along_last_dim(st_mixed_raw_layer, 3)
            st_attention_output = st_dense(mixed_value_layer)
            if self.training:
                st_attention_output = st_output_dropout(st_attention_output)

        hidden_states = hidden_states + gate_t * st_attention_output # ((b,(t n),d)

        # cross attention (b,(t n),d)
        if self.is_decoder:
            cross_attention_input = layer.post_attention_layernorm(hidden_states)
            assert 'cross_attention_mask' in kwargs and 'encoder_outputs' in kwargs
            cross_attention_output = layer.cross_attention(cross_attention_input, **kwargs)
            hidden_states = hidden_states + cross_attention_output
            mlp_input = layer.post_cross_attention_layernorm(hidden_states)
        else:
            mlp_input = layer.post_attention_layernorm(hidden_states)

        # mlp (b,(t n),d)
        mlp_input = modulate(mlp_input, shift_mlp, scale_mlp)
        mlp_output = layer.mlp(mlp_input, **kwargs)
        if self.transformer.layernorm_order == 'sandwich':
            mlp_output = layer.fourth_layernorm(mlp_output)
        hidden_states = hidden_states + gate_mlp * mlp_output # (b,(t n),d)
        
        hidden_states = rearrange(hidden_states, 'b (t n) d -> b t n d', t=T)
        return hidden_states

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                     attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):

        attention_fn_default = HOOKS_DEFAULT["attention_fn"]
        if self.qk_ln:
            query_layer = self.query_layernorm(query_layer)
            key_layer = self.key_layernorm(key_layer)

        return attention_fn_default(query_layer, key_layer, value_layer, attention_mask,
                                    attention_dropout=attention_dropout,
                                    log_attention_weights=log_attention_weights,
                                    scaling_attention_score=scaling_attention_score,
                                    **kwargs)


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
        num_classes=None,
        modules={},
        input_time='adaln',
        adm_in_channels=None,
        parallel_output=True,
        compress_frame=True,
        pad_first=False,
        **kwargs
    ):
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = latent_width * latent_height * num_frames // patch_size**3 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.model_channels = hidden_size
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.is_decoder = transformer_args.is_decoder
        self.compress_frame = compress_frame
        self.pad_first = pad_first
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
        super().__init__(args=transformer_args, transformer=None, layernorm=partial(LayerNorm, elementwise_affine=False, eps=1e-6),  **kwargs)

        module_configs = modules
        self._build_modules(module_configs)


    def _build_modules(self, module_configs):
        model_channels = self.hidden_size
        # time_embed_dim = model_channels * 4
        time_embed_dim = model_channels
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

        if self.pad_first:
            compressed_num_frames = (self.num_frames+1)//self.patch_size
        else:
            compressed_num_frames = self.num_frames//self.patch_size if self.compress_frame else self.num_frames

        pos_embed_config = module_configs.pop('pos_embed_config')
        self.add_mixin('pos_embed', instantiate_from_config(pos_embed_config, height=self.latent_height//self.patch_size,
                                                            width=self.latent_width//self.patch_size,
                                                            compressed_num_frames=compressed_num_frames, hidden_size=self.hidden_size), reinit=True)
        
        patch_embed_config = module_configs.pop('patch_embed_config')
        self.add_mixin('patch_embed', instantiate_from_config(patch_embed_config,
                                                              patch_size=self.patch_size,
                                                              hidden_size=self.hidden_size,
                                                              in_channels=self.in_channels,
                                                              compress_frame=self.compress_frame,
                                                              pad_first=self.pad_first), reinit=True)
        if self.input_time == 'adaln':
            adaln_layer_config = module_configs.pop('adaln_layer_config')
            self.add_mixin('adaln_layer', instantiate_from_config(adaln_layer_config,
                                                                  height=self.latent_height//self.patch_size,
                                                                  width=self.latent_width//self.patch_size,
                                                                  hidden_size=self.hidden_size, num_layers=self.num_layers,
                                                                  compressed_num_frames=compressed_num_frames,
                                                                  hidden_size_head=self.hidden_size//self.num_attention_heads,
                                                                  is_decoder=self.is_decoder))
        else:
            raise NotImplementedError
        final_layer_config = module_configs.pop('final_layer_config')
        self.add_mixin('final_layer', instantiate_from_config(final_layer_config, hidden_size=self.hidden_size, patch_size=self.patch_size, num_patches=self.num_patches,
                                                              out_channels=self.out_channels, num_frames=self.num_frames,
                                                              latent_width=self.latent_width, latent_height=self.latent_height,
                                                              compress_frame=self.compress_frame,
                                                              pad_first=self.pad_first), reinit=True)

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
        kwargs['cross_attention_mask'] = torch.ones(context.shape[:2], dtype=x.dtype)

        kwargs['input_ids'] = kwargs['position_ids'] = kwargs['attention_mask'] = torch.ones((1, 1)).to(x.dtype)
        output = super().forward(**kwargs)[0]
        return output

