import os
import sys
import json
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union, Optional
from omegaconf import ListConfig, OmegaConf
from functools import partial
from einops import rearrange, repeat

import torch
from torch import nn
import torch.nn.functional as F

from sat.model.base_model import BaseModel
from sat.model.mixins import BaseMixin
from sat.ops.layernorm import LayerNorm
from sat.transformer_defaults import HOOKS_DEFAULT
from sat.mpu.mappings import copy_to_model_parallel_region
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
        reg_token_num=0
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.append_emb = append_emb
        self.add_emb = add_emb

        self.reg_token_num  = reg_token_num
        if reg_token_num > 0:
            self.register_parameter('reg_token_emb', nn.Parameter(torch.zeros(reg_token_num, hidden_size)))
            nn.init.normal_(self.reg_token_emb, mean=0., std=0.02)
    
    def word_embedding_forward(self, input_ids, **kwargs):
        images = kwargs["images"]
        emb = self.proj(images)
        emb = emb.flatten(2).transpose(1, 2)
        if self.append_emb:
            emb = torch.cat((kwargs["emb"][:, None, :], emb), dim=1)
        if self.reg_token_num > 0:
            emb = torch.cat((self.reg_token_emb[None, ...].repeat(emb.shape[0], 1, 1), emb), dim=1)
        if self.add_emb:
            emb = emb + kwargs["emb"][:, None, :]
        return emb

    def reinit(self, parent_model=None):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)
        del self.transformer.word_embeddings


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

class RotaryPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        num_patches,
        hidden_size,
        hidden_size_head,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        rot_v=False,
        qk_ln=False,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.rot_v = rot_v
        self.qk_ln = qk_ln

        if qk_ln:
            self.query_layernorm = LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=False)
            self.key_layernorm = LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=False)

        pt_seq_len = int(num_patches**0.5)
        dim = hidden_size_head // 2

        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        
        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        ###
        freqs = freqs.permute(2, 0, 1).flatten(1).transpose(0, 1)

        freqs_cos = freqs.contiguous().cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.contiguous().sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def rotary(self, t):
        return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin
    
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
        attention_fn_default = HOOKS_DEFAULT["attention_fn"]
        if self.qk_ln:
            query_layer = self.query_layernorm(query_layer)
            key_layer = self.key_layernorm(key_layer)
        if query_layer.shape[-2] == key_layer.shape[-2]: # only for self attention
            query_layer = torch.cat((query_layer[:, :, :-self.num_patches, :], self.rotary(query_layer[:, :, -self.num_patches:, :])), dim=2)
            key_layer = torch.cat((key_layer[:, :, :-self.num_patches, :], self.rotary(key_layer[:, :, -self.num_patches:, :])), dim=2)
            if self.rot_v:
                value_layer = torch.cat((value_layer[:, :, :-self.num_patches, :], self.rotary(value_layer[:, :, -self.num_patches:, :])), dim=2)
        
        return attention_fn_default(query_layer, key_layer, value_layer, attention_mask,
                                    attention_dropout=attention_dropout, 
                                    log_attention_weights=log_attention_weights, 
                                    scaling_attention_score=scaling_attention_score, 
                                    **kwargs)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def unpatchify(x, c, p, rope_position_ids=None):
    """
    x: (N, T, patch_size**2 * C)
    imgs: (N, H, W, C)
    """
    if rope_position_ids is not None:
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum('nlpqc->ncplq', x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    
    return imgs

class FinalLayerMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        patch_size,
        num_patches,
        out_channels,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def final_forward(self, logits, **kwargs):
        x, emb = logits[:, -self.num_patches:, :], kwargs['emb']
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return unpatchify(x, c=self.out_channels, p=self.patch_size, rope_position_ids=kwargs.get('rope_position_ids', None))

    def reinit(self, parent_model=None):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.linear.weight, 0)
        # nn.init.constant_(self.linear.bias, 0)
        # nn.init.xavier_uniform_(self.adaLN_modulation[-1].weight)
        # nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

class AdaLNMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        num_layers,
        nogate=False,
        cross_adaln=False,
        skip_connection=False,
    ):
        super().__init__()
        self.nogate = nogate
        self.cross_adaln = cross_adaln
        if nogate:
            out_times = 4
        else:
            out_times = 6
        if cross_adaln:
            out_times = (out_times // 2) * 3
            
        self.adaLN_modulations = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, out_times * hidden_size)
            ) for _ in range(num_layers)
        ])

        if skip_connection:
            self.skip_connection = nn.ModuleList([
                nn.Linear(hidden_size * 2, hidden_size) for _ in range(num_layers)
            ])
        else:
            self.skip_connection = None
    
    def layer_forward(
        self, 
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        layer = self.transformer.layers[kwargs['layer_id']]
        adaLN_modulation = self.adaLN_modulations[kwargs['layer_id']]
        if self.nogate and self.cross_adaln:
            shift_msa, scale_msa, shift_cross, scale_cross, shift_mlp, scale_mlp = adaLN_modulation(kwargs['emb']).chunk(6, dim=1)
            gate_msa = gate_cross = gate_mlp = 1
        elif self.nogate and not self.cross_adaln:
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_modulation(kwargs['emb']).chunk(4, dim=1)
            gate_msa = gate_mlp = 1
        elif not self.nogate and self.cross_adaln:
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(kwargs['emb']).chunk(9, dim=1)
            gate_msa, gate_cross, gate_mlp = gate_msa.unsqueeze(1), gate_cross.unsqueeze(1), gate_mlp.unsqueeze(1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(kwargs['emb']).chunk(6, dim=1)
            gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        
        if self.skip_connection is not None:
            assert 'hs' in kwargs
            hidden_states = torch.concat((hidden_states, kwargs['hs'][kwargs['layer_id']]), dim=-1)
            hidden_states = self.skip_connection[kwargs['layer_id']](hidden_states)

        # self attention
        attention_input = layer.input_layernorm(hidden_states)
        attention_input = modulate(attention_input, shift_msa, scale_msa)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        if self.transformer.layernorm_order == 'sandwich':
            attention_output = layer.third_layernorm(attention_output)
        hidden_states = hidden_states + gate_msa * attention_output

        cross_attention_input = layer.post_attention_layernorm(hidden_states)
        if self.cross_adaln:
            cross_attention_input = modulate(cross_attention_input, shift_cross, scale_cross)
        assert 'cross_attention_mask' in kwargs and 'encoder_outputs' in kwargs
        cross_attention_output = layer.cross_attention(cross_attention_input, **kwargs)
        if self.cross_adaln:
            cross_attention_output = gate_cross * cross_attention_output
        hidden_states = hidden_states + cross_attention_output
        mlp_input = layer.post_cross_attention_layernorm(hidden_states)

        mlp_input = modulate(mlp_input, shift_mlp, scale_mlp)
        mlp_output = layer.mlp(mlp_input, **kwargs)
        if self.transformer.layernorm_order == 'sandwich':
            mlp_output = layer.fourth_layernorm(mlp_output)
        hidden_states = hidden_states + gate_mlp * mlp_output
        return hidden_states
    
    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

str_to_dtype = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

class DiffusionTransformer(BaseModel):
    def __init__(
        self, 
        transformer_args,
        image_size,
        patch_size,
        in_channels,
        out_channels,
        hidden_size,
        num_classes=None,
        modules={},
        input_time='adaln',
        adm_in_channels=None,
        parallel_output=True,
        **kwargs
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.model_channels = hidden_size
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time

        try:
            self.dtype = str_to_dtype[kwargs.pop('dtype')]
        except:
            self.dtype = torch.float32

        if 'activation_func' not in kwargs:
            approx_gelu = nn.GELU(approximate='tanh')
            kwargs['activation_func'] = approx_gelu
        super().__init__(args=transformer_args, transformer=None, parallel_output=parallel_output, layernorm=partial(LayerNorm, elementwise_affine=False, eps=1e-6),  **kwargs)

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
            
        pos_embed_config = module_configs.pop('pos_embed_config')
        self.add_mixin('pos_embed', instantiate_from_config(pos_embed_config), reinit=True)
        

        patch_embed_config = module_configs.pop('patch_embed_config')
        self.add_mixin('patch_embed', instantiate_from_config(patch_embed_config), reinit=True)
        if self.input_time == 'adaln':
            adaln_layer_config = module_configs.pop('adaln_layer_config')
            self.add_mixin('adaln_layer', instantiate_from_config(adaln_layer_config))
        else:
            raise NotImplementedError
        final_layer_config = module_configs.pop('final_layer_config')
        self.add_mixin('final_layer', instantiate_from_config(final_layer_config))

        return
    
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
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

        kwargs['input_ids'] = kwargs['position_ids'] = kwargs['attention_mask'] = torch.ones((1,1)).to(x.dtype)

        output = super().forward(**kwargs)[0]
        return output


class EncoderFinalMixin(BaseMixin):
    def final_forward(self, logits, **kwargs):
        logits = copy_to_model_parallel_region(logits)
        return logits
    
class DecoderWordEmbeddingMixin(BaseMixin):
    def word_embedding_forward(self, input_ids, **kwargs):
        return kwargs["images"]


class EncDecDiffusionTransformer(nn.Module):
    def __init__(
        self, 
        transformer_args,
        image_size,
        patch_size,
        in_channels,
        out_channels,
        hidden_size,
        num_classes=None,
        modules={},
        input_time='adaln',
        adm_in_channels=None,
        parallel_output=True,
        **kwargs
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.model_channels = hidden_size
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time

        try:
            self.dtype = str_to_dtype[kwargs.pop('dtype')]
        except:
            self.dtype = torch.float32
        if 'activation_func' not in kwargs:
            approx_gelu = nn.GELU(approximate='tanh')
            kwargs['activation_func'] = approx_gelu

        self.encoder = BaseModel(
            args=transformer_args, 
            transformer=None, 
            parallel_output=parallel_output, 
            layernorm=partial(LayerNorm, elementwise_affine=False, eps=1e-6),  
            **kwargs)
        
        self.decoder = BaseModel(
            args=transformer_args, 
            transformer=None, 
            parallel_output=parallel_output, 
            layernorm=partial(LayerNorm, elementwise_affine=False, eps=1e-6),  
            **kwargs)

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

        self.encoder.add_mixin('final', EncoderFinalMixin())
        self.decoder.add_mixin('word_embed', DecoderWordEmbeddingMixin())

        pos_embed_config = module_configs.pop('pos_embed_config')
        self.encoder.add_mixin('pos_embed', instantiate_from_config(pos_embed_config), reinit=True)
        self.decoder.add_mixin('pos_embed', instantiate_from_config(pos_embed_config), reinit=True)

        patch_embed_config = module_configs.pop('patch_embed_config')
        self.encoder.add_mixin('patch_embed', instantiate_from_config(patch_embed_config), reinit=True)

        if self.input_time == 'adaln':
            adaln_layer_config = module_configs.pop('adaln_layer_config')
            self.encoder.add_mixin('adaln_layer', instantiate_from_config(adaln_layer_config))
            
            decoder_adaln_layer_config = adaln_layer_config
            # decoder_adaln_layer_config['params']['skip_connection'] = True
            self.decoder.add_mixin('adaln_layer', instantiate_from_config(decoder_adaln_layer_config))
        else:
            raise NotImplementedError
        
        final_layer_config = module_configs.pop('final_layer_config')
        self.decoder.add_mixin('final_layer', instantiate_from_config(final_layer_config))

        return

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        input_ids = position_ids = attention_mask = torch.ones((1,1)).to(x.dtype)

        encoder_outputs = self.encoder(
            images=x,
            emb=emb,
            encoder_outputs=context,
            cross_attention_mask=torch.ones(context.shape[:2], dtype=x.dtype),
            # output_hidden_states=True
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask,
        )

        x = encoder_outputs[0]
        decoder_outputs = self.decoder(
            images=x,
            emb=emb,
            encoder_outputs=context,
            cross_attention_mask=torch.ones(context.shape[:2], dtype=x.dtype),
            # hs=[outputs['hidden_states'] for outputs in encoder_outputs[1:][::-1]],
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask,
        )
        return decoder_outputs[0]
             