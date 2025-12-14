from functools import partial, reduce
from operator import mul
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from sat import mpu
from sat.model.base_model import BaseModel, non_conflict
from sat.model.mixins import BaseMixin
from sat.mpu.layers import (
    ColumnParallelLinear,
    get_model_parallel_world_size,
)
from sat.mpu.utils import (
    divide,
    split_tensor_along_last_dim,
    unscaled_init_method,
)
from sat.ops.layernorm import LayerNorm
from sat.transformer_defaults import attention_fn_default

from sgm.modules.diffusionmodules.openaimodel import Timestep
from sgm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
    zero_module,
)
from sgm.util import instantiate_from_config

class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True, tp_shared=False):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        if tp_shared:
            self.weight.register_hook(self._reduce_grads)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.elementwise_affine:
            hidden_states = self.weight * hidden_states
        return (hidden_states).to(input_dtype)

    def _reduce_grads(self, grad):
        if mpu.get_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(grad, group=mpu.get_model_parallel_group())
        return grad


class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(self, in_channels, hidden_size, patch_size, bias=True, use_conv=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_conv = use_conv
        if use_conv:
            self.proj = nn.Conv3d(
                in_channels,
                hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
            self.proj_pose = nn.Conv3d(
                in_channels,
                hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        else:
            self.proj = nn.Linear(in_channels * reduce(mul, patch_size), hidden_size)

    def word_embedding_forward(self, input_ids, **kwargs):
        # now is 3d patch
        images = kwargs["images"]  # (b,t,c,h,w)
        ref_concat = kwargs["ref_concat"]  # (b,t,c,h,w)
        pose_emb = kwargs["concat_smpl_render"]  # (b,t,c,h,w)
        if self.use_conv:
            # emb = rearrange(images, 'b t c h w -> b c t h w') # (b,c,t+1,h,w)
            # emb = self.proj(emb) # (b,c,t,h/2,w/2)
            # emb_ref = rearrange(ref_concat, 'b t c h w -> b c t h w')
            # emb_ref = self.proj_ref(emb_ref)
            # emb = torch.cat([emb_ref, emb], dim=2) # (b,c,t+1,h/2,w/2)
            emb = rearrange(
                torch.cat([ref_concat, images], dim=1), "b t c h w -> b c t h w"
            )  # (b,c,t+1,h,w)
            emb = self.proj(emb)  # (b,c,t,h/2,w/2)
            pose_emb = rearrange(pose_emb, "b t c h w -> b c t h w")
            pose_emb = self.proj_pose(pose_emb)
            # print(f"debug: emb.shape {emb.shape}")
            # print(f"debug: pose_emb.shape {pose_emb.shape}")
            emb = torch.cat(
                [
                    rearrange(emb, "b c t h w -> b (t h w) c"),
                    rearrange(pose_emb, "b c t h w -> b (t h w) c"),
                ],
                dim=1,
            )

        else:
            raise NotImplementedError("Not implemented")

        emb = emb.contiguous()
        return emb  # (b,n_t+t*n_i,d)

    def reinit(self, parent_model=None):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)
        del self.transformer.word_embeddings


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    t_size,
    cls_token=False,
    height_interpolation=1.0,
    width_interpolation=1.0,
    time_interpolation=1.0,
):
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
    grid_h = np.arange(grid_height, dtype=np.float32) / height_interpolation
    grid_w = np.arange(grid_width, dtype=np.float32) / width_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32) / time_interpolation
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_height * grid_width, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    # pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    return pos_embed  # [T, H*W, D]


def get_2d_sincos_pos_embed(
    embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0
):
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
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
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
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Basic2DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self, height, width, compressed_num_frames, hidden_size, text_length=0
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.spatial_length = height * width
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.spatial_length), int(hidden_size)),
            requires_grad=False,
        )

    def position_embedding_forward(self, position_ids, **kwargs):
        return self.pos_embedding

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embedding.shape[-1], self.height, self.width
        )
        self.pos_embedding.data[:, -self.spatial_length :].copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )


class Basic3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.compressed_num_frames = compressed_num_frames
        self.spatial_length = height * width
        self.num_patches = height * width * compressed_num_frames
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(self.num_patches), int(hidden_size)), requires_grad=False
        )
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation
        raise NotImplementedError(
            "Bug not solved yet, please use Rotary3DPositionEmbeddingMixin instead"
        )

    def position_embedding_forward(self, position_ids, **kwargs):
        if kwargs["images"].shape[1] == 1:
            return self.pos_embedding[:, self.spatial_length]

        return self.pos_embedding[:, kwargs["seq_length"]]

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embedding.shape[-1],
            self.height,
            self.width,
            self.compressed_num_frames,
            height_interpolation=self.height_interpolation,
            width_interpolation=self.width_interpolation,
            time_interpolation=self.time_interpolation,
        )
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed = rearrange(pos_embed, "t n d -> (t n) d")
        self.pos_embedding.data[:, -self.num_patches :].copy_(pos_embed)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (
        "invalid dimensions for broadcastable concatenation"
    )
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def rotate_half_false(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


from sat.mpu.ulysses_attn_layer import UlyssesAttention


class UlyessAttentionMixin(BaseMixin):
    def __init__(self):
        super().__init__()
        self.ulysses_attention = UlyssesAttention()

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        **kwargs,
    ):
        fn = partial(
            attention_fn_default,
            None,
            None,
            None,
            attention_mask,
            attention_dropout,
            log_attention_weights,
            scaling_attention_score,
        )
        return self.ulysses_attention(
            query_layer, key_layer, value_layer, fa_fn=fn, need_transposed=True
        )


class Rotary3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        hidden_size_head,
        theta=10000,
        rot_v=False,
        pnp=False,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
        learnable_pos_embed=False,
        patch_size=None,
        interleaved_rope=False,
    ):
        super().__init__()
        self.rot_v = rot_v
        self.interleaved_rope = interleaved_rope

        dim_t = hidden_size_head - 4 * (hidden_size_head // 6)
        dim_h = (hidden_size_head // 6) * 2
        dim_w = (hidden_size_head // 6) * 2

        # scale = 4
        # height = height * scale
        # width = width * scale
        # compressed_num_frames = compressed_num_frames * scale

        # 'lang':
        freqs_t = 1.0 / (
            theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t)
        )
        freqs_h = 1.0 / (
            theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h)
        )
        freqs_w = 1.0 / (
            theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w)
        )

        grid_t = torch.arange(1, compressed_num_frames + 1, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)  # 多加一点，防止越界
        grid_w = torch.arange(width + 120, dtype=torch.float32)

        grid_extended_t = torch.tensor([0], dtype=torch.float32)
        grid_extended_h = torch.arange(height, dtype=torch.float32)
        grid_extended_w = torch.arange(width, dtype=torch.float32)

        # grid_extended_t_2 = torch.arange(2, compressed_num_frames + 2, dtype=torch.float32)
        # grid_extended_h_2 = torch.arange(90, height + 90, dtype=torch.float32)
        # grid_extended_w_2 = torch.arange(90, width + 90, dtype=torch.float32)

        freqs_extended_t = torch.einsum("..., f -> ... f", grid_extended_t, freqs_t)
        freqs_extended_h = torch.einsum("..., f -> ... f", grid_extended_h, freqs_h)
        freqs_extended_w = torch.einsum("..., f -> ... f", grid_extended_w, freqs_w)

        # freqs_extended_t_2 = torch.einsum('..., f -> ... f', grid_extended_t_2, freqs_t)
        # freqs_extended_h_2 = torch.einsum('..., f -> ... f', grid_extended_h_2, freqs_h)
        # freqs_extended_w_2 = torch.einsum('..., f -> ... f', grid_extended_w_2, freqs_w)

        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        if self.interleaved_rope:
            freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
            freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
            freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
            freqs = broadcat(
                (
                    freqs_t[:, None, None, :],
                    freqs_h[None, :, None, :],
                    freqs_w[None, None, :, :],
                ),
                dim=-1,
            )

            freqs_extended_t = repeat(freqs_extended_t, "... n -> ... (n r)", r=2)
            freqs_extended_h = repeat(freqs_extended_h, "... n -> ... (n r)", r=2)
            freqs_extended_w = repeat(freqs_extended_w, "... n -> ... (n r)", r=2)
            freqs_extended = broadcat(
                (
                    freqs_extended_t[:, None, None, :],
                    freqs_extended_h[None, :, None, :],
                    freqs_extended_w[None, None, :, :],
                ),
                dim=-1,
            )

            # freqs_extended_t_2 = repeat(freqs_extended_t_2, '... n -> ... (n r)', r = 2)
            # freqs_extended_h_2 = repeat(freqs_extended_h_2, '... n -> ... (n r)', r = 2)
            # freqs_extended_w_2 = repeat(freqs_extended_w_2, '... n -> ... (n r)', r = 2)
            # freqs_extended_2 = broadcat((freqs_extended_t_2[:,None,None,:], freqs_extended_h_2[None,:,None,:], freqs_extended_w_2[None,None,:,:]), dim=-1)
        else:
            freqs = broadcat(
                (
                    freqs_t[:, None, None, :],
                    freqs_h[None, :, None, :],
                    freqs_w[None, None, :, :],
                ),
                dim=-1,
            )
            # (T H W D)
            freqs = repeat(freqs, "... n -> ... (r n)", r=2)
            # (T H W D)
            freqs_extended = broadcat(
                (
                    freqs_extended_t[:, None, None, :],
                    freqs_extended_h[None, :, None, :],
                    freqs_extended_w[None, None, :, :],
                ),
                dim=-1,
            )
            # (T H W D)
            freqs_extended = repeat(freqs_extended, "... n -> ... (r n)", r=2)
            # # (T H W D)
            # freqs_extended_2 = broadcat((freqs_extended_t_2[:, None, None, :], freqs_extended_h_2[None, :, None, :], freqs_extended_w_2[None, None, :, :]), dim=-1)
            # # (T H W D)
            # freqs_extended_2 = repeat(freqs_extended_2, '... n -> ... (r n)', r=2)

        self.pnp = pnp

        # if not self.pnp:
        #     freqs = rearrange(freqs, 't h w d -> (t h w) d')

        freqs = freqs.contiguous()
        self.freqs_sin = freqs.sin().cuda()
        self.freqs_cos = freqs.cos().cuda()
        self.freqs_extended_sin = freqs_extended.sin().cuda()
        self.freqs_extended_cos = freqs_extended.cos().cuda()
        # self.freqs_extended_2_sin = freqs_extended_2.sin().cuda()
        # self.freqs_extended_2_cos = freqs_extended_2.cos().cuda()
        # self.register_buffer('freqs_sin', freqs_sin)
        # self.register_buffer('freqs_cos', freqs_cos)
        # torch.register_after_fork(self, self._set_freqs)

        # freqs_cos = freqs.contiguous().cos()
        # freqs_sin = freqs.contiguous().sin()
        # self.freqs_cos = freqs_cos
        # self.freqs_sin = freqs_sin

    def rotary(self, t, **kwargs):
        if self.pnp:
            t_coords = kwargs["rope_position_ids"][:, :, 0]
            x_coords = kwargs["rope_position_ids"][:, :, 1]
            y_coords = kwargs["rope_position_ids"][:, :, 2]
            # 创建布尔掩码，标记哪些位置的 coords 值不为 -1
            mask = (x_coords != -1) & (y_coords != -1) & (t_coords != -1)
            # 仅对 mask 为 True 的位置应用频率编码
            freqs = torch.zeros(
                [t.shape[0], t.shape[2], t.shape[3]], dtype=t.dtype, device=t.device
            )
            freqs[mask] = self.freqs[t_coords[mask], x_coords[mask], y_coords[mask]]
            freqs = freqs.unsqueeze(1)
            # freqs_cos = self.freqs_cos[t_coords[mask], x_coords[mask], y_coords[mask]].unsqueeze(1)
            # freqs_sin = self.freqs_sin[t_coords[mask], x_coords[mask], y_coords[mask]].unsqueeze(1)

        else:

            def reshape_freq(freqs):
                freqs = freqs[
                    : kwargs["rope_T"],
                    kwargs["rope_H_shift"] : kwargs["rope_H"] + kwargs["rope_H_shift"],
                    kwargs["rope_W_shift"] : kwargs["rope_W"] + kwargs["rope_W_shift"],
                ].contiguous()
                freqs = rearrange(freqs, "t h w d -> (t h w) d")
                freqs = freqs.unsqueeze(0).unsqueeze(0)
                return freqs

            freqs_cos = reshape_freq(self.freqs_cos).to(t.dtype)
            freqs_sin = reshape_freq(self.freqs_sin).to(t.dtype)

        if self.interleaved_rope:
            return t * freqs_cos + rotate_half(t) * freqs_sin
        else:
            return t * freqs_cos + rotate_half_false(t) * freqs_sin

    def rotary_ref(self, t, **kwargs):
        if self.pnp:
            t_coords = kwargs["rope_position_ids"][:, :, 0]
            x_coords = kwargs["rope_position_ids"][:, :, 1]
            y_coords = kwargs["rope_position_ids"][:, :, 2]
            # 创建布尔掩码，标记哪些位置的 coords 值不为 -1
            mask = (x_coords != -1) & (y_coords != -1) & (t_coords != -1)
            # 仅对 mask 为 True 的位置应用频率编码
            freqs = torch.zeros(
                [t.shape[0], t.shape[2], t.shape[3]], dtype=t.dtype, device=t.device
            )
            freqs[mask] = self.freqs[t_coords[mask], x_coords[mask], y_coords[mask]]
            freqs = freqs.unsqueeze(1)
            # freqs_cos = self.freqs_cos[t_coords[mask], x_coords[mask], y_coords[mask]].unsqueeze(1)
            # freqs_sin = self.freqs_sin[t_coords[mask], x_coords[mask], y_coords[mask]].unsqueeze(1)

        else:

            def reshape_freq(freqs_extended):
                # 修改: RoPE的T尺寸+1，并且H W额外做shift
                freqs = freqs_extended[
                    0:1,
                    kwargs["rope_H_shift"] : kwargs["rope_H"] + kwargs["rope_H_shift"],
                    kwargs["rope_W_shift"] : kwargs["rope_W"] + kwargs["rope_W_shift"],
                ].contiguous()
                freqs = rearrange(freqs, "t h w d -> (t h w) d")
                freqs = freqs.unsqueeze(0).unsqueeze(0)
                return freqs

            freqs_cos = reshape_freq(self.freqs_extended_cos).to(t.dtype)
            freqs_sin = reshape_freq(self.freqs_extended_sin).to(t.dtype)

        if self.interleaved_rope:
            return t * freqs_cos + rotate_half(t) * freqs_sin
        else:
            return t * freqs_cos + rotate_half_false(t) * freqs_sin

    def rotary_pose(self, t, **kwargs):
        if self.pnp:
            t_coords = kwargs["rope_position_ids"][:, :, 0]
            x_coords = kwargs["rope_position_ids"][:, :, 1]
            y_coords = kwargs["rope_position_ids"][:, :, 2]
            # 创建布尔掩码，标记哪些位置的 coords 值不为 -1
            mask = (x_coords != -1) & (y_coords != -1) & (t_coords != -1)
            # 仅对 mask 为 True 的位置应用频率编码
            freqs = torch.zeros(
                [t.shape[0], t.shape[2], t.shape[3]], dtype=t.dtype, device=t.device
            )
            freqs[mask] = self.freqs[t_coords[mask], x_coords[mask], y_coords[mask]]
            freqs = freqs.unsqueeze(1)
            # freqs_cos = self.freqs_cos[t_coords[mask], x_coords[mask], y_coords[mask]].unsqueeze(1)
            # freqs_sin = self.freqs_sin[t_coords[mask], x_coords[mask], y_coords[mask]].unsqueeze(1)

        else:

            def reshape_freq(freqs):
                freqs = freqs[
                    : kwargs["rope_T"],
                    kwargs["global_rope_H"] + kwargs["rope_H_shift"] : kwargs[
                        "global_rope_H"
                    ]
                    + kwargs["rope_H"]
                    + kwargs["rope_H_shift"],
                    kwargs["global_rope_W"] + kwargs["rope_W_shift"] : kwargs[
                        "global_rope_W"
                    ]
                    + kwargs["rope_W"]
                    + kwargs["rope_W_shift"],
                ].contiguous()
                freqs = F.avg_pool2d(
                    freqs.permute(0, 3, 1, 2), kernel_size=2, stride=2
                ).permute(
                    0, 2, 3, 1
                )  # T H W D -> T D H W -> T D H/2 W/2 -> T H/2 W/2 D
                freqs = rearrange(freqs, "t h w d -> (t h w) d")
                freqs = freqs.unsqueeze(0).unsqueeze(0)
                return freqs

            freqs_cos = reshape_freq(self.freqs_cos).to(t.dtype)
            freqs_sin = reshape_freq(self.freqs_sin).to(t.dtype)

        if self.interleaved_rope:
            return t * freqs_cos + rotate_half(t) * freqs_sin
        else:
            return t * freqs_cos + rotate_half_false(t) * freqs_sin

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings

    def position_embedding_forward(self, position_ids, **kwargs):
        return None

    @non_conflict
    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        old_impl=attention_fn_default,
        **kwargs,
    ):
        # attention_fn_default = HOOKS_DEFAULT["attention_fn"]

        if not (
            "cross_attention" in kwargs.keys() and kwargs["cross_attention"] == True
        ):  # 对带有cross_attention的输入不进行rotary, 仅对self-attention进行rotary
            query_layer = torch.cat(
                (
                    self.rotary_ref(
                        query_layer[
                            :,
                            :,
                            : kwargs["ref_length"],
                        ],
                        **kwargs,
                    ),
                    self.rotary(
                        query_layer[
                            :,
                            :,
                            kwargs["ref_length"] : kwargs["ref_length"]
                            + kwargs["seq_length"],
                        ],
                        **kwargs,
                    ),
                    self.rotary_pose(
                        query_layer[:, :, -kwargs["pose_length"] :], **kwargs
                    ),
                ),
                dim=2,
            )
            key_layer = torch.cat(
                (
                    self.rotary_ref(
                        key_layer[
                            :,
                            :,
                            : kwargs["ref_length"],
                        ],
                        **kwargs,
                    ),
                    self.rotary(
                        key_layer[
                            :,
                            :,
                            kwargs["ref_length"] : kwargs["ref_length"]
                            + kwargs["seq_length"],
                        ],
                        **kwargs,
                    ),
                    self.rotary_pose(
                        key_layer[:, :, -kwargs["pose_length"] :], **kwargs
                    ),
                ),
                dim=2,
            )
            if self.rot_v:
                value_layer = torch.cat(
                    (
                        self.rotary_ref(
                            value_layer[
                                :,
                                :,
                                : kwargs["ref_length"],
                            ],
                            **kwargs,
                        ),
                        self.rotary(
                            value_layer[
                                :,
                                :,
                                kwargs["ref_length"] : kwargs["ref_length"]
                                + kwargs["seq_length"],
                            ],
                            **kwargs,
                        ),
                        self.rotary_pose(
                            value_layer[:, :, -kwargs["pose_length"] :], **kwargs
                        ),
                    ),
                    dim=2,
                )

        return old_impl(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def unpatchify(x, c, patch_size, w, h, **kwargs):
    """
    x: (N, T/2 * S, patch_size**3 * C)
    imgs: (N, T, H, W, C)
    """

    b = x.shape[0]
    x = x[:, kwargs["ref_length"] : kwargs["ref_length"] + kwargs["seq_length"], :]
    imgs = rearrange(
        x,
        "b (t h w) (o p q c) -> b (t o) c (h p) (w q)",
        c=c,
        o=patch_size[0],
        p=patch_size[1],
        q=patch_size[2],
        t=kwargs["rope_T"],
        h=kwargs["rope_H"],
        w=kwargs["rope_W"],
    )

    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        time_embed_dim,
        patch_size,
        out_channels,
        elementwise_affine,
        layernorm_epsilon,
        share_adaln,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=elementwise_affine, eps=layernorm_epsilon
        )
        self.linear = nn.Linear(
            hidden_size, reduce(mul, patch_size) * out_channels, bias=True
        )
        self.share_adaln = share_adaln
        if not self.share_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Parameter(
                torch.randn(1, 2, hidden_size) / hidden_size**0.5
            )

    def final_forward(self, logits, **kwargs):
        x, emb = logits, kwargs["final_layer_emb"]  # x:(b,(t n),d)
        if not self.share_adaln:
            shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        else:
            shift, scale = (emb.unsqueeze(1) + self.adaLN_modulation).chunk(2, dim=1)

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return unpatchify(
            x,
            c=self.out_channels,
            patch_size=self.patch_size,
            w=kwargs["rope_W"],
            h=kwargs["rope_H"],
            **kwargs,
        )

    def reinit(self, parent_model=None):
        # nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class AdaLNMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        num_layers,
        time_embed_dim,
        compressed_num_frames,
        transformer_args,
        qk_ln=True,
        qk_ln_affine=None,
        hidden_size_head=None,
        params_dtype=torch.float,
        device=torch.device("cpu"),
        elementwise_affine=True,
        share_adaln=False,
        use_i2v_clip=False,
    ):
        super().__init__()
        layernorm_epsilon = transformer_args.layernorm_epsilon
        num_multi_query_heads = transformer_args.num_multi_query_heads
        cross_num_multi_query_heads = transformer_args.cross_num_multi_query_heads
        inner_hidden_size = transformer_args.inner_hidden_size
        num_attention_heads = transformer_args.num_attention_heads
        is_gated_mlp = transformer_args.is_gated_mlp

        self.is_gated_mlp = is_gated_mlp
        self.num_layers = num_layers
        self.compressed_num_frames = compressed_num_frames
        self.num_multi_query_heads = num_multi_query_heads
        self.cross_num_multi_query_heads = cross_num_multi_query_heads
        world_size = get_model_parallel_world_size()
        self.share_adaln = share_adaln
        self.use_i2v_clip = use_i2v_clip

        init_method = unscaled_init_method(0.02)

        if not self.share_adaln:
            self.adaLN_modulations = nn.ModuleList(
                [
                    nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 6 * hidden_size))
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.adaLN_modulations = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size**0.5)
                    for _ in range(self.num_layers)
                ]
            )

        self.qk_ln = qk_ln
        if qk_ln:
            if qk_ln_affine is None:
                qk_ln_affine = elementwise_affine
            self.query_layernorm_list = nn.ModuleList(
                [
                    RMSNorm(
                        hidden_size_head,
                        eps=layernorm_epsilon,
                        elementwise_affine=qk_ln_affine,
                        tp_shared=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    RMSNorm(
                        hidden_size_head,
                        eps=layernorm_epsilon,
                        elementwise_affine=qk_ln_affine,
                        tp_shared=True,
                    )
                    for _ in range(num_layers)
                ]
            )

            self.cross_query_layernorm_list = nn.ModuleList(
                [
                    RMSNorm(
                        hidden_size_head,
                        eps=layernorm_epsilon,
                        elementwise_affine=qk_ln_affine,
                        tp_shared=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.cross_key_layernorm_list = nn.ModuleList(
                [
                    RMSNorm(
                        hidden_size_head,
                        eps=layernorm_epsilon,
                        elementwise_affine=qk_ln_affine,
                        tp_shared=True,
                    )
                    for _ in range(num_layers)
                ]
            )

            if self.use_i2v_clip:
                self.clip_feature_key_layernorm_list = nn.ModuleList(
                    [
                        RMSNorm(
                            hidden_size_head,
                            eps=layernorm_epsilon,
                            elementwise_affine=qk_ln_affine,
                            tp_shared=True,
                        )
                        for _ in range(num_layers)
                    ]
                )

        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.num_multi_query_heads_per_partition = divide(
            num_multi_query_heads, world_size
        )
        self.cross_num_multi_query_heads_per_partition = divide(
            cross_num_multi_query_heads, world_size
        )
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        if num_multi_query_heads == 0:
            qkv_size = 3 * hidden_size
            self.stride = 3
        else:  # multi-query
            qkv_size = (
                hidden_size
                + self.hidden_size_per_attention_head * self.num_multi_query_heads * 2
            )
            self.stride = [
                self.num_attention_heads_per_partition,
                self.num_multi_query_heads_per_partition,
                self.num_multi_query_heads_per_partition,
            ]
        # Strided linear layer.
        if cross_num_multi_query_heads == 0:
            kv_size = 2 * hidden_size
        else:  # multi-query
            kv_size = (
                self.hidden_size_per_attention_head
                * self.cross_num_multi_query_heads
                * 2
            )

        if self.use_i2v_clip:
            self.clip_feature_key_value_list = nn.ModuleList(
                [
                    ColumnParallelLinear(
                        hidden_size,
                        kv_size,
                        stride=2,
                        gather_output=False,
                        init_method=init_method,
                        bias=True,
                        params_dtype=params_dtype,
                        module=self,
                        name="clip_feature_key_value",
                        skip_init=False,
                        device=device,
                    )
                    for _ in range(self.num_layers)
                ]
            )

    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        layer_id = kwargs["layer_id"]
        layer = self.transformer.layers[layer_id]
        adaLN_modulation = self.adaLN_modulations[layer_id]

        if not self.share_adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                adaLN_modulation(kwargs["emb"]).unsqueeze(1).chunk(6, dim=2)
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                kwargs["emb"].unflatten(1, (6, adaLN_modulation.shape[-1]))
                + adaLN_modulation
            ).chunk(6, dim=1)

        # self full attention (b,(t n),d)
        attention_input = layer.input_layernorm(hidden_states)
        attention_input = modulate(attention_input, shift_msa, scale_msa)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        if self.transformer.layernorm_order == "sandwich":
            attention_output = layer.third_layernorm(attention_output)
        hidden_states = hidden_states + gate_msa * attention_output

        assert self.transformer.is_decoder
        cross_attention_input = layer.post_cross_attention_layernorm(hidden_states)
        assert "cross_attention_mask" in kwargs and "encoder_outputs" in kwargs
        cross_attention_output = layer.cross_attention(cross_attention_input, **kwargs)
        hidden_states = hidden_states + cross_attention_output

        # mlp (b,(t n),d)
        mlp_input = layer.post_attention_layernorm(hidden_states)
        mlp_input = modulate(mlp_input, shift_mlp, scale_mlp)
        mlp_output = layer.mlp(mlp_input, **kwargs)
        if self.transformer.layernorm_order == "sandwich":
            mlp_output = layer.fourth_layernorm(mlp_output)
        hidden_states = hidden_states + gate_mlp * mlp_output
        return hidden_states

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    def attention_forward(self, hidden_states, mask, **kw_args):
        mixin_self = self
        self = self.transformer.layers[kw_args["layer_id"]].attention
        attention_fn = attention_fn_default
        if "attention_fn" in self.hooks:
            attention_fn = self.hooks["attention_fn"]

        mixed_raw_layer = self.query_key_value(hidden_states)
        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = (
            split_tensor_along_last_dim(mixed_raw_layer, self.stride)
        )

        if mixin_self.qk_ln:
            query_layernorm = mixin_self.query_layernorm_list[kw_args["layer_id"]]
            key_layernorm = mixin_self.key_layernorm_list[kw_args["layer_id"]]
            mixed_query_layer = query_layernorm(mixed_query_layer)
            mixed_key_layer = key_layernorm(mixed_key_layer)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # rotary position embedding
        if self.transformer.is_rotary_emb:
            query_layer, key_layer = self.transformer.position_embeddings(
                query_layer,
                key_layer,
                kw_args["position_ids"],
                max_seqlen=kw_args["position_ids"].max() + 1,
                layer_id=kw_args["layer_id"],
            )

        context_layer = attention_fn(
            query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

    def cross_attention_forward(
        self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args
    ):
        mixin_self = self
        self = self.transformer.layers[kw_args["layer_id"]].cross_attention
        attention_fn = attention_fn_default
        if "attention_fn" in self.hooks:
            attention_fn = self.hooks["attention_fn"]

        mixed_query_layer = self.query(hidden_states)
        mixed_x_layer = self.key_value(encoder_outputs)
        (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(
            mixed_x_layer, 2
        )
        if mixin_self.use_i2v_clip:
            clip_feature_key_value = mixin_self.clip_feature_key_value_list[
                kw_args["layer_id"]
            ]
            clip_feature_mixed_x_layer = clip_feature_key_value(
                kw_args["image_clip_features"]
            )
            (clip_feature_mixed_key_layer, clip_feature_mixed_value_layer) = (
                split_tensor_along_last_dim(clip_feature_mixed_x_layer, 2)
            )
        if mixin_self.qk_ln:
            query_layernorm = mixin_self.cross_query_layernorm_list[kw_args["layer_id"]]
            key_layernorm = mixin_self.cross_key_layernorm_list[kw_args["layer_id"]]
            mixed_query_layer = query_layernorm(mixed_query_layer)
            mixed_key_layer = key_layernorm(mixed_key_layer)
            if mixin_self.use_i2v_clip:
                clip_feature_key_layernorm = mixin_self.clip_feature_key_layernorm_list[
                    kw_args["layer_id"]
                ]
                clip_feature_mixed_key_layer = clip_feature_key_layernorm(
                    clip_feature_mixed_key_layer
                )

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        dropout_fn = self.attention_dropout if self.training else None
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        mem_cross = (key_layer, value_layer)
        if mixin_self.use_i2v_clip:
            clip_feature_key_layer = self._transpose_for_scores(
                clip_feature_mixed_key_layer
            )
            clip_feature_value_layer = self._transpose_for_scores(
                clip_feature_mixed_value_layer
            )
            clip_feature_mem_cross = (clip_feature_key_layer, clip_feature_value_layer)

        context_layer = attention_fn(
            query_layer,
            key_layer,
            value_layer,
            cross_attention_mask,
            dropout_fn,
            cross_attention=True,
            mem_cross=mem_cross,
            **kw_args,
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        if mixin_self.use_i2v_clip:
            clip_feature_context_layer = attention_fn(
                query_layer,
                clip_feature_key_layer,
                clip_feature_value_layer,
                cross_attention_mask,
                dropout_fn,
                cross_attention=True,
                mem_cross=clip_feature_mem_cross,
                **kw_args,
            )
            clip_feature_context_layer = clip_feature_context_layer.permute(
                0, 2, 1, 3
            ).contiguous()
            new_clip_feature_context_layer_shape = clip_feature_context_layer.size()[
                :-2
            ] + (self.hidden_size_per_partition,)
            # [b, s, hp]
            clip_feature_context_layer = clip_feature_context_layer.view(
                *new_clip_feature_context_layer_shape
            )
            if "seq_length" in kw_args and "ref_length" in kw_args:
                seq_len = kw_args["seq_length"]
                ref_len = kw_args["ref_length"]
                total_video_len = seq_len + ref_len
                if total_video_len < clip_feature_context_layer.shape[1]:
                    print(f"debug: total_video_len={total_video_len}")
                    print(f"debug: clip_feature_context_layer.shape={clip_feature_context_layer.shape}")
                    print(clip_feature_context_layer[:, total_video_len:, :].max())
                    clip_feature_context_layer[:, total_video_len:, :] = 0
            context_layer = context_layer + clip_feature_context_layer

        # Output. [b, s, h]
        output = self.dense(context_layer)
        if self.training:
            output = self.output_dropout(output)
        return output


str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


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
        text_dim,
        num_layers,
        num_attention_heads,
        elementwise_affine,
        num_multi_query_heads=0,
        cross_num_multi_query_heads=0,
        time_freq_dim=None,
        time_embed_dim=None,
        num_classes=None,
        modules={},
        input_time="adaln",
        share_adaln=False,
        adm_in_channels=None,
        parallel_output=True,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
        use_SwiGLU=False,
        use_RMSNorm=False,
        cfg_embed_dim=None,
        ofs_embed_dim=None,
        layernorm_epsilon=1e-6,
        inner_hidden_size=None,
        use_i2v_clip=False,
        **kwargs,
    ):
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.time_compressed_rate = time_compressed_rate
        self.spatial_length = (
            latent_width * latent_height // reduce(mul, patch_size[1:])
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.text_dim = text_dim
        self.model_channels = hidden_size
        self.time_embed_dim = (
            time_embed_dim if time_embed_dim is not None else hidden_size
        )
        self.time_freq_dim = (
            time_freq_dim if time_freq_dim is not None else self.time_embed_dim
        )
        self.cfg_embed_dim = cfg_embed_dim
        self.ofs_embed_dim = ofs_embed_dim
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time
        self.share_adaln = share_adaln
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.is_decoder = transformer_args.is_decoder
        self.elementwise_affine = elementwise_affine
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation
        self.layernorm_epsilon = layernorm_epsilon
        self.num_multi_query_heads = num_multi_query_heads
        self.cross_num_multi_query_heads = cross_num_multi_query_heads
        self.use_i2v_clip = use_i2v_clip
        if inner_hidden_size is not None:
            self.inner_hidden_size = inner_hidden_size
        else:
            self.inner_hidden_size = hidden_size * 4
        try:
            self.dtype = str_to_dtype[kwargs.pop("dtype")]
        except:
            self.dtype = torch.float32

        if use_SwiGLU:
            kwargs["activation_func"] = F.silu
            transformer_args.is_gated_mlp = True
        elif "activation_func" not in kwargs:
            approx_gelu = nn.GELU(approximate="tanh")
            kwargs["activation_func"] = approx_gelu
            transformer_args.is_gated_mlp = False

        if use_RMSNorm:
            kwargs["layernorm"] = partial(
                RMSNorm, elementwise_affine=elementwise_affine, eps=layernorm_epsilon
            )
        else:
            kwargs["layernorm"] = partial(
                LayerNorm, elementwise_affine=elementwise_affine, eps=layernorm_epsilon
            )

        transformer_args.num_layers = self.num_layers
        transformer_args.hidden_size = hidden_size
        transformer_args.num_attention_heads = num_attention_heads
        transformer_args.num_multi_query_heads = num_multi_query_heads
        transformer_args.cross_num_multi_query_heads = cross_num_multi_query_heads
        transformer_args.parallel_output = parallel_output
        transformer_args.layernorm_epsilon = layernorm_epsilon
        transformer_args.inner_hidden_size = self.inner_hidden_size
        transformer_args.use_final_layernorm = False
        super().__init__(args=transformer_args, transformer=None, **kwargs)

        module_configs = modules
        self._build_modules(module_configs, transformer_args)

    def _build_modules(self, module_configs, transformer_args):
        model_channels = self.hidden_size
        # time_embed_dim = model_channels * 4
        time_embed_dim = self.time_embed_dim
        self.time_embed = nn.Sequential(
            linear(self.time_freq_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.share_adaln:
            self.adaln_projection = nn.Sequential(
                nn.SiLU(), linear(time_embed_dim, model_channels * 6)
            )

        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, model_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_channels, model_channels),
        )

        if self.ofs_embed_dim is not None:
            self.ofs_embed = nn.Sequential(
                linear(self.ofs_embed_dim, self.ofs_embed_dim),
                nn.SiLU(),
                zero_module(linear(self.ofs_embed_dim, self.ofs_embed_dim)),
            )

        if self.cfg_embed_dim is not None:
            self.cfg_embed = nn.Sequential(
                linear(self.time_freq_dim, self.cfg_embed_dim),
                nn.SiLU(),
                zero_module(linear(self.cfg_embed_dim, self.cfg_embed_dim)),
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
                    linear(self.adm_in_channels, time_embed_dim),
                    nn.SiLU(),
                    linear(time_embed_dim, time_embed_dim),
                )
            else:
                raise ValueError()

        if self.use_i2v_clip:
            self.clip_proj = MLPProj(1280, self.hidden_size)
        self.add_mixin("ulysse", UlyessAttentionMixin())
        pos_embed_config = module_configs["pos_embed_config"]
        self.add_mixin(
            "pos_embed",
            instantiate_from_config(
                pos_embed_config,
                height=self.latent_height // self.patch_size[1],
                width=self.latent_width // self.patch_size[2],
                compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate
                + 1,
                hidden_size=self.hidden_size,  # 修改：不用修改，compressed_num_frames传入RoPE后再加1
                height_interpolation=self.height_interpolation,
                width_interpolation=self.width_interpolation,
                time_interpolation=self.time_interpolation,
                patch_size=self.patch_size,
            ),
            reinit=True,
        )

        patch_embed_config = module_configs["patch_embed_config"]
        self.add_mixin(
            "patch_embed",
            instantiate_from_config(
                patch_embed_config,
                patch_size=self.patch_size,
                hidden_size=self.hidden_size,
                in_channels=self.in_channels,
            ),
            reinit=True,
        )
        if self.input_time == "adaln":
            adaln_layer_config = module_configs["adaln_layer_config"]
            self.add_mixin(
                "adaln_layer",
                instantiate_from_config(
                    adaln_layer_config,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    compressed_num_frames=(self.num_frames - 1)
                    // self.time_compressed_rate
                    + 1,
                    time_embed_dim=self.time_embed_dim,
                    elementwise_affine=self.elementwise_affine,
                    transformer_args=transformer_args,
                    share_adaln=self.share_adaln,
                    use_i2v_clip=self.use_i2v_clip,
                ),
            )
        else:
            raise NotImplementedError
        final_layer_config = module_configs["final_layer_config"]
        self.add_mixin(
            "final_layer",
            instantiate_from_config(
                final_layer_config,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                time_embed_dim=self.time_embed_dim,
                elementwise_affine=self.elementwise_affine,
                layernorm_epsilon=self.layernorm_epsilon,
                share_adaln=self.share_adaln,
            ),
            reinit=True,
        )

        return

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        b, t, d, h, w = x.shape
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        assert kwargs.get("ref_concat", None) is not None, "must specify ref_concat"
        if kwargs.get("concat_images", None) is not None:  # btchw
            if kwargs["concat_images"].shape[0] != x.shape[0]:
                concat_images = kwargs["concat_images"].repeat(2, 1, 1, 1, 1)
            else:
                concat_images = kwargs["concat_images"]
            if kwargs.get("history_mask", None) is not None:
                if kwargs["history_mask"].shape[0] != x.shape[0]:
                    mask = kwargs["history_mask"].repeat(2, 1, 1, 1, 1)
                else:
                    mask = kwargs["history_mask"]
            else:
                mask = torch.zeros(b, t, 4, h, w, device=x.device, dtype=x.dtype)

            if kwargs.get("concat_cheek_hands", None) is not None:
                if kwargs["concat_cheek_hands"].shape[0] != x.shape[0]:
                    concat_cheek_hands = kwargs["concat_cheek_hands"].repeat(
                        2, 1, 1, 1, 1
                    )
                else:
                    concat_cheek_hands = kwargs["concat_cheek_hands"]

            if kwargs.get("ref_concat", None) is not None:
                if kwargs["ref_concat"].shape[0] != x.shape[0]:
                    ref_concat = kwargs["ref_concat"].repeat(2, 1, 1, 1, 1)
                else:
                    ref_concat = kwargs["ref_concat"]
                ref_concat_mask = torch.ones(
                    b, 1, 4, h, w, device=x.device, dtype=x.dtype
                )
                ref_concat = torch.cat([ref_concat, ref_concat_mask], dim=2)
                kwargs["ref_concat"] = ref_concat

            if kwargs.get("concat_smpl_render", None) is not None:
                if kwargs["concat_smpl_render"].shape[0] != x.shape[0]:
                    concat_smpl_render = kwargs["concat_smpl_render"].repeat(
                        2, 1, 1, 1, 1
                    )
                else:
                    concat_smpl_render = kwargs["concat_smpl_render"]
                pose_mask = torch.ones(
                    b, t, 4, h // 2, w // 2, device=x.device, dtype=x.dtype
                )
                kwargs["concat_smpl_render"] = torch.cat(
                    [concat_smpl_render, pose_mask], dim=2
                )

            x = torch.cat([x, mask], dim=2)
        # text
        context = self.text_embedding(context)
        # i2v clip
        if self.use_i2v_clip:
            assert kwargs.get("image_clip_features", None) is not None
            kwargs["image_clip_features"] = self.clip_proj(
                kwargs["image_clip_features"].to(x.device)
            )  # b, 257, d
            if kwargs["image_clip_features"].shape[0] != x.shape[0]:
                kwargs["image_clip_features"] = kwargs["image_clip_features"].repeat(
                    2, 1, 1
                )

        assert (y is not None) == (self.num_classes is not None), (
            "must specify y if and only if the model is class-conditional"
        )
        # with amp.autocast(dtype=torch.float32):
        t_emb = timestep_embedding(
            timesteps, self.time_freq_dim, repeat_only=False, dtype=self.dtype
        )
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            # assert y.shape[0] == x.shape[0]
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        if self.ofs_embed_dim is not None:
            ofs_emb = timestep_embedding(
                kwargs["ofs"], self.ofs_embed_dim, repeat_only=False, dtype=self.dtype
            )
            ofs_emb = self.ofs_embed(ofs_emb)
            emb = emb + ofs_emb
        if self.cfg_embed_dim is not None:
            cfg_scale = kwargs["cfg_scale"]
            cfg_scale = (
                torch.tensor([cfg_scale], device=x.device)
                if not isinstance(cfg_scale, torch.Tensor)
                else cfg_scale.to(x.device)
            )
            cfg_emb = timestep_embedding(
                cfg_scale, self.time_freq_dim, repeat_only=False, dtype=self.dtype
            )
            cfg_emb = self.cfg_embed(cfg_emb)
            emb = emb + cfg_emb

        kwargs["final_layer_emb"] = emb

        if self.share_adaln:
            # with amp.autocast(dtype=torch.float32):
            adaln_emb = self.adaln_projection(emb)

        kwargs["seq_length"] = t * h * w // reduce(mul, self.patch_size)
        kwargs["pose_length"] = t * (h // 2) * (w // 2) // reduce(mul, self.patch_size)
        kwargs["ref_length"] = 1 * h * w // reduce(mul, self.patch_size)
        kwargs["images"] = x
        kwargs["emb"] = adaln_emb if self.share_adaln else emb
        kwargs["encoder_outputs"] = context
        kwargs["cross_attention_mask"] = torch.ones(context.shape[:2], dtype=x.dtype)
        kwargs["text_length"] = context.shape[1]

        kwargs["rope_T"] = t // self.patch_size[0]
        kwargs["rope_H"] = h // self.patch_size[1]
        kwargs["rope_W"] = w // self.patch_size[2]

        kwargs["global_rope_H"] = 0
        kwargs["global_rope_W"] = 120

        kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = (
            torch.ones((1, 1)).to(x.dtype)
        )
        kwargs["rope_H_shift"] = 0
        kwargs["rope_W_shift"] = 0
        if "chunk_dim" in kwargs and kwargs["chunk_dim"] is not None:
            local_rank = mpu.get_sequence_parallel_rank()
            if kwargs["chunk_dim"] == 3:  # h
                kwargs["rope_H_shift"] = local_rank * (h // self.patch_size[1])
            elif kwargs["chunk_dim"] == 4:
                kwargs["rope_W_shift"] = (w // self.patch_size[2]) * local_rank
            else:
                raise NotImplementedError
        output = super().forward(**kwargs)[0]
        return output
