from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs,
    ):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        if "return_cache" in additional_model_inputs and additional_model_inputs["return_cache"]:
            model_output, *output_per_layers = network(input * c_in, c_noise, cond, **additional_model_inputs)
            return model_output * c_out + input * c_skip, output_per_layers
        else:
            model_output = network(input * c_in, c_noise, cond, **additional_model_inputs)
            return model_output * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.sigmas = sigmas
        # self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise


class DiscreteDenoiser_TASD(DiscreteDenoiser):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def sigma_to_idx(self, sigma):
        dists = sigma.unsqueeze(0) - self.sigmas.to(sigma.device)[:, None, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)
