import math
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union, Optional
from omegaconf import ListConfig, OmegaConf


import torch
from torch import nn
import torch.nn.functional as F

from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
    SeededNoise
)

from sat import mpu
from sat.helpers import print_rank0

class SATVideoDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get('log_keys', None)
        input_key = model_config.get('input_key', 'mp4')
        network_config = model_config.get('network_config', None)
        network_wrapper = model_config.get('network_wrapper', None)
        denoiser_config = model_config.get('denoiser_config', None)
        sampler_config = model_config.get('sampler_config', None)
        conditioner_config = model_config.get('conditioner_config', None)
        first_stage_config = model_config.get('first_stage_config', None)
        loss_fn_config = model_config.get('loss_fn_config', None)
        scale_factor = model_config.get('scale_factor', 1.0)
        disable_first_stage_autocast = model_config.get('disable_first_stage_autocast', False)
        no_cond_log = model_config.get('disable_first_stage_autocast', False)
        untrainable_prefixs = model_config.get('untrainable_prefixs', ['first_stage_model', 'conditioner'])
        compile_model = model_config.get('compile_model', False)
        en_and_decode_n_samples_a_time = model_config.get('en_and_decode_n_samples_a_time', None)
        lr_scale = model_config.get('lr_scale', None)
        lora_train = model_config.get('lora_train', False)

        self.log_keys = log_keys
        self.input_key = input_key
        self.untrainable_prefixs = untrainable_prefixs
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.lora_train = lora_train

        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        network_config['params']['dtype'] = dtype_str
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )

        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

        from sgm.modules.diffusionmodules.sampling import SeededNoiseEDMSampler
        if isinstance(self.sampler, SeededNoiseEDMSampler):
            self.seeded_noise = SeededNoise(seeds=[seed + 1234 for seed in self.sampler.seeded_noise.seeds],
                                            weights=self.sampler.seeded_noise.weights)


    def disable_untrainable_params(self):
        total_trainable = 0
        if self.lora_train:
            for n, p in self.named_parameters():
                if p.requires_grad == False:
                    continue
                if 'lora_layer' not in n:
                    p.lr_scale = 0
                else:
                    total_trainable += p.numel()
        else:
            for n, p in self.named_parameters():
                if p.requires_grad == False:
                    continue
                flag = False
                for prefix in self.untrainable_prefixs:
                    if n.startswith(prefix) and 'lora' not in n:
                        flag = True
                        break
                if flag:
                    p.requires_grad_(False)
                else:
                    total_trainable += p.numel()

        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples: (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples: (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch):
        # [B, T, C, H, W]
        time0 = time.time()
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            frame_pos = 0
            for batch_frame_lens in batch["frame_lens"]:
                for frame_len in batch_frame_lens:
                    if frame_len <= 9:
                        use_cp = False
                    else:
                        use_cp = True
                    out = self.first_stage_model.encode(
                        x[:, frame_pos: frame_pos + frame_len], use_cp=use_cp
                    )
                    frame_pos += frame_len
                    all_out.append(out)
        z = torch.cat(all_out, dim=1)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        if self.lr_scale is not None:
            lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
            lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
            lr_z = self.encode_first_stage(lr_x, batch)
            batch['lr_input'] = lr_z

        b, t = x.shape[:2]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # x = x.view(-1, *x.shape[2:])
        x = self.encode_first_stage(x, batch)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # x = x.view(b, t, *x.shape[1:])
        # batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)
        if hasattr(self.loss_fn, "block_scale") and self.loss_fn.block_scale is not None:
            randn = self.loss_fn.get_blk_noise(randn)

        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, **addtional_model_inputs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        samples = samples.to(self.dtype)
        return samples

    @torch.no_grad()
    def sample_sdedit(
            self,
            image: torch.Tensor,
            cond: Dict,
            uc: Union[Dict, None] = None,
            edit_ratio: float = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            **kwargs,
    ):
        image = image.expand(batch_size, *shape)
        randn = torch.randn(batch_size, *shape).to(self.dtype).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, image, randn, cond, uc=uc, edit_ratio=edit_ratio)
        return samples

    @torch.no_grad()
    def sample_relay(
            self,
            image: torch.Tensor,
            cond: Dict,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.dtype).to(self.device)
        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, image, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                    (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_video(
            self,
            batch: Dict,
            N: int = 8,
            sample: bool = True,
            ucg_keys: List[str] = None,
            only_log_video_latents = False,
            **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x.to(torch.float32)
        b, t = x.shape[:2]
        # x = x.view(-1, *x.shape[2:])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encode_first_stage(x, batch)
        if not only_log_video_latents:
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            log["reconstructions"] = log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
        z = z.permute(0, 2, 1, 3, 4).contiguous()

        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        from sgm.modules.diffusionmodules.sigma_sampling import PartialDiscreteSampling
        if sample and isinstance(self.loss_fn.sigma_sampler, PartialDiscreteSampling):
            sdedit_samples = self.sample_sdedit(
                z, c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )
            sdedit_samples = self.decode_first_stage(sdedit_samples)
            log['sdedit_samples'] = sdedit_samples
        elif sample and self.lr_scale is not None:
            lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
            lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
            log["lr_inputs"] = lr_x
            lr_z = self.encode_first_stage(lr_x, batch)
            samples = self.sample_relay(
                lr_z, c, shape=lr_z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        elif sample:
            samples = self.sample(
                c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            ) # b t c h w
            b, t = samples.shape[:2]
            # samples = samples.view(-1, *samples.shape[2:])
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                # samples = samples.view(b, t, *samples.shape[1:])
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                log["samples"] = samples
        return log