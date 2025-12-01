import torch

from sat import mpu

from .loss import StandardDiffusionLoss, get_3d_position_ids, get_lin_function, time_shift
from ...util import append_dims

class CausalRFLoss(StandardDiffusionLoss):
    def __init__(self, schedule_shift=False, **kwargs):
        self.schedule_shift = schedule_shift
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        # input b, t, d, h, w
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        t_indices = self.sigma_sampler(input.shape[:2])
        t_indices = t_indices.to(input.device)

        noise = torch.randn_like(input)
        if self.schedule_shift:
            for index in range(t_indices.shape[0]):
                image_seq_len = input.shape[-1] * input.shape[-2] // network.diffusion_model.patch_size[-1] // network.diffusion_model.patch_size[-2]
                mu = get_lin_function(y1=0.5, y2=1.15)(image_seq_len)
                t_indices[index] = time_shift(mu, t_indices[index], mode='normal')

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(t_indices, src=src, group=mpu.get_model_parallel_group())

        scaled_input = input.float() * append_dims(1 - t_indices, input.ndim)
        scaled_noise = noise.float() * append_dims(t_indices, input.ndim)
        noised_input = scaled_input + scaled_noise

        # get position ids
        patch_size = network.diffusion_model.patch_size
        position_ids = get_3d_position_ids(input.shape[1]//patch_size[0], input.shape[3]//patch_size[1], input.shape[4]//patch_size[2]).reshape(-1, 3)
        position_ids = position_ids.unsqueeze(0).expand(input.shape[0], -1, -1)
        additional_model_inputs['rope_position_ids'] = position_ids
        
        model_output = denoiser(
            network, noised_input, t_indices, cond, **additional_model_inputs
        )

        model_label = noise - input

        return self.get_loss(model_output, model_label, 1)

    def get_loss(self, model_output, target, w):
        return torch.mean(
            (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
        )
