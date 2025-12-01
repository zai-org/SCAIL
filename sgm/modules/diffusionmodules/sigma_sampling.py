import torch
import torch.distributed
import math
from sat import mpu

from ...util import default, instantiate_from_config


def time_shift(mu, t):
    return 1 / (1 + math.exp(mu)/t - math.exp(mu))

class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, uniform_sampling=False, group_num=0):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        world_size = mpu.get_data_parallel_world_size()
        if world_size <= 8:
            uniform_sampling = False
        self.uniform_sampling = uniform_sampling
        self.group_num = group_num
        if self.uniform_sampling:
            assert self.group_num > 0
            assert world_size % group_num == 0
            self.group_width = world_size // group_num # the number of rank in one group
            self.sigma_interval = self.num_idx // self.group_num

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None, return_idx=False):
        if self.uniform_sampling:
            rank = mpu.get_data_parallel_rank()
            group_index = rank // self.group_width
            idx = default(
                rand,
                torch.randint(group_index * self.sigma_interval, (group_index + 1) * self.sigma_interval, (n_samples,)),
            )
        else:
            idx = default(
                rand,
                torch.randint(0, self.num_idx, (n_samples,)),
            )
        if return_idx:
            return self.idx_to_sigma(idx), idx
        else:
            return self.idx_to_sigma(idx)

class DiscreteSampling_TASD:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, uniform_sampling=False, group_num=0):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        world_size = mpu.get_data_parallel_world_size()
        if world_size <= 8:
            uniform_sampling = False
        self.uniform_sampling = uniform_sampling
        self.group_num = group_num
        if self.uniform_sampling:
            assert self.group_num > 0
            assert world_size % group_num == 0
            self.group_width = world_size // group_num # the number of rank in one group
            self.sigma_interval = self.num_idx // self.group_num

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples_shape, rand=None, return_idx=False):
        idx = default(
            rand,
            torch.randint(0, self.num_idx, n_samples_shape),
        )
        if return_idx:
            return self.idx_to_sigma(idx), idx
        else:
            return self.idx_to_sigma(idx)

class PartialDiscreteSampling:
    def __init__(self, discretization_config, total_num_idx, partial_num_idx, do_append_zero=False, flip=True):
        self.total_num_idx = total_num_idx
        self.partial_num_idx = partial_num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            total_num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            # torch.randint(self.total_num_idx-self.partial_num_idx, self.total_num_idx, (n_samples,)),
            torch.randint(0, self.partial_num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)
    

class RFSampling:
    def __init__(self, p_mean=0.0, p_std=1.0):
        self.p_mean = p_mean
        self.p_std = p_std
        self.distribution = torch.distributions.LogisticNormal(torch.tensor([self.p_mean]), torch.tensor([self.p_std]))

    def __call__(self, n_samples, rand=None):
        sigma = self.distribution.sample((n_samples,))[:, 0]
        return sigma

class RFSampling_TASD:
    def __init__(self, p_mean=0.0, p_std=1.0):
        self.p_mean = p_mean
        self.p_std = p_std
        self.distribution = torch.distributions.LogisticNormal(torch.tensor([self.p_mean]), torch.tensor([self.p_std]))

    def __call__(self, n_samples, rand=None):
        sigma = self.distribution.sample((*n_samples, ))[:, :, 0]
        return sigma
