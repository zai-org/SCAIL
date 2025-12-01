import torch
import deepspeed

from typing import Union
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from sat import mpu

try:
    from torch.distributed._tensor import DTensor, distribute_tensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

def _get_main_grad_attr(param: torch.nn.Parameter, use_custom_fsdp: bool = False):
    # Actually only 'grad' is used in deepspeed
    if use_custom_fsdp:
        return "fsdp_managed_main_grad"
    if hasattr(param, "main_grad"):
        return "main_grad"
    return "grad"

def _unshard_if_dtensor(tensor: Union[torch.Tensor, "DTensor"]) -> torch.Tensor:
    """
    Unshards the input tensor if it is a DTensor and otherwise returns the
    tensor unmodified.

    Args:
        tensor (Union[torch.Tensor, DTensor]): The tensor to potentially unshard.

    Returns:
        An unsharded version of the input tensor if it is a DTensor, or the
        input tensor unmodified if it is not a DTensor.
    """
    if HAVE_DTENSOR and isinstance(tensor, DTensor):
        unsharded_tensor = tensor.full_tensor()
        for k, v in vars(tensor).items():
            setattr(unsharded_tensor, k, v)
        return unsharded_tensor
    return tensor

def _reshard_if_dtensor(
    tensor_to_shard: torch.Tensor, reference_tensor: Union[torch.Tensor, "DTensor"]
) -> Union[torch.Tensor, "DTensor"]:
    """
    Reshards the input tensor to match the sharding configuration of the
    reference tensor if the reference tensor is a DTensor. Otherwise, returns
    the reference tensor unmodified.

    Args:
        tensor_to_shard (torch.Tensor): The tensor to be potentially sharded.
        reference_tensor (Union[torch.Tensor, DTensor]): The reference tensor
            for the sharding configuration.

    Returns:
        Union[torch.Tensor, DTensor]: The sharded tensor matching the reference tensor's
        configuration, or the reference tensor itself if it is not a DTensor.
    """
    if HAVE_DTENSOR and isinstance(reference_tensor, DTensor):
        sharded_tensor = distribute_tensor(
            tensor_to_shard,
            device_mesh=reference_tensor.device_mesh,
            placements=reference_tensor.placements,
        )
        for k, v in vars(reference_tensor).items():
            setattr(sharded_tensor, k, v)
        return sharded_tensor
    return reference_tensor

def _allreduce_layernorm_grads(model: deepspeed.runtime.engine.DeepSpeedEngine):
    """
    All-reduce layernorm grads (for tensor parallelism).
    fsdp / ZeRO-3 not supported
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when tensor parallelism is used
    if mpu.get_model_parallel_world_size() > 1:
        params = []
        grads = []
        #for model_chunk in model:
        for name, param in model.module.named_parameters():
            if param.requires_grad and (
                'q_layernorm' in name
                or 'k_layernorm' in name
                or 'query_layernorm' in name
                or 'key_layernorm' in name
            ) and getattr(param, _get_main_grad_attr(param)) is not None:
                #try:
                params.append(param)
                grad_attr = _get_main_grad_attr(param)
                grad = getattr(param, grad_attr)
                grad = _unshard_if_dtensor(grad)
                grads.append(grad.data)
                #except:
                #    print('grad append failed:', name, param, grad, vars(param))
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=mpu.get_model_parallel_group()
            )
            for param, buf, synced in zip(
                params, grads, _unflatten_dense_tensors(coalesced, grads)
            ):
                #if torch.distributed.get_rank() == 0:
                #    print('before sync: buf, synced, orig_grad:', buf, synced, getattr(param, _get_main_grad_attr(param)))
                buf.copy_(synced)
                grad_attr = _get_main_grad_attr(param)
                orig_grad = getattr(param, grad_attr)
                setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))
                #if torch.distributed.get_rank() == 0:
                #    print('after sync: buf, synced, orig_grad:', buf, synced, getattr(param, _get_main_grad_attr(param)))

def finalize_model_grads(model_engine):
    # Reduce layernorm grads
    _allreduce_layernorm_grads(model_engine)
