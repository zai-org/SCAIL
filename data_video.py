import io
import re
import os
import sys
import json
import numpy as np
from PIL import Image
from functools import partial
import math
import tarfile
from braceexpand import braceexpand
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import multiprocessing
import torchvision.transforms as TT
from torch.utils.data import default_collate

from sat import mpu
# from sat.data_utils.webds import MetaDistributedWebDataset
from sgm.webds import MetaDistributedWebDataset
import webdataset as wds
from webdataset import ResampledShards, DataPipeline
from webdataset.utils import pytorch_worker_seed
from webdataset.filters import pipelinefilter
from webdataset.tariterators import url_opener, group_by_keys
from webdataset.handlers import reraise_exception

import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torch.utils.data import IterableDataset
# from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torchvision.io.video import av
import PIL
import numpy as np
import torch
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms import Compose
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode

from sgm.util import instantiate_from_config
from sat.helpers import print_rank0
import decord
from decord import VideoReader
import imageio

def rectangle_crop(arr, image_size, reshape_mode='center'):
    h, w = arr.shape[2], arr.shape[3]
    new_h, new_w = image_size

    delta_h = h - new_h
    delta_w = w - new_w

    if reshape_mode == 'center':
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=new_h, width=new_w
    )
    return arr

def load_video_with_pose(video_data, pose_data, ori_height, ori_width, image_size, motion_indices=None, ref_image_indices=None):
    # num_frames: wanted frames in wanted fps; image_size: [H, W]
    if ori_height < 0 or ori_width < 0:
        new_height = -1
        new_width = -1
    if ori_width / ori_height > image_size[1] / image_size[0]:
        new_height = image_size[0]      # 根据height缩放
        new_width = int(ori_width * new_height / ori_height)
    else:
        new_width = image_size[1]       # 根据width缩放
        new_height = int(ori_height * new_width / ori_width)


    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=new_height, width=new_width)
    vr_pose = VideoReader(uri=pose_data, height=new_height, width=new_width)
    assert len(vr) == len(vr_pose), f"frames and pose frames should have the same length, but here has len(vr): {len(vr)} and len(vr_pose): {len(vr_pose)}"
    
    if motion_indices is not None:
        ref_indices = [random.choice(ref_image_indices)]
        temp_frms = vr.get_batch(motion_indices)
        temp_frms_pose = vr_pose.get_batch(motion_indices)
    else:
        ref_indices = [0]
        indices = list(range(len(vr)))
        temp_frms = vr.get_batch(indices)
        temp_frms_pose = vr_pose.get_batch(indices)
   
    ref_frms = vr.get_batch(ref_indices)
    ref_frms_pose = vr_pose.get_batch(ref_indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms_pose = torch.from_numpy(temp_frms_pose) if type(temp_frms_pose) is not torch.Tensor else temp_frms_pose
    tensor_ref_frms = torch.from_numpy(ref_frms) if type(ref_frms) is not torch.Tensor else ref_frms
    tensor_ref_frms_pose = torch.from_numpy(ref_frms_pose) if type(ref_frms_pose) is not torch.Tensor else ref_frms_pose
    
    # --- copy and modify the image process ---
    tensor_frms = tensor_frms.permute(0, 3, 1, 2) # T H W C -> [T, C, H, W]
    tensor_frms_pose = tensor_frms_pose.permute(0, 3, 1, 2)
    tensor_ref_frms = tensor_ref_frms.permute(0, 3, 1, 2)
    tensor_ref_frms_pose = tensor_ref_frms_pose.permute(0, 3, 1, 2)
    tensor_frms = (tensor_frms - 127.5) / 127.5
    tensor_frms_pose = (tensor_frms_pose - 127.5) / 127.5
    tensor_ref_frms = (tensor_ref_frms - 127.5) / 127.5
    tensor_ref_frms_pose = (tensor_ref_frms_pose - 127.5) / 127.5
    tensor_frms = rectangle_crop(tensor_frms, image_size, reshape_mode='center')
    tensor_frms_pose = rectangle_crop(tensor_frms_pose, image_size, reshape_mode='center')
    tensor_ref_frms = rectangle_crop(tensor_ref_frms, image_size, reshape_mode='center')
    tensor_ref_frms_pose = rectangle_crop(tensor_ref_frms_pose, image_size, reshape_mode='center')

    return tensor_frms, tensor_frms_pose, tensor_ref_frms, tensor_ref_frms_pose

def load_video_with_pose_with_timeout(*args, **kwargs):
    # 创建一个Thread对象，目标函数是load_video
    video_container = {}
    def target_function():
        video, pose, ref, ref_pose = load_video_with_pose(*args, **kwargs)
        video_container['video'] = video
        video_container['pose'] = pose
        video_container['ref'] = ref
        video_container['ref_pose'] = ref_pose

    # 启动线程
    thread = threading.Thread(target=target_function)
    thread.start()
    # 等待线程完成或超时
    # timeout = 10
    timeout = 100
    thread.join(timeout)
    if thread.is_alive():
        # stop_thread(thread)
        # thread.join()
        print("Loading video timed out")
        raise TimeoutError
        # return None  # 可以抛出异常或返回特定值表示超时
    return video_container.get('video', None).contiguous(), video_container.get('pose', None).contiguous(), video_container.get('ref', None).contiguous(), video_container.get('ref_pose', None).contiguous()

def resize_for_rectangle_crop(arr, image_size, reshape_mode='random'):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(arr, size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])], interpolation=InterpolationMode.BICUBIC)
    else:
        arr = resize(arr, size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]], interpolation=InterpolationMode.BICUBIC)

    h, w = arr.shape[2], arr.shape[3]

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == 'random' or reshape_mode == 'none':
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == 'center':
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=image_size[0], width=image_size[1]
    )
    return arr

def pad_last_frame(tensor, sampling_frms_num):
    # T, H, W, C
    if tensor.shape[0] < sampling_frms_num:
        # 复制最后一帧
        last_frame = tensor[-int(sampling_frms_num-tensor.shape[0]):]
        # 将最后一帧添加到第二个维度
        padded_tensor = torch.cat([tensor, last_frame], dim=0)
        return padded_tensor
    else:
        return tensor[:sampling_frms_num]


def load_video(video_data, sampling="uniform", duration=None, num_frames=4, wanted_fps=None, actual_fps=None,
               skip_frms_num=0., ori_height=None, ori_width=None, image_size=None):

    # num_frames: wanted frames in wanted fps; image_size: [H, W]
    if ori_width / ori_height > image_size[1] / image_size[0]:
        new_height = image_size[0]
        new_width = int(ori_width * new_height / ori_height)
    else:
        new_width = image_size[1]
        new_height = int(ori_height * new_width / ori_width)

    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=new_height, width=new_width)
    ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    start = skip_frms_num
    end = int(start + num_frames / wanted_fps * actual_fps)

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / num_frames).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(0, end))
    assert temp_frms is not None
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[indices.tolist()]

    return pad_last_frame(tensor_frms, num_frames)

import threading
import ctypes
import inspect

def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)

def load_video_with_timeout(*args, **kwargs):
    # 创建一个Thread对象，目标函数是load_video
    video_container = {}
    def target_function():
        video = load_video(*args, **kwargs)
        video_container['video'] = video

    # 启动线程
    thread = threading.Thread(target=target_function)
    thread.start()
    # 等待线程完成或超时
    timeout = 10
    # timeout = 30
    thread.join(timeout)
    if thread.is_alive():
        # stop_thread(thread)
        # thread.join()
        print("Loading video timed out")
        raise TimeoutError
        # return None  # 可以抛出异常或返回特定值表示超时
    return video_container.get('video', None).contiguous()