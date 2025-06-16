# MooreAA 同样API
import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
import decord
from decord import VideoReader, cpu, gpu
import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import time
from fractions import Fraction
import cv2
import jsonlines
import random


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)
        

def save_videos_from_pil(pil_images, path, fps=8):
    if fps is None or fps <= 0 or fps > 240:
        print(f"Warning: Invalid FPS {fps}")
        return

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        try:
            codec = "libx264"
            container = av.open(path, "w")
            stream = container.add_stream(codec, rate=fps)

            stream.width = width
            stream.height = height

            for pil_image in pil_images:
                # pil_image = Image.fromarray(image_arr).convert("RGB")
                av_frame = av.VideoFrame.from_image(pil_image)
                container.mux(stream.encode(av_frame))
            container.mux(stream.encode())
            container.close()
        except Exception as e:
            print(f"Unexpected error while saving video {path}: {e}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Corrupted file {path} removed successfully.")
                except Exception as rm_e:
                    print(f"Failed to remove corrupted file {path}: {rm_e}")

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:    # x: b c h w
        if x.shape[0] != 1:
            x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        else:
            x = x.squeeze(0)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    try:
        # 使用 decord 打开视频
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = []

        # 逐帧解码
        for i in range(len(vr)):
            frame = vr[i]  # 获取帧，返回的是 mx.ndarray
            image = Image.fromarray(frame.asnumpy())  # 转换为 PIL 格式
            frames.append(image)

        return frames

    except Exception as e:
        print(f"Error reading frames from {video_path}: {e}")
        return None  # 返回 None 避免代码崩溃


# pyav版本的
# def read_frames(video_path):  
#     start_time = time.time()  # 开始计时
#     container = av.open(video_path)

#     video_stream = next(s for s in container.streams if s.type == "video")
#     frames = []
#     for packet in container.demux(video_stream):
#         for frame in packet.decode():
#             image = Image.frombytes(
#                 "RGB",
#                 (frame.width, frame.height),
#                 frame.to_rgb().to_ndarray(),
#             )
#             frames.append(image)

#     end_time = time.time()  # 结束计时
#     total_time = end_time - start_time
#     print(f"Total time taken: {(end_time - start_time):.4f} seconds")
#     return frames

def get_fps(video_path):

    # container = av.open(video_path)
    # video_stream = next(s for s in container.streams if s.type == "video")
    # fps = video_stream.average_rate
    # container.close()
    # print("pyav_fps")
    # print(fps)
    try:
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        return Fraction(fps).limit_denominator(1001)
    except Exception as e:
        print(f"Error reading FPS from {video_path}: {e}")
        return None  # 返回 None 避免代码崩溃


def read_frames_and_fps(video_path):
    try:
        # 使用 decord 打开视频
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        frames = []

        # 逐帧解码
        for i in range(len(vr)):
            frame = vr[i]  # 获取帧，返回的是 mx.ndarray
            image = Image.fromarray(frame.asnumpy())  # 转换为 PIL 格式
            frames.append(image)

        processed_fps = Fraction(fps).limit_denominator(1001)
        return frames, processed_fps

    except Exception as e:
        print(f"Error reading frames from {video_path}: {e}")
        return None, None  # 返回 None 避免代码崩溃

def read_frames_and_fps_as_np(video_path):
    try:
        # 使用 decord 打开视频
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        frames = []

        # 逐帧解码
        for i in range(len(vr)):
            frame = vr[i]  # 获取帧，返回的是 mx.ndarray
            image = frame.asnumpy()  # 转换为 PIL 格式
            frames.append(image)
        processed_fps = Fraction(fps).limit_denominator(1001)
        return frames, processed_fps

    except Exception as e:
        print(f"Error reading frames from {video_path}: {e}")
        return None, None  # 返回 None 避免代码崩溃



def save_videos_from_pil(pil_images, path, fps=8):
    if fps is None or fps <= 0 or fps > 240:
        print(f"Warning: Invalid FPS {fps}")
        return

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        try:
            codec = "libx264"
            container = av.open(path, "w")
            stream = container.add_stream(codec, rate=fps)

            stream.width = width
            stream.height = height

            for pil_image in pil_images:
                # pil_image = Image.fromarray(image_arr).convert("RGB")
                av_frame = av.VideoFrame.from_image(pil_image)
                container.mux(stream.encode(av_frame))
            container.mux(stream.encode())
            container.close()
        except Exception as e:
            print(f"Unexpected error while saving video {path}: {e}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Corrupted file {path} removed successfully.")
                except Exception as rm_e:
                    print(f"Failed to remove corrupted file {path}: {rm_e}")

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")
    

def load_video_with_pose_from_first_frame(video_data, pose_data, sampling="uniform", duration=None, num_frames=99, wanted_fps=None, actual_fps=None,
               skip_frms_num=4., nb_read_frames=None):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    vr_pose = VideoReader(uri=pose_data, height=-1, width=-1)

    start = 0
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames + 1     # 要取到num_frames帧, +1把第一帧需要额外拿出来处理，让第一帧和后续帧不同

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C

    temp_frms = vr.get_batch(np.arange(start, end))
    temp_frms_pose = vr_pose.get_batch(np.arange(start, end))

    assert temp_frms is not None
    assert temp_frms_pose is not None

    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    tensor_frms_pose = torch.from_numpy(temp_frms_pose) if type(temp_frms_pose) is not torch.Tensor else temp_frms_pose
    tensor_frms_pose = temp_frms_pose[torch.tensor((indices - start).tolist())]

    # print(f"n_frms: {n_frms}; tensor_frms.shape: {tensor_frms.shape} tensor_frms_pose.shape: {tensor_frms_pose.shape}")
    return pad_last_frame(tensor_frms, n_frms), pad_last_frame(tensor_frms_pose, n_frms)


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
    

def load_video_sampling(video_data, pose_data, num_frames, wanted_fps):
    decord.bridge.set_bridge("torch")
    # 以video_data的为准
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    actual_fps = vr.get_avg_fps()
    if video_data:
        video, pose = load_video_with_pose_from_first_frame(video_data, pose_data, sampling="uniform", duration=100000, num_frames=num_frames, wanted_fps=wanted_fps, actual_fps=actual_fps, skip_frms_num=0, nb_read_frames=None)
        return video, pose
    else:
        raise ValueError("mooreAA should have video data")