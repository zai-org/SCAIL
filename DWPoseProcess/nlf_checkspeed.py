import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.checkUtils import *
from collections import deque
import shutil
import torch
import yaml
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, TimeoutError
from decord import VideoReader
from fractions import Fraction
import io
import gc
from PIL import Image
from multiprocessing import Process
import json
import jsonlines
from webdataset import TarWriter
import math
import glob
import pickle
import copy
from NLFPoseExtract.nlf_render import render_nlf_as_images
from NLFPoseExtract.nlf_draw import preview_nlf_2d_new
from NLFPoseExtract.reshape_utils_3d import reshapePool3d
from pose_draw.draw_pose_main import draw_pose_to_canvas_np
from NLFPoseExtract.smpl_joint_xyz import compute_motion_speed, collect_nlf_for_speed_select
from NLFPoseExtract.nlf_draw import intrinsic_matrix_from_field_of_view
import traceback
import decord
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import torchvision.transforms as TT
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy
import imageio


def resize_rectangle_crop_for_pose(arr, image_size, reshape_mode='center'):    
    h, w = arr.shape[2], arr.shape[3]   # T C H W
    target_h, target_w = image_size

    scale_factor_h = image_size[0]/h
    scale_factor_w = image_size[1]/w

    if scale_factor_h < scale_factor_w:
        new_w = image_size[1]
        new_h = int(image_size[1] * h / w )
    else:
        new_h = image_size[0]
        new_w = int(image_size[0] * w / h )
        
    arr = resize(arr, size=[new_h, new_w], interpolation=InterpolationMode.BICUBIC)

    delta_h = new_h - target_h
    delta_w = new_w - target_w

    if reshape_mode == 'center':
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=target_h, width=target_w
    )
    return arr

def process_fn_video(src, meta_dict=None):
    worker_info = torch.utils.data.get_worker_info()
    for i, r in enumerate(src):
        if worker_info is not None:
            if i % worker_info.num_workers != worker_info.id:
                continue

        meta = meta_dict.get(r['__key__'], None)
        if meta is None:
            print(f"skip {r['__key__']}, no meta")
            continue
        r.update(meta)
        ori_meta = meta.get('ori_meta', {})
        if isinstance(ori_meta, dict):
            r.update(ori_meta)
        
        motion_indices = r.get('motion_indices', None)
        if motion_indices is None:
            print(f"skip {r['__key__']}, no motion_indices")
            continue

        yield {'__key__': r['__key__'], 'mp4': r['mp4'], 'motion_indices': motion_indices}

def calc_smpl_speed(wds_path, save_dir_smpl):
    meta_dict = {}
    meta_file = wds_path.replace('.tar', '.meta.jsonl')
    meta_lines = open(meta_file).readlines()
    decord.bridge.set_bridge("torch")
    

    for meta_line in meta_lines:
        meta_line = meta_line.strip()
        try:
            meta = json.loads(meta_line)
        except Exception as e:
            print(e)
            print('json load error: ', meta_file)
            continue
        meta_dict[meta['key']] = meta
    dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
            partial(process_fn_video, meta_dict=meta_dict),
        )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=lambda x: x[0])
    output_root = '/workspace/yanwenhao/dwpose_draw/preview_videos'
    for data in tqdm(dataloader):
        key = data['__key__']
        motion_indices = data['motion_indices']
        mp4_bytes = data['mp4']
        smpl_path = os.path.join(save_dir_smpl, f"{key}.pkl")
        if not os.path.exists(smpl_path):
            print(f"skip {smpl_path}, not exist")
            continue
        with open(smpl_path, 'rb') as f:
            ori_smpl = pickle.load(f)
        collected_nlf = collect_nlf_for_speed_select(ori_smpl)
        smpl_ori_data = [
            torch.stack(collected_nlf[i]) if len(collected_nlf[i]) > 0 
            else torch.empty((0, 24, 3)) 
            for i in range(len(collected_nlf))
        ]
        height = ori_smpl[0]['video_height']
        width = ori_smpl[0]['video_width']
        camera = intrinsic_matrix_from_field_of_view([height, width])
        speed = compute_motion_speed(smpl_ori_data, height, width, camera)
        print(f"Speed for {key}: {speed}")
        if speed is None or speed == 0:
            speed = 0
            target_subdir = os.path.join(output_root, 'None')
        elif speed < 10:
            target_subdir = os.path.join(output_root, 'Slow')
        elif speed < 20:
            target_subdir = os.path.join(output_root, 'Normal')
        elif speed < 60:
            target_subdir = os.path.join(output_root, 'Fast')
        else:
            target_subdir = os.path.join(output_root, 'VeryFast')
        os.makedirs(target_subdir, exist_ok=True)
        vr = VideoReader(io.BytesIO(mp4_bytes))  
        frames = vr.get_batch(motion_indices) # T H W C
        if frames.shape[1] < frames.shape[2]:
            image_size = [720, 1280]
        else:
            image_size = [1280, 720]
        frames = resize_rectangle_crop_for_pose(frames.permute(0, 3, 1, 2), image_size).permute(0, 2, 3, 1)
        imageio.mimsave(os.path.join(target_subdir, f"{key}.mp4"), frames, fps=16)


    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/workspace/yanwenhao/dwpose_draw/DWPoseExtractConfig/motionarray_nowat.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0929_step3',
                        help='Input root')

    args = parser.parse_args()
    config = load_config(args.config)

    wds_root = config.get('wds_root', '')
    video_root = config.get('video_root', '')
    

    save_dir_keypoints = os.path.join(video_root, 'keypoints')
    save_dir_bboxes = os.path.join(video_root, 'bboxes')
    save_dir_dwpose_mp4 = os.path.join(video_root, 'dwpose')
    save_dir_dwpose_reshape_mp4 = os.path.join(video_root, 'dwpose_reshape')
    save_dir_hands = os.path.join(video_root, 'hands')
    save_dir_faces = os.path.join(video_root, 'faces')
    save_dir_caption = os.path.join(video_root, 'caption')
    save_dir_caption_multi = os.path.join(video_root, 'caption_multi')
    save_dir_smpl = os.path.join(video_root, 'smpl')
    save_dir_smpl_render = os.path.join(video_root, 'smpl_render')

    input_dir = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    # Split wds_list into chunks
    input_tar_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True))
    random.shuffle(input_tar_paths)
    for wds_path in input_tar_paths:
        calc_smpl_speed(wds_path, save_dir_smpl)







    
