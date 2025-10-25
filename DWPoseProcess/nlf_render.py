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
import traceback
import taichi as ti
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy

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

        yield {'__key__': r['__key__'], 'motion_indices': motion_indices}


def nlf_2d_drawing(nlf_results, poses):
    height = nlf_results[0]['video_height']
    width = nlf_results[0]['video_width']
    frames_np_rgba = preview_nlf_2d_new(nlf_results)
    canvas_2d = draw_pose_to_canvas_np(poses, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True)
    for i in range(len(frames_np_rgba)):
        frame_img = frames_np_rgba[i]
        canvas_img = canvas_2d[i]
        mask = canvas_img != 0
        frame_img[:, :, :3][mask] = canvas_img[mask]
        frames_np_rgba[i] = frame_img
    return frames_np_rgba

def render_to_smpl_path(wds_path, save_dir_smpl, save_dir_smpl_render, save_dir_keypoints, reshape_type, save_dir_caption, save_dir_caption_multi):
    meta_dict = {}
    meta_file = wds_path.replace('.tar', '.meta.jsonl')
    meta_lines = open(meta_file).readlines()
    tmp_dir = '/dev/shm/tmp'
    ti.reset()       # 清理taichi的缓存
    tarname = os.path.basename(wds_path).replace('.tar', '')    
    ti.init(arch=ti.cuda, offline_cache_file_path=f'{tmp_dir}/{tarname}')
    os.makedirs(save_dir_smpl_render, exist_ok=True)
    os.makedirs(save_dir_smpl_render.replace('smpl_render', 'smpl_render_aug'), exist_ok=True)
    os.makedirs(save_dir_smpl_render.replace('smpl_render', 'smpl_render_noface'), exist_ok=True)
    os.makedirs(save_dir_smpl_render.replace('smpl_render', 'smpl_2d'), exist_ok=True)

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
    for data in tqdm(dataloader):
        key = data['__key__']
        motion_indices = data['motion_indices']
        smpl_data = pickle.load(open(os.path.join(save_dir_smpl, key + '.pkl'), 'rb'))
        poses = torch.load(os.path.join(save_dir_keypoints, key + '.pt'))  # (T, 72)
        multi_person_path = os.path.join(save_dir_caption_multi, key + '.txt')
        single_person_path = os.path.join(save_dir_caption, key + '.txt')
        multi_person = False
        if os.path.exists(multi_person_path):
            # print(f"debug: {key} is multi person")
            multi_person = True
        elif os.path.exists(single_person_path):
            multi_person = False
        if multi_person:
            reshape_type = "low"
        out_path_smpl_render = os.path.join(save_dir_smpl_render, key + '.mp4')
        out_path_smpl_2d = os.path.join(save_dir_smpl_render.replace('smpl_render', 'smpl_2d'), key + '.mp4')
        out_path_smpl_render_aug = os.path.join(save_dir_smpl_render.replace('smpl_render', 'smpl_render_aug'), key + '.mp4')
        out_path_smpl_render_noface = os.path.join(save_dir_smpl_render.replace('smpl_render', 'smpl_render_noface'), key + '.mp4')

        height, width = smpl_data[0]["video_height"], smpl_data[0]["video_width"]
        reshape_pool_3d = reshapePool3d(reshape_type=reshape_type, height=height, width=width)
        reshape_pool_3d_no_face = reshapePool3d(reshape_type=reshape_type, height=height, width=width) if random.random() < 0.6 else None

        # 完全用DWPose的手/脸
        smpl_data_in_motion = [smpl_data[i] for i in motion_indices]
        poses_in_motion = [poses[i] for i in motion_indices]
        smpl_2d_data = nlf_2d_drawing(copy.deepcopy(smpl_data_in_motion), copy.deepcopy(poses_in_motion))
        smpl_render_data = render_nlf_as_images(copy.deepcopy(smpl_data_in_motion), copy.deepcopy(poses_in_motion), reshape_pool=None, aug_2d=False, aug_cam=False)
        smpl_render_data_aug = render_nlf_as_images(copy.deepcopy(smpl_data_in_motion), copy.deepcopy(poses_in_motion), reshape_pool=reshape_pool_3d, aug_2d=True, aug_cam=True)
        smpl_render_aug_no_face = render_nlf_as_images(copy.deepcopy(smpl_data_in_motion), copy.deepcopy(poses_in_motion), reshape_pool=reshape_pool_3d_no_face, draw_2d=False, aug_2d=True, aug_cam=False)
        mpy.ImageSequenceClip(smpl_2d_data, fps=16).write_videofile(out_path_smpl_2d)
        mpy.ImageSequenceClip(smpl_render_data, fps=16).write_videofile(out_path_smpl_render)
        mpy.ImageSequenceClip(smpl_render_data_aug, fps=16).write_videofile(out_path_smpl_render_aug)
        mpy.ImageSequenceClip(smpl_render_aug_no_face, fps=16).write_videofile(out_path_smpl_render_noface)

def process_tar_chunk(chunk, save_dir_smpl, save_dir_smpl_render, save_dir_keypoints, reshape_type, save_dir_caption, save_dir_caption_multi):
    for wds_path in chunk:
        render_to_smpl_path(wds_path, save_dir_smpl, save_dir_smpl_render, save_dir_keypoints, reshape_type, save_dir_caption, save_dir_caption_multi)
        gc.collect()
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0929_step3',
                        help='Input root')
    parser.add_argument('--max_processes', type=int, default=8,
                        help='Max processes')
    parser.add_argument('--current_process', type=int, default=0,
                        help='Current process')

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

    processes = []  # 存储进程的列表
    max_processes = args.max_processes

    input_dir = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    # Split wds_list into chunks
    input_tar_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True))
    filter_args = config.get('filter_args', {})
    reshape_type = filter_args.get('reshape', 'normal')
    print(f"reshape_type: {reshape_type} for data: {video_root}")

    current_process = args.current_process
    current_tar_paths = input_tar_paths[current_process::max_processes]
    if len(current_tar_paths) == 0:
        print("No chunks to process")
    process_tar_chunk(current_tar_paths, save_dir_smpl, save_dir_smpl_render, save_dir_keypoints, reshape_type, save_dir_caption, save_dir_caption_multi)







    
