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
from pose_draw.draw_pose_main import draw_pose_to_canvas_np, reshapePool
from extractUtils import check_single_human_requirements, check_multi_human_requirements, human_select
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
import traceback
from DWPoseProcess.extract_mp4_hybrid_eval import get_hybrid_video
from VITPoseExtract.pipeline import VITPosePipeline
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

        item = r.copy()
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
        
        item.update({'motion_indices': motion_indices})

        yield item

def reshape_render_to_wds(wds_path, output_wds_path, save_dir_keypoints, save_dir_dwpose_mp4, save_dir_smpl, save_dir_smpl_render):
    obj_list = []
    meta_dict = {}
    meta_file = wds_path.replace('.tar', '.meta.jsonl')
    meta_lines = open(meta_file).readlines()

    for meta_line in meta_lines:
        meta_line = meta_line.strip()
        try:
            meta = json.loads(meta_line)
        except Exception as e:
            print(e)
            print('json load error: ', meta_file)
            continue
        meta_dict[meta['key']] = meta
    output_meta_file = output_wds_path.replace('.tar', '.meta.jsonl')
    dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
            partial(process_fn_video, meta_dict=meta_dict),
        )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, collate_fn=lambda x: x[0])
    if os.path.exists(output_wds_path):   # 提前清除旧tar
        os.remove(output_wds_path)
    with TarWriter(output_wds_path) as writer:
        for data in tqdm(dataloader):
            key = data['__key__']
            mp4_bytes = data['mp4']
            motion_indices = data['motion_indices']
            vr = VideoReader(io.BytesIO(mp4_bytes))   # h w c
            frames_np_motion = vr.get_batch(motion_indices).asnumpy()
            first_frame_np = frames_np_motion[0]
            try:
                keypoints = torch.load(os.path.join(save_dir_keypoints, key + '.pt'), weights_only=False)
                keypoints_in_motion = [keypoints[idx] for idx in motion_indices]
                smpl_rendered_path = os.path.join(save_dir_smpl_render, key + '.mp4')
                if not os.path.exists(smpl_rendered_path):
                    print(f"skip {key}, no smpl rendered")
                    continue
                height = vr[0].shape[0]
                width = vr[0].shape[1]
                tmp_dir = '/dev/shm/tmp'
                os.makedirs(tmp_dir, exist_ok=True)

                # hybrid_cheek_video_aug = get_hybrid_video(first_frame_np, copy.deepcopy(keypoints_in_motion), copy.deepcopy(tpl_pose_metas_motion), height, width, reshape_scale=0.6, only_cheek=True)
                # hybrid_cheek_video_no_aug = get_hybrid_video(first_frame_np, copy.deepcopy(keypoints_in_motion), copy.deepcopy(tpl_pose_metas_motion), height, width, reshape_scale=0, only_cheek=True)
                # hybrid_video_full_aug = get_hybrid_video(first_frame_np, copy.deepcopy(keypoints_in_motion), copy.deepcopy(tpl_pose_metas_motion), height, width, reshape_scale=0.6, only_cheek=False)
                # hybrid_video_full_no_aug = get_hybrid_video(first_frame_np, copy.deepcopy(keypoints_in_motion), copy.deepcopy(tpl_pose_metas_motion), height, width, reshape_scale=0, only_cheek=False)
                reshape_pool = reshapePool(reshape_scale=0.6, H=height, W=width)

                hybrid_cheek_video_aug = draw_pose_to_canvas_np(copy.deepcopy(keypoints_in_motion), pool=reshape_pool, H=height, W=width, reshape_scale=0.6, show_body_flag=False, show_cheek_flag=True)
                hybrid_cheek_video_no_aug = draw_pose_to_canvas_np(copy.deepcopy(keypoints_in_motion), pool=None, H=height, W=width, reshape_scale=0, show_body_flag=False, show_cheek_flag=True)
                hybrid_video_full_aug = draw_pose_to_canvas_np(copy.deepcopy(keypoints_in_motion), pool=reshape_pool, H=height, W=width, reshape_scale=0.6, show_body_flag=True, show_cheek_flag=False)
                # hybrid_video_full_no_aug = draw_pose_to_canvas_np(copy.deepcopy(keypoints_in_motion), pool=None, H=height, W=width, reshape_scale=0, show_body_flag=True, show_cheek_flag=False)

                hybrid_cheek_video_aug_path = os.path.join(tmp_dir, f'{key}_hybrid_cheek_aug.mp4')
                hybrid_cheek_video_no_aug_path = os.path.join(tmp_dir, f'{key}_hybrid_cheek_no_aug.mp4')
                hybrid_video_full_aug_path = os.path.join(tmp_dir, f'{key}_hybrid_full_aug.mp4')
                # hybrid_video_full_no_aug_path = os.path.join(tmp_dir, f'{key}_hybrid_full_no_aug.mp4')
                mpy.ImageSequenceClip(hybrid_cheek_video_aug, fps=16).write_videofile(hybrid_cheek_video_aug_path)
                mpy.ImageSequenceClip(hybrid_cheek_video_no_aug, fps=16).write_videofile(hybrid_cheek_video_no_aug_path)
                mpy.ImageSequenceClip(hybrid_video_full_aug, fps=16).write_videofile(hybrid_video_full_aug_path)
                # mpy.ImageSequenceClip(hybrid_video_full_no_aug, fps=16).write_videofile(hybrid_video_full_no_aug_path)
                with open(hybrid_cheek_video_aug_path, "rb") as f:
                    hybrid_cheek_video_aug_data = f.read()
                with open(hybrid_cheek_video_no_aug_path, "rb") as f:
                    hybrid_cheek_video_no_aug_data = f.read()
                with open(hybrid_video_full_aug_path, "rb") as f:
                    hybrid_video_full_aug_data = f.read()
                # with open(hybrid_video_full_no_aug_path, "rb") as f:
                #     hybrid_video_full_no_aug_data = f.read()
                with open(smpl_rendered_path, "rb") as f:
                    smpl_render_data = f.read()

                # if random.random() < 0.8:
                    # data['append_dwpose_noreshape'] = hybrid_video_full_no_aug_data  # pose现在不能替换，因为长度不一致
                data['append_dwpose_reshape'] = hybrid_video_full_aug_data
                data['append_smpl_render'] = smpl_render_data
                data['append_dwpose_reshape_cheek_hands'] = hybrid_cheek_video_aug_data
                data['append_dwpose_noreshape_cheek_hands'] = hybrid_cheek_video_no_aug_data
                # 清除临时文件
                os.remove(hybrid_cheek_video_aug_path)
                os.remove(hybrid_cheek_video_no_aug_path)
                os.remove(hybrid_video_full_aug_path)
                # os.remove(hybrid_video_full_no_aug_path)

                data.pop('motion_indices')
                obj_list.append(meta_dict.get(key, None))
                writer.write(data)
            except Exception as e:
                print(e)
                print(f"skip {key}, error")
                continue
    with open(output_meta_file, 'w', encoding='utf-8') as outfile:
        writer = jsonlines.Writer(outfile)
        writer.write_all(obj_list)
        writer.close()
        

def process_tar_chunk(chunk, input_root, output_root, save_dir_keypoints, save_dir_dwpose_mp4, save_dir_smpl, save_dir_smpl_render):
    for wds_path in chunk:
        rel_path = os.path.relpath(wds_path, input_root)
        output_wds_path = os.path.join(output_root, rel_path)
        reshape_render_to_wds(wds_path, output_wds_path, save_dir_keypoints, save_dir_dwpose_mp4, save_dir_smpl, save_dir_smpl_render)
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
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0929_step5',
                        help='Output root')
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
    output_dir = os.path.join(args.output_root, os.path.basename(os.path.normpath(video_root)))
    os.makedirs(output_dir, exist_ok=True)
    # Split wds_list into chunks
    input_tar_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True))

    current_process = args.current_process
    current_tar_paths = input_tar_paths[current_process::max_processes]
    if len(current_tar_paths) == 0:
        print("No chunks to process")
    process_tar_chunk(current_tar_paths, input_dir, output_dir, save_dir_keypoints, save_dir_dwpose_reshape_mp4, save_dir_smpl, save_dir_smpl_render)







    
