import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.AAUtils import save_videos_from_pil
from DWPoseProcess.checkUtils import *
from collections import deque
import shutil
import torch
import yaml
from pose_draw.draw_pose_main import draw_pose_to_canvas, reshapePool
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
from NLFPoseExtract.nlf_render import render_nlf_as_images

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
    meta_dict = {}
    meta_file = wds_path.replace('.tar', '.meta.jsonl')
    meta_lines = open(meta_file).readlines()
    tmp_dir = '/dev/shm/tmp'
    os.makedirs(os.path.join(tmp_dir, 'faces_hands'), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, 'cheek_hands'), exist_ok=True)
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
    with TarWriter(output_wds_path) as writer:
        for data in tqdm(dataloader):
            key = data['__key__']
            mp4_bytes = data['mp4']
            motion_indices = data['motion_indices']
            vr = VideoReader(io.BytesIO(mp4_bytes))   # h w c
            height = vr[0].shape[0]
            width = vr[0].shape[1]
            keypoints = torch.load(os.path.join(save_dir_keypoints, key + '.pt'), weights_only=False)
            smpl_data = pickle.load(open(os.path.join(save_dir_smpl, key + '.pkl'), 'rb'))

            out_path_dwpose_mp4 = os.path.join(save_dir_dwpose_mp4, key + '.mp4')
            out_path_smpl_render = os.path.join(save_dir_smpl_render, key + '.mp4')
            out_path_dwpose_mp4_face_hands = os.path.join(tmp_dir, 'faces_hands', key + '.mp4')
            out_path_dwpose_mp4_cheek_hands = os.path.join(tmp_dir, 'cheek_hands', key + '.mp4')
            # out_path_dwpose_mp4_face_hands = os.path.join(save_dir_smpl_render, key + '_face_hands.mp4')
            # out_path_dwpose_mp4_cheek_hands = os.path.join(save_dir_smpl_render, key + '_cheek_hands.mp4')

            
            
            reshape_scale = 0.6
            pool = reshapePool(alpha=reshape_scale)
            motion_keypoints = [keypoints[idx] for idx in motion_indices]
            smpl_render_data = render_nlf_as_images(smpl_data, motion_indices, out_path_smpl_render)
            motion_reshape_results = draw_pose_to_canvas(motion_keypoints, pool=pool, H=height, W=width, reshape_scale=reshape_scale, points_only_flag=False, show_feet_flag=False, aug_body_draw=False)
            motion_reshape_results_face_hands = draw_pose_to_canvas(motion_keypoints, pool=pool, H=height, W=width, reshape_scale=reshape_scale, points_only_flag=False, show_feet_flag=False, aug_body_draw=False, show_body_flag=False)
            motion_noreshape_results_cheek_hands = draw_pose_to_canvas(motion_keypoints, pool=None, H=height, W=width, reshape_scale=0, points_only_flag=False, show_feet_flag=False, aug_body_draw=False, show_body_flag=False, show_face_flag=False, show_cheek_flag=True)


            # save_videos_from_pil(motion_reshape_results, out_path_dwpose_mp4, fps=16) 
            # save_videos_from_pil(motion_reshape_results_face_hands, out_path_dwpose_mp4_face_hands, fps=16)
            # save_videos_from_pil(motion_noreshape_results_cheek_hands, out_path_dwpose_mp4_cheek_hands, fps=16)
            t1 = threading.Thread(target=save_videos_from_pil,
                      args=(motion_reshape_results, out_path_dwpose_mp4, 16))
            t2 = threading.Thread(target=save_videos_from_pil,
                                args=(motion_reshape_results_face_hands, out_path_dwpose_mp4_face_hands, 16))
            t3 = threading.Thread(target=save_videos_from_pil,
                                args=(motion_noreshape_results_cheek_hands, out_path_dwpose_mp4_cheek_hands, 16))
            t1.start()
            t2.start()
            t3.start()
            t1.join()
            t2.join()
            t3.join()

            with open(out_path_dwpose_mp4, "rb") as f:
                dwpose_mp4_data = f.read()
            with open(out_path_smpl_render, "rb") as f:
                smpl_render_data = f.read()
            with open(out_path_dwpose_mp4_face_hands, "rb") as f:
                dwpose_mp4_face_hands = f.read()
            with open(out_path_dwpose_mp4_cheek_hands, "rb") as f:
                dwpose_mp4_cheek_hands = f.read()
            data['append_dwpose_reshape'] = dwpose_mp4_data
            data['append_smpl_render'] = smpl_render_data
            data['append_dwpose_reshape_face_hands'] = dwpose_mp4_face_hands
            data['append_dwpose_noreshape_cheek_hands'] = dwpose_mp4_cheek_hands
            os.remove(out_path_dwpose_mp4_face_hands)    # 清除临时文件
            os.remove(out_path_dwpose_mp4_cheek_hands)    # 清除临时文件
            data.pop('motion_indices')
            writer.write(data)
    with open(output_meta_file, 'w', encoding='utf-8') as outfile:
        writer = jsonlines.Writer(outfile)
        writer.write_all(meta_dict)
        writer.close()

def process_tar_chunk(chunk, input_root, output_root, save_dir_keypoints, save_dir_dwpose_mp4, save_dir_smpl, save_dir_smpl_render):
    for wds_path in chunk:
        rel_path = os.path.relpath(wds_path, input_root)
        output_wds_path = os.path.join(output_root, rel_path)
        reshape_render_to_wds(wds_path, output_wds_path, save_dir_keypoints, save_dir_dwpose_mp4, save_dir_smpl, save_dir_smpl_render)
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0923_step3',
                        help='Input root')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0923_step4',
                        help='Output root')
    parser.add_argument('--max_processes', type=int, default=32,
                        help='Max processes')

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
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Split wds_list into chunks
    input_tar_paths = glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True)
    chunk_size = len(input_tar_paths) // max_processes
    if len(input_tar_paths) % max_processes != 0:
        chunk_size += 1
    chunks = [input_tar_paths[i:i + chunk_size] for i in range(0, len(input_tar_paths), chunk_size)]

    # 测试
    # process_tar_chunk(chunks[0], input_dir, output_dir, save_dir_keypoints, save_dir_dwpose_reshape_mp4, save_dir_smpl, save_dir_smpl_render)
    
    for chunk_idx, chunk in enumerate(chunks):
        p = Process(
            target=process_tar_chunk,
            args=(chunk, input_dir, output_dir, save_dir_keypoints, save_dir_dwpose_reshape_mp4, save_dir_smpl, save_dir_smpl_render)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        gc.collect()






    
