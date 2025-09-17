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
from pose_draw.draw_pose_main import draw_pose_to_canvas
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


def process_video_to_indices_no_filter(keypoint_path, bbox_path, height, width, fps, multi_person):
    try:
        ori_poses = torch.load(keypoint_path, weights_only=False)
        ori_bboxes = torch.load(bbox_path, weights_only=False)
        target_fps = 16
        pick_indices = np.arange(0, len(ori_poses), fps / target_fps).astype(int)
        possible_lengths = [65, 81, 100, 130, 146, 162, 200]
        if len(pick_indices) < 65:
            return None
        else:
            for length in possible_lengths:
                if len(pick_indices) >= length:
                    # 从pick_indices中随机选择一个起始位置，然后取连续的length帧
                    max_start = len(pick_indices) - length
                    start_idx = np.random.randint(0, max_start + 1)
                    selected_indices = pick_indices[start_idx:start_idx + length]
                    
                    # 从当前帧之前的30帧范围内随机选择参考帧
                    ref_start = max(0, start_idx - 30)
                    ref_end = start_idx
                    selected_ref_indices = pick_indices[ref_start:ref_end + 1]
                    
                    return selected_indices.tolist(), selected_ref_indices.tolist()  # 返回选中的连续帧索引和随机选择的参考帧
    except Exception as e:
        print(f"Error processing video {keypoint_path}: {e}, continue")
    return None

def process_video_to_indices(keypoint_path, bbox_path, height, width, fps, multi_person):   # TODO: 还是修改一下，16fps的interval在这里就可以取了
    try:
        ori_poses = torch.load(keypoint_path, weights_only=False)
        ori_bboxes = torch.load(bbox_path, weights_only=False)
        target_fps = 16

        H, W = height, width
        max_slide_attempts = 30
        # 定义可选的 motion_part_len 值
        possible_lengths = [49, 65, 81, 130, 146, 162, 200]
        pick_indices = np.arange(0, len(ori_poses), fps / target_fps).astype(int)  # 比如orilist 0-10， downsample成 newlist [0 2 4 6 8], 那么newlist[1] 对应原来 orilist[2]，直接用即可
        poses = [ori_poses[index] for index in pick_indices]
        bboxes = [ori_bboxes[index] for index in pick_indices]
        valid_lengths = [length for length in possible_lengths if length < len(poses) - 1]
        valid_lengths.sort(reverse=True)


        final_motion_indices = None

        for motion_part_len in valid_lengths:
            start_index = 8
            for _ in range(max_slide_attempts):
                end = int(start_index + motion_part_len)
                if end >= len(poses):
                    start_index -= 2    # 如果走太多就先后退一步
                    if start_index <= 0:
                        break   # 跳出内层sliding loop
                    continue    # 不用while，记做总体重复次数

                motion_part_indices = np.arange(start_index, end, 1).astype(int)
                ref_part_indices = np.arange(max(start_index-15, 0), start_index + 1, 1).astype(int)

                motion_part_poses = [poses[index] for index in motion_part_indices]
                motion_part_bboxes = [bboxes[index] for index in motion_part_indices]
                ref_part_poses = [poses[index] for index in ref_part_indices]
                ref_part_bboxes = [bboxes[index] for index in ref_part_indices]

                # 下面这四个筛选逻辑，除了bbox本身的，其他只对第0个bbox里的骨骼进行筛选
                motion_part_bbox_check_result = check_from_keypoints_bbox(motion_part_poses, motion_part_bboxes, IoU_thresthold=0.3, reference_width=W, reference_height=H, multi_person=multi_person)
                motion_part_core_check_result = check_from_keypoints_core_keypoints(motion_part_poses, motion_part_bboxes)
                ref_part_check_indices = get_valid_indice_from_keypoints(ref_part_poses, ref_part_indices)
                
                if len(ref_part_check_indices) > 0 and motion_part_bbox_check_result and motion_part_core_check_result:    # 这里只要满足正脸条件就可以，其它都留着
                    if multi_person:
                        final_ref_image_indice = select_ref_from_keypoints_bbox_multi(ref_part_indices, ref_part_bboxes, motion_part_bboxes)
                        if final_ref_image_indice is None:
                            start_index += random.randint(3, 4)
                            continue
                        delta_check_result = check_from_keypoints_stick_movement(motion_part_poses, angle_threshold=0.05)
                        if not delta_check_result:
                            start_index += random.randint(3, 4)
                            continue
                        final_ref_image_indices = [final_ref_image_indice]
                    else:
                        delta_check_result = check_from_keypoints_stick_movement(motion_part_poses, angle_threshold=0.06)
                        if not delta_check_result:
                            start_index += random.randint(3, 4)
                            continue
                        final_ref_image_indices = ref_part_check_indices
                    # 转换索引
                    final_motion_indices = motion_part_indices.tolist()
                    break
                else:
                    start_index += random.randint(3, 4)
                    continue

            if final_motion_indices is None:
                continue
            else:
                final_motion_indices = [int(pick_indices[idx]) for idx in final_motion_indices]
                final_ref_image_indices = [int(pick_indices[idx]) for idx in final_ref_image_indices]
                return final_motion_indices, final_ref_image_indices
    except Exception as e:
        print(f"Error processing video {keypoint_path}: {e}, continue")
    return None


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
        
        h = r.get('height', None)
        w = r.get('width', None)
        fps = r.get('fps', None)
        if h is None or w is None or fps is None:
            print(f"skip {r['__key__']}, no width or height or fps")
            continue
        
        item.update({'height': h, 'width': w, 'fps': fps})

        yield item

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_tar(wds_chunk, chunk_id, output_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, filter_args, eval_list):
    obj_list = []
    sample_list = []
    shard_size = 100
    shard_id = 0
    output_pattern = "%06d"
    for _, wds_path in tqdm(enumerate(wds_chunk), total=len(wds_chunk), disable=(chunk_id != 0)):
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
        dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
            partial(process_fn_video, meta_dict=meta_dict),
        )
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
        data_iter = iter(dataloader)

        for data_batch in data_iter:
            data = {}
            for k, v in data_batch.items():
                data[k] = v[0]
            try:
                key = data['__key__']
                height = data.get('height', None)
                width = data.get('width', None)
                fps = data.get('fps', None)
                multi_path = os.path.join(save_dir_caption_multi, key + '.txt')
                single_path = os.path.join(save_dir_caption, key + '.txt')
                if key in eval_list:
                    print(f"exclude {key}, in eval list")
                    continue
                out_path_keypoint = os.path.join(save_dir_keypoints, key + '.pt')
                out_path_bbox = os.path.join(save_dir_bboxes, key + '.pt')
                out_path_hands = os.path.join(save_dir_hands, key + '.pt')
                out_path_faces = os.path.join(save_dir_faces, key + '.pt')
                out_path_mp4 = os.path.join(save_dir_dwpose_mp4, key + '.mp4')

                if os.path.exists(multi_path):
                    multi_person = True
                    with open(multi_path, "r", encoding="utf-8") as f:
                        txt_data = f.read()
                elif os.path.exists(single_path):
                    multi_person = False
                    with open(single_path, "r", encoding="utf-8") as f:
                        txt_data = f.read()
                else:
                    continue
                if os.path.exists(out_path_bbox) and os.path.exists(out_path_hands) and os.path.exists(out_path_faces):
                    pass
                else:
                    continue
                with open(out_path_bbox, "rb") as f:
                    bbox_data = f.read()
                with open(out_path_hands, "rb") as f:
                    hands_data = f.read()
                with open(out_path_faces, "rb") as f:
                    faces_data = f.read()

                if filter_args is None or filter_args.get('use_filter', False) == True:
                    process_result = process_video_to_indices(out_path_keypoint, out_path_bbox, height, width, fps, multi_person)              
                    if process_result is None:
                        continue
                elif filter_args.get('use_filter', False) == False:
                    process_result = process_video_to_indices_no_filter(out_path_keypoint, out_path_bbox, height, width, fps, multi_person)
                    if process_result is None:
                        continue

                final_motion_indices, final_ref_image_indices = process_result
                obj = meta_dict.get(key, None)
                if obj is None:
                    print(f"skip {key}, no meta")
                    continue
                obj.update({'motion_indices': final_motion_indices, 'ref_image_indices': final_ref_image_indices})
                with open(out_path_mp4, "rb") as f:
                    mp4_data = f.read()
                data['dwpose'] = mp4_data
                data['recaption'] = txt_data
                data['bbox'] = bbox_data
                data['hands'] = hands_data
                data['faces'] = faces_data
                data.pop('height', None)
                data.pop('width', None)
                data.pop('fps', None)
                sample_list.append(data)
                obj_list.append(obj)
                
                # Write shard when it reaches the target size
                if len(sample_list) >= shard_size:
                    shard_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.tar'
                    jsonl_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.meta.jsonl'
                    
                    with TarWriter(shard_file) as writer:
                        for sample in sample_list:
                            writer.write(sample)
                    with open(jsonl_file, 'w', encoding='utf-8') as outfile:
                        writer = jsonlines.Writer(outfile)
                        writer.write_all(obj_list)
                        writer.close()
                    
                    print(f"Written shard {shard_id} with {len(sample_list)} samples")
                    sample_list = []
                    obj_list = []
                    shard_id += 1
            
            except Exception as e:
                print(f"Error processing video {key}: {e}")
                continue

    # Write remaining samples if any
    if len(sample_list) > 0:
        shard_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.tar'
        jsonl_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.meta.jsonl'
        
        with TarWriter(shard_file) as writer:
            for sample in sample_list:
                writer.write(sample)
        with open(jsonl_file, 'w', encoding='utf-8') as outfile:
            writer = jsonlines.Writer(outfile)
            writer.write_all(obj_list)
            writer.close()
        
        print(f"Written final shard {shard_id} with {len(sample_list)} samples")

def get_eval_list():
    eval_list = []
    eval_dirs = ["/workspace/ys_data/evaluation_300/DWPose/videos"]
    clean_eval_dirs = ["/workspace/ywh_data/evaluation_80_clear/videos"]
    for eval_dir in eval_dirs:
        for video_name in os.listdir(eval_dir):
            eval_list.append(os.path.splitext(video_name)[0].split('_', 1)[1])
    for clean_eval_dir in clean_eval_dirs:
        for video_name in os.listdir(clean_eval_dir):
            eval_list.append(os.path.splitext(video_name)[0])
    return eval_list

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--force_no_filter', action='store_true', default=False,
                        help='Force no filter')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_default',
                        help='Output root')
    parser.add_argument('--max_processes', type=int, default=8,
                        help='Max processes')

    args = parser.parse_args()
    config = load_config(args.config)

    wds_root = config.get('wds_root', '')
    tar_paths = glob.glob(os.path.join(wds_root, "**", "*.tar"), recursive=True)
    video_root = config.get('video_root', '')
    filter_args = config.get('filter_args', None)
    if args.force_no_filter:
        filter_args['use_filter'] = False
    output_root = os.path.join(args.output_root, os.path.basename(os.path.normpath(video_root)))
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    save_dir_keypoints = os.path.join(video_root, 'keypoints')
    save_dir_bboxes = os.path.join(video_root, 'bboxes')
    save_dir_dwpose_mp4 = os.path.join(video_root, 'dwpose')
    save_dir_hands = os.path.join(video_root, 'hands')
    save_dir_faces = os.path.join(video_root, 'faces')
    save_dir_caption = os.path.join(video_root, 'caption')
    save_dir_caption_multi = os.path.join(video_root, 'caption_multi')

    os.makedirs(save_dir_keypoints, exist_ok=True)
    os.makedirs(save_dir_bboxes, exist_ok=True)
    os.makedirs(save_dir_dwpose_mp4, exist_ok=True)
    os.makedirs(save_dir_hands, exist_ok=True)
    os.makedirs(save_dir_faces, exist_ok=True)
    os.makedirs(save_dir_caption, exist_ok=True)
    os.makedirs(save_dir_caption_multi, exist_ok=True)


    processes = []  # 存储进程的列表
    max_processes = args.max_processes

    # Split wds_list into chunks
    random.shuffle(tar_paths)
    chunk_size = len(tar_paths) // max_processes
    if len(tar_paths) % max_processes != 0:
        chunk_size += 1
    chunks = [tar_paths[i:i + chunk_size] for i in range(0, len(tar_paths), chunk_size)]
    eval_list = get_eval_list()
    
    for chunk_idx, chunk in enumerate(chunks):
        p = Process(
            target=process_tar,
            args=(chunk, chunk_idx, output_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, filter_args, eval_list)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        gc.collect()






    
