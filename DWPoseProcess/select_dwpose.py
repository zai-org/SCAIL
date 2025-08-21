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


def process_video_to_indices(keypoint_path, bbox_path, height, width, multi_person):
    poses = torch.load(keypoint_path)
    bboxes = torch.load(bbox_path)

    H, W = height, width
    max_slide_attempts = 30
    start_index = 8
    # 定义可选的 motion_part_len 值
    possible_lengths = [60, 75, 90, 100, 110, 135, 150, 180, 210, 250]
    valid_lengths = [length for length in possible_lengths if length < len(poses) - 1]
    valid_lengths.sort(reverse=True)


    final_motion_indices = None

    for motion_part_len in valid_lengths:
        for attempt in range(max_slide_attempts):
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
                    delta_check_result = check_from_keypoints_stick_movement(motion_part_poses, angle_threshold=0.03)
                    if not delta_check_result:
                        start_index += random.randint(3, 4)
                        continue
                    final_ref_image_indices = [final_ref_image_indice]
                else:
                    delta_check_result = check_from_keypoints_stick_movement(motion_part_poses, angle_threshold=0.03)
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
            return final_motion_indices, final_ref_image_indices
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

def process_tar(wds_chunk, chunk_id, output_root, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, save_dir_caption, save_dir_caption_multi, random_slow_play):
    meta_dict = {}
    obj_list = []
    sample_list = []
    for _, wds_path in enumerate(wds_chunk):
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
        wds.SimpleShardList(wds_chunk, seed=None),
        wds.tarfile_to_samples(),
        partial(process_fn_video, meta_dict=meta_dict),
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, prefetch_factor=8)
    data_iter = iter(dataloader)

    for data_batch in tqdm(data_iter):
        data = {}
        for k, v in data_batch.items():
            data[k] = v[0]
        try:
            key = data['__key__']
            height = data.get('height', None)
            width = data.get('width', None)
            multi_path = os.path.join(save_dir_caption_multi, key + '.txt')
            single_path = os.path.join(save_dir_caption, key + '.txt')

            out_path_keypoint = os.path.join(save_dir_keypoints, key + '.pt')
            out_path_bbox = os.path.join(save_dir_bboxes, key + '.pt')
            out_path_mp4 = os.path.join(save_dir_mp4, key + '.mp4')

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
            with open(out_path_bbox, "rb") as f:
                bbox_data = f.read()

            process_result = process_video_to_indices(out_path_keypoint, out_path_bbox, height, width, multi_person)
            if process_result is None:
                continue
            else:
                final_motion_indices, final_ref_image_indices = process_result
            obj = meta_dict.get(key, None)
            if obj is None:
                print(f"skip {key}, no meta")
                continue
            obj.update({'motion_indices': final_motion_indices, 'ref_image_indices': final_ref_image_indices, 'random_slow_play': random_slow_play})
            with open(out_path_mp4, "rb") as f:
                mp4_data = f.read()
            data['dwpose'] = mp4_data
            data['recaption'] = txt_data
            data['bbox'] = bbox_data
            data.pop('height', None)
            data.pop('width', None)
            data.pop('fps', None)
            sample_list.append(data)
            obj_list.append(obj)
            
        except Exception as e:
            print(f"Error processing video {key}: {e}")
            continue

    # 2) 分 shard
    chunk_size=100
    output_pattern = "%06d"
    total_count = len(obj_list)
    print(f"Total samples: {total_count}")
     # 计算分片数量 (可用 total_count // chunk_size)
    num_shards = math.ceil(total_count / chunk_size)
    print(f"Will produce {num_shards} shards, each up to {chunk_size} samples")

    # 3) 对分好的shard进行处理
    for shard_id in range(num_shards):
        start_idx = shard_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_count)
        sample_shards = sample_list[start_idx:end_idx]
        obj_shards = obj_list[start_idx:end_idx]

        shard_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.tar'
        jsonl_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.meta.jsonl'

        with TarWriter(shard_file) as writer:
            for sample in sample_shards:
                writer.write(sample)
        with open(jsonl_file, 'w', encoding='utf-8') as outfile:
                writer = jsonlines.Writer(outfile)
                writer.write_all(obj_shards)
                writer.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')

    args = parser.parse_args()
    config = load_config(args.config)

    wds_root = config.get('wds_root', '')
    output_root = config.get('output_root', '')
    os.makedirs(output_root, exist_ok=True)
    tar_paths = [file for file in os.listdir(wds_root) if file.endswith('.tar')]
    video_root = config.get('video_root', '')
    random_slow_play = config.get('random_slow_play', 0)
    name_args = config.get('name_args', {'keypoint_suffix_name': 'keypoints', 'bbox_suffix_name': 'bboxes', 'mp4_suffix_name': 'dwpose', 'caption_suffix_name': 'caption', 'caption_suffix_name_multi': 'caption_multi'})

    save_dir_keypoints = os.path.join(video_root, name_args['keypoint_suffix_name'])
    save_dir_bboxes = os.path.join(video_root, name_args['bbox_suffix_name'])
    save_dir_mp4 = os.path.join(video_root, name_args['mp4_suffix_name'])
    save_dir_caption = os.path.join(video_root, name_args['caption_suffix_name'])
    save_dir_caption_multi = os.path.join(video_root, name_args['caption_suffix_name_multi'])

    os.makedirs(save_dir_keypoints, exist_ok=True)
    os.makedirs(save_dir_bboxes, exist_ok=True)
    os.makedirs(save_dir_mp4, exist_ok=True)
    os.makedirs(save_dir_caption, exist_ok=True)
    os.makedirs(save_dir_caption_multi, exist_ok=True)

    wds_list = [os.path.join(wds_root, file) for file in tar_paths]
    # 分chunk
    max_items_per_chunk = 2500
    chunks = []
    current_chunk = []
    current_count = 0

    # 先统计每个 wds_path 对应的 item 数
    wds_info = []
    for wds_path in wds_list:
        meta_file = wds_path.replace('.tar', '.meta.jsonl')
        item_num = sum(1 for _ in open(meta_file))
        wds_info.append((wds_path, item_num))

    # 分 chunk
    for path, count in wds_info:
        if count > max_items_per_chunk:
            chunks.append([path])
            continue

        if current_count + count <= max_items_per_chunk:
            current_chunk.append(path)
            current_count += count
        else:
            chunks.append(current_chunk)
            current_chunk = [path]
            current_count = count
        
    # 把最后一个 chunk 加进去
    if current_chunk:
        chunks.append(current_chunk)
    # for chunk_idx, chunk in tqdm(enumerate(chunks), desc='Processing chunks', total=len(chunks)):
    #     process_tar(chunk, chunk_idx, output_root, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, save_dir_caption, save_dir_caption_multi, random_slow_play)
    processes = []  # 存储进程的列表
    max_processes = 24  # 最大并发进程数
    for chunk_idx, chunk in tqdm(enumerate(chunks), desc='Processing chunks', total=len(chunks)):
        p = Process(
            target=process_tar,
            args=(chunk, chunk_idx, output_root, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, save_dir_caption, save_dir_caption_multi, random_slow_play)
        )
        p.start()
        processes.append(p)
        if len(processes) >= max_processes:
            processes[0].join()
            processes.pop(0)
            gc.collect()

    for p in processes:
        p.join()
        gc.collect()






    
