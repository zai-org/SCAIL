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
import pickle
import copy
from NLFPoseExtract.nlf_draw import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_to_p2d
from NLFPoseExtract.smpl_joint_xyz import compute_motion_speed, compute_motion_range


def project_dwpose_to_3d(dwpose_keypoint, original_threed_keypoint, focal, princpt, H, W):
    # 相机内参
    # fx, fy = focal, focal
    fx, fy = focal
    cx, cy = princpt

    # 2D 关键点坐标
    x_2d, y_2d = dwpose_keypoint[0] * W, dwpose_keypoint[1] * H

    # 原始 3D 点（相机坐标系下）
    ori_x, ori_y, ori_z = original_threed_keypoint

    # 使用新的 2D 点和原始深度反投影计算新的 3D 点
    # 公式: x = (u - cx) * z / fx
    new_x = (x_2d - cx) * ori_z / fx
    new_y = (y_2d - cy) * ori_z / fy
    new_z = ori_z  # 保持深度不变

    return [new_x, new_y, new_z]

def check_2d_3d_match(dwpose_data, video_height, video_width, smpl_extracted_data, multi_person):
    match_t = []
    nozero_t = []

    limb_seq = [
        [1, 2],    # 0 Neck -> R. Shoulder
        [1, 5],    # 1 Neck -> L. Shoulder
        [2, 3],    # 2 R. Shoulder -> R. Elbow
        [3, 4],    # 3 R. Elbow -> R. Wrist
        [5, 6],    # 4 L. Shoulder -> L. Elbow
        [6, 7],    # 5 L. Elbow -> L. Wrist
        [1, 8],    # 6 Neck -> R. Hip
        [8, 9],    # 7 R. Hip -> R. Knee
        [9, 10],   # 8 R. Knee -> R. Ankle
        [1, 11],   # 9 Neck -> L. Hip
        [11, 12],  # 10 L. Hip -> L. Knee
        [12, 13],  # 11 L. Knee -> L. Ankle
        [1, 0],    # 12 Neck -> Nose
        # [0, 14],   # 13 Nose -> R. Eye
        # [14, 16],  # 14 R. Eye -> R. Ear
        # [0, 15],   # 15 Nose -> L. Eye
        # [15, 17],  # 16 L. Eye -> L. Ear
    ]

    for t in range(len(dwpose_data)):
        dwpose_kpts = copy.deepcopy(dwpose_data[t]['bodies']['candidate'])
        smpl_kpts = smpl_extracted_data[t]
        # 目前只支持单人
        if not multi_person:
            if dwpose_kpts.shape[0] == 0 or smpl_kpts.shape[0] == 0:
                continue
            else:
                nozero_t.append(t)
                dwpose_2d_joints = dwpose_kpts[0]
                smpl_3d_joints = smpl_kpts[0].cpu().numpy()   # torch->numpy
                camera_matrix = intrinsic_matrix_from_field_of_view((video_height, video_width))
                focal = camera_matrix[0, 0], camera_matrix[1, 1]
                princpt = camera_matrix[0, 2], camera_matrix[1, 2]
                smpl_3d_joints = process_data_to_COCO_format(smpl_3d_joints)[:14]   # 24->18点->14点
                smpl_2d_joints = p3d_to_p2d(smpl_3d_joints, video_height, video_width)[0]
                def out_of_screen(joint):
                    if joint[0] < 0 or joint[0] > video_width or joint[1] < 0 or joint[1] > video_height:
                        return True
                    else:
                        return False
                dwpose_3d_joints = np.zeros((18, 3), dtype=smpl_3d_joints.dtype)
                for j in range(14):  # 只取关键的几个点
                    if dwpose_2d_joints[j][0] == -1 or dwpose_2d_joints[j][1] == -1:
                        continue
                    else:
                        dwpose_3d_joints[j] = project_dwpose_to_3d(dwpose_2d_joints[j], smpl_3d_joints[j], focal, princpt, video_height, video_width)
                match_flag = True
                for line_idx in limb_seq:
                    start, end = line_idx[0], line_idx[1]
                    if np.sum(dwpose_3d_joints[start]) == 0 or np.sum(dwpose_3d_joints[end]) == 0 or out_of_screen(smpl_2d_joints[start]) or out_of_screen(smpl_2d_joints[end]):  # 没有识别出的点，或在屏幕外的点
                        continue
                    else:
                        vec_dwpose = np.array(dwpose_3d_joints[end]) - np.array(dwpose_3d_joints[start])
                        vec_smpl = np.array(smpl_3d_joints[end]) - np.array(smpl_3d_joints[start])
                        vec_dwpose_len = np.linalg.norm(vec_dwpose)
                        vec_smpl_len = np.linalg.norm(vec_smpl)
                        if vec_dwpose_len > vec_smpl_len * 2.2 or vec_dwpose_len < vec_smpl_len * 0.4:
                            match_flag = False
                            break
                if match_flag:
                    match_t.append(t)
        else:
            if dwpose_kpts.shape[0] == 0 or smpl_kpts.shape[0] == 0:
                continue
            elif dwpose_kpts.shape[0] != smpl_kpts.shape[0]:
                continue
            else:
                nozero_t.append(t)
                match_t.append(t)
    return match_t, nozero_t

def process_video_to_indices(keypoint_path, bbox_path, smpl_path, height, width, fps, multi_person, use_filter=True): 
    try:
        ori_poses = torch.load(keypoint_path, weights_only=False)
        ori_bboxes = torch.load(bbox_path, weights_only=False)
        ori_smpl = pickle.load(open(smpl_path, 'rb'))
        smpl_data = ori_smpl['pose']['joints3d_nonparam']
        target_fps = 16

        H, W = height, width
        max_slide_attempts = 80
        # 定义可选的 motion_part_len 值
        possible_lengths = [65, 81, 162]
        if use_filter:
            pick_indices = np.arange(2, len(ori_poses)-2, fps / target_fps).astype(int)  # 比如orilist 0-10， downsample成 newlist [0 2 4 6 8], 那么newlist[1] 对应原来 orilist[2]，直接用即可 需要去掉前后两帧
        else:
            pick_indices = np.arange(0, len(ori_poses), 1).astype(int)   # 不去掉首尾帧，并且固定fps=16（因为是生成的，尽量保持81帧数都能取到）

        poses = [ori_poses[index] for index in pick_indices]
        bboxes = [ori_bboxes[index] for index in pick_indices]
        smpl_extracted_data = [smpl_data[index] for index in pick_indices]
        valid_lengths = [length for length in possible_lengths if length <= len(poses)]
        valid_lengths.sort(reverse=True)
        final_motion_indices = None
        check_2d_3d_match_result, nozero_t_result = check_2d_3d_match(poses.copy(), H, W, smpl_extracted_data, multi_person)   # 取了Pick_indices之后的t

        for motion_part_len in valid_lengths:
            start_index = 8
            for _ in range(max_slide_attempts):
                end = int(start_index + motion_part_len)
                if end > len(poses):
                    start_index -= 2    # 如果走太多就先后退一步，对no_filter也可以退到0
                    if start_index < 0:
                        break   # 跳出内层sliding loop
                    continue    # 不用while，记做总体重复次数

                motion_part_indices = np.arange(start_index, end, 1).astype(int)
                ref_part_indices = np.arange(max(start_index-15, 0), start_index + 1, 1).astype(int)  # 对Synthetic视频，就是首帧
                motion_part_indices_in_nozero_t = [index for index in motion_part_indices if index in nozero_t_result]
                if len(motion_part_indices_in_nozero_t) < 0.5 * motion_part_len:   # 正确的太少了
                    start_index += random.randint(3, 4)
                    continue

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
                        final_ref_image_indices = [final_ref_image_indice]
                    else:
                        final_ref_image_indices = ref_part_check_indices
                    # 转换索引
                    joints3d = ori_smpl['pose']['joints3d_nonparam']
                    joints3d = [joints3d[i] for i in motion_part_indices.tolist()]
                    motion_speed = compute_motion_speed(joints3d)
                    if motion_speed is None or motion_speed < 16:
                        start_index += random.randint(3, 4)
                        continue
                    else:
                        final_motion_indices = motion_part_indices.tolist()
                        break   # 退出循环，返回
                else:
                    start_index += random.randint(3, 4)
                    continue

            if final_motion_indices is None:
                continue
            else:
                final_motion_indices = [int(pick_indices[idx]) for idx in final_motion_indices]
                final_ref_image_indices = [int(pick_indices[idx]) for idx in final_ref_image_indices]
                final_check_2d_3d_match_result = [int(pick_indices[idx]) for idx in check_2d_3d_match_result]
                return final_motion_indices, final_ref_image_indices, final_check_2d_3d_match_result, motion_speed
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

def process_tar(wds_chunk, chunk_id, output_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_smpl, save_dir_caption, save_dir_caption_multi, filter_args, eval_list):
    print(f"Processing chunk {chunk_id}")
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
                out_path_smpl = os.path.join(save_dir_smpl, key + '.pkl')

                if os.path.exists(multi_path):
                    multi_person = True
                elif os.path.exists(single_path):
                    multi_person = False
                else:
                    continue
                if not os.path.exists(out_path_keypoint):
                    print(f"skip {key}, no keypoint")
                    continue

                if filter_args is None or filter_args.get('use_filter', False) == True:
                    process_result = process_video_to_indices(out_path_keypoint, out_path_bbox, out_path_smpl, height, width, fps, multi_person, use_filter=True)              
                elif filter_args.get('use_filter', False) == False:
                    process_result = process_video_to_indices(out_path_keypoint, out_path_bbox, out_path_smpl, height, width, fps, multi_person, use_filter=False)
                if process_result is None:
                    continue

                final_motion_indices, final_ref_image_indices, final_check_2d_3d_match_result, motion_speed = process_result
                obj = meta_dict.get(key, None)
                if obj is None:
                    print(f"skip {key}, no meta")
                    continue
                obj.update({'motion_indices': final_motion_indices, 'ref_image_indices': final_ref_image_indices, 'check_2d_3d_match_result': final_check_2d_3d_match_result, 'motion_speed': motion_speed})

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
    clean_eval_dirs = ["/workspace/ywh_data/eval_hq_v2/videos"]
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
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_pack_wds_0923_step1',
                        help='Input root')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_default',
                        help='Output root')
    parser.add_argument('--max_processes', type=int, default=8,
                        help='Max processes')

    args = parser.parse_args()
    config = load_config(args.config)

    video_root = config.get('video_root', '')
    wds_root = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    tar_paths = glob.glob(os.path.join(wds_root, "**", "*.tar"), recursive=True)
    filter_args = config.get('filter_args', None)
    output_root = os.path.join(args.output_root, os.path.basename(os.path.normpath(video_root)))
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    save_dir_keypoints = os.path.join(video_root, 'keypoints')
    save_dir_smpl = os.path.join(video_root, 'smpl')
    save_dir_bboxes = os.path.join(video_root, 'bboxes')
    save_dir_dwpose_mp4 = os.path.join(video_root, 'dwpose')
    save_dir_hands = os.path.join(video_root, 'hands')
    save_dir_faces = os.path.join(video_root, 'faces')
    save_dir_caption = os.path.join(video_root, 'caption')
    save_dir_caption_multi = os.path.join(video_root, 'caption_multi')

    os.makedirs(save_dir_keypoints, exist_ok=True)
    os.makedirs(save_dir_smpl, exist_ok=True)
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

    # 串行
    # process_tar(tar_paths, 0, output_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_smpl, save_dir_caption, save_dir_caption_multi, filter_args, eval_list)

    # 并行
    for chunk_idx, chunk in enumerate(chunks):
        p = Process(
            target=process_tar,
            args=(chunk, chunk_idx, output_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_smpl, save_dir_caption, save_dir_caption_multi, filter_args, eval_list)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        gc.collect()







    
