import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.AAUtils import save_videos_from_pil
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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, TimeoutError, as_completed
from AAUtils import read_frames
from decord import VideoReader
from fractions import Fraction
import io
import gc
from PIL import Image
from multiprocessing import Process
import decord
import json
import glob
import sys

def get_bbox_from_position_list(position_list):
    # 输入list of xy (x y 0-1), 输出bbox (x1 y1 x2 y2)
    x_list = [xy[0] for xy in position_list]
    y_list = [xy[1] for xy in position_list]
    x1 = min(x_list)
    y1 = min(y_list)
    x2 = max(x_list)
    y2 = max(y_list)
    return x1, y1, x2, y2


def convert_scores_to_specific_bboxes(poses, scores, type='hands', score_type='hand_score', score_threshold=0.75):
    # 输入scores, a list of dict, 每个dict包含body_score, hand_score, face_score, body_score是[n, 24]的矩阵，hand_score是[2*n, 21]的矩阵，face_score是[n, 68]的矩阵
    # 如果平均值>0.3，再根据这些分数框bbox
    specific_det_bboxes = [[] for _ in range(len(poses))]
    for i, (pose, score) in enumerate(zip(poses, scores)):
        part_score = score[score_type]
        for part_idx, part_score in enumerate(part_score):     # 可能有左右手，但是不影响此次遍历
            mean_part_score = np.mean(part_score)
            if mean_part_score > score_threshold:
                part_bbox = get_bbox_from_position_list(pose[type][part_idx])
                specific_det_bboxes[i].append(part_bbox)
    return specific_det_bboxes

def calculate_video_mean_and_std_pil(frames):
    variances = []
    means = []
    
    for frame in frames:
        # 将 PIL 图像转换为灰度图
        gray_frame = frame.convert('L')
        
        # 转为 NumPy 数组
        gray_array = np.array(gray_frame)
        
        # 计算每帧的均值和标准差
        frame_mean = np.mean(gray_array)
        frame_std = np.std(gray_array)
        
        means.append(frame_mean)
        variances.append(frame_std)
    
    # 计算平均均值和平均标准差
    average_mean = np.mean(means)
    average_std = np.mean(variances)
    
    return average_std, average_mean

def process_single_video_with_timeout(*args, **kwargs):
    timeout_seconds = 180

    def task():
        process_single_video(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        try:
            future.result(timeout=timeout_seconds)
        except TimeoutError:
            print(f"超时：超过 {timeout_seconds} 秒，已跳过。")
        except Exception as e:
            print(f"处理视频出错：{str(e)}")


def process_single_video(detector, key, video_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, filter_args):
    use_filter=filter_args['use_filter']
    save_mp4=filter_args['save_mp4']

    try:
        tmp_dir = "/dev/shm/tmp"
        pt_path = os.path.join(tmp_dir, key + '.pt')
        frames_tensor = torch.load(pt_path)
        os.unlink(pt_path)
    except Exception as e:
        print(f"Load Tensor Failed: {str(e)}")
        return

    multi_path =  os.path.join(video_root, 'labels_multi', key + '.txt')
    single_path = os.path.join(video_root, 'labels', key + '.txt')
    fail_path = os.path.join(video_root, 'labels_person_fail', key + '.txt')
    target_multi_path = os.path.join(save_dir_caption_multi, key + '.txt')
    target_single_path = os.path.join(save_dir_caption, key + '.txt')

    out_path_keypoint = os.path.join(save_dir_keypoints, key + '.pt')
    out_path_bbox = os.path.join(save_dir_bboxes, key + '.pt')
    out_path_hands = os.path.join(save_dir_hands, key + '.pt')
    out_path_faces = os.path.join(save_dir_faces, key + '.pt')
    out_path_dwpose_mp4 = os.path.join(save_dir_dwpose_mp4, key + '.mp4')

    # output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    # if not output_dir.exists():
    #     output_dir.mkdir(parents=True, exist_ok=True)

    detector_return_list = []
    multi_person = True
    if os.path.exists(single_path):
        multi_person = False

    # 逐帧解码
    pil_frames = []
    for i in range(len(frames_tensor)):
        pil_frame = Image.fromarray(frames_tensor[i].numpy())
        pil_frames.append(pil_frame)
        detector_result = detector(pil_frame)
        _, _, det_result = detector_result
        if use_filter:
            if multi_person:
                if not check_multi_human_requirements(det_result):
                    return
            else:
                if not check_single_human_requirements(det_result):
                    return

        detector_return_list.append(detector_result)


    W, H = pil_frames[0].size
    
    if use_filter:
        mean, std = calculate_video_mean_and_std_pil(pil_frames)
        if mean < 30:
            return
    del pil_frames

    if multi_person:
        if os.path.exists(multi_path):
            shutil.copyfile(multi_path, target_multi_path)
        elif os.path.exists(fail_path):
            shutil.copyfile(fail_path, target_multi_path)
    else:
        shutil.copyfile(single_path, target_single_path)

    poses, scores, det_results = zip(*detector_return_list) # 这里存的是整个视频的poses
    hands_bboxes = convert_scores_to_specific_bboxes(poses, scores, type='hands', score_type='hand_score', score_threshold=0.72)
    faces_bboxes = convert_scores_to_specific_bboxes(poses, scores, type='faces', score_type='face_score', score_threshold=0.9)
    poses, det_results = human_select(poses, det_results, multi_person)

    if save_mp4:
        mp4_results = draw_pose_to_canvas(poses, pool=None, H=H, W=W, reshape_scale=0, points_only_flag=False, show_feet_flag=False)
        save_videos_from_pil(mp4_results, out_path_dwpose_mp4, fps=16) # 实际可能fps不一致，需要视频fps存，这个是否会影响训练？
        # 暂时不会 --> 因为视频训练时对视频取indice，frames和pose frames长度上必定一致
    
    torch.save(poses, out_path_keypoint)
    torch.save(det_results, out_path_bbox)
    torch.save(hands_bboxes, out_path_hands)
    torch.save(faces_bboxes, out_path_faces)



def process_fn_video(src, meta_dict=None, video_root=None, filter_args=None):
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
        key = r['__key__']

        if os.path.exists(os.path.join(video_root, 'labels', key + '.txt')) or os.path.exists(os.path.join(video_root, 'labels_multi', key + '.txt')) or os.path.exists(os.path.join(video_root, 'labels_person_fail', key + '.txt')):
            if filter_args['use_fail_prompt'] == False:
                if os.path.exists(os.path.join(video_root, 'labels_person_fail', key + '.txt')):
                    continue
            mp4_bytes = r.get("mp4", None)
            h = r.get('height', None)
            w = r.get('width', None)
            fps = r.get('fps', None)
            if mp4_bytes is None or h is None or w is None or fps is None:
                continue
            else:
                try:
                    decord.bridge.set_bridge("torch")
                    tmp_dir = "/dev/shm/tmp"

                    vr = VideoReader(io.BytesIO(mp4_bytes))
                    vr_len = len(vr)
                    if vr_len / fps > 24 or vr_len > 1000:   # 视频过长
                        del vr
                        continue
                    frame_indices = list(range(vr_len))
                    frames = vr.get_batch(frame_indices)
                    frames = torch.from_numpy(frames) if type(frames) is not torch.Tensor else frames
                    torch.save(frames, os.path.join(tmp_dir, key + '.pt'))
                    del vr
                except Exception as e:
                    print(e)
                    print('load video error: ', key)
                    continue
                item = {'__key__': key, 'height': h, 'width': w}
                yield item
        else:
            continue
        
        

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def gpu_worker(gpu_id, task_queue, video_root, save_dir_keypoints, save_dir_bboxes,
               save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, filter_args, max_detector_threads):
    detector = DWposeDetector(use_batch=False).to(gpu_id)
    futures = set()

    def process_task(task_queue, video_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, filter_args):
        while True:
            key = task_queue.get()
            if key is None:
                break
            try:
                process_single_video_with_timeout(
                    detector, key, video_root,
                    save_dir_keypoints, save_dir_bboxes,
                    save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, filter_args
                )
            except Exception as e:
                print(f"Task failed: {e}")

    with ThreadPoolExecutor(max_workers=max_detector_threads) as executor:
        for _ in range(max_detector_threads):
            futures.add(executor.submit(process_task, task_queue, video_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, filter_args))
        for future in futures:
            future.result()

def process_tar(wds_path, task_queue, video_root, filter_args, total_count):
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
        partial(process_fn_video, meta_dict=meta_dict, video_root=video_root, filter_args=filter_args),
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, prefetch_factor=2, collate_fn=lambda b: b[0])
    for data_batch in dataloader:
        key = data_batch['__key__']
        task_queue.put(key)
        total_count += 1

    return total_count

def producer_worker_wds(tar_paths, task_queue, video_root, filter_args, max_samples_per_gpu):
    total_count = 0
    for _, tar_path in tqdm(enumerate(tar_paths), desc="Processing tar files", total=len(tar_paths)):
        total_count = process_tar(tar_path, task_queue, video_root, filter_args, total_count)
        if total_count > max_samples_per_gpu:
            break
        else:
            print(f"Has Processed Total count: {total_count}")
        gc.collect()
    gc.collect()
            
        
if __name__ == "__main__":
    import argparse

    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))

    wds_root = config.get('wds_root', '')
    video_root = config.get('video_root', '')
    filter_args = config.get('filter_args', {})
    max_detector_threads = config.get('max_detector_threads', 8)
    tar_paths = glob.glob(os.path.join(wds_root, "**", "*.tar"), recursive=True)
    max_samples_per_gpu = config.get('max_samples_per_gpu', 1000000)

    save_dir_keypoints = os.path.join(video_root, 'keypoints')
    save_dir_bboxes = os.path.join(video_root, 'bboxes')
    save_dir_dwpose_mp4 = os.path.join(video_root, 'dwpose')
    save_dir_hands = os.path.join(video_root, 'hands')
    save_dir_faces = os.path.join(video_root, 'faces')
    save_dir_caption = os.path.join(video_root, 'caption')
    save_dir_caption_multi = os.path.join(video_root, 'caption_multi')
    tmp_dir = os.path.join("/dev/shm/tmp")

    os.makedirs(save_dir_keypoints, exist_ok=True)
    os.makedirs(save_dir_bboxes, exist_ok=True)
    os.makedirs(save_dir_hands, exist_ok=True)
    os.makedirs(save_dir_faces, exist_ok=True)
    os.makedirs(save_dir_dwpose_mp4, exist_ok=True)
    os.makedirs(save_dir_caption, exist_ok=True)
    os.makedirs(save_dir_caption_multi, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)


    # 建立 task queue
    task_queue = multiprocessing.Queue(maxsize=max_detector_threads*4)
    # 启动 GPU worker
    processes = []
    producer_processes = []

    p = multiprocessing.Process(target=gpu_worker, args=(local_rank, task_queue, video_root, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, filter_args, max_detector_threads))
    p.start()

    # 生产者进程（mp4/wds）
    tar_shard = tar_paths[rank::world_size]
    random.shuffle(tar_shard)
    producer_worker_wds(tar_shard, task_queue, video_root, filter_args, max_samples_per_gpu)

    for _ in range(max_detector_threads):
        task_queue.put(None)  # 发送结束标记
    
    # 此时队列里应该最多有32个要处理，理论上几分钟可以处理完
    p.join(timeout=6000)  # 等待2h，确保消费、销毁完成
    if p.is_alive():
        print("警告：GPU worker进程未在预期时间内结束")
        p.terminate()

    
