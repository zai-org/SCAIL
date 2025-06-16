import concurrent.futures
import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.AAUtils import get_fps, read_frames, save_videos_from_pil
from collections import deque
import shutil
import torch
import yaml
from pose_draw.draw_pose_main import draw_pose_to_canvas
from extractUtils import check_single_human_requirements, check_multi_human_requirements, human_select

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

def process_single_video(video_path, detector, relative_path, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, filter_args):
    use_filter=filter_args['use_filter']
    save_mp4=filter_args['save_mp4']
    multi_person=filter_args['multi_person']
    out_path_keypoint = os.path.join(save_dir_keypoints, relative_path)
    out_path_bbox = os.path.join(save_dir_bboxes, relative_path)
    out_path_mp4 = os.path.join(save_dir_mp4, relative_path)
    if os.path.exists(out_path_keypoint) and os.path.exists(out_path_bbox):
        return

    # output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    # if not output_dir.exists():
    #     output_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path)
    frames = read_frames(video_path)
    W, H = frames[0].size

    if fps is None or frames is None:
        return
    else:
        print(f"Processing: {video_path} fps: {int(fps)}")


    detector_return_list = []
    for i, frame_pil in enumerate(frames):
        detector_result = detector(frame_pil)
        _, _, det_result = detector_result
        if use_filter:
            if multi_person:
                if not check_multi_human_requirements(det_result):
                    return
            else:
                if not check_single_human_requirements(det_result):
                    return
        detector_return_list.append(detector_result)

    poses, scores, det_results = zip(*detector_return_list) # 这里存的是整个视频的poses
    poses, det_results = human_select(poses, det_results, multi_person)
    # 存raw poses
    assert len(poses) == len(frames), "frames must match"

    if use_filter:
        mean, std = calculate_video_mean_and_std_pil(frames)
        if mean < 30:
            return

    if save_mp4:
        mp4_results = draw_pose_to_canvas(poses, pool=None, H=H, W=W, reshape_scale=0, points_only_flag=False, show_feet_flag=False)
        save_videos_from_pil(mp4_results, out_path_mp4, fps=16)
    torch.save(poses, out_path_keypoint.replace(".mp4", ".pt"))
    torch.save(det_results, out_path_bbox.replace(".mp4", ".pt"))

def process_single_video_with_timeout(video_path, detector, relative_path, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, filter_args):
    if "timeout_per_video" in filter_args:
        timeout_seconds = filter_args["timeout_per_video"]
    else:
        timeout_seconds = 150

    def task():
        process_single_video(video_path, detector, relative_path,
                             save_dir_keypoints, save_dir_bboxes, save_dir_mp4,
                             filter_args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        try:
            future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            print(f"超时：处理视频 {video_path} 超过 {timeout_seconds} 秒，已跳过。")
        except Exception as e:
            print(f"处理视频 {video_path} 出错：{str(e)}")


def process_batch_videos(video_list, detector, filter_args, name_args):
    for i, video_path in enumerate(video_list):
        video_root = os.path.dirname(video_path)
        relative_path = os.path.relpath(video_path, video_root)
        save_dir_keypoints = video_root + name_args['keypoint_suffix_name']
        save_dir_bboxes = video_root + name_args['bbox_suffix_name']
        save_dir_mp4 = video_root + name_args['mp4_suffix_name']
        print(f"Process {i}/{len(video_list)} video")
        process_single_video_with_timeout(video_path, detector, relative_path, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, filter_args=filter_args)

# 对每张卡，对chunk里的视频串行
def process_per_gpu(mp4_path_list, gpu_id, num_workers_per_proc, filter_args, name_args):
    detector = DWposeDetector(use_batch=False)
    detector = detector.to(gpu_id)
    # 再把mp4_path_list分成video_chunks_per_proc，每块gpu多个进程
    perproc_batch_size = (len(mp4_path_list) + num_workers_per_proc - 1) // num_workers_per_proc
    video_chunks_per_proc = [
        mp4_path_list[i : i + perproc_batch_size]
        for i in range(0, len(mp4_path_list), perproc_batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(video_chunks_per_proc):
            futures.append(
                executor.submit(process_batch_videos, chunk, detector, filter_args, name_args)
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"GPU {gpu_id} Done")
    del detector
        

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # -----
    # NOTE:
    # python tools/extract_dwpose_from_vid.py --video_root /path/to/video_dir
    # -----
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    gpu_ids = [0,1,2,3,4,5,6,7]

    video_roots = config.get('video_roots', [])
    num_workers_per_proc = config.get('num_workers_per_proc', 8)
    flag_remove_last = config.get('remove_last', False)
    single_gpu_test = config.get('single_gpu_test', False)
    filter_args = config.get('filter_args', True)
    name_args = config.get('name_args', {})



    if len(video_roots) == 0:
        raise ValueError("No video roots specified in the configuration file.")
    
        
    # collect all video_folder paths
    video_mp4_paths = set()

    for video_root in video_roots:
        save_dir_keypoints = video_root + name_args['keypoint_suffix_name']
        save_dir_bboxes = video_root + name_args['bbox_suffix_name']
        save_dir_mp4 = video_root + name_args['mp4_suffix_name']

        if flag_remove_last:
            if os.path.exists(save_dir_keypoints):
                shutil.rmtree(save_dir_keypoints)
            if os.path.exists(save_dir_bboxes):
                shutil.rmtree(save_dir_bboxes)
            if filter_args['save_mp4']:
                if os.path.exists(save_dir_mp4):
                    shutil.rmtree(save_dir_mp4)
        if not os.path.exists(save_dir_keypoints):
            os.makedirs(save_dir_keypoints)
        if not os.path.exists(save_dir_bboxes):
            os.makedirs(save_dir_bboxes)
        if filter_args['save_mp4']:
            if not os.path.exists(save_dir_mp4):
                os.makedirs(save_dir_mp4)
        for root, dirs, files in os.walk(video_root):
            for name in files:
                if name.endswith(".mp4"):
                    video_mp4_paths.add(os.path.join(root, name))

    video_mp4_paths = list(video_mp4_paths)
    random.shuffle(video_mp4_paths)
    print(f"all videos num {len(video_mp4_paths)}")

    gpu_chunks = [[] for _ in gpu_ids]
    for idx, video in enumerate(video_mp4_paths):
        gpu_idx = idx % len(gpu_ids)
        gpu_chunks[gpu_idx].append(video)

    # 单卡串行debug
    if single_gpu_test:
        for gpu_chunk in gpu_chunks:
            process_per_gpu(gpu_chunk, 0, num_workers_per_proc, filter_args, name_args)
        
    # 每张卡一个进程
    else:
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            p = multiprocessing.Process(
                target=process_per_gpu,
                args=(gpu_chunks[i], i, num_workers_per_proc, filter_args, name_args),
            )
            p.start()
            processes.append(p)

        # 等待所有进程完成
        for p in processes:
            p.join()
        # process_per_proc(video_chunks[0], gpu_id, num_workers_per_proc)
        print("All Done")







    
