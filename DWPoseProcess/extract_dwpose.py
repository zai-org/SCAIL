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
from pose_draw.draw_pose import draw_pose_to_canvas

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

def process_single_video(video_path, detector, relative_path, save_dir_keypoints, save_dir_bboxes, save_dir_refined, infer_batch_size, filter_args):
    use_filter=filter_args['use_filter']
    save_mp4_refine=filter_args['save_mp4_refine']
    out_path_keypoint = os.path.join(save_dir_keypoints, relative_path)
    out_path_bbox = os.path.join(save_dir_bboxes, relative_path)
    out_path_refined = os.path.join(save_dir_refined, relative_path)
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

    if use_filter:
        mean, std = calculate_video_mean_and_std_pil(frames)
        if mean < 30:
            return

    if infer_batch_size == 1:
        for i, frame_pil in enumerate(frames):
            detector_result = detector(frame_pil)
            detector_return_list.append(detector_result)
    else:
        # 分割frames为chunks，每个chunk包含batch_size帧
        for i in range(0, len(frames), infer_batch_size):
            try:
                frame_chunk = frames[i:i + infer_batch_size]
                detector_result = detector(frame_chunk)
                detector_return_list.extend(detector_result)  # 调用detector写入结果
            except Exception as e:
                print(f"[ERROR] 视频 `{video_path}` 处理失败，跳过！错误信息：{e}")
                return

    results, poses, scores, det_results = zip(*detector_return_list)
    # 存raw poses
    assert len(poses) == len(frames), "frames must match"
    if save_mp4_refine:    # refine模式
        refined_results = draw_pose_to_canvas(poses, pool=None, H=H, W=W, reshape_flag=False, points_only_flag=False)
        save_videos_from_pil(refined_results, out_path_refined, fps=16) # 会自动makedirs
    else:
        torch.save(poses, out_path_keypoint.replace(".mp4", ".pt"))
        torch.save(det_results, out_path_bbox.replace(".mp4", ".pt"))

    
def process_batch_videos(video_list, detector, infer_batch_size, filter_args):
    for i, video_path in enumerate(video_list):
        video_root = os.path.dirname(video_path)
        relative_path = os.path.relpath(video_path, video_root)
        save_dir_keypoints = video_root + "_keypoints"
        save_dir_bboxes = video_root + "_bboxes"
        save_dir_refined = video_root + "_refined_dwpose"
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, relative_path, save_dir_keypoints, save_dir_bboxes, save_dir_refined, infer_batch_size=infer_batch_size, filter_args=filter_args)


# 对每个gpu串行执行，执行一定次数后重启detector，防止内存泄漏
def process_per_proc(mp4_path_chunks, gpu_id, num_workers_per_proc, filter_args):
    detector = DWposeDetector(use_batch=(infer_batch_size>1))
    detector = detector.to(gpu_id)
    # split into worker chunks
    perproc_batch_size = (len(mp4_path_chunks) + num_workers_per_proc - 1) // num_workers_per_proc
    video_chunks_per_proc = [
        mp4_path_chunks[i : i + perproc_batch_size]
        for i in range(0, len(mp4_path_chunks), perproc_batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(video_chunks_per_proc):
            futures.append(
                executor.submit(process_batch_videos, chunk, detector, infer_batch_size, filter_args)
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
    del detector

def process_per_gpu(mp4_path_chunks_list, gpu_id, num_workers_per_proc, filter_args):
    for mp4_path_chunks in mp4_path_chunks_list:
        process_per_proc(mp4_path_chunks, gpu_id, num_workers_per_proc, filter_args)

    

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
    infer_batch_size = config.get('infer_batch_size', 1)
    videos_per_worker = config.get('videos_per_worker', 8)
    num_workers_per_proc = config.get('num_workers_per_proc', 8)
    flag_remove_last = config.get('remove_last', False)
    single_gpu_test = config.get('single_gpu_test', False)
    filter_args = config.get('filter_args', True)



    if len(video_roots) == 0:
        raise ValueError("No video roots specified in the configuration file.")
    
        
    # collect all video_folder paths
    video_mp4_paths = set()

    for video_root in video_roots:
        if filter_args['save_mp4_refine']:
            save_dir_refined = video_root + "_refined_dwpose"
            if not os.path.exists(save_dir_refined):
                os.makedirs(save_dir_refined)

        else:
            save_dir_keypoints = video_root + "_keypoints"
            save_dir_bboxes = video_root + "_bboxes"
            if flag_remove_last:
                if os.path.exists(save_dir_keypoints):
                    shutil.rmtree(save_dir_keypoints)
                if os.path.exists(save_dir_bboxes):
                    shutil.rmtree(save_dir_bboxes)
            if not os.path.exists(save_dir_keypoints):
                os.makedirs(save_dir_keypoints)
            if not os.path.exists(save_dir_bboxes):
                os.makedirs(save_dir_bboxes)

        for root, dirs, files in os.walk(video_root):
            for name in files:
                if name.endswith(".mp4"):
                    video_mp4_paths.add(os.path.join(root, name))

    video_mp4_paths = list(video_mp4_paths)
    random.shuffle(video_mp4_paths)

    print(f"all videos num {len(video_mp4_paths)}")

    # 每个gpu一次处理这么多
    loader_batch_size = num_workers_per_proc * videos_per_worker
    # video_chunks 为总共需要处理的次数
    video_chunks = [
        video_mp4_paths[i : i + loader_batch_size]
        for i in range(0, len(video_mp4_paths), loader_batch_size)
    ]
    # 每个gpu分一些video_chunk去串行处理
    gpu_chunks_list = [video_chunks[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    processes = []


    # 单卡debug
    if single_gpu_test:
        process_per_gpu(gpu_chunks_list[0], 0, num_workers_per_proc, filter_args)
        
    # 每张卡一个进程
    else:
        for i, gpu_id in enumerate(gpu_ids):
            p = multiprocessing.Process(
                target=process_per_gpu,
                args=(gpu_chunks_list[i], i, num_workers_per_proc, filter_args),
            )
            p.start()
            processes.append(p)

        # 等待所有进程完成
        for p in processes:
            p.join()
        # process_per_proc(video_chunks[0], gpu_id, num_workers_per_proc)
        print("All Done")







    
