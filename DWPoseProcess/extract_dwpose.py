import concurrent.futures
import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.AAUtils import get_fps, read_frames, save_videos_from_pil
from DWPoseProcess.checkUtils import check_frame, check_frames_list
from collections import deque
import shutil
import torch
import yaml

# Extract dwpose mp4 videos from raw videos
# /path/to/video_dataset/*/*.mp4 -> /path/to/video_dataset_dwpose/*/*.mp4



def process_single_video(video_path, detector, relative_path, save_dir, infer_batch_size, filter_args):
    beta=filter_args['beta']
    consistant_thresthold=filter_args['consistant_thresthold']
    use_filter=filter_args['use_filter']

    out_path = os.path.join(save_dir, relative_path)
    if os.path.exists(out_path):
        return

    # output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    # if not output_dir.exists():
    #     output_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path)
    frames = read_frames(video_path)
    if fps is None or frames is None:
        return
    else:
        print(f"Processing: {video_path} fps: {int(fps)}")


    # 对于每个视频的检测而言，以下参数是与推理形式无关的，用于检测帧之间的关系
    # last_det_result: 上一帧结果，用于检测连续性
    # skeleton_init_sequence: 骨骼初始8帧的检测窗口（大小为3
    # skeleton_window_sequence: 过程中骨骼检测窗口
    # last_mean_score 上一帧得分

    last_det_result = np.zeros(4)
    skeleton_init_sequence = deque(maxlen=3)
    skeleton_window_sequence = deque(maxlen=4)
    last_mean_score = [0]

    detector_return_list = []

    if infer_batch_size == 1:
        for i, frame_pil in enumerate(frames):
            detector_result = detector(frame_pil)
            if use_filter:
                check_result = check_frame(i, last_det_result, skeleton_init_sequence, skeleton_window_sequence, last_mean_score, video_path, detector_result, beta, consistant_thresthold)
                if not check_result:
                    return
            detector_return_list.append(detector_result)
    else:
        # 分割frames为chunks，每个chunk包含batch_size帧
        for i in range(0, len(frames), infer_batch_size):
            try:
                frame_chunk = frames[i:i + infer_batch_size]
                detector_result = detector(frame_chunk)
                if use_filter:
                    check_result = check_frames_list(i, infer_batch_size, last_det_result, skeleton_init_sequence, skeleton_window_sequence, last_mean_score, video_path, detector_result, beta, consistant_thresthold)
                    if not check_result:
                        return
                detector_return_list.extend(detector_result)  # 调用detector写入结果
            except Exception as e:
                print(f"[ERROR] 视频 `{video_path}` 处理失败，跳过！错误信息：{e}")
                return

    results, poses, scores, det_results = zip(*detector_return_list)
    # 存raw poses
    assert len(poses) == len(frames), "frames must match"
    torch.save(poses, out_path.replace(".mp4", ".pt"))
    # 直接存mp4
    save_videos_from_pil(results, out_path, fps=fps)

    
def process_batch_videos(video_list, detector, infer_batch_size, filter_args):
    for i, video_path in enumerate(video_list):
        video_root = os.path.dirname(video_path)
        relative_path = os.path.relpath(video_path, video_root)
        save_dir = video_root + "_dwpose"
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, relative_path, save_dir, infer_batch_size=infer_batch_size, filter_args=filter_args)


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
    # del detector

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
        save_dir = video_root + "_dwpose"
        if flag_remove_last:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

    # 每张卡一个进程
    if not single_gpu_test:
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
    
    # 单卡debug
    else:
        process_per_gpu(gpu_chunks_list[0], 0, num_workers_per_proc, filter_args)







    
