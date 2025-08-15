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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, TimeoutError
from decord import VideoReader
from fractions import Fraction
import io
import gc
from PIL import Image
from multiprocessing import Process




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
    timeout_seconds = 150

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


def process_single_video(vr, detector, key, video_root, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, save_dir_caption, save_dir_caption_multi, filter_args):
    use_filter=filter_args['use_filter']
    save_mp4=filter_args['save_mp4']
    multi_path =  os.path.join(video_root, 'labels_multi', key + '.txt')
    single_path = os.path.join(video_root, 'labels', key + '.txt')
    fail_path = os.path.join(video_root, 'labels_person_fail', key + '.txt')
    target_multi_path = os.path.join(save_dir_caption_multi, key + '.txt')
    target_single_path = os.path.join(save_dir_caption, key + '.txt')

    out_path_keypoint = os.path.join(save_dir_keypoints, key + '.pt')
    out_path_bbox = os.path.join(save_dir_bboxes, key + '.pt')
    out_path_mp4 = os.path.join(save_dir_mp4, key + '.mp4')

    # output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    # if not output_dir.exists():
    #     output_dir.mkdir(parents=True, exist_ok=True)

    # 使用 decord 打开视频
    frames = []
    detector_return_list = []
    multi_person = True
    if os.path.exists(single_path):
        multi_person = False

    # 逐帧解码
    for i in range(len(vr)):
        frame = vr[i]  # 获取帧，返回的是 mx.ndarray
        frame_pil = Image.fromarray(frame.asnumpy())  # 转换为 PIL 格式
        frames.append(frame_pil)
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


    W, H = frames[0].size
    
    if use_filter:
        mean, std = calculate_video_mean_and_std_pil(frames)
        if mean < 30:
            return

    if multi_person:
        if os.path.exists(multi_path):
            shutil.copyfile(multi_path, target_multi_path)
        elif os.path.exists(fail_path):
            shutil.copyfile(fail_path, target_multi_path)
    else:
        shutil.copyfile(single_path, target_single_path)

    poses, scores, det_results = zip(*detector_return_list) # 这里存的是整个视频的poses
    poses, det_results = human_select(poses, det_results, multi_person)
    # 存raw poses
    del frames
    del detector_return_list  # 清理detector返回列表

    if save_mp4:
        mp4_results = draw_pose_to_canvas(poses, pool=None, H=H, W=W, reshape_scale=0, points_only_flag=False, show_feet_flag=False)
        save_videos_from_pil(mp4_results, out_path_mp4, fps=16) # 实际可能不一致
        del mp4_results  # 清理mp4结果
    
    torch.save(poses, out_path_keypoint)
    torch.save(det_results, out_path_bbox)
    del poses, scores, det_results

def process_fn_video(src):
    worker_info = torch.utils.data.get_worker_info()
    for i, r in enumerate(src):
        if worker_info is not None:
            if i % worker_info.num_workers != worker_info.id:
                continue
        yield r

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_tar(wds_list, video_root, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, save_dir_caption, save_dir_caption_multi, filter_args):
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    
    detector = DWposeDetector(use_batch=False)
    detector = detector.to(local_rank%8)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(wds_list, seed=None),
        wds.tarfile_to_samples(),
        partial(process_fn_video)
    )
    max_workers = 4
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, prefetch_factor=2)
    data_iter = iter(dataloader)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for data_batch in tqdm(data_iter):
            data = {}
            for k, v in data_batch.items():
                data[k] = v[0]
            try:
                key = data['__key__']
                if os.path.exists(os.path.join(video_root, 'labels', key + '.txt')) or os.path.exists(os.path.join(video_root, 'labels_multi', key + '.txt')) or os.path.exists(os.path.join(video_root, 'labels_person_fail', key + '.txt')):
                    if filter_args['use_fail_prompt'] == False:
                        if os.path.exists(os.path.join(video_root, 'labels_person_fail', key + '.txt')):
                            continue

                    vr = VideoReader(io.BytesIO(data['mp4']))
                    process_single_video_with_timeout(vr, detector, key, video_root,
                                                        save_dir_keypoints, save_dir_bboxes,
                                                        save_dir_mp4, save_dir_caption, save_dir_caption_multi, filter_args)
                del data
                future = executor.submit(
                    process_single_video_with_timeout,
                    vr, detector, key, video_root,
                    save_dir_keypoints, save_dir_bboxes,
                    save_dir_mp4, save_dir_caption, save_dir_caption_multi, filter_args
                )
                futures.add(future)

                # 如果任务太多，就等到至少有一个完成，相当于pop(0)，但是这个pop的是最先完成的，气泡更少
                if len(futures) >= max_workers:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)  # 更新futures为未完成的
                    for f in done:
                        try:
                            f.result()
                        except Exception as e:
                            print(f"Task failed: {e}")

            except Exception as e:
                print(f"Error processing video {key}: {e}")
                continue

        # 等剩下的全部完成
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"Task failed: {e}")
    

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
    device = torch.device(f'cuda:{local_rank%8}')

    wds_root = config.get('wds_root', '')
    tar_paths = [file for file in os.listdir(wds_root) if file.endswith('.tar')]
    tar_paths = tar_paths[rank::world_size]
    video_root = config.get('video_root', '')
    filter_args = config.get('filter_args', {})
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
    max_items_per_chunk = 300
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

    processes = []  # 存储进程的列表
    max_processes = 4  # 最大并发进程数
    for chunk_idx, chunk in tqdm(enumerate(chunks), desc='Processing chunks', total=len(chunks)):
        p = Process(
            target=process_tar,
            args=(chunk, video_root, save_dir_keypoints, save_dir_bboxes, save_dir_mp4, save_dir_caption, save_dir_caption_multi, filter_args)
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
    






    
