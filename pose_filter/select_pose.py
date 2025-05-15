import cv2
import numpy as np
from PIL import Image
import pose_draw.draw_utils as util
import torch
from DWPoseProcess.AAUtils import read_frames_and_fps_as_np, save_videos_from_pil, resize_image
from DWPoseProcess.checkUtils import *
import random
import shutil
import argparse
import yaml  # Add this import
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count



####### 这一部分逻辑针对多人也需要重写

def process_video(mp4_path, wanted_fps=None):
    keypoint_path = mp4_path.replace("videos", "videos_keypoints")
    poses = torch.load(keypoint_path.replace(".mp4", ".pt"))
    bbox_path = mp4_path.replace("videos", "videos_bboxes")
    bboxes = torch.load(bbox_path.replace(".mp4", ".pt"))

    # filter_result_indexes = check_valid(poses, bboxes)
    frames, fps = read_frames_and_fps_as_np(mp4_path)

    assert len(poses) == len(frames), "poses and frames should have the same length"

    initial_frame = frames[0]
    H, W, C = resize_image(initial_frame).shape
    max_attempts = 50
    start_index = 0
    motion_part_len = 65
    ref_part_len = 15
    final_motion_indices = None
    applied_wanted_fps = wanted_fps
    final_max_delta_list = []

    while (motion_part_len + ref_part_len) / applied_wanted_fps * fps > len(poses):
        ref_part_len -= 3
        if ref_part_len < 10:   # 还是太少，就降fps
            applied_wanted_fps = wanted_fps * 3 / 4 # 降一些提高产出率
            if (motion_part_len + ref_part_len) / applied_wanted_fps * fps > len(poses):  # 降了还是不行，就退出了，如果可以就跳出循环
                # print("filtered: 视频太短，无法满足采样要求，跳过")
                return
    
    num_frames = motion_part_len + ref_part_len
    for attempt in range(max_attempts):
        end = int(start_index + num_frames / applied_wanted_fps * fps)
        if end >= len(frames):
            start_index -= 3    # 后退一步
            if start_index <= 0:   # 如果后退到头了，就退出
                # print("filtered: 视频无法满足采样要求，跳过")
                return
            continue

        indices = np.arange(start_index, end, (end - start_index) / num_frames).astype(int)
        if len(indices) != num_frames:
            continue

        ref_part_indices = indices[:ref_part_len]
        motion_part_indices = indices[ref_part_len:]

        motion_part_poses = [poses[index] for index in motion_part_indices]
        motion_part_bboxes = [bboxes[index] for index in motion_part_indices]
        ref_part_poses = [poses[index] for index in ref_part_indices]

        motion_part_bbox_check_result = check_from_keypoints_bbox(motion_part_poses, motion_part_bboxes, IoU_thresthold=0.3, reference_width=W, reference_height=H)
        motion_part_core_check_result = check_from_keypoints_core_keypoints(motion_part_poses, motion_part_bboxes, IoU_thresthold=0.3, reference_width=W, reference_height=H)
        max_delta_list = check_from_keypoints_stick_movement(motion_part_poses, motion_part_bboxes, IoU_thresthold=0.3, reference_width=W, reference_height=H)

        ref_part_check_indices = get_valid_indice_from_keypoints(ref_part_poses, ref_part_indices)
        if len(ref_part_check_indices) > 0 and motion_part_bbox_check_result and motion_part_core_check_result:    # 这里只要满足正脸条件就可以，其它都留着
            final_ref_image_indice = random.choice(ref_part_check_indices)
            final_motion_indices = motion_part_indices
            final_max_delta_list = max_delta_list
            break
        else:
            start_index += random.randint(4, 5) # 没选中就随机增加start，然后记录错误的数据
            continue

    if final_motion_indices is None:
        # 没找到
        return

    else:
        # 找到
        poses = [poses[i] for i in final_motion_indices]
        ref_frame = Image.fromarray(frames[final_ref_image_indice])
        frames = [Image.fromarray(frames[i]) for i in final_motion_indices]
        bboxes = [bboxes[i] for i in final_motion_indices]

    
    # save dwpose keypoints and selected frames / reference frame
    save_videos_from_pil(frames, keypoint_path.replace("videos_keypoints", "videos_filtered"), wanted_fps)
    ref_frame_path = keypoint_path.replace("videos_keypoints", "videos_ref").replace(".mp4", ".jpg")
    meta_path = keypoint_path.replace("videos_keypoints", "videos_moving_info").replace(".mp4", ".txt")
    filtered_keypoints_path = keypoint_path.replace("videos_keypoints", "videos_keypoints_filtered")
    os.makedirs(os.path.dirname(ref_frame_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    os.makedirs(os.path.dirname(filtered_keypoints_path), exist_ok=True)
    ref_frame.save(ref_frame_path)
    torch.save(poses, filtered_keypoints_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for item in final_max_delta_list:
            f.write(str(item) + "\n")
    

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video directories based on YAML config')
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    args = parser.parse_args()
    # Load configuration
    config = load_config(args.config)

    directories = config.get("directories")
    remove_last_flag = config.get("remove_last_flag", False)

    mp4_paths = []
    for directory in directories:
        if remove_last_flag:
            # 删除 directory 中所有文件
            filtered_keypoints_path = directory.replace("videos", "videos_keypoints_filtered")
            filtered_video_path = directory.replace("videos", "videos_filtered")
            ref_image_path = directory.replace("videos", "videos_ref")
            info_path = directory.replace("videos", "videos_moving_info")
            if os.path.exists(filtered_keypoints_path):
                shutil.rmtree(filtered_keypoints_path)
            if os.path.exists(filtered_video_path):
                shutil.rmtree(filtered_video_path)
            if os.path.exists(ref_image_path):
                shutil.rmtree(ref_image_path)
            if os.path.exists(info_path):
                shutil.rmtree(info_path)
            print("已清除上次产生的")


        keypoint_files = []
        for root, dirs, files in os.walk(directory.replace("videos", "videos_keypoints")):
            for file in files:
                if file.lower().endswith('.pt'):  # 只查找 .mp4 文件
                    keypoint_files.append(file.replace(".pt", ".mp4"))  # 获取绝对路径

        print(f"Processing directory: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file in keypoint_files and file.lower().endswith('.mp4'):  # 只查找 .mp4 文件
                    full_path = os.path.join(root, file)  # 获取绝对路径
                    mp4_paths.append(full_path)
    
    # 串行
        for mp4_path in tqdm(mp4_paths, desc="Processing videos", unit="video"):
            process_video(mp4_path, wanted_fps=16)
    # 并行
    # with Pool(64) as p:
    #     p.starmap(process_video, [(mp4_path, reshape_flag, points_only_flag, original_resolution_flag, 16, True, representation_dirname) for mp4_path in mp4_paths])


    


