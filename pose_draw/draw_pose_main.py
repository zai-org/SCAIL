import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pose_draw.draw_utils as util
from pose_draw.draw_3d_utils import *
from pose_draw.reshape_utils import *
from DWPoseProcess.AAUtils import read_frames_and_fps_as_np, save_videos_from_pil
from DWPoseProcess.checkUtils import *
import random
import shutil
import argparse
import yaml  # Add this import
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from decord import VideoReader
import copy


def draw_pose(pose, H, W, show_feet=False, show_body=True, show_hand=True, show_face=True, show_cheek=False, dw_bgr=False, dw_hand=False, aug_body_draw=False, optimized_face=False):
    final_canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for i in range(len(pose["bodies"]["candidate"])):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        bodies = pose["bodies"]
        faces = pose["faces"][i:i+1]
        hands = pose["hands"][2*i:2*i+2]
        candidate = bodies["candidate"][i]
        subset = bodies["subset"][i:i+1]   # subset是认为的有效点

        if show_body:
            if len(subset[0]) <= 18 or show_feet == False:
                if aug_body_draw:
                    rand = random.random()
                    if rand < 0.035:  # 非常小概率
                        canvas = util.draw_bodypose_augmentation(canvas, candidate, subset, drop_aug=False, shift_aug=True)
                    elif rand < 0.3:
                        canvas = util.draw_bodypose_augmentation(canvas, candidate, subset, drop_aug=True, shift_aug=False)   # 本身里面只有50%概率drop，也就是5帧可能有一帧drop
                    else:
                        canvas = util.draw_bodypose_augmentation(canvas, candidate, subset, drop_aug=False, shift_aug=False)
                else:
                    canvas = util.draw_bodypose(canvas, candidate, subset)
            else:
                canvas = util.draw_bodypose_with_feet(canvas, candidate, subset)
            if dw_bgr:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if show_cheek:
            assert show_body == False, "show_cheek and show_body cannot be True at the same time"
            canvas = util.draw_bodypose_augmentation(canvas, candidate, subset,  drop_aug=True, shift_aug=False, all_cheek_aug=True)
        if show_hand:
            if not dw_hand:
                canvas = util.draw_handpose_lr(canvas, hands)
            else:
                canvas = util.draw_handpose(canvas, hands)
        if show_face:
            canvas = util.draw_facepose(canvas, faces, optimized_face=optimized_face)
        final_canvas = final_canvas + canvas
    return final_canvas


def scale_image_hw_keep_size(img, scale_h, scale_w):
    """分别按 scale_h, scale_w 缩放图像，保持输出尺寸不变。"""
    H, W = img.shape[:2]
    new_H, new_W = int(H * scale_h), int(W * scale_w)
    scaled = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    result = np.zeros_like(img)

    # 计算在目标图上的放置范围
    # --- Y方向 ---
    if new_H >= H:
        y_start_src = (new_H - H) // 2
        y_end_src = y_start_src + H
        y_start_dst = 0
        y_end_dst = H
    else:
        y_start_src = 0
        y_end_src = new_H
        y_start_dst = (H - new_H) // 2
        y_end_dst = y_start_dst + new_H

    # --- X方向 ---
    if new_W >= W:
        x_start_src = (new_W - W) // 2
        x_end_src = x_start_src + W
        x_start_dst = 0
        x_end_dst = W
    else:
        x_start_src = 0
        x_end_src = new_W
        x_start_dst = (W - new_W) // 2
        x_end_dst = x_start_dst + new_W

    # 将 scaled 映射到 result
    result[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = scaled[y_start_src:y_end_src, x_start_src:x_end_src]

    return result

def draw_pose_to_canvas_np(poses, pool, H, W, reshape_scale, show_feet_flag=False, show_body_flag=True, show_hand_flag=True, show_face_flag=True, show_cheek_flag=False, dw_bgr=False, dw_hand=False, aug_body_draw=False):
    canvas_np_lst = []
    for pose in poses:
        if reshape_scale > 0:
            pool.apply_random_reshapes(pose)
        canvas = draw_pose(pose, H, W, show_feet_flag, show_body_flag, show_hand_flag, show_face_flag, show_cheek_flag, dw_bgr, dw_hand, aug_body_draw, optimized_face=True)
        canvas_np_lst.append(canvas)
    return canvas_np_lst


def draw_pose_to_canvas(poses, pool, H, W, reshape_scale, points_only_flag, show_feet_flag, show_body_flag=True, show_hand_flag=True, show_face_flag=True, show_cheek_flag=False, dw_bgr=False, dw_hand=False, aug_body_draw=False):
    canvas_lst = []
    for pose in poses:
        if reshape_scale > 0:
            pool.apply_random_reshapes(pose)
        canvas = draw_pose(pose, H, W, show_feet_flag, show_body_flag, show_hand_flag, show_face_flag, show_cheek_flag, dw_bgr, dw_hand, aug_body_draw, optimized_face=True)
        canvas_img = Image.fromarray(canvas)
        canvas_lst.append(canvas_img)
    return canvas_lst


def get_mp4_filenames_from_directory(dwpose_keypoints_dir):
    mp4_filenames_dwpose = []
    # 通过keypoints和mp4的交集取所有可用的mp4
    if dwpose_keypoints_dir:
        for root, dirs, files in os.walk(dwpose_keypoints_dir):
            for file in files:
                if file.lower().endswith('.pt'):  # 只查找 .mp4 文件
                    mp4_filenames_dwpose.append(file.replace(".pt", ".mp4"))  # 获取绝对路径
    return mp4_filenames_dwpose

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



def process_video(mp4_path, dwpose_keypoint_path, threed_keypoint_pair, reshape_scale, points_only_flag, show_feet_flag, wanted_fps=None, output_dirname=None, pose_type="dwpose"):
    frames, fps = read_frames_and_fps_as_np(mp4_path)
    initial_frame = frames[0]
    output_path = os.path.join(output_dirname, os.path.basename(mp4_path))
    os.makedirs(output_dirname, exist_ok=True)

    if "3dpose" in pose_type:
        raise NotImplementedError("3dpose is not implemented")
    else:
        poses = torch.load(dwpose_keypoint_path)
        pool = reshapePool(alpha=reshape_scale)
        canvas_lst = draw_pose_to_canvas(poses, pool, initial_frame.shape[0], initial_frame.shape[1], reshape_scale, points_only_flag, show_feet_flag, show_body_flag=True)
        save_videos_from_pil(canvas_lst, output_path, wanted_fps)


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
    threed_kpt_dirs = config.get("threed_kpt_dirs", None)
    if threed_kpt_dirs:
        assert len(threed_kpt_dirs) == len(directories), "threed_kpt_dirs must have the same length as directories"
    reshape_scale = config.get("reshape_scale", 0)
    points_only_flag = config.get("points_only_flag", False)
    remove_last_flag = config.get("remove_last_flag", False)
    show_feet_flag = config.get("show_feet_flag", False)
    pose_type = config.get("pose_type", "dwpose")
    target_representation_dirname = config.get("target_representation_suffix", None)
    keypoints_suffix_dwpose = config.get("keypoints_suffix_dwpose", "_keypoints")
    parallel_flag = config.get("parallel_flag", False)




    for dir_idx, directory in enumerate(directories):
        mp4_paths = []
        dwpose_keypoint_paths = []
        threed_keypoint_paths = []
        output_representation_dir = directory + target_representation_dirname
        if remove_last_flag:
            # 删除 directory 中所有文件
            if os.path.exists(output_representation_dir):
                shutil.rmtree(output_representation_dir)
            print(f"已清除上次产生的{output_representation_dir}文件夹")

        video_directory_name = directory.split("/")[-1]
        # video_directory_name 是 directory的最后一层子目录
        dwpose_keypoints_dir = directory.replace(video_directory_name, f"{video_directory_name}{keypoints_suffix_dwpose}")
        mp4_filenames_dwpose = get_mp4_filenames_from_directory(dwpose_keypoints_dir)
        if "3dpose" in pose_type:
            threed_kpt_dir = threed_kpt_dirs[dir_idx]

        print(f"Processing directory: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if "3dpose" in pose_type:
                    if file in mp4_filenames_dwpose and file.lower().endswith('.mp4'):  # 只查找 .mp4 文件
                        full_path = os.path.join(root, file)  # 获取绝对路径
                        full_dwpose_path = os.path.join(dwpose_keypoints_dir, file.replace(".mp4", ".pt"))
                        full_threed_path = (os.path.join(threed_kpt_dir, file.replace(".mp4", ""), "keypoints_3d.pt"), os.path.join(threed_kpt_dir, file.replace(".mp4", ""), "camera.json"))
                        if os.path.exists(full_threed_path[0]) and os.path.exists(full_threed_path[1]):
                            mp4_paths.append(full_path)
                            dwpose_keypoint_paths.append(full_dwpose_path)
                            threed_keypoint_paths.append(full_threed_path)

                elif "dwpose" in pose_type:
                    if file in mp4_filenames_dwpose and file.lower().endswith('.mp4'):  # 只查找 .mp4 文件
                        full_path = os.path.join(root, file)  # 获取绝对路径
                        full_dwpose_path = os.path.join(dwpose_keypoints_dir, file.replace(".mp4", ".pt"))
                        mp4_paths.append(full_path)
                        dwpose_keypoint_paths.append(full_dwpose_path)
                        threed_keypoint_paths.append(None)

        
        # 并行
        if parallel_flag:
            with Pool(64) as p:
                p.starmap(process_video, [(mp4_path, dwpose_keypoint_paths[path_idx], threed_keypoint_paths[path_idx], reshape_scale, points_only_flag, show_feet_flag, 16, output_representation_dir, pose_type) for path_idx, mp4_path in enumerate(mp4_paths)])
        else: # 串行
            for path_idx, mp4_path in tqdm(enumerate(mp4_paths), desc="Processing videos", unit="video"):
                process_video(mp4_path, dwpose_keypoint_paths[path_idx], threed_keypoint_paths[path_idx], reshape_scale, points_only_flag, show_feet_flag, wanted_fps=16, output_dirname=output_representation_dir, pose_type=pose_type)


    


