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
from render_3d.render_cylinder import render_colored_cylinders
from decord import VideoReader




def draw_pose_points_only(pose, H, W, show_feet=False):
    raise NotImplementedError("draw_pose_points_only is not implemented")

def draw_pose(pose, H, W, show_feet=False, show_body=True):
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
                canvas = util.draw_bodypose(canvas, candidate, subset)
            else:
                canvas = util.draw_bodypose_with_feet(canvas, candidate, subset)

        canvas = util.draw_handpose_lr(canvas, hands)

        canvas = util.draw_facepose(canvas, faces)
        final_canvas = final_canvas + canvas
    return final_canvas

def draw_pose_to_canvas(poses, pool, H, W, reshape_scale, points_only_flag, show_feet_flag, show_body_flag=True):
    canvas_lst = []
    for pose in poses:
        if reshape_scale > 0:
            pool.apply_random_reshapes(pose)
        if points_only_flag:
            canvas = draw_pose_points_only(pose, H, W, show_feet_flag, show_body_flag)
        else:
            canvas = draw_pose(pose, H, W, show_feet_flag, show_body_flag)
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
    fx, fy = focal, focal
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

def render_3d_pose(canvas_lst, dwpose_keypoint_path, threed_keypoint_pair, output_path):
    keypoint_pt, camera_json = threed_keypoint_pair
    import json
    keypoints = torch.load(keypoint_pt)
    dwpose_keypoint_dicts = torch.load(dwpose_keypoint_path)
    camera_obj = json.load(open(camera_json, "r"))
    extrinsics_rotate = camera_obj["Rotation_cam2world"]
    extrinsics_translate = camera_obj["Translation_cam2world"]
    img_focal = camera_obj["img_focal"]
    img_princpt = camera_obj["img_center"]
    W, H = canvas_lst[0].size

    openpose_to_jointnames_map = [
        [0, 55],   # Nose
        [1, 12],   # Neck
        [2, 17],   # R. Shoulder
        [3, 19],   # R. Elbow
        [4, 21],   # R. Wrist
        [5, 16],   # L. Shoulder
        [6, 18],   # L. Elbow
        [7, 20],   # L. Wrist
        [8, 2],    # R. Hip
        [9, 5],    # R. Knee
        [10, 8],   # R. Ankle
        [11, 1],   # L. Hip
        [12, 4],   # L. Knee
        [13, 7],   # L. Ankle
        [14, 56],  # R. Eye
        [15, 57],  # L. Eye
        [16, 58],  # R. Ear
        [17, 59],  # L. Ear
    ]

    limb_seq = [
        [1, 2],    # Neck -> R. Shoulder
        [1, 5],    # Neck -> L. Shoulder
        [2, 3],    # R. Shoulder -> R. Elbow
        [3, 4],    # R. Elbow -> R. Wrist
        [5, 6],    # L. Shoulder -> L. Elbow
        [6, 7],    # L. Elbow -> L. Wrist
        [1, 8],    # Neck -> R. Hip
        [8, 9],    # R. Hip -> R. Knee
        [9, 10],   # R. Knee -> R. Ankle
        [1, 11],   # Neck -> L. Hip
        [11, 12],  # L. Hip -> L. Knee
        [12, 13],  # L. Knee -> L. Ankle
        [1, 0],    # Neck -> Nose
        [0, 14],   # Nose -> R. Eye
        [14, 16],  # R. Eye -> R. Ear
        [0, 15],   # Nose -> L. Eye
        [15, 17],  # L. Eye -> L. Ear
    ]

    base_colors_255_dict = {
        # Warm Colors for Right Side (R.) - Red, Orange, Yellow
        "Red": [255, 0, 0],
        "Orange": [255, 85, 0],
        "Golden Orange": [255, 170, 0],
        "Yellow": [255, 240, 0],
        "Yellow-Green": [180, 255, 0],
        # Cool Colors for Left Side (L.) - Green, Blue, Purple
        "Bright Green": [0, 255, 0],
        "Light Green-Blue": [0, 255, 85],
        "Aqua": [0, 255, 170],
        "Cyan": [0, 255, 255],
        "Sky Blue": [0, 170, 255],
        "Medium Blue": [0, 85, 255],
        "Pure Blue": [0, 0, 255],
        "Purple-Blue": [85, 0, 255],
        "Medium Purple": [170, 0, 255],
        # Neutral/Central Colors (e.g., for Neck, Nose, Eyes, Ears)
        "Grey": [150, 150, 150],
        "Pink-Magenta": [255, 0, 170],
        "Dark Pink": [255, 0, 85],
        "Violet": [100, 0, 255],
        "Dark Violet": [50, 0, 255],
    }

    ordered_colors_255 = [
        base_colors_255_dict["Red"],              # Neck -> R. Shoulder (Red)
        base_colors_255_dict["Cyan"],             # Neck -> L. Shoulder (Cyan)
        base_colors_255_dict["Orange"],           # R. Shoulder -> R. Elbow (Orange)
        base_colors_255_dict["Golden Orange"],    # R. Elbow -> R. Wrist (Golden Orange)
        base_colors_255_dict["Sky Blue"],         # L. Shoulder -> L. Elbow (Sky Blue)
        base_colors_255_dict["Medium Blue"],      # L. Elbow -> L. Wrist (Medium Blue)
        base_colors_255_dict["Yellow-Green"],       # Neck -> R. Hip ( Yellow-Green)
        base_colors_255_dict["Bright Green"],     # R. Hip -> R. Knee (Bright Green - transitioning warm to cool spectrum)
        base_colors_255_dict["Light Green-Blue"], # R. Knee -> R. Ankle (Light Green-Blue - transitioning)
        base_colors_255_dict["Pure Blue"],        # Neck -> L. Hip (Pure Blue)
        base_colors_255_dict["Purple-Blue"],      # L. Hip -> L. Knee (Purple-Blue)
        base_colors_255_dict["Medium Purple"],    # L. Knee -> L. Ankle (Medium Purple)
        base_colors_255_dict["Grey"],             # Neck -> Nose (Grey)
        base_colors_255_dict["Pink-Magenta"],     # Nose -> R. Eye (Pink/Magenta)
        base_colors_255_dict["Dark Violet"],        # R. Eye -> R. Ear (Dark Pink)
        base_colors_255_dict["Pink-Magenta"],           # Nose -> L. Eye (Violet)
        base_colors_255_dict["Dark Violet"],      # L. Eye -> L. Ear (Dark Violet)
    ]

    colors = [[c / 300 + 0.15 for c in color_rgb] + [0.8] for color_rgb in ordered_colors_255]

    render_images = []
    for frame_idx in range(keypoints.shape[0]):
        world_kpts = keypoints[frame_idx][0] # shape: [127, 3], 先处理单人
        dwpose_kpts = dwpose_keypoint_dicts[frame_idx]['bodies']['candidate'][0]   # shape: [24, 2]，先处理单人
        img = np.array(canvas_lst[frame_idx])
        body_3d_keypoints = [None for _ in range(len(openpose_to_jointnames_map))]
        for mapping in openpose_to_jointnames_map:
            body_3d_keypoints[mapping[0]] = np.array(world_kpts[mapping[1]])
        camera_R = extrinsics_rotate[frame_idx]
        camera_T = extrinsics_translate[frame_idx]
        cylinder_specs = []
        for line_idx, line in enumerate(limb_seq):
            start, end = line[0], line[1]
            start_3d = body_3d_keypoints[start] # shape: [3]
            end_3d = body_3d_keypoints[end] # shape: [3]
            start_3d = np.dot(camera_R, start_3d) + camera_T
            end_3d = np.dot(camera_R, end_3d) + camera_T
            if dwpose_kpts[start][0] < 0 or dwpose_kpts[start][1] < 0 or dwpose_kpts[end][0] < 0 or dwpose_kpts[end][1] < 0:
                continue
            start_3d = project_dwpose_to_3d(dwpose_kpts[start], start_3d, img_focal, img_princpt, H, W)
            end_3d = project_dwpose_to_3d(dwpose_kpts[end], end_3d, img_focal, img_princpt, H, W)
            cylinder_specs.append((start_3d, end_3d, colors[line_idx]))
        
        render_image = render_colored_cylinders(cylinder_specs, img_focal, img_princpt, image_size=(H, W), img=img)
        render_images.append(render_image)

    file_path = output_path.replace(".mp4", f"_3d_pose.mp4")
    save_videos_from_pil(render_images, file_path, 16)
    print(f"3d pose video saved to {file_path}")


def process_video(mp4_path, dwpose_keypoint_path, threed_keypoint_pair, reshape_scale, points_only_flag, show_feet_flag, wanted_fps=None, output_dirname=None, pose_type="dwpose"):
    frames, fps = read_frames_and_fps_as_np(mp4_path)
    initial_frame = frames[0]
    output_path = os.path.join(output_dirname, os.path.basename(mp4_path))
    os.makedirs(output_dirname, exist_ok=True)

    if "3dpose" in pose_type:
        poses = torch.load(dwpose_keypoint_path)
        pool = None
        canvas_lst = draw_pose_to_canvas(poses, pool, initial_frame.shape[0], initial_frame.shape[1], reshape_scale, points_only_flag, show_feet_flag, show_body_flag=False)
        render_3d_pose(canvas_lst, dwpose_keypoint_path, threed_keypoint_pair, output_path)
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


    mp4_paths = []
    dwpose_keypoint_paths = []
    threed_keypoint_paths = []

    for dir_idx, directory in enumerate(directories):
        output_representation_dir = directory + target_representation_dirname
        if remove_last_flag:
            # 删除 directory 中所有文件
            if os.path.exists(output_representation_dir):
                shutil.rmtree(output_representation_dir)
            print(f"已清除上次产生的{output_representation_dir}文件夹")

        video_directory_name = directory.split("/")[-1]

        # video_directory_name 是 directory的最后一层子目录
        dwpose_keypoints_dir = directory.replace(video_directory_name, f"{video_directory_name}{keypoints_suffix_dwpose}")   # TODO: 暂时修改
        mp4_filenames_dwpose = get_mp4_filenames_from_directory(dwpose_keypoints_dir)
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

        # 串行
        for path_idx, mp4_path in tqdm(enumerate(mp4_paths), desc="Processing videos", unit="video"):
            process_video(mp4_path, dwpose_keypoint_paths[path_idx], threed_keypoint_paths[path_idx], reshape_scale, points_only_flag, show_feet_flag, wanted_fps=16, output_dirname=output_representation_dir, pose_type=pose_type)
    # 并行
        # with Pool(64) as p:
        #     p.starmap(process_video, [(mp4_path, dwpose_keypoint_paths[path_idx], threed_keypoint_paths[path_idx], reshape_scale, points_only_flag, show_feet_flag, 16, output_representation_dir, pose_type) for path_idx, mp4_path in enumerate(mp4_paths)])


    


