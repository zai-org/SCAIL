import cv2
import numpy as np
import math
from PIL import Image
from render_3d.taichi_cylinder import render_whole
from NLFPoseExtract.nlf_draw import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, preview_nlf_2d
from concurrent.futures import ProcessPoolExecutor, as_completed
from DWPoseProcess.dwpose.util import draw_bodypose, draw_handpose
import torch.multiprocessing as mp
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import trimesh
import copy

def get_single_pose_cylinder_specs(args):
    """渲染单个pose的辅助函数，用于并行处理"""
    idx, pose, focal, princpt, height, width, colors, limb_seq, draw_seq = args
    cylinder_specs = []
    
    for joints3d in pose:  # 多人
        joints3d = joints3d.cpu().numpy()
        joints3d = process_data_to_COCO_format(joints3d)
        for line_idx in draw_seq:
            line = limb_seq[line_idx]
            start, end = line[0], line[1]
            if np.sum(joints3d[start]) == 0 or np.sum(joints3d[end]) == 0:
                continue
            else:
                cylinder_specs.append((joints3d[start], joints3d[end], colors[line_idx]))
    return cylinder_specs
    



def render_nlf_as_images(data, motion_indices, dwpose_kpt_seq=None, reshape_pool=None):
    """ return a list of images """
    height, width = data['video_height'], data['video_width']
    video_length = data['video_length']

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
        [0, 14],   # 13 Nose -> R. Eye
        [14, 16],  # 14 R. Eye -> R. Ear
        [0, 15],   # 15 Nose -> L. Eye
        [15, 17],  # 16 L. Eye -> L. Ear
    ]

    draw_seq = [0, 2, 3, # Neck -> R. Shoulder -> R. Elbow -> R. Wrist
                1, 4, 5, # Neck -> L. Shoulder -> L. Elbow -> L. Wrist
                6, 7, 8, # Neck -> R. Hip -> R. Knee -> R. Ankle
                9, 10, 11, # Neck -> L. Hip -> L. Knee -> L. Ankle
                12, # Neck -> Nose
                13, 14, # Nose -> R. Eye -> R. Ear
                15, 16, # Nose -> L. Eye -> L. Ear
                ]   # 从近心端往外扩展

    colors = [[c / 300 + 0.15 for c in color_rgb] + [0.8] for color_rgb in ordered_colors_255]
    intrinsic_matrix = intrinsic_matrix_from_field_of_view((height, width))
    focal = intrinsic_matrix[0,0]
    princpt = (intrinsic_matrix[0,2], intrinsic_matrix[1,2])  # 主点 (cx, cy)
    smpl_poses = data['pose']['joints3d_nonparam']

    # 获取min_z
    min_z = float('inf')
    for frame_idx in range(len(smpl_poses)):
        for person_idx in range(len(smpl_poses[frame_idx])):
            for joint_idx in range(len(smpl_poses[frame_idx][person_idx])):
                z_value = smpl_poses[frame_idx][person_idx][joint_idx][2].item()
                if z_value < min_z:
                    min_z = z_value

    if reshape_pool is not None:
        reshape_pool.set_offset_3d_z(min_z)
        for i in range(video_length):
            if i in motion_indices:
                joints_list = smpl_poses[i]
                dwpose = dwpose_kpt_seq[i]
                reshape_pool.apply_random_reshapes(joints_list, dwpose)   # 对joints_list和dwpose应用形变
                smpl_poses[i] = joints_list
                dwpose_kpt_seq[i] = dwpose

    # 串行
    cylinder_specs_list = []
    for i in range(video_length):
        if i in motion_indices:
            cylinder_specs = get_single_pose_cylinder_specs((i, smpl_poses[i], focal, princpt, height, width, colors, limb_seq, draw_seq))
            cylinder_specs_list.append(cylinder_specs)



    data['pose']['joints3d_nonparam'] = [smpl_poses[i] for i in motion_indices]
    frames_np_rgba = render_whole(cylinder_specs_list, H=height, W=width, fx=focal, fy=focal, cx=princpt[0], cy=princpt[1])
    PIL_frames = []
    PIL_frames_dw = []
    for idx, frame_np in enumerate(frames_np_rgba):
        frame_np = frame_np[:, :, :3]
        final_dw_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        dwpose = dwpose_kpt_seq[idx]
        for i in range(len(dwpose["bodies"]["candidate"])):
            bodies = dwpose["bodies"]
            candidate = bodies["candidate"][i]
            subset = bodies["subset"][i:i+1]   # subset是认为的有效点
            subset[:, 1:14] = -1
            hands = dwpose["hands"][2*i:2*i+2]

            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas = draw_bodypose(canvas, candidate, subset)
            canvas = draw_handpose(canvas, hands)
            final_dw_canvas = final_dw_canvas + canvas
        PIL_frames.append(Image.fromarray(frame_np))
        PIL_frames_dw.append(Image.fromarray(final_dw_canvas))


    return PIL_frames, PIL_frames_dw