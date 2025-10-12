import cv2
import numpy as np
import math
from PIL import Image
from render_3d.taichi_cylinder import render_whole
from NLFPoseExtract.nlf_draw import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, preview_nlf_2d
from concurrent.futures import ProcessPoolExecutor, as_completed
from pose_draw.draw_pose_main import draw_pose_to_canvas_np, scale_image_hw_keep_size
import torch.multiprocessing as mp
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import trimesh
import copy
import random
import torch

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
    



def render_nlf_as_images(data, poses, reshape_pool=None):
    """ return a list of images """
    height, width = data[0]['video_height'], data[0]['video_width']
    video_length = len(data)

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
    uncollected_smpl_poses = [item['nlfpose'] for item in data]

    # 获取min_z，并重新收集poses
    min_z = float('inf')
    smpl_poses = [[] for _ in range(len(uncollected_smpl_poses))]
    for frame_idx in range(len(uncollected_smpl_poses)):
        for person_idx in range(len(uncollected_smpl_poses[frame_idx])):  # 每个人（每个bbox）只给出一个pose
            if len(uncollected_smpl_poses[frame_idx][person_idx]) > 0:    # 有返回的骨骼
                smpl_poses[frame_idx].append(uncollected_smpl_poses[frame_idx][person_idx][0])
                for joint_idx in range(len(uncollected_smpl_poses[frame_idx][person_idx][0])):
                    z_value = uncollected_smpl_poses[frame_idx][person_idx][0][joint_idx][2].item()
                    if z_value < min_z:
                        min_z = z_value
            else:
                smpl_poses[frame_idx].append(torch.zeros((24, 3), dtype=torch.float32))  # 没有检测到人，就放一个全0的


    aligned_poses = copy.deepcopy(poses)
    if reshape_pool is not None:
        reshape_pool.set_offset_3d_z(min_z)
        for i in range(video_length):
            persons_joints_list = smpl_poses[i]
            poses_list = aligned_poses[i]
            # 对里面每一个人，取关节并进行变形；并且修改2d；如果3d不存在，把2d的手/脸也去掉
            for person_idx, person_joints in enumerate(persons_joints_list):
                candidate = poses_list['bodies']['candidate'][person_idx]
                subset = poses_list['bodies']['subset'][person_idx]
                face = poses_list["faces"][person_idx]
                right_hand = poses_list["hands"][2 * person_idx]
                left_hand = poses_list["hands"][2 * person_idx + 1]
                # print(f"debug: person_joints.shape: {person_joints.shape}")
                reshape_pool.apply_random_reshapes(person_joints, candidate, left_hand, right_hand, face, subset)

                
    # 串行获取每一帧的cylinder_specs
    cylinder_specs_list = []
    for i in range(video_length):
        cylinder_specs = get_single_pose_cylinder_specs((i, smpl_poses[i], focal, princpt, height, width, colors, limb_seq, draw_seq))
        cylinder_specs_list.append(cylinder_specs)


    frames_np_rgba = render_whole(cylinder_specs_list, H=height, W=width, fx=focal, fy=focal, cx=princpt[0], cy=princpt[1])
    canvas_2d = draw_pose_to_canvas_np(aligned_poses, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True)

    # 覆盖 + rescale
    scale_h = random.uniform(0.85, 1.04)
    scale_w = random.uniform(0.85, 1.04)
    rescale_flag = random.random() < 0.4 if reshape_pool is None else False
    for i in range(len(frames_np_rgba)):
        frame_img = frames_np_rgba[i]
        canvas_img = canvas_2d[i]
        mask = canvas_img != 0
        frame_img[:, :, :3][mask] = canvas_img[mask]
        frames_np_rgba[i] = frame_img
        if rescale_flag:
            frames_np_rgba[i]  = scale_image_hw_keep_size(frames_np_rgba[i], scale_h, scale_w)
        if reshape_pool is not None:
            # 4%的概率完全消除某些帧
            if random.random() < 0.04:
                frames_np_rgba[i][:, :, 0:3] = 0

    return frames_np_rgba