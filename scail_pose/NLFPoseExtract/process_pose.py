import os
import sys

# 动态添加项目根目录到 sys.path，这样就不需要 export PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # SCAIL_Pose 目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import torch
import pickle
import torchvision
import shutil
import glob
import random
from tqdm import tqdm   
import decord
from decord import VideoReader, cpu, gpu
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import argparse
from NLFPoseExtract.nlf_render import render_nlf_as_images, collect_smpl_poses, shift_dwpose_according_to_nlf, p3d_single_p2d
from NLFPoseExtract.nlf_draw import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_to_p2d
from DWPoseProcess.dwpose import DWposeDetector
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from NLFPoseExtract.extract_nlfpose_batch import process_video_nlf
from NLFPoseExtract.reshape_utils_3d import reshapePool3d
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
import copy
from NLFPoseExtract.align3d import solve_new_camera_params_central, solve_new_camera_params_down


def recollect_nlf(data):
    new_data = []
    for item in data:
        new_item = item.copy()
        if len(item['bboxes']) > 0:
            new_item['bboxes'] = item['bboxes'][:1]
            new_item['nlfpose'] = item['nlfpose'][:1]
        new_data.append(new_item)
    return new_data

def recollect_dwposes(poses):
    new_poses = []
    for pose in poses:
        new_pose = pose.copy()
        for i in range(1):
            bodies = pose["bodies"]
            faces = pose["faces"][i:i+1]
            hands = pose["hands"][2*i:2*i+2]
            candidate = bodies["candidate"][i:i+1]  # candidate是所有点的坐标和置信度
            subset = bodies["subset"][i:i+1]   # subset是认为的有效点
            new_pose = {
                "bodies": {
                    "candidate": candidate,
                    "subset": subset
                },
                "faces": faces,
                "hands": hands
            }
        new_poses.append(new_pose)
    return new_poses



def resize_for_rectangle_crop(arr, image_size, reshape_mode='random'):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(arr, size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])], interpolation=InterpolationMode.BICUBIC)
    else:
        arr = resize(arr, size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]], interpolation=InterpolationMode.BICUBIC)

    h, w = arr.shape[2], arr.shape[3]

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == 'random' or reshape_mode == 'none':
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == 'center':
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=image_size[0], width=image_size[1]
    )
    return arr

def scale_faces(poses, pose_2d_ref):
    # 输入：两个list of dict，poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2
    # 根据脸部的中心点，对poses中的脸部关键点进行缩放
    # 也即：计算ref里面脸部中心点(idx: 30)到其他脸部关键点的中心距离， 计算poses里面脸部中心点到其他脸部关键点的中心距离，得到scale_n
    # 对scale_n 取一下0.8-1.5的上下界，然后应用在poses上
    # 注意：需要inplace改变poses

    ref = pose_2d_ref[0]
    pose_0 = poses[0]
        

    face_0 = pose_0['faces']  # shape: (1, 68, 2)
    face_ref = ref['faces']

    # 提取 numpy 数组
    face_0 = np.array(face_0[0])      # (68, 2)
    face_ref = np.array(face_ref[0])

    # 中心点（鼻尖或面部中心）
    center_idx = 30
    center_0 = face_0[center_idx]
    center_ref = face_ref[center_idx]

    # 计算到中心点的距离
    dist = np.linalg.norm(face_0 - center_0, axis=1)
    dist_ref = np.linalg.norm(face_ref - center_ref, axis=1)

    # 避免中心点自身的 0 距离影响
    dist = np.delete(dist, center_idx)
    dist_ref = np.delete(dist_ref, center_idx)

    mean_dist = np.mean(dist)
    mean_dist_ref = np.mean(dist_ref)

    if mean_dist < 1e-6:
        scale_n = 1.0
    else:
        scale_n = mean_dist_ref / mean_dist

    # 限制在 [0.8, 1.5]
    scale_n = np.clip(scale_n, 0.8, 1.5)

    for i, pose in enumerate(poses):
        face = pose['faces']
        # 提取 numpy 数组
        face = np.array(face[0])      # (68, 2)
        center = face[center_idx]
        scaled_face = (face - center) * scale_n + center
        poses[i]['faces'][0] = scaled_face

        body = pose['bodies']
        candidate = body['candidate']
        candidate_np = np.array(candidate[0])   # (14, 2)
        body_center = candidate_np[0]
        scaled_candidate = (candidate_np - body_center) * scale_n + body_center
        poses[i]['bodies']['candidate'][0] = scaled_candidate

    # inplace 修改
    pose['faces'][0] = scaled_face
    
    return scale_n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video with NLF pose estimation')
    parser.add_argument('--subdir', type=str, default="../examples/001", help='Path to the subdirectory to process')
    parser.add_argument('--model_path', type=str, default='pretrained_weights/nlf_l_multi_0.3.2.torchscript', 
                        help='Path to NLF model')
    parser.add_argument('--use_align', action='store_true', help='Whether to use 2D keypoints from reference image for alignment')
    parser.add_argument('--resolution', type=int, nargs=2, default=[512, 896], 
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Target resolution as [height, width], currently only [512, 896] are supported')
    args = parser.parse_args()
    
    subdir = args.subdir
    model_nlf = torch.jit.load(args.model_path).cuda().eval()
    decord.bridge.set_bridge("torch")

    # 设置路径
    mp4_path = os.path.join(subdir, 'driving.mp4')
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"No video file found in {subdir}")
    
    if args.use_align:
        out_path_aligned = os.path.join(subdir, 'rendered_aligned.mp4')
    else:
        out_path_aligned = os.path.join(subdir, 'rendered.mp4')
    
    ref_image_path = os.path.join(subdir, 'ref_image.jpg')
    if not os.path.exists(ref_image_path):
        ref_image_path = os.path.join(subdir, 'ref_image.png')
    if not os.path.exists(ref_image_path):
        ref_image_path = os.path.join(subdir, 'ref.jpg')
    if not os.path.exists(ref_image_path):
        raise FileNotFoundError(f"No reference image found in {subdir}")

    print(f"Processing: {subdir}")
    print(f"Video: {mp4_path}")
    print(f"Reference: {ref_image_path}")
    print(f"Resolution: {args.resolution}")

    # 读取视频
    vr = VideoReader(mp4_path)
    vr_frames = vr.get_batch(list(range(len(vr))))   # T H W C
    sampling_image_size = args.resolution
    if vr_frames.shape[1] < vr_frames.shape[2]:
        target_H, target_W = sampling_image_size
    else:
        target_W, target_H = sampling_image_size
    vr_frames = resize_for_rectangle_crop(vr_frames.permute(0, 3, 1, 2), [target_H, target_W], reshape_mode='center').permute(0, 2, 3, 1)  # T H W C ->T C H W -> T H W C

    # 读取参考图片
    img_ref = cv2.imread(ref_image_path)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    vr_frames_ref = torch.from_numpy(img_ref).unsqueeze(0)
    vr_frames_ref = resize_for_rectangle_crop(vr_frames_ref.permute(0, 3, 1, 2), [target_H, target_W], reshape_mode='center').permute(0, 2, 3, 1)  # 1 H W C ->1 C H W -> 1 H W C

    # 初始化检测器
    detector = DWposeDetector(use_batch=False).to(0)

    # 处理Driving视频
    print("Processing driving video...")
    detector_return_list = []
    pil_frames = []
    for i in tqdm(range(len(vr_frames)), desc="Detecting poses in video"):
        pil_frame = Image.fromarray(vr_frames[i].numpy())
        pil_frames.append(pil_frame)
        detector_result = detector(pil_frame)
        detector_return_list.append(detector_result)

    W, H = pil_frames[0].size
    poses, scores, det_results = zip(*detector_return_list)
    
    print("Running NLF on driving video...")
    nlf_results = process_video_nlf(model_nlf, vr_frames, det_results)

    # 处理ref图片
    print("Processing reference image...")
    detector_return_list_ref = []
    pil_frames_ref = []
    for i in range(len(vr_frames_ref)):
        pil_frame = Image.fromarray(vr_frames_ref[i].numpy())
        pil_frames_ref.append(pil_frame)
        detector_result = detector(pil_frame)
        detector_return_list_ref.append(detector_result)

    poses_ref, scores_ref, det_results_ref = zip(*detector_return_list_ref)
    
    print("Running NLF on reference image...")
    nlf_results_ref = process_video_nlf(model_nlf, vr_frames_ref, det_results_ref)

    # 进行对齐和渲染
    print("Aligning and rendering...")
    ori_camera_pose = intrinsic_matrix_from_field_of_view([target_H, target_W])
    ori_focal = ori_camera_pose[0, 0]
    pose_3d_first_driving_frame = nlf_results[0]['nlfpose'][0][0].cpu().numpy()  # 3D点 frame-idx bbox-idx detect-idx
    pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)

    if args.use_align:
        poses_2d_ref = poses_ref[0]['bodies']['candidate'][0][:14]
        poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * target_W
        poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * target_H

        poses_2d_subset = poses_ref[0]['bodies']['subset'][0][:14]
        pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

        valid_indices = []
        valid_upper_indices = []
        valid_lower_indices = []
        upper_body_indices = [0, 2, 3, 5, 6]
        lower_body_indices = [9, 10, 12, 13]
        excluded_indices = [3, 4, 6, 7]  # 去除手
        for i in range(len(poses_2d_subset)):
            if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                if i in upper_body_indices:
                    valid_upper_indices.append(i)
                if i in lower_body_indices:
                    valid_lower_indices.append(i)
        
        if len(valid_lower_indices) >= 4:
            print("Align feet")
            valid_indices = [1] + valid_lower_indices
        else:
            print("Align body")
            valid_indices = [1] + valid_upper_indices

        pose_2d_ref = poses_2d_ref[valid_indices]
        pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]
        
        if len(valid_lower_indices) >= 4:
            new_camera_intrinsics, scale_m = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [target_H, target_W], pose_2d_ref)
        else:
            new_camera_intrinsics, scale_m = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [target_H, target_W], pose_2d_ref)
        
        # m 代表缩放了多少
        scale_face = scale_faces(list(poses), list(poses_ref))   # poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2

        print(f"Scale - m: {scale_m}, face: {scale_face}")

        nlf_results = recollect_nlf(nlf_results)
        poses = recollect_dwposes(list(poses))
        shift_dwpose_according_to_nlf(collect_smpl_poses(nlf_results), poses, ori_camera_pose, new_camera_intrinsics, target_H, target_W)
        
        print("Rendering final video...")
        frames_np = render_nlf_as_images(nlf_results, poses, reshape_pool=None, intrinsic_matrix=new_camera_intrinsics)

    else:
        nlf_results = recollect_nlf(nlf_results)
        print("Rendering final video...")
        frames_np = render_nlf_as_images(nlf_results, poses, reshape_pool=None, intrinsic_matrix=ori_camera_pose)

    mpy.ImageSequenceClip(frames_np, fps=16).write_videofile(out_path_aligned)
    print(f"Done! Output saved to: {out_path_aligned}")

        

