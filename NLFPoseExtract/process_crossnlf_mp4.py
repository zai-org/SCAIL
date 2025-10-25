import os
import sys
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
from NLFPoseExtract.nlf_render import render_nlf_as_images, collect_smpl_poses, shift_dwpose_according_to_nlf, p3d_single_p2d
from NLFPoseExtract.nlf_draw import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_to_p2d
from DWPoseProcess.dwpose import DWposeDetector
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from DWPoseProcess.extract_nlfpose import process_video_nlf
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


if __name__ == '__main__':
    model_nlf = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()

    evaluation_dir = "/workspace/ywh_data/EvalCross/cross_pair_eval100"
    wan_dir = "/workspace/ywh_data/EvalCross/Results/wan2.2-animate"
    decord.bridge.set_bridge("torch")

    for subdir_idx, subdir in tqdm(enumerate(sorted(os.listdir(evaluation_dir)))):
        mp4_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        out_path_aligned = os.path.join(evaluation_dir, subdir, 'smpl_aligned.mp4')
        ref_image_path = os.path.join(evaluation_dir, subdir, 'ref_image.jpg')
        meta_cache_dir = os.path.join(evaluation_dir, subdir, 'meta')
        poses_cache_path = os.path.join(meta_cache_dir, 'keypoints.pt')
        det_cache_path = os.path.join(meta_cache_dir, 'bboxes.pt')
        nlf_cache_path = os.path.join(meta_cache_dir, 'nlf_results.pkl')
        poses_ref_cache_path = os.path.join(meta_cache_dir, 'keypoints_ref.pt')
        nlf_ref_cache_path = os.path.join(meta_cache_dir, 'nlf_results_ref.pkl')
        os.makedirs(meta_cache_dir, exist_ok=True)
        src_pose_path = os.path.join(wan_dir, subdir, 'src_pose.mp4')
        shutil.copy(src_pose_path, os.path.join(evaluation_dir, subdir, 'wan_src_pose.mp4'))

        vr = VideoReader(mp4_path)
        vr_frames = vr.get_batch(list(range(len(vr))))   # T H W C
        sampling_image_size = [512, 896]
        if vr_frames.shape[1] < vr_frames.shape[2]:
            target_H, target_W = sampling_image_size
        else:
            target_W, target_H = sampling_image_size
        vr_frames = resize_for_rectangle_crop(vr_frames.permute(0, 3, 1, 2), [target_H, target_W], reshape_mode='center').permute(0, 2, 3, 1)  # T H W C ->T C H W -> T H W C

        img_ref = cv2.imread(ref_image_path)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        vr_frames_ref =  torch.from_numpy(img_ref).unsqueeze(0)
        vr_frames_ref = resize_for_rectangle_crop(vr_frames_ref.permute(0, 3, 1, 2), [target_H, target_W], reshape_mode='center').permute(0, 2, 3, 1)  # 1 H W C ->1 C H W -> 1 H W C


        if os.path.exists(poses_cache_path) and os.path.exists(det_cache_path) and os.path.exists(nlf_cache_path):
            poses = torch.load(poses_cache_path)
            poses_ref = torch.load(poses_ref_cache_path)
            bboxes = torch.load(det_cache_path)
            with open(nlf_cache_path, 'rb') as f:
                nlf_results = pickle.load(f)
            with open(nlf_ref_cache_path, 'rb') as f:
                nlf_results_ref = pickle.load(f)
            ori_camera_pose = intrinsic_matrix_from_field_of_view([target_H, target_W])
            ori_focal = ori_camera_pose[0,0]
            pose_3d_first_driving_frame = nlf_results[0]['nlfpose'][0][0].cpu().numpy()  # 3D点 frame-idx bbox-idx detect-idx
            pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)



            use_ref_2d = True
            if use_ref_2d:
                poses_2d_ref = poses_ref[0]['bodies']['candidate'][0][:14]
                poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * target_W
                poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * target_H
            else:
                poses_3d_ref = nlf_results_ref[0]['nlfpose'][0][0].cpu().numpy()
                poses_3d_ref_coco = process_data_to_COCO_format(poses_3d_ref)[:14]
                poses_2d_ref = p3d_to_p2d(poses_3d_ref_coco, target_H, target_W)[0][:, :2]

            poses_2d_subset = poses_ref[0]['bodies']['subset'][0][:14]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

            valid_indices = []
            valid_upper_indices = []
            valid_lower_indices = []
            upper_body_indices = [0, 2, 3, 5, 6]
            lower_body_indices = [9, 10, 12, 13]
            excluded_indices = [3, 4, 6, 7] # 去除手
            for i in range(len(poses_2d_subset)):
                if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                    if i in upper_body_indices:
                        valid_upper_indices.append(i)
                    if i in lower_body_indices:
                        valid_lower_indices.append(i)
            if len(valid_lower_indices) >= 4:
                print("align feet")
                valid_indices = [1] + valid_lower_indices
            else:
                print("align body")
                valid_indices = [1] + valid_upper_indices

            pose_2d_ref = poses_2d_ref[valid_indices]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]
            
            if len(valid_lower_indices) >= 4:
                new_camera_intrinsics = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [target_H, target_W], pose_2d_ref)
            else:
                new_camera_intrinsics = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [target_H, target_W], pose_2d_ref)

            nlf_results = recollect_nlf(nlf_results)
            poses = recollect_dwposes(poses)      
            shift_dwpose_according_to_nlf(collect_smpl_poses(nlf_results), poses, ori_camera_pose, new_camera_intrinsics, target_H, target_W)      
            frames_np = render_nlf_as_images(nlf_results, poses, reshape_pool=None, intrinsic_matrix=new_camera_intrinsics)
            mpy.ImageSequenceClip(frames_np, fps=16).write_videofile(out_path_aligned)
        else:
            detector = DWposeDetector(use_batch=False).to(0)


            # 处理Driving视频
            detector_return_list = []
            pil_frames = []
            for i in range(len(vr_frames)):
                pil_frame = Image.fromarray(vr_frames[i].numpy())
                pil_frames.append(pil_frame)
                detector_result = detector(pil_frame)
                detector_return_list.append(detector_result)


            W, H = pil_frames[0].size

            poses, scores, det_results = zip(*detector_return_list) # 这里存的是整个视频的poses
            nlf_results = process_video_nlf(model_nlf, vr_frames, det_results)

            torch.save(poses, poses_cache_path)
            torch.save(det_results, det_cache_path)
            with open(nlf_cache_path, 'wb') as f:
                pickle.dump(nlf_results, f)


            # 处理ref图片
            detector_return_list = []
            pil_frames = []
            for i in range(len(vr_frames_ref)):
                pil_frame = Image.fromarray(vr_frames_ref[i].numpy())
                pil_frames.append(pil_frame)
                detector_result = detector(pil_frame)
                detector_return_list.append(detector_result)


            W, H = pil_frames[0].size

            poses_ref, scores_ref, det_results_ref = zip(*detector_return_list) # 这里存的是整个视频的poses
            nlf_results_ref = process_video_nlf(model_nlf, vr_frames_ref, det_results_ref)

            torch.save(poses_ref, poses_ref_cache_path)
            with open(nlf_ref_cache_path, 'wb') as f:
                pickle.dump(nlf_results_ref, f)


        

