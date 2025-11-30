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
from flask import Flask, request, jsonify
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


app = Flask(__name__)

# Global variables for model and detector
model_nlf = None
detector = None
MODEL_PATH = 'pretrained_weights/nlf_l_multi_0.3.2.torchscript'
DEFAULT_RESOLUTION = [512, 896]


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


def process_example(example_id, use_align=True, resolution=None):
    """
    Process a single example with given example_id
    
    Args:
        example_id: The example ID to process
        use_align: Whether to use alignment (default: True)
        resolution: Target resolution as [height, width] (default: [512, 896])
    
    Returns:
        dict: Status and message
    """
    global model_nlf, detector
    
    if resolution is None:
        resolution = DEFAULT_RESOLUTION
    
    # Set up paths
    subdir = os.path.join("../examples", example_id)
    
    if not os.path.exists(subdir):
        return {
            "status": "error",
            "message": f"Example directory not found: {subdir}"
        }
    
    mp4_path = os.path.join(subdir, 'driving.mp4')
    if not os.path.exists(mp4_path):
        return {
            "status": "error",
            "message": f"No video file found: {mp4_path}"
        }
    
    if use_align:
        out_path_aligned = os.path.join(subdir, 'rendered_aligned.mp4')
    else:
        out_path_aligned = os.path.join(subdir, 'rendered.mp4')
    
    # Find reference image
    ref_image_path = None
    for ref_name in ['ref_image.jpg', 'ref_image.png', 'ref.jpg']:
        path = os.path.join(subdir, ref_name)
        if os.path.exists(path):
            ref_image_path = path
            break
    
    if ref_image_path is None:
        return {
            "status": "error",
            "message": f"No reference image found in {subdir}"
        }
    
    try:
        print(f"Processing: {subdir}")
        print(f"Video: {mp4_path}")
        print(f"Reference: {ref_image_path}")
        print(f"Resolution: {resolution}")
        
        # Read video
        decord.bridge.set_bridge("torch")
        vr = VideoReader(mp4_path)
        vr_frames = vr.get_batch(list(range(len(vr))))   # T H W C
        sampling_image_size = resolution
        if vr_frames.shape[1] < vr_frames.shape[2]:
            target_H, target_W = sampling_image_size
        else:
            target_W, target_H = sampling_image_size
        vr_frames = resize_for_rectangle_crop(vr_frames.permute(0, 3, 1, 2), [target_H, target_W], reshape_mode='center').permute(0, 2, 3, 1)  # T H W C ->T C H W -> T H W C

        # Read reference image
        img_ref = cv2.imread(ref_image_path)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        vr_frames_ref = torch.from_numpy(img_ref).unsqueeze(0)
        vr_frames_ref = resize_for_rectangle_crop(vr_frames_ref.permute(0, 3, 1, 2), [target_H, target_W], reshape_mode='center').permute(0, 2, 3, 1)  # 1 H W C ->1 C H W -> 1 H W C

        # Process driving video
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

        # Process reference image
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

        # Align and render
        print("Aligning and rendering...")
        ori_camera_pose = intrinsic_matrix_from_field_of_view([target_H, target_W])
        ori_focal = ori_camera_pose[0, 0]
        pose_3d_first_driving_frame = nlf_results[0]['nlfpose'][0][0].cpu().numpy()  # 3D点 frame-idx bbox-idx detect-idx
        pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)

        if use_align:
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
        
        return {
            "status": "success",
            "message": f"Processing completed successfully",
            "output_path": out_path_aligned
        }
        
    except Exception as e:
        print(f"Error processing example {example_id}: {str(e)}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }


@app.route('/process', methods=['POST'])
def process():
    """
    API endpoint to process an example
    
    Expected JSON payload:
    {
        "example_id": "001",
        "use_align": true,  // optional, default: true
        "resolution": [512, 896]  // optional, default: [512, 896]
    }
    
    Returns:
    {
        "status": "success" | "error",
        "message": "...",
        "output_path": "..."  // only on success
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'example_id' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing required field: example_id"
            }), 400
        
        example_id = data['example_id']
        use_align = data.get('use_align', True)  # Default to True
        resolution = data.get('resolution', DEFAULT_RESOLUTION)
        
        result = process_example(example_id, use_align, resolution)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_nlf is not None,
        "detector_loaded": detector is not None
    }), 200


def initialize_models():
    """Initialize the models on server startup"""
    global model_nlf, detector
    
    print("Loading NLF model...")
    model_nlf = torch.jit.load(MODEL_PATH).cuda().eval()
    
    print("Initializing DWPose detector...")
    detector = DWposeDetector(use_batch=False).to(0)
    
    print("Models loaded successfully!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process pose API server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5999, help='Port to bind to')
    parser.add_argument('--model_path', type=str, default='pretrained_weights/nlf_l_multi_0.3.2.torchscript', 
                        help='Path to NLF model')
    args = parser.parse_args()
    
    MODEL_PATH = args.model_path
    
    # Initialize models before starting the server
    initialize_models()
    
    # Start the Flask server
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
