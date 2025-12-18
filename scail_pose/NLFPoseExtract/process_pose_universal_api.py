
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
import gc
from tqdm import tqdm   
import decord
from decord import VideoReader, cpu, gpu
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import string
import datetime
from werkzeug.utils import secure_filename
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
from NLFPoseExtract.extract_nlfpose_batch import process_video_multi_nlf
from NLFPoseExtract.nlf_render import render_multi_nlf_as_images

# Add sam2 to path
if os.path.join(project_root, "sam2") not in sys.path:
    sys.path.append(os.path.join(project_root, "sam2"))
from sam2.build_sam import build_sam2_video_predictor


app = Flask(__name__)

# Global variables for model and detector
model_nlf = None
detector = None
MODEL_PATH = 'pretrained_weights/nlf_l_multi_0.3.2.torchscript'
DEFAULT_RESOLUTION = [512, 896]
EXAMPLES_DIR = '../examples'
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4'}


def generate_example_id():
    """Generate a unique 9-digit example ID"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # Take last 9 digits to ensure it's 9 digits
    example_id = timestamp[-9:]
    
    # Ensure uniqueness by checking if directory exists
    while os.path.exists(os.path.join(EXAMPLES_DIR, example_id)):
        # Add a random suffix if collision occurs
        import time
        time.sleep(0.1)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        example_id = timestamp[-9:]
    
    return example_id


def allowed_file(filename, allowed_extensions):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def process_driving_video(input_path, output_path, max_frames=81, target_fps=16):
    """
    Process driving video based on frame count rules:
    - < 65 frames: Error
    - 65 <= frames < 81: Sample to 65 frames
    - 81 <= frames < 96: Take first 81 frames
    - >= 96 frames: Sample to 81 frames
    
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        max_frames: Maximum number of frames (default: 81) - Used as the target for sampling/trimming
        target_fps: Target FPS for downsampling (default: 16)
    
    Returns:
        dict: Processing info with status and message
    """
    try:
        # Read video properties
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Failed to open video"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        decord.bridge.set_bridge("torch")
        
        print(f"Video info: {total_frames} frames, {original_fps:.2f} fps, {width}x{height}")
        
        if total_frames < 65:
            return {
                "status": "error", 
                "message": f"Video too short: {total_frames} frames. Minimum 65 frames required."
            }
            
        vr = VideoReader(input_path)
        
        if 65 <= total_frames < 81:
            # 65-81帧，直接按65帧采样
            print(f"Sampling 65 frames from {total_frames} frames")
            indices = np.linspace(0, total_frames - 1, 65, dtype=int)
            frames = vr.get_batch(indices)
            out_fps = target_fps
            
        elif 81 <= total_frames < 96:
            # <96帧 >=81帧，直接取前81帧
            print(f"Taking first 81 frames from {total_frames} frames")
            indices = list(range(81))
            frames = vr.get_batch(indices)
            out_fps = original_fps
            
        else: # total_frames >= 96
            # >96帧，直接均匀取样到81帧
            print(f"Sampling 81 frames from {total_frames} frames")
            indices = np.linspace(0, total_frames - 1, 81, dtype=int)
            frames = vr.get_batch(indices)
            out_fps = target_fps
            
        # Save using moviepy
        frames_np = [frames[i].numpy() for i in range(len(frames))]
        clip = mpy.ImageSequenceClip(frames_np, fps=out_fps)
        clip.write_videofile(output_path, codec='libx264', logger=None)
        
        return {
            "status": "success",
            "message": f"Processed: {total_frames} -> {len(frames)} frames",
            "total_frames": len(frames),
            "fps": out_fps
        }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Video processing failed: {str(e)}"
        }


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


def get_largest_bbox_indices(bboxes, num_bboxes=2):
    # 计算每个bbox的面积
    def calculate_area(bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    # 计算每个bbox的面积，并保留原索引
    bboxes_with_area = [(i, calculate_area(bbox)) for i, bbox in enumerate(bboxes)]
    
    # 根据面积从大到小排序
    bboxes_with_area.sort(key=lambda x: x[1], reverse=True)
    
    # 取出面积最大的 num_bboxes 个索引
    largest_indices = [idx for idx, _ in bboxes_with_area[:num_bboxes]]
    
    return largest_indices


def change_poses_to_limit_num(poses, bboxes, num_bboxes=2):
    bboxes = list(bboxes)  # ✅ 转换为可变列表
    for idx, (pose, bbox) in enumerate(zip(poses, bboxes)):
        if len(bbox) == 0:
            continue
        largest_indices = get_largest_bbox_indices(bbox, num_bboxes)
        
        # 过滤 subset、hands、faces
        pose['bodies']['subset'] = pose['bodies']['subset'][largest_indices]
        
        new_hands = []
        for i in largest_indices:
            if 2*i+1 < len(pose['hands']):
                new_hands.append(pose['hands'][2*i])
                new_hands.append(pose['hands'][2*i+1])
        pose['hands'] = new_hands
        
        pose['faces'] = [pose['faces'][i] for i in largest_indices if i < len(pose['faces'])]
        
        bboxes[idx] = [bbox[i] for i in largest_indices]

    return poses, bboxes


def get_samurai_crop_video(video_input_path, video_output_root, bboxes_0, final_keypoints_list, predictor=None, use_green_background=True):
    # 用 decord 读取视频帧
    if video_input_path.endswith(".mp4"):
        vr = VideoReader(video_input_path)
        decord.bridge.set_bridge("torch")
        loaded_frames = vr.get_batch(list(range(len(vr)))).numpy()
        height, width = loaded_frames[0].shape[:2]

    # 每个人一个输出视频
    num_persons = len(final_keypoints_list)
    print(f"Detected {num_persons} persons, will save {num_persons} videos.")

    prompts = {fid: ((x1, y1, x2, y2), 0) for fid, (x1, y1, x2, y2) in enumerate(bboxes_0)}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for person_idx in range(num_persons):
            print(f"Processing person {person_idx + 1}/{num_persons}...")

            state = predictor.init_state(video_input_path, offload_video_to_cpu=True)
            bbox, track_label = prompts[person_idx]
            bbox = (bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height)
            points = copy.deepcopy(final_keypoints_list[person_idx])
            points[:, 0] *= width
            points[:, 1] *= height
            
            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, points=points, labels=np.ones(points.shape[0]), frame_idx=0, obj_id=0)

            output_frames = []
            output_mask_frames = []
            repeat_flag = False

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                img = loaded_frames[frame_idx].copy()
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy() > 0.0   # 更新 mask
                    mask_log = np.zeros_like(img)
                    if use_green_background:
                        mask_img = np.full_like(img, (30, 60, 30))
                    else:
                        mask_img = np.zeros_like(img)
                    mask_img[mask] = img[mask]
                    mask_log[mask] = 255
                output_frames.append(mask_img)    # mask_img: array of [h, w, 3]
                output_mask_frames.append(mask_log)    # mask: array of [h, w]

            del state
            gc.collect()
            torch.cuda.empty_cache()

            # 用 moviepy 保存视频
            output_name = os.path.join(video_output_root, f"{person_idx+1}.mp4")
            clip = mpy.ImageSequenceClip(output_frames, fps=16)
            clip.write_videofile(output_name, codec="libx264", audio=False)
            print(f"Saved {output_name}")

    # del predictor # Do not delete predictor here as it might be reused or managed outside
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


def process_example(example_id, use_align=True, resolution=None, use_multi=False):
    """
    Process a single example with given example_id
    
    Args:
        example_id: The example ID to process
        use_align: Whether to use alignment (default: True)
        resolution: Target resolution as [height, width] (default: [512, 896])
        use_multi: Whether to use multi-person processing (default: False)
    
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
    
    if use_multi:
        out_path_aligned = os.path.join(subdir, 'rendered.mp4')
        # Multi-person processing logic
        try:
            print(f"Processing multi-person: {subdir}")
            print(f"Video: {mp4_path}")
            print(f"Resolution: {resolution}")
            
            # 1. Extract DWpose and BBoxes
            print("Extracting DWpose and BBoxes...")
            # detector is global
            
            vr = VideoReader(mp4_path)
            decord.bridge.set_bridge("torch")
            vr_frames = vr.get_batch(list(range(len(vr)))).numpy() # T H W C
            
            detector_return_list = []
            pil_frames = []
            for i in tqdm(range(len(vr_frames)), desc="Detecting poses"):
                pil_frame = Image.fromarray(vr_frames[i])
                pil_frames.append(pil_frame)
                detector_result = detector(pil_frame)
                detector_return_list.append(detector_result)
            
            poses, scores, det_results = zip(*detector_return_list)
            
            # Save meta
            meta_dir = os.path.join(subdir, "meta")
            os.makedirs(meta_dir, exist_ok=True)
            torch.save(poses, os.path.join(meta_dir, "keypoints.pt"))
            torch.save(det_results, os.path.join(meta_dir, "bboxes.pt"))
            
            # 2. Run Samurai Segmentation
            print("Running Samurai Segmentation...")
            samurai_output_root = os.path.join(subdir, "samurai")
            if os.path.exists(samurai_output_root):
                shutil.rmtree(samurai_output_root)
            os.makedirs(samurai_output_root, exist_ok=True)
            
            device = "cuda:0"
            # Assuming sam2 config and checkpoint paths relative to project root                
            predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2/checkpoints/sam2.1_hiera_large.pt", device=device)
            
            # Prepare inputs for samurai
            bboxes_0 = det_results[0]
            indices = get_largest_bbox_indices(bboxes_0)
            bboxes_0 = [bboxes_0[index] for index in indices]
            
            keypoints_0 = poses[0]['bodies']['candidate']
            subset_0 = poses[0]['bodies']['subset']
            chosen_keypoints = keypoints_0[indices]
            
            final_keypoints_list = []
            for i in range(len(chosen_keypoints)):
                keypoints_for_person = chosen_keypoints[i]
                subset_for_person = subset_0[i]
                considered_points = [0, 1, 14, 15]
                subset_for_person_mod = subset_for_person.copy() 
                for k in range(len(subset_for_person_mod)):
                    if k not in considered_points:
                        subset_for_person_mod[k] = -1
                new_keypoints = keypoints_for_person[subset_for_person_mod != -1]
                final_keypoints_list.append(new_keypoints)
                
            get_samurai_crop_video(mp4_path, samurai_output_root, bboxes_0, final_keypoints_list, predictor=predictor)
            
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
            
            # 3. Render Multi NLF
            print("Rendering Multi NLF...")
            # model_nlf is global
            decord.bridge.set_bridge("torch")
            
            vr_frames_list = []
            for samurai_mp4_path in sorted(glob.glob(os.path.join(samurai_output_root, '*.mp4'))):
                vr_tmp = VideoReader(samurai_mp4_path)
                vr_frames_tmp = vr_tmp.get_batch(list(range(len(vr_tmp))))
                vr_frames_list.append(vr_frames_tmp)
                
            poses_list = list(poses)
            det_results_list = list(det_results)
            
            poses_list, det_results_list = change_poses_to_limit_num(poses_list, det_results_list)
            
            nlf_results = process_video_multi_nlf(model_nlf, vr_frames_list)
            frames_ori_np = render_multi_nlf_as_images(nlf_results, poses_list, reshape_pool=None)
            
            mpy.ImageSequenceClip(frames_ori_np, fps=16).write_videofile(out_path_aligned)
            print(f"Done! Output saved to: {out_path_aligned}")
            
            return {
                "status": "success",
                "message": f"Processing completed successfully (multi-person)",
                "output_path": out_path_aligned
            }
            
        except Exception as e:
            print(f"Error processing example {example_id}: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Processing failed: {str(e)}"
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
        "resolution": [512, 896],  // optional, default: [512, 896]
        "use_multi": false // optional, default: false
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
        use_multi = data.get('use_multi', False) # Default to False
        
        result = process_example(example_id, use_align, resolution, use_multi)
        
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


@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload reference image and driving video
    
    Expected form data:
    - ref_image: image file (jpg/png)
    - driving_video: video file (mp4)
    
    Returns:
    {
        "status": "success" | "error",
        "message": "...",
        "example_id": "..."  // only on success
    }
    """
    try:
        # Check if files are in request
        if 'ref_image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "Missing required file: ref_image"
            }), 400
        
        if 'driving_video' not in request.files:
            return jsonify({
                "status": "error",
                "message": "Missing required file: driving_video"
            }), 400
        
        ref_image = request.files['ref_image']
        driving_video = request.files['driving_video']
        
        # Check if files are selected
        if ref_image.filename == '':
            return jsonify({
                "status": "error",
                "message": "No reference image selected"
            }), 400
        
        if driving_video.filename == '':
            return jsonify({
                "status": "error",
                "message": "No driving video selected"
            }), 400
        
        # Validate file extensions
        if not allowed_file(ref_image.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                "status": "error",
                "message": f"Invalid image format. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            }), 400
        
        if not allowed_file(driving_video.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({
                "status": "error",
                "message": f"Invalid video format. Allowed formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            }), 400
        
        # Generate unique example_id
        example_id = generate_example_id()
        
        # Create directory for this example
        example_dir = os.path.join(EXAMPLES_DIR, example_id)
        os.makedirs(example_dir, exist_ok=True)
        
        # Save reference image
        ref_image_ext = ref_image.filename.rsplit('.', 1)[1].lower()
        ref_image_path = os.path.join(example_dir, f'ref_image.{ref_image_ext}')
        ref_image.save(ref_image_path)
        
        # Save driving video to temporary location first
        temp_video_path = os.path.join(example_dir, 'driving_temp.mp4')
        driving_video_path = os.path.join(example_dir, 'driving.mp4')
        driving_video.save(temp_video_path)
        
        # Process video (downsample/trim if needed)
        print(f"Processing driving video...")
        video_result = process_driving_video(temp_video_path, driving_video_path, max_frames=81, target_fps=16)
        
        # Remove temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        if video_result['status'] != 'success':
            # Clean up on failure
            if os.path.exists(example_dir):
                shutil.rmtree(example_dir)
            return jsonify({
                "status": "error",
                "message": video_result['message']
            }), 500
        
        print(f"Files uploaded successfully for example_id: {example_id}")
        print(f"  - Reference image: {ref_image_path}")
        print(f"  - Driving video: {driving_video_path}")
        print(f"  - Video processing: {video_result['message']}")
        
        return jsonify({
            "status": "success",
            "message": "Files uploaded successfully",
            "example_id": example_id,
            "video_info": {
                "total_frames": video_result.get('total_frames'),
                "fps": video_result.get('fps'),
                "processing": video_result.get('message')
            }
        }), 200
        
    except Exception as e:
        print(f"Error uploading files: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Upload failed: {str(e)}"
        }), 500


def initialize_models():
    """Initialize the models on server startup"""
    global model_nlf, detector
    
    print("Loading NLF model...")
    model_nlf = torch.jit.load(MODEL_PATH).cuda().eval()
    
    print("Initializing DWPose detector...")
    detector = DWposeDetector(use_batch=False).to(0)
    
    print("Models loaded successfully!")


# Initialize models when module is loaded (for Gunicorn workers)
def init_app():
    """Initialize application (called by Gunicorn on worker start)"""
    global MODEL_PATH, EXAMPLES_DIR
    
    # Read configuration from environment variables
    MODEL_PATH = os.environ.get('MODEL_PATH', 'pretrained_weights/nlf_l_multi_0.3.2.torchscript')
    EXAMPLES_DIR = os.environ.get('EXAMPLES_DIR', '../examples')
    
    # Create examples directory if it doesn't exist
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    print(f"Initializing worker (PID: {os.getpid()})...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Examples directory: {EXAMPLES_DIR}")
    
    # Initialize models
    initialize_models()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process pose API server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5999, help='Port to bind to')
    parser.add_argument('--model_path', type=str, default='pretrained_weights/nlf_l_multi_0.3.2.torchscript', 
                        help='Path to NLF model')
    parser.add_argument('--examples_dir', type=str, default='../examples',
                        help='Directory to store uploaded examples')
    args = parser.parse_args()
    
    MODEL_PATH = args.model_path
    EXAMPLES_DIR = args.examples_dir
    
    # Create examples directory if it doesn't exist
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    # Initialize models before starting the server
    initialize_models()
    
    # Start the Flask development server (NOT recommended for production)
    print(f"Starting development server on {args.host}:{args.port}")
    print(f"Examples directory: {EXAMPLES_DIR}")
    print("WARNING: For production use, please use Gunicorn instead:")
    print(f"  gunicorn -c gunicorn_config.py NLFPoseExtract.process_pose_api_server:app")
    app.run(host=args.host, port=args.port, debug=False, threaded=False, use_reloader=False)
