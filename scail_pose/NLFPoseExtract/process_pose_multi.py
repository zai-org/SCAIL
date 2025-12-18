import argparse
import os
import os.path as osp
import shutil
import numpy as np
import torch
import gc
import sys
import cv2
from tqdm import tqdm
from PIL import Image
import decord
from decord import VideoReader, cpu
try:
    import moviepy.editor as mpy
except ImportError:
    import moviepy as mpy
import copy
import glob

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.path.join(project_root, "sam2"))
from sam2.build_sam import build_sam2_video_predictor

from DWPoseProcess.dwpose import DWposeDetector
from NLFPoseExtract.extract_nlfpose_batch import process_video_multi_nlf
from NLFPoseExtract.nlf_render import render_multi_nlf_as_images

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
    decord.bridge.set_bridge("torch")
    # 用 decord 读取视频帧
    if video_input_path.endswith(".mp4"):
        vr = VideoReader(video_input_path)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', type=str, required=True, help='Path to the subdirectory containing GT.mp4')
    parser.add_argument('--model_path', type=str, default='pretrained_weights/nlf_l_multi_0.3.2.torchscript', 
                    help='Path to NLF model')
    parser.add_argument('--resolution', type=int, nargs=2, default=[512, 512], help='Resolution [H, W]')
    args = parser.parse_args()

    subdir = args.subdir
    model_path = args.model_path
    resolution = args.resolution

    video_input_path = osp.join(subdir, "driving.mp4")
    if not osp.exists(video_input_path):
        print(f"Error: {video_input_path} does not exist.")

    # 1. Extract DWpose and BBoxes
    print("Extracting DWpose and BBoxes...")
    detector = DWposeDetector(use_batch=False).to(0)
    
    vr = VideoReader(video_input_path)
    vr_frames = vr.get_batch(list(range(len(vr)))).asnumpy() # T H W C
    
    # Resize if needed? process_pose.py does resize. 
    # But run_samurai_mp4.py seems to use original video for samurai.
    # process_multinlf_after_samurai.py uses samurai output which is same resolution as input?
    # Let's stick to original resolution for extraction to match run_samurai_mp4 logic which uses original video.
    
    detector_return_list = []
    pil_frames = []
    for i in tqdm(range(len(vr_frames)), desc="Detecting poses"):
        pil_frame = Image.fromarray(vr_frames[i])
        pil_frames.append(pil_frame)
        detector_result = detector(pil_frame)
        detector_return_list.append(detector_result)
    
    poses, scores, det_results = zip(*detector_return_list)
    # poses is tuple of dicts, det_results is tuple of lists of bboxes
    
    # Save meta if needed, or just use in memory. 
    # run_samurai_mp4.py saves to meta/keypoints.pt and meta/bboxes.pt
    meta_dir = osp.join(subdir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    torch.save(poses, osp.join(meta_dir, "keypoints.pt"))
    torch.save(det_results, osp.join(meta_dir, "bboxes.pt"))
    
    # 2. Run Samurai Segmentation
    print("Running Samurai Segmentation...")
    samurai_output_root = osp.join(subdir, "samurai")
    if osp.exists(samurai_output_root):
        shutil.rmtree(samurai_output_root)
    os.makedirs(samurai_output_root, exist_ok=True)
    
    device = "cuda:0"
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
        # Create a copy to avoid modifying original if needed, though subset_for_person is from tensor
        subset_for_person_mod = subset_for_person.copy() 
        for k in range(len(subset_for_person_mod)):
            if k not in considered_points:
                subset_for_person_mod[k] = -1
        new_keypoints = keypoints_for_person[subset_for_person_mod != -1]
        final_keypoints_list.append(new_keypoints)
        
    get_samurai_crop_video(video_input_path, samurai_output_root, bboxes_0, final_keypoints_list, predictor=predictor)
    
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Render Multi NLF
    print("Rendering Multi NLF...")
    model_nlf = torch.jit.load(model_path).cuda().eval()
    decord.bridge.set_bridge("torch")
    
    vr_frames_list = []
    for samurai_mp4_path in sorted(glob.glob(osp.join(samurai_output_root, '*.mp4'))):
        vr_tmp = VideoReader(samurai_mp4_path)
        vr_frames_tmp = vr_tmp.get_batch(list(range(len(vr_tmp))))
        vr_frames_list.append(vr_frames_tmp)
        
    # Filter poses for rendering
    # change_poses_to_limit_num modifies poses in-place or returns new ones?
    # It returns poses, bboxes. And it modifies the lists passed to it?
    # poses is a tuple from zip, convert to list
    poses_list = list(poses)
    det_results_list = list(det_results)
    
    poses_list, det_results_list = change_poses_to_limit_num(poses_list, det_results_list)
    
    nlf_results = process_video_multi_nlf(model_nlf, vr_frames_list)
    frames_ori_np = render_multi_nlf_as_images(nlf_results, poses_list, reshape_pool=None)
    
    out_path = osp.join(subdir, 'rendered.mp4')
    mpy.ImageSequenceClip(frames_ori_np, fps=16).write_videofile(out_path)
    print(f"Done! Output saved to: {out_path}")

