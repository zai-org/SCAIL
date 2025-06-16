import cv2
import numpy as np
from PIL import Image
import torch
from DWPoseProcess.AAUtils import read_frames_and_fps_as_np, save_videos_from_pil
from DWPoseProcess.checkUtils import *
import random
import shutil
import argparse
import yaml  # Add this import
import os
from tqdm import tqdm
from multiprocessing import Process, Queue
from pose_draw.draw_pose_main import draw_pose_to_canvas
import concurrent.futures


def process_video(mp4_path, keypoint_path, bbox_path, target_video_path, target_pose_video_path, target_ref_image_path, target_keypoints_path, wanted_fps=None, multi_person=False, draw_pose=False):
    poses = torch.load(keypoint_path)
    bboxes = torch.load(bbox_path)

    # filter_result_indexes = check_valid(poses, bboxes)
    frames, fps = read_frames_and_fps_as_np(mp4_path)

    initial_frame = frames[0]
    H, W, C = initial_frame.shape
    max_attempts = 60
    start_index = 0
    motion_part_len = 65
    ref_part_len = 15
    final_motion_indices = None
    applied_wanted_fps = wanted_fps

    while (motion_part_len + ref_part_len) / wanted_fps * fps > len(poses):
        ref_part_len -= 3
        if ref_part_len < 4:   # 还是太少，就降fps
            applied_wanted_fps = wanted_fps * 3 / 4 # 降一些提高产出率
            if (motion_part_len + ref_part_len) / applied_wanted_fps * fps > len(poses):  # 降了还是不行，就退出了，如果可以就跳出循环
                # print("filtered: 视频太短，无法满足采样要求，跳过")
                return
    
    num_frames = ref_part_len + motion_part_len 
    for attempt in range(max_attempts):
        end = int(start_index + num_frames / applied_wanted_fps * fps)
        if end >= len(frames):
            start_index -= 2    # 如果走太多就先后退一步
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
        ref_part_bboxes = [bboxes[index] for index in ref_part_indices]

        # 下面这四个筛选逻辑，除了bbox本身的，其他只对第0个bbox里的骨骼进行筛选
        motion_part_bbox_check_result = check_from_keypoints_bbox(motion_part_poses, motion_part_bboxes, IoU_thresthold=0.3, reference_width=W, reference_height=H, multi_person=multi_person)
        motion_part_core_check_result = check_from_keypoints_core_keypoints(motion_part_poses, motion_part_bboxes)
        ref_part_check_indices = get_valid_indice_from_keypoints(ref_part_poses, ref_part_indices)
        
        if len(ref_part_check_indices) > 0 and motion_part_bbox_check_result and motion_part_core_check_result:    # 这里只要满足正脸条件就可以，其它都留着
            if multi_person:
                final_ref_image_indice = select_ref_from_keypoints_bbox_multi(ref_part_indices, ref_part_bboxes, motion_part_bboxes)
                if final_ref_image_indice is None:
                    start_index += random.randint(3, 4)
                    continue
                delta_check_result = check_from_keypoints_stick_movement(motion_part_poses, angle_threshold=0.04)
                if not delta_check_result:
                    start_index += random.randint(3, 4)
                    continue
            else:
                final_ref_image_indice = random.choice(ref_part_check_indices)
                delta_check_result = check_from_keypoints_stick_movement(motion_part_poses, angle_threshold=0.04)
                if not delta_check_result:
                    start_index += random.randint(3, 4)
                    continue
            final_motion_indices = motion_part_indices
            break
        else:
            start_index += random.randint(3, 4)
            continue

    if final_motion_indices is None:
        return
    else:
        # 找到
        poses = [poses[i] for i in final_motion_indices]
        ref_frame = Image.fromarray(frames[final_ref_image_indice])
        frames = [Image.fromarray(frames[i]) for i in final_motion_indices]
        bboxes = [bboxes[i] for i in final_motion_indices]
        # save dwpose keypoints and selected frames / reference frame
        save_videos_from_pil(frames, target_video_path, wanted_fps)
        ref_frame.save(target_ref_image_path)
        torch.save(poses, target_keypoints_path)
        if draw_pose:
            pose_frames = draw_pose_to_canvas(poses, pool=None, H=H, W=W, reshape_scale=0, points_only_flag=False, show_feet_flag=False)
            save_videos_from_pil(pose_frames, target_pose_video_path, wanted_fps)
        return
    
def process_video_with_timeout(mp4_path, keypoint_path, bbox_path, target_video_path, target_pose_video_path, target_ref_image_path, target_keypoints_path, wanted_fps=None, multi_person=False, draw_pose=False):
    timeout_seconds = 150
    def task():
        process_video(mp4_path, keypoint_path, bbox_path, target_video_path, target_pose_video_path, target_ref_image_path, target_keypoints_path, wanted_fps, multi_person, draw_pose)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        try:
            future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            print(f"超时：处理视频 {mp4_path} 超过 {timeout_seconds} 秒，已跳过。")
        except Exception as e:
            print(f"处理视频 {mp4_path} 出错：{str(e)}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def worker(input_queue, directory, original_keypoints_dir, original_bboxes_dir, filtered_keypoints_dir, filtered_video_dir, filtered_pose_video_dir, ref_image_dir, draw_pose):
    while True:
        item = input_queue.get()
        if item == "STOP":
            break
        try:
            mp4_filename = item  # 如果只传递键名
            mp4_path = os.path.join(directory, mp4_filename)
            keypoint_path = os.path.join(original_keypoints_dir, mp4_filename.replace(".mp4", ".pt"))
            bbox_path = os.path.join(original_bboxes_dir, mp4_filename.replace(".mp4", ".pt"))
            target_video_path = os.path.join(filtered_video_dir, mp4_filename)
            target_pose_video_path = os.path.join(filtered_pose_video_dir, mp4_filename)
            target_ref_image_path = os.path.join(ref_image_dir, mp4_filename.replace(".mp4", ".jpg"))
            target_keypoints_path = os.path.join(filtered_keypoints_dir, mp4_filename.replace(".mp4", ".pt"))
            if os.path.exists(mp4_path) and os.path.exists(keypoint_path) and os.path.exists(bbox_path):
                process_video_with_timeout(mp4_path, keypoint_path, bbox_path, target_video_path, target_pose_video_path, target_ref_image_path, target_keypoints_path, wanted_fps=16, multi_person=multi_person, draw_pose=draw_pose)
        except Exception as e:
            print(f"Error processing {item}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video directories based on YAML config')
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    args = parser.parse_args()
    # Load configuration
    config = load_config(args.config)

    directories = config.get("directories")
    remove_last_flag = config.get("remove_last_flag", False)
    multi_person = config.get("multi_person", False)
    draw_pose = config.get("draw_pose", False)

    mp4_paths = []
    for directory in directories:
        original_keypoints_dir = directory + config.get("keypoint_suffix_name")
        original_bboxes_dir = directory + config.get("bbox_suffix_name")
        filtered_keypoints_dir = directory + config.get("target_keypoint_suffix_name")
        filtered_video_dir = directory + config.get("target_video_suffix_name")
        filtered_pose_video_dir = directory + config.get("target_pose_video_suffix_name")
        ref_image_dir = directory + config.get("target_ref_image_suffix_name")
        if remove_last_flag:
            # 删除 directory 中所有文件
            if os.path.exists(filtered_keypoints_dir):
                shutil.rmtree(filtered_keypoints_dir)
            if os.path.exists(filtered_video_dir):
                shutil.rmtree(filtered_video_dir)
            if os.path.exists(ref_image_dir):
                shutil.rmtree(ref_image_dir)
            if draw_pose:
                if os.path.exists(filtered_pose_video_dir):
                    shutil.rmtree(filtered_pose_video_dir)
            print(f"对{filtered_keypoints_dir}/{filtered_video_dir}/{ref_image_dir}已清除上次产生的")
        os.makedirs(filtered_keypoints_dir, exist_ok=True)
        os.makedirs(filtered_video_dir, exist_ok=True)
        os.makedirs(ref_image_dir, exist_ok=True)
        if draw_pose:
            os.makedirs(filtered_pose_video_dir, exist_ok=True)


        mp4_filenames = []
        for root, dirs, files in os.walk(original_keypoints_dir):
            for file in files:
                if file.lower().endswith('.pt'):  # 只查找 .mp4 文件
                    mp4_filenames.append(file.replace(".pt", ".mp4"))  # 获取绝对路径
        
        


    
        # 串行
        # for mp4_filename in tqdm(mp4_filenames, desc="Processing videos", unit="video"):
        #     mp4_path = os.path.join(directory, mp4_filename)
        #     keypoint_path = os.path.join(original_keypoints_dir, mp4_filename.replace(".mp4", ".pt"))
        #     bbox_path = os.path.join(original_bboxes_dir, mp4_filename.replace(".mp4", ".pt"))
        #     target_video_path = os.path.join(filtered_video_dir, mp4_filename)
        #     target_pose_video_path = os.path.join(filtered_pose_video_dir, mp4_filename)
        #     target_ref_image_path = os.path.join(ref_image_dir, mp4_filename.replace(".mp4", ".jpg"))
        #     target_keypoints_path = os.path.join(filtered_keypoints_dir, mp4_filename.replace(".mp4", ".pt"))
        #     if os.path.exists(mp4_path) and os.path.exists(keypoint_path) and os.path.exists(bbox_path):
        #         process_video_with_timeout(mp4_path, keypoint_path, bbox_path, target_video_path, target_pose_video_path, target_ref_image_path, target_keypoints_path, wanted_fps=16, multi_person=multi_person, draw_pose=draw_pose)

        # 并行
        max_tasks_buffer = 24
        task_queue = Queue(maxsize=max_tasks_buffer)
        workers = []
        for _ in range(max_tasks_buffer):
            p = Process(target=worker, args=(task_queue, directory, original_keypoints_dir, original_bboxes_dir, filtered_keypoints_dir, filtered_video_dir, filtered_pose_video_dir, ref_image_dir, draw_pose))
            p.start()
            workers.append(p)
        for mp4_filename in tqdm(mp4_filenames, desc="Processing videos"):
            task_queue.put(mp4_filename)  # 只放入key，worker可以实现阻塞，但是如果使用pdb调试则必须使用串行，因为worker不在主进程中
        
        # 添加停止信号
        for _ in range(max_tasks_buffer):
            task_queue.put("STOP")
        
        # 等待所有worker完成
        for p in workers:
            p.join()


    


