import decord
from decord import VideoReader
import numpy as np
import torch
import os
import cv2
from pathlib import Path
from tqdm import tqdm



def load_multiple_videos(video_data, pose_data_A, pose_data_B):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    vr_pose_A = VideoReader(uri=pose_data_A, height=-1, width=-1)
    vr_pose_B = VideoReader(uri=pose_data_B, height=-1, width=-1)

    indices = np.arange(0, len(vr))
    temp_frms = vr.get_batch(indices)
    temp_frms_pose_A = vr_pose_A.get_batch(indices)
    temp_frms_pose_B = vr_pose_B.get_batch(indices)
    
    return temp_frms, temp_frms_pose_A, temp_frms_pose_B


def create_grid(video0, video1, video2, dst_dir=None):
    # Load videos
    try:
        frames0, frames1, frames2 = load_multiple_videos(video0, video1, video2)
        
        # Resize frames1 and frames2 to match frames0's shape
        if frames1.shape[1:] != frames0.shape[1:]:
            frames1 = torch.nn.functional.interpolate(frames1.permute(0,3,1,2), 
                                                    size=(frames0.shape[1], frames0.shape[2])).permute(0,2,3,1)
    except Exception as e:
        print(f"Error loading video frames: {str(e)}")
        return
    
    # Convert to numpy arrays
    frames0 = frames0.numpy()
    frames1 = frames1.numpy()
    frames2 = frames2.numpy()
    
    # Create output directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)
    
    # Process each frame
    for i in range(min(len(frames1), len(frames2))):
        frame0 = frames0[i]
        frame1 = frames1[i]
        frame2 = frames2[i]
        
        # Create overlay
        overlay = frame1 + frame2*0.5
        
        if frame0.shape[0] > frame0.shape[1]:
            grid = np.hstack((frame1, frame2, overlay, frame0+frame1, frame0+frame2))
        else:
            grid = np.vstack((frame1, frame2, overlay, frame0+frame1, frame0+frame2))
        
        # Save frame
        output_path = os.path.join(dst_dir, f"{Path(video1).stem}_frame_{i:04d}.png")
        cv2.imwrite(output_path, grid)

def main():
    directory_V = "/workspace/ywh_data/DataProcess/bilibili_dance_shuping_0929/videos_filtered"
    directory_A = "/workspace/ywh_data/DataProcess/bilibili_dance_shuping_0929/videos_dwpose_filtered"
    directory_B = "/workspace/ys_data/filtered_data/3dpose_paired_data/skeleton/MP4_bilibili_dance_shuping_0929/videos_3dpose"
    dst_dir = "/workspace/ywh_data/DataProcess/poseLap"

    # Get list of video files from both directories
    videos_V =  [f for f in os.listdir(directory_V) if f.endswith('.mp4')]
    videos_A = [f for f in os.listdir(directory_A) if f.endswith('.mp4')]
    videos_B = [f for f in os.listdir(directory_B) if f.endswith('.mp4')]
    
    # Find matching video names
    matching_videos = set(videos_V) & set(videos_A) & set(videos_B)
    
    # Process each matching video
    for video_name in tqdm(matching_videos):
        video_path_V = os.path.join(directory_V, video_name)
        video_path_A = os.path.join(directory_A, video_name)
        video_path_B = os.path.join(directory_B, video_name)
        
        print(f"Processing {video_name}...")
        create_grid(video_path_V, video_path_A, video_path_B, dst_dir)

if __name__ == "__main__":
    main()

