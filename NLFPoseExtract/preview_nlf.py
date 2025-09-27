import os
import sys
import cv2
import torch
import pickle
import torchvision
import shutil
import glob
from tqdm import tqdm   
import decord
from decord import VideoReader, cpu, gpu
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
from NLFPoseExtract.nlf_draw import preview_nlf_as_images
from DWPoseProcess.AAUtils import save_videos_from_pil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from DWPoseProcess.extract_nlfpose import process_video_nlf


def concat_images(images, direction='horizontal', pad=0, pad_value=0):
    if len(images) == 1:
        return images[0]
    is_pil = isinstance(images[0], Image.Image)
    if is_pil:
        images = [np.array(image) for image in images]
    if direction == 'horizontal':
        height = max([image.shape[0] for image in images])
        width = sum([image.shape[1] for image in images]) + pad * (len(images) - 1)
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[1]
            new_image[: image.shape[0], begin:end] = image
            begin = end + pad
    elif direction == 'vertical':
        height = sum([image.shape[0] for image in images]) + pad * (len(images) - 1)
        width = max([image.shape[1] for image in images])
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[0]
            new_image[begin:end, : image.shape[1]] = image
            begin = end + pad
    else:
        assert False
    if is_pil:
        new_image = Image.fromarray(new_image)
    return new_image


def process_video(file, input_dir, output_dir):
    if file.lower().endswith(('.pkl')):
        key = file.replace('.pkl', '')
        input_pkl_path = os.path.join(input_dir, file)
        output_mp4_path = os.path.join(output_dir, key + '.mp4')
        output_data = pickle.load(open(input_pkl_path, 'rb'))
        output_preview_images = preview_nlf_as_images(output_data)
        save_videos_from_pil(output_preview_images, output_mp4_path, fps=16)

def video_worker(task_queue):
    while True:
        item = task_queue.get()
        if item == "STOP":
            break
        try:
            process_video(item['file'], item['input_dir'], item['output_dir'])
        except Exception as e:
            # trace
            traceback.print_exc()
            print(f"Error processing file: {e}")
    

if __name__ == '__main__':
    root_dir = "/workspace/ywh_data/DataProcessNew"
    for dir in os.listdir(root_dir):
        print("Processing dir: ", dir)
        input_dir = os.path.join(root_dir, dir, "smpl")
        if not os.path.exists(input_dir):
            continue
        output_dir = os.path.join(root_dir, dir, "smpl_render")
        os.makedirs(output_dir, exist_ok=True)

        # 串行
        # for file in tqdm(os.listdir(input_dir)):
        #     process_video(file, input_dir, output_dir)

