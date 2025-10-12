# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import numpy as np
import shutil
import torch
import cv2
from PIL import Image
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy
import decord
from decord import VideoReader

from VITPoseExtract.pose2d import Pose2d
from VITPoseExtract.pose2d_utils import AAPoseMeta
from VITPoseExtract.human_visualization import draw_aapose_by_meta_new

def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[1:] * (w, h)

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]



def get_cond_images(first_frame_np, tpl_pose_metas, only_cheek=True, draw_hand=True, dw_bgr=False):
        cond_images = []

        for idx, meta_list in enumerate(tpl_pose_metas):
            tpl_pose_metas_list = [AAPoseMeta.from_humanapi_meta(meta) for meta in meta_list]
            canvas = np.zeros_like(first_frame_np)
            for _, meta in enumerate(tpl_pose_metas_list):
                canvas = draw_aapose_by_meta_new(canvas, meta, only_cheek=only_cheek, draw_hand=draw_hand)
                if dw_bgr:
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cond_images.append(canvas)
        return cond_images


class VITPosePipeline():
    def __init__(self, det_checkpoint_path, pose2d_checkpoint_path):
        self.pose2d = Pose2d(checkpoint=pose2d_checkpoint_path, detector_checkpoint=det_checkpoint_path)

    def __call__(self, frames_np):
        tpl_pose_metas = self.pose2d(frames_np)

        return tpl_pose_metas
    

    def get_mask(self, frames, th_step, kp2ds_all):
        frame_num = len(frames)
        if frame_num < th_step:
            num_step = 1
        else:
            num_step = (frame_num + th_step) // th_step

        all_mask = []
        for index in range(num_step):
            each_frames = frames[index * th_step:(index + 1) * th_step]
    
            kp2ds = kp2ds_all[index * th_step:(index + 1) * th_step]
            if len(each_frames) > 4:
                key_frame_num = 4
            elif 4 >= len(each_frames) > 0:
                key_frame_num = 1
            else:
                continue

            key_frame_step = len(kp2ds) // key_frame_num
            key_frame_index_list = list(range(0, len(kp2ds), key_frame_step))

            key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]
            key_frame_body_points_list = []
            for key_frame_index in key_frame_index_list:
                keypoints_body_list = []
                body_key_points = kp2ds[key_frame_index]['keypoints_body']
                for each_index in key_points_index:
                    each_keypoint = body_key_points[each_index]
                    if None is each_keypoint:
                        continue
                    keypoints_body_list.append(each_keypoint)

                keypoints_body = np.array(keypoints_body_list)[:, :2]
                wh = np.array([[kp2ds[0]['width'], kp2ds[0]['height']]])
                points = (keypoints_body * wh).astype(np.int32)
                key_frame_body_points_list.append(points)

            inference_state = self.predictor.init_state_v2(frames=each_frames)
            self.predictor.reset_state(inference_state)
            ann_obj_id = 1
            for ann_frame_idx, points in zip(key_frame_index_list, key_frame_body_points_list):
                labels = np.array([1] * points.shape[0], np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            for out_frame_idx in range(len(video_segments)):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    out_mask = out_mask[0].astype(np.uint8)
                    all_mask.append(out_mask)

        return all_mask
    
    def convert_list_to_array(self, metas):
        metas_list = []
        for meta in metas:
            for key, value in meta.items():
                if type(value) is list:
                    value = np.array(value)
                meta[key] = value
            metas_list.append(meta)
        return metas_list

