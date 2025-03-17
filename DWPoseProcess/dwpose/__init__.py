# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self, use_batch=False):
        self.use_batch = use_batch
        pass

    def to(self, device):
        self.pose_estimation = Wholebody(device, self.use_batch)
        return self

    # def cal_height(self, input_image):
    #     input_image = cv2.cvtColor(
    #         np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
    #     )

    #     input_image = HWC3(input_image)
    #     H, W, C = input_image.shape
    #     with torch.no_grad():
    #         candidate, subset = self.pose_estimation(input_image)
    #         nums, keys, locs = candidate.shape
    #         # candidate[..., 0] /= float(W)
    #         # candidate[..., 1] /= float(H)
    #         body = candidate
    #     return body[0, ..., 1].min(), body[..., 1].max() - body[..., 1].min()

    def _get_result_from_est(self, input_image, candidate, subset, det_result, image_resolution, output_type, H, W):
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        score = subset[:, :18]              # 前18个是躯干骨骼  score(n, 18)
        max_ind = np.mean(score, axis=-1).argmax(axis=0)    # 返回分数最高的锚框对应的骨骼
        score = score[[max_ind]]
        body = candidate[:, :18].copy()
        body = body[[max_ind]]
        nums = 1
        body = body.reshape(nums * 18, locs)    # Moore-AA只有一个人体, 0-18表示body
        body_score = copy.deepcopy(score)       # 已经去过max_ind
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18 * i + j)
                else:
                    score[i][j] = -1    # 躯干中去除掉不可见的骨骼

        un_visible = subset < 0.3       
        candidate[un_visible] = -1      # 全部关键点中去掉不可见骨骼

        foot = candidate[:, 18:24]

        faces = candidate[[max_ind], 24:92]

        hands = candidate[[max_ind], 92:113]
        hands = np.vstack([hands, candidate[[max_ind], 113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        detected_map = draw_pose(pose, H, W)
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_LINEAR
        )

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map, pose, body_score, det_result     # body_score是原始的躯干骨骼分数

    def __call__(
        self,
        input,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        **kwargs,
    ):                          # 默认是会做下采样的
        if not self.use_batch:
            # 这边改成 batch的
            input = cv2.cvtColor(
                np.array(input, dtype=np.uint8), cv2.COLOR_RGB2BGR
            )

            input = HWC3(input)
            input = resize_image(input, detect_resolution)
            H, W, C = input.shape


            with torch.no_grad():
                candidate, subset, det_result = self.pose_estimation(input)   # candidate (n, 134, 2) 候选点 / subset (n, 134) 得分
                return self._get_result_from_est(input, candidate, subset, det_result, image_resolution, output_type, H, W)
        else:
            input_batch_images = input
            # breakpoint()
            batch_input = [resize_image(HWC3(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)), detect_resolution) for img in input_batch_images]
            H, W, C = batch_input[0].shape
            with torch.no_grad():
                candidates, subsets, det_results = zip(*self.pose_estimation(batch_input))   # candidate (n, 134, 2) 候选点 / subset (n, 134) 得分
                candidates = list(candidates)
                # print("subsets")
                # subsets = list(subsets)
                # for subset in subsets:
                #     print(subset.shape)   # 这里subset里可能没检测到骨骼 subsets为空 导致是(0,134) 其实是取索引取到0了
                    
                det_results = list(det_results)
                return [self._get_result_from_est(input, candidate, subset, det_result, image_resolution, output_type, H, W) 
                        for input, candidate, subset, det_result in zip(batch_input, candidates, subsets, det_results)]





            

