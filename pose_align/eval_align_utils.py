import numpy as np
import argparse
import torch
import copy
import cv2
import os
from tqdm import tqdm


def warpAffine_kps(kps, M):
    a = M[:,:2]
    t = M[:,2]
    kps = np.dot(kps, a.T) + t
    return kps

def align_img(pose_ori, align_args_list, video_ratio):   #     video_ratio = W / H
    assert len(align_args_list) == len(pose_ori['bodies']['candidate']), f"num of bodies in align_args_list {len(align_args_list)} should be the same as the num of bodies in the following driving frames {len(pose_ori['bodies']['candidate'])}"
    body_pose_multi = copy.deepcopy(pose_ori['bodies']['candidate'])
    hands_multi = copy.deepcopy(pose_ori['hands'])
    faces_multi = copy.deepcopy(pose_ori['faces'])

    for body_idx, scales in enumerate(align_args_list):
        body_pose = body_pose_multi[body_idx]
        left_hand = hands_multi[2*body_idx+1]
        right_hand = hands_multi[2*body_idx]
        face = faces_multi[body_idx]

        '''
        计算逻辑:
        0. 该函数内进行绝对变换，始终保持人体中心点 body_pose[1] 不变
        1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
        2. 用点在图中的实际坐标来计算。
        3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
        4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
        注意：dwpose 输出是 (w, h)
        '''

        # h不变，w缩放到原比例  
        body_pose[:, 0]  = body_pose[:, 0] * video_ratio
        left_hand[:, 0] = left_hand[:, 0] * video_ratio
        right_hand[:, 0] = right_hand[:, 0] * video_ratio
        face[:, 0] = face[:, 0] * video_ratio

        # scales of 10 body parts 
        scale_neck      = scales["scale_neck"] 
        scale_face      = scales["scale_face"]
        scale_shoulder  = scales["scale_shoulder"]
        scale_arm_upper = scales["scale_arm_upper"]
        scale_arm_lower = scales["scale_arm_lower"]
        scale_hand      = scales["scale_hand"]
        scale_body_len  = scales["scale_body_len"]
        scale_leg_upper = scales["scale_leg_upper"]
        scale_leg_lower = scales["scale_leg_lower"]
        scale_real_face = scales["scale_real_face"]

        scale_sum = 0
        count = 0
        # TODO: 防止inf的逻辑
        scale_list = [scale_neck, scale_face, scale_shoulder, scale_arm_upper, scale_arm_lower, scale_hand, scale_body_len, scale_leg_upper, scale_leg_lower]
        for i in range(len(scale_list)):
            if not np.isinf(scale_list[i]):
                scale_sum = scale_sum + scale_list[i]
                count = count + 1
        for i in range(len(scale_list)):
            if np.isinf(scale_list[i]):   
                scale_list[i] = scale_sum/count

        # offsets of each part 
        offset = dict()
        offset["14_15_16_17_to_0"] = body_pose[[14,15,16,17], :] - body_pose[[0], :] 
        offset["3_to_2"] = body_pose[[3], :] - body_pose[[2], :] 
        offset["4_to_3"] = body_pose[[4], :] - body_pose[[3], :] 
        offset["6_to_5"] = body_pose[[6], :] - body_pose[[5], :] 
        offset["7_to_6"] = body_pose[[7], :] - body_pose[[6], :] 
        offset["9_to_8"] = body_pose[[9], :] - body_pose[[8], :] 
        offset["10_to_9"] = body_pose[[10], :] - body_pose[[9], :] 
        offset["12_to_11"] = body_pose[[12], :] - body_pose[[11], :] 
        offset["13_to_12"] = body_pose[[13], :] - body_pose[[12], :] 
        offset["hand_left_to_4"] = left_hand - body_pose[[4], :]
        offset["hand_right_to_7"] = right_hand - body_pose[[7], :]
        offset["nose_to_neck"] = face[33] - body_pose[[0], :]
        offset["face_to_nose"] = face - face[33]

        # neck
        c_ = body_pose[1]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_neck)
        neck = body_pose[[0], :]    # 先得到一个正确位置的人中
        neck = warpAffine_kps(neck, M)
        body_pose[[0], :] = neck

        # up_shoulder
        c_ = body_pose[0]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face)
        body_pose_up_shoulder = offset["14_15_16_17_to_0"] + body_pose[[0], :]
        body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
        body_pose[[14,15,16,17], :] = body_pose_up_shoulder

        # face
        nose = offset["nose_to_neck"] + body_pose[[0], :]
        nose = warpAffine_kps(nose, M)
        face[33] = nose
        if np.any(face != -1):  # 检查是否有有效的面部关键点
            c_ = face[33]
            cx = c_[0]
            cy = c_[1]
            M = cv2.getRotationMatrix2D((cx,cy), 0, scale_real_face)
            if not np.all(face == -1):
                face_transformed = offset["face_to_nose"] + nose
                face_transformed = warpAffine_kps(face_transformed, M)
                face[:] = face_transformed

        # shoulder 
        c_ = body_pose[1]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_shoulder)
        body_pose_shoulder = body_pose[[2,5], :] 
        body_pose_shoulder = warpAffine_kps(body_pose_shoulder, M) 
        body_pose[[2,5], :] = body_pose_shoulder

        # arm upper left
        c_ = body_pose[2]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
        elbow = offset["3_to_2"] + body_pose[[2], :]
        elbow = warpAffine_kps(elbow, M)
        body_pose[[3], :] = elbow

        # arm lower left
        c_ = body_pose[3]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
        wrist = offset["4_to_3"] + body_pose[[3], :]
        wrist = warpAffine_kps(wrist, M)
        body_pose[[4], :] = wrist

        # hand left
        c_ = body_pose[4]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
        left_hand_transformed = offset["hand_left_to_4"] + body_pose[[4], :]
        left_hand_transformed = warpAffine_kps(left_hand_transformed, M)
        left_hand[:] = left_hand_transformed

        # arm upper right
        c_ = body_pose[5]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
    
        elbow = offset["6_to_5"] + body_pose[[5], :]
        elbow = warpAffine_kps(elbow, M)
        body_pose[[6], :] = elbow

        # arm lower right
        c_ = body_pose[6]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
    
        wrist = offset["7_to_6"] + body_pose[[6], :]
        wrist = warpAffine_kps(wrist, M)
        body_pose[[7], :] = wrist

        # hand right
        c_ = body_pose[7]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
        right_hand_transformed = offset["hand_right_to_7"] + body_pose[[7], :]
        right_hand_transformed = warpAffine_kps(right_hand_transformed, M)
        right_hand[:] = right_hand_transformed

        # body len
        c_ = body_pose[1]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_body_len)
        body_len = body_pose[[8,11], :] 
        body_len = warpAffine_kps(body_len, M)
        body_pose[[8,11], :] = body_len

        # leg upper left
        c_ = body_pose[8]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
        knee = offset["9_to_8"] + body_pose[[8], :]
        knee = warpAffine_kps(knee, M)
        body_pose[[9], :] = knee

        # leg lower left
        c_ = body_pose[9]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
        ankle = offset["10_to_9"] + body_pose[[9], :]
        ankle = warpAffine_kps(ankle, M)
        body_pose[[10], :] = ankle

        # leg upper right
        c_ = body_pose[11]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
        knee = offset["12_to_11"] + body_pose[[11], :]
        knee = warpAffine_kps(knee, M)
        body_pose[[12], :] = knee

        # leg lower right
        c_ = body_pose[12]
        cx = c_[0]
        cy = c_[1]
        M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
        ankle = offset["13_to_12"] + body_pose[[12], :]
        ankle = warpAffine_kps(ankle, M)
        body_pose[[13], :] = ankle

        # none part
        body_pose_none = pose_ori['bodies']['candidate'][body_idx]
        left_hand_none = pose_ori['hands'][2*body_idx+1]
        right_hand_none = pose_ori['hands'][2*body_idx]
        face_none = pose_ori['faces'][body_idx]

        body_pose_none = body_pose_none == -1.
        left_hand_none = left_hand_none == -1.
        right_hand_none = right_hand_none == -1.
        face_none = face_none == -1.

        body_pose[body_pose_none] = -1.
        left_hand[left_hand_none] = -1. 
        right_hand[right_hand_none] = -1. 
        face[face_none] = -1.

    # last check nan -> -1.
    body_pose_multi = np.nan_to_num(body_pose_multi, nan=-1.)
    hands_multi = np.nan_to_num(hands_multi, nan=-1.)
    faces_multi = np.nan_to_num(faces_multi, nan=-1.)

    # return
    pose_align = copy.deepcopy(pose_ori)
    pose_align['bodies']['candidate'] = body_pose_multi
    pose_align['hands'] = hands_multi
    pose_align['faces'] = faces_multi

    return pose_align

def get_align_args_and_offset(body_ref_img, body_1st_img, hands_ref_img, faces_ref_img, hands_1st_img, faces_1st_img):
    align_args = dict()
    dist_1st_img = np.linalg.norm(body_1st_img[0]-body_1st_img[1])   # 0.078   
    dist_ref_img = np.linalg.norm(body_ref_img[0]-body_ref_img[1])   # 0.106
    align_args["scale_neck"] = dist_ref_img / dist_1st_img  # align / pose = ref / 1st

    dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[17])
    dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[17])
    align_args["scale_face"] = dist_ref_img / dist_1st_img

    dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[5])  # 0.112
    dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[5])  # 0.174
    align_args["scale_shoulder"] = dist_ref_img / dist_1st_img  

    dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[3])  # 0.895
    dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[3])  # 0.134
    s1 = dist_ref_img / dist_1st_img
    dist_1st_img = np.linalg.norm(body_1st_img[5]-body_1st_img[6])
    dist_ref_img = np.linalg.norm(body_ref_img[5]-body_ref_img[6])
    s2 = dist_ref_img / dist_1st_img
    align_args["scale_arm_upper"] = (s1+s2)/2 # 1.548

    dist_1st_img = np.linalg.norm(body_1st_img[3]-body_1st_img[4])
    dist_ref_img = np.linalg.norm(body_ref_img[3]-body_ref_img[4])
    s1 = dist_ref_img / dist_1st_img
    dist_1st_img = np.linalg.norm(body_1st_img[6]-body_1st_img[7])
    dist_ref_img = np.linalg.norm(body_ref_img[6]-body_ref_img[7])
    s2 = dist_ref_img / dist_1st_img
    align_args["scale_arm_lower"] = (s1+s2)/2


    # nose
    dict_1st_img = np.linalg.norm(faces_1st_img[0][33]-body_1st_img[0])
    dict_ref_img = np.linalg.norm(faces_ref_img[0][33]-body_ref_img[0])
    align_args["scale_nose"] = dist_ref_img / dist_1st_img

    # face
    dict_1st_img = np.zeros(68)
    dict_ref_img = np.zeros(68)

    for i in range(68):
        dict_1st_img[i] = np.linalg.norm(faces_1st_img[0][i]-faces_1st_img[0][33])
        dict_ref_img[i] = np.linalg.norm(faces_ref_img[0][i]-faces_ref_img[0][33])
    
    ratio = 0   
    count = 0
    for i in range (68): 
        if dict_1st_img[i] != 0:
            ratio = ratio + dict_ref_img[i]/dict_1st_img[i]
            count = count + 1
    if count!=0:
        align_args["scale_real_face"] = ratio / count
    else:
        align_args["scale_real_face"] = 1

    # hand
    dist_1st_img = np.zeros(10)
    dist_ref_img = np.zeros(10)      
        
    dist_1st_img[0] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,1])
    dist_1st_img[1] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,5])
    dist_1st_img[2] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,9])
    dist_1st_img[3] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,13])
    dist_1st_img[4] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,17])
    dist_1st_img[5] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,1])
    dist_1st_img[6] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,5])
    dist_1st_img[7] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,9])
    dist_1st_img[8] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,13])
    dist_1st_img[9] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,17])

    dist_ref_img[0] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,1])
    dist_ref_img[1] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,5])
    dist_ref_img[2] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,9])
    dist_ref_img[3] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,13])
    dist_ref_img[4] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,17])
    dist_ref_img[5] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,1])
    dist_ref_img[6] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,5])
    dist_ref_img[7] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,9])
    dist_ref_img[8] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,13])
    dist_ref_img[9] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,17])

    ratio = 0   
    count = 0
    for i in range (10): 
        if dist_1st_img[i] != 0:
            ratio = ratio + dist_ref_img[i]/dist_1st_img[i]
            count = count + 1
    if count!=0:
        align_args["scale_hand"] = (ratio/count+align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/3
    else:
        align_args["scale_hand"] = (align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/2

    # body 
    dist_1st_img = np.linalg.norm(body_1st_img[1] - (body_1st_img[8] + body_1st_img[11])/2 )
    dist_ref_img = np.linalg.norm(body_ref_img[1] - (body_ref_img[8] + body_ref_img[11])/2 )
    align_args["scale_body_len"]=dist_ref_img / dist_1st_img

    dist_1st_img = np.linalg.norm(body_1st_img[8]-body_1st_img[9])
    dist_ref_img = np.linalg.norm(body_ref_img[8]-body_ref_img[9])
    s1 = dist_ref_img / dist_1st_img
    dist_1st_img = np.linalg.norm(body_1st_img[11]-body_1st_img[12])
    dist_ref_img = np.linalg.norm(body_ref_img[11]-body_ref_img[12])
    s2 = dist_ref_img / dist_1st_img
    align_args["scale_leg_upper"] = (s1+s2)/2

    dist_1st_img = np.linalg.norm(body_1st_img[9]-body_1st_img[10])
    dist_ref_img = np.linalg.norm(body_ref_img[9]-body_ref_img[10])
    s1 = dist_ref_img / dist_1st_img
    dist_1st_img = np.linalg.norm(body_1st_img[12]-body_1st_img[13])
    dist_ref_img = np.linalg.norm(body_ref_img[12]-body_ref_img[13])
    s2 = dist_ref_img / dist_1st_img
    align_args["scale_leg_lower"] = (s1+s2)/2

    ####################
    ####################
    # need adjust nan
    for k,v in align_args.items():
        if np.isnan(v):
            align_args[k]=1

    # centre offset (the offset of key point 1)
    offset = body_ref_img[1] - body_1st_img[1]
    for k in align_args:
        align_args[k] = np.clip(align_args[k], 0.5, 1.5)
    return align_args, offset




def run_align_video(vid_height, vid_width, ref_height, ref_width, pose_refer, pose_video):
    # 需要先把 image 和 video 缩放到 H 一致
    body_ref_img_multi  = pose_refer['bodies']['candidate']
    hands_ref_img_multi = pose_refer['hands']
    faces_ref_img_multi = pose_refer['faces']

    pose_list= []

    pose_1st_img_multi =  copy.deepcopy(pose_video)[0]
    body_1st_img_multi = pose_1st_img_multi['bodies']['candidate']
    hands_1st_img_multi = pose_1st_img_multi['hands']
    faces_1st_img_multi = pose_1st_img_multi['faces']

    align_args_list = [dict() for _ in range(len(body_ref_img_multi))]
    offset_list = [[0, 0] for _ in range(len(body_ref_img_multi))]
    assert len(body_ref_img_multi) == len(body_1st_img_multi), "person num in ref and driving video should be the same"

    for body_idx in range(len(body_ref_img_multi)):
        body_ref_img = body_ref_img_multi[body_idx]
        body_1st_img = body_1st_img_multi[body_idx]
        hands_ref_img = hands_ref_img_multi[2*body_idx: 2*body_idx+2]
        faces_ref_img = faces_ref_img_multi[body_idx: body_idx+1]
        hands_1st_img = hands_1st_img_multi[2*body_idx: 2*body_idx+2]
        faces_1st_img = faces_1st_img_multi[body_idx: body_idx+1]

        '''
        计算逻辑:
        1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
        2. 用点在图中的实际坐标来计算。
        3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
        4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
        注意：dwpose 输出是 (w, h)
        '''
        
        # h不变，w缩放到原比例
        ref_ratio = ref_width / ref_height
        body_ref_img[:, 0]  = body_ref_img[:, 0] * ref_ratio
        hands_ref_img[:, :, 0] = hands_ref_img[:, :, 0] * ref_ratio
        faces_ref_img[:, :, 0] = faces_ref_img[:, :, 0] * ref_ratio

        video_ratio = vid_width / vid_height
        body_1st_img[:, 0]  = body_1st_img[:, 0] * video_ratio
        hands_1st_img[:, :, 0] = hands_1st_img[:, :, 0] * video_ratio
        faces_1st_img[:, :, 0] = faces_1st_img[:, :, 0] * video_ratio

        align_args, offset = get_align_args_and_offset(body_ref_img, body_1st_img, hands_ref_img, faces_ref_img, hands_1st_img, faces_1st_img)
        align_args_list[body_idx] = align_args
        offset_list[body_idx] = offset
        
        

    for i in range(len(pose_video)):
        # estimate scale parameters by the 1st frame in the video
        # pose align
        pose_ori = pose_video[i]
        pose_align = align_img(pose_ori, align_args_list, video_ratio)

        # 这里也要改
        # add centre offset
        pose = copy.deepcopy(pose_align)
        for body_idx in range(len(body_ref_img_multi)):
            offset = offset_list[body_idx]
            pose['bodies']['candidate'][body_idx] = pose['bodies']['candidate'][body_idx] + offset
            pose['hands'][2*body_idx: 2*body_idx+2] = pose['hands'][2*body_idx: 2*body_idx+2] + offset
            pose['faces'][body_idx] = pose['faces'][body_idx] + offset

        # h不变，w从绝对坐标缩放回0-1 注意这里要回到ref的坐标系
        pose['bodies']['candidate'][:, :, 0] = pose['bodies']['candidate'][:, :, 0] / ref_ratio
        pose['hands'][:, :, 0] = pose['hands'][:, :, 0] / ref_ratio
        pose['faces'][:, :, 0] = pose['faces'][:, :, 0] / ref_ratio
        pose_list.append(pose)

    return pose_list