import numpy as np
from collections import deque
import numpy as np
from collections import deque
import math

def get_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def check_consistant(boxA, boxB, scoreA_lst, scoreB_lst, beta, all_threshold):
    """
    计算两个锚框之间的 IoU（交并比）以及分数的变化比例，来判断连续性
    """
    # 计算交集框的坐标
    scoreA = scoreA_lst[0]
    scoreB = scoreB_lst[0]
    iou = get_IoU(boxA, boxB)

    reduction_ratio = (scoreA - scoreB) / (scoreA + scoreB) if scoreA > scoreB else 0  # 如果分数减少的很多，就更不连续

    return iou - reduction_ratio * beta > all_threshold


def get_IoU(boxA, boxB):
    """
    计算两个锚框之间的 IoU（交并比）以及分数的变化比例，来判断连续性
    """
    # 计算交集框的坐标
    x1_int = max(boxA[0], boxB[0])
    y1_int = max(boxA[1], boxB[1])
    x2_int = min(boxA[2], boxB[2])
    y2_int = min(boxA[3], boxB[3])

    # 计算交集的面积
    inter_width = max(0, x2_int - x1_int)
    inter_height = max(0, y2_int - y1_int)
    inter_area = inter_width * inter_height

    # 计算两个锚框的面积
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集的面积
    union_area = areaA + areaB - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0.0

    return iou

def check_bbox_single_for_video(bbox, reference_width, reference_height, min_bbox_width=1/6, min_bbox_area=1/30):
    """
    每一帧都要检查，判断 bbox 是否符合视频格式下的要求
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # 防止非法 bbox
    if bbox_width <= 0 or bbox_height <= 0:
        return False

    bbox_area = bbox_width * bbox_height

    if bbox_area < min_bbox_area:
        # print("filtered: bbox too small or too large")
        return False

    # 横屏额外筛
    if reference_width > reference_height:
        if bbox_width < min_bbox_width:
            # print("filtered: bbox too wide")
            return False
    return True



##############################判断点是否满足################################

def part5_valid(valid_joints):
    """
    判断身体五块是不是都有点
    
    参数:
    valid_joints:布尔数组
    
    返回:
    bool: 如果满足要求返回True，否则返回False。
    """

    ###  认为以下的视频是满足我们采样需要的：
    ###  A只有上半身手部动作的：手部有可能在挥舞过程中移出屏幕，但是上半身应该一直在屏幕内，此时1-0, 1-2, 1-5这三条骨骼应该都存在；14-17应该至少有一个点存在
    ###  B全身动作：1-0, 1-2, 1-5, 1-8, 1-11这五条骨骼应该都存在
    
    top_core_joints = valid_joints[[0,1,2,5]]
    top_nece_joints = valid_joints[[14,15,16,17]]
    if all(top_core_joints) and any(top_nece_joints):
        return True
    
    wholebody_core_joints = valid_joints[[0,1,2,5,8,11]]
    if all(wholebody_core_joints):
        return True
    return False
    
    
def check_valid_sequence(valid_keypoints, threshold=0.3):
    valid_joints = np.zeros(24)
    for valid_keypoint in valid_keypoints:
        valid_joints += valid_keypoint
    return part5_valid(valid_joints)


def get_valid_indice_from_keypoints(ref_part_poses, ref_part_indices):
    # ref_part_poses: poses序列，ref_part_indices poses序列里面每个值对应的在整个序列里的index
    # return: 每个pose序列里面，满足要求的pose的index
    valid_indice = []
    for i, (keypoint_all, indice) in enumerate(zip(ref_part_poses, ref_part_indices)):
        body_subset = keypoint_all["bodies"]["subset"][0]
        valid_joints = body_subset > -1 # 得到一个布尔索引

        if not part5_valid(valid_joints):
            continue

        faces = keypoint_all["faces"][0]    

        left_eye = faces[36:42]  # 左眼关键点   5个关键点有4个认为有左眼
        right_eye = faces[42:48]  # 右眼关键点  5个关键点有4个认为有右眼
        nose = faces[27:36]  # 鼻子关键点   8个关键点有5个认为有鼻子
        mouth = faces[48:68]  # 嘴巴关键点  21个关键点有15个认为有嘴巴

        # 计算每个部位有效的关键点数
        left_eye_valid = sum(1 for point in left_eye if point[0] > 0 and point[1] > 0)
        right_eye_valid = sum(1 for point in right_eye if point[0] > 0 and point[1] > 0)
        nose_valid = sum(1 for point in nose if point[0] > 0 and point[1] > 0)
        mouth_valid = sum(1 for point in mouth if point[0] > 0 and point[1] > 0)

        # 如果有两个或以上部位有效，则认为是正脸
        valid_face_parts = 0
        if left_eye_valid >= 4:
            valid_face_parts += 1
        if right_eye_valid >= 4:
            valid_face_parts += 1
        if nose_valid >= 5:
            valid_face_parts += 1
        if mouth_valid >= 15:
            valid_face_parts += 1

        if valid_face_parts >= 2:
            valid_indice.append(int(indice))
    
    return valid_indice
    
def check_from_keypoints_core_keypoints(keypoints, bboxs):
    # 用于keypoints版本，根据每一帧的18个keypoints和bbox iou来判断是否满足要求
    valid_sequence = deque(maxlen=4)
    for i, (keypoint_all, bbox_all) in enumerate(zip(keypoints, bboxs)):
        body_subset = keypoint_all["bodies"]["subset"][0]
        valid_joints = body_subset > -1 # 得到一个布尔索引
        if len(valid_sequence) == 4:
            if not check_valid_sequence(valid_sequence):
                # print("filtered: 骨骼不满足要求")
                return False   # 关键点异常
        valid_sequence.append(valid_joints)
    return True

def select_ref_from_keypoints_bbox_multi(ref_part_indices, ref_part_bboxes, bboxs):
    for ref_index, ref_bbox in zip(ref_part_indices, ref_part_bboxes):
        bbox_areas_ref = [get_bbox_area(bbox) for bbox in ref_bbox]
        max_bbox_area_ref = max(bbox_areas_ref)
        num_human_ref = sum(1 for bbox in ref_bbox if get_bbox_area(bbox) > max_bbox_area_ref * 0.5)
        if num_human_ref < 2 or num_human_ref > 5:
            continue
        driving_bbox_ok = True
        for i, bbox_all in enumerate(bboxs):
            bbox_areas = [get_bbox_area(bbox) for bbox in bbox_all]
            max_bbox_area = max(bbox_areas)
            num_human = sum(1 for bbox in bbox_all if get_bbox_area(bbox) > max_bbox_area * 0.5)
            if num_human != num_human_ref:
                driving_bbox_ok = False
                break
        if driving_bbox_ok:
            return int(ref_index)
        else:
            continue
    return None

def check_from_keypoints_bbox(keypoints, bboxs, IoU_thresthold, reference_width, reference_height, multi_person=False):
    # 用于keypoints版本，根据每一帧的18个keypoints和bbox iou来判断是否满足要求
    last_bbox = None
    for i, (keypoint_all, bbox_all) in enumerate(zip(keypoints, bboxs)):
        if not len(bbox_all):
            return False
        else:
            if multi_person:
                for bbox in bbox_all:
                    if not check_bbox_single_for_video(bbox, reference_width, reference_height, min_bbox_width=1/6):
                        return False
            else:
                bbox = bbox_all[0]
                if not check_bbox_single_for_video(bbox, reference_width, reference_height, min_bbox_width=1/7):
                    return False   # bbox大小异常
                if last_bbox is not None:
                    if not get_IoU(bbox, last_bbox) > IoU_thresthold:
                        return False   # IoU异常
                last_bbox = bbox
    return True



def check_from_keypoints_stick_movement(keypoints, angle_threshold):
    # 骨骼选择：列表中每个元组表示由两个关节确定一条骨骼：格式 (joint_a, joint_b)
    # bones = [(1, 0), (1, 2), (1, 5), (1, 8), (1, 11)]
    bones = [(1, 0), (1, 2), (1, 5), (1, 8), (1, 11), (2, 3), (5, 6), (8, 9), (11, 12)]
    max_delta_list = []
    # 遍历从第二帧开始，对比前一帧和当前帧
    human_num_list = [len(keypoints[idx]["bodies"]["candidate"]) for idx in range(0, len(keypoints))]
    min_human_num = min(human_num_list)
    for human_idx in range(min_human_num):
        for i in range(1, len(keypoints)):
            # 获取上一帧和当前帧的关键点数据（格式为 (18,3) 数组）
            prev_frame_subset = keypoints[i-1]["bodies"]["subset"][human_idx]
            curr_frame_subset = keypoints[i]["bodies"]["subset"][human_idx]
            prev_frame_keypoints = keypoints[i-1]["bodies"]["candidate"][human_idx]
            curr_frame_keypoints = keypoints[i]["bodies"]["candidate"][human_idx]

            
            max_delta = 0
            for (j1, j2) in bones:
                # 检查上一帧中两个关节是否有效（假设 x, y 坐标需大于 0 才认为有效）
                if prev_frame_subset[j1] < 0 or prev_frame_subset[j2] < 0:
                    continue
                if curr_frame_subset[j1] < 0 or curr_frame_subset[j2] < 0:
                    continue

                # 计算上一帧和当前帧中对应骨骼的向量（方向一致，均从 j1 指向 j2）
                vec_prev = np.array([prev_frame_keypoints[j2][0] - prev_frame_keypoints[j1][0],
                                    prev_frame_keypoints[j2][1] - prev_frame_keypoints[j1][1]])
                vec_curr = np.array([curr_frame_keypoints[j2][0] - curr_frame_keypoints[j1][0],
                                    curr_frame_keypoints[j2][1] - curr_frame_keypoints[j1][1]])
                
                # 如果向量模长为0，则无法计算角度，跳过
                if np.linalg.norm(vec_prev) == 0 or np.linalg.norm(vec_curr) == 0:
                    continue
                # 计算向量对应的角度（弧度制）
                angle_prev = math.atan2(vec_prev[1], vec_prev[0])
                angle_curr = math.atan2(vec_curr[1], vec_curr[0])
                
                # 计算角度差，并规范到 [0, pi] 范围
                delta = abs(angle_curr - angle_prev)
                if delta > math.pi:
                    delta = 2 * math.pi - delta
                    
                max_delta = max(delta, max_delta)
            max_delta_list.append(max_delta)
        max_delta_list = sorted(max_delta_list)
        max_delta_list = max_delta_list[len(max_delta_list)//8:-len(max_delta_list)//8] # 去掉两端8分之一的值
        avg_movement = sum(max_delta_list) / len(max_delta_list)
        if avg_movement < angle_threshold:  # 筛去过小的动作
            return False
    return True
    