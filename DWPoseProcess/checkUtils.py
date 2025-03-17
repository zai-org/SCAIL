import numpy as np

def check_consistant(boxA, boxB, scoreA_lst, scoreB_lst, beta, all_threshold):
    """
    计算两个锚框之间的 IoU（交并比）以及分数的变化比例，来判断连续性
    """
    # 计算交集框的坐标
    scoreA = scoreA_lst[0]
    scoreB = scoreB_lst[0]
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

    reduction_ratio = (scoreA - scoreB) / (scoreA + scoreB) if scoreA > scoreB else 0  # 如果分数减少的很多，就更不连续

    return iou - reduction_ratio * beta > all_threshold

def part5_valid(valid_joints):
    """
    判断身体五块是不是都有点
    
    参数:
    valid_joints:布尔数组
    
    返回:
    bool: 如果满足要求返回True，否则返回False。
    """
    
    result = 0
    # 1. 核心关节存在 (1)
    core_joints = valid_joints[[1]]
    result += any(core_joints) * 5

    
    # 2. 头部关节至少存在一个 (0, 14, 15, 16, 17)
    head_joints = valid_joints[[0, 14, 15, 16, 17]]
    result += any(head_joints)
    
    # 3. 肩部关节至少存在一个 (2, 5)
    shoulder_joints = valid_joints[[2, 5]]
    result += any(shoulder_joints) * 2
    
    # 4. 手部关节至少存在一个 (3, 4, 6, 7)
    hand_joints = valid_joints[[3, 4, 6, 7]]
    result += any(hand_joints)
    
    # 5. 下半身关节至少存在一个 (8, 9, 10, 11, 12, 13)
    lower_body_joints = valid_joints[[8, 9, 10, 11, 12, 13]]
    result += any(lower_body_joints)
    
    return result >= 8


def is_pose_valid(pose_score, threshold=0.3):
    """
    判断输入的18点姿态是否满足要求。
    
    参数:
    pose_scores (np.array): 形状为(18,)的张量，表示每个关节的得分。
    threshold (float): 得分阈值，默认0.3。
    
    返回:
    bool: 如果满足要求返回True，否则返回False。
    """
    # 1. 每个关节的得分 > 0.3 才认为是有效的
    valid_joints = pose_score > threshold
    return part5_valid(valid_joints)
    
    
def check_skeleton_init_seq(pose_scores, threshold=0.3):
    valid_joints = np.zeros(18)
    for pose_score in pose_scores:
        valid_joints += (pose_score > threshold)
    return part5_valid(valid_joints)


def check_skeleton_window_seq(pose_scores, threshold=0.3):
    valid_joints = np.zeros(18)
    for pose_score in pose_scores:
        valid_joints += pose_score > threshold
    return np.count_nonzero(valid_joints) > 8


def check_frame(i, last_det_result, skeleton_init_sequence, skeleton_window_sequence, last_mean_score, video_path, detector_return_list, beta, consistant_thresthold):
    # 后面都要改的，在draw_pose里面check，这里的逻辑在送进去的时候取的就不一定对了
    # 应该在送进draw_pose之前筛连续性和粗的骨骼序列，比如windows_size为8这样
    # 送进去draw_pose之后筛细一些的骨骼序列，比如windows_size为3-4这样，通过滑动保证最终保证取的65帧里满足需要即可

    result, pose, score, det_result = detector_return_list
    mean_score = [np.mean(score, axis=-1)]
    if det_result is None or len(det_result)==0:
        pass
        # print(f"第{i}帧锚框异常，跳过此帧")
        # print(det_result)
    else:
        if last_det_result.any():
            if not check_consistant(last_det_result, det_result[0], last_mean_score, mean_score, beta, consistant_thresthold) :
                print(f"{video_path} 过程中不满足连续性限制")
                return 0                    
        last_det_result = det_result[0]
    
    if len(skeleton_window_sequence)>=4:
        if not check_skeleton_window_seq(skeleton_window_sequence):
            print(f"{video_path} 过程中第不满足骨骼限制")
            return 0
    
    if i == 0:   # 首帧
        if not is_pose_valid(score[0]):
            print(f"{video_path} 首帧不满足骨骼限制")
            return 0
    
    if i <= 8:  # 初始8帧
        if len(skeleton_init_sequence)==3:
            if not check_skeleton_init_seq(skeleton_init_sequence):
                print(f"{video_path} 第一个batch不满足骨骼限制")
                return 0
        skeleton_init_sequence.append(score[0])

    skeleton_window_sequence.append(score[0])
    last_mean_score[0] = mean_score[0]

    return 1


def check_frames_list(batch_idx, infer_batch_size, last_det_result, skeleton_init_sequence, skeleton_window_sequence, last_mean_score, video_path, detector_return_list, beta, consistant_thresthold):
    """
    判断输入的18点姿态是否满足要求。
    
    参数:
    last_det_result: 上一帧结果，用于检测连续性
    skeleton_init_sequence: 骨骼初始8帧的检测窗口（大小为3
    skeleton_window_sequence: 过程中骨骼检测窗口（大小为4
    threshold (float): 得分阈值，默认0.3
    ...
    
    返回:
    bool: 如果满足要求返回True，否则返回False。
    """
    results, pose, scores, det_results = zip(*detector_return_list)
    results = list(results)
    scores = list(scores)
    det_results = list(det_results)
    for i, (result, score, det_result) in enumerate(zip(results, scores, det_results)):     # score, det_result均为[[]]格式，判断时需取0
        mean_score = [np.mean(score, axis=-1)]
        if det_result is None or len(det_result)==0:
            pass
            # print(f"第{i}帧锚框异常，跳过此帧")
            # print(det_result)
        else:
            if last_det_result.any():
                if not check_consistant(last_det_result, det_result[0], last_mean_score, mean_score, beta, consistant_thresthold) :
                    print(f"{video_path} 过程中不满足连续性限制")
                    return 0                    
            last_det_result = det_result[0]
        
        if len(skeleton_window_sequence)>=4:
            if not check_skeleton_window_seq(skeleton_window_sequence):
                print(f"{video_path} 过程中第不满足骨骼限制")
                return 0
        
        if batch_idx * infer_batch_size + i == 0:   # 首帧
            if not is_pose_valid(score[0]):
                print(f"{video_path} 首帧不满足骨骼限制")
                return 0

        if batch_idx * infer_batch_size + i <= 8:   # 对初始8帧才会影响skeleton_init_sequence
            if len(skeleton_init_sequence)==3:  # keleton_init_sequence满
                if not check_skeleton_init_seq(skeleton_init_sequence):
                    print(f"{video_path} 第一个batch不满足骨骼限制")
                    return 0
            skeleton_init_sequence.append(score[0])

        skeleton_window_sequence.append(score[0])
        last_mean_score[0] = mean_score[0]

    return 1