import numpy as np

def get_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def check_single_human_requirements(det_result):
    # filter results
    if len(det_result) > 3 or len(det_result) == 0:
        return False
    elif len(det_result) == 1:
        return True
    elif len(det_result) > 1: # [2, 3]
        bbox_areas = [get_bbox_area(bbox) for bbox in det_result]
        # 获取最大 bbox 面积的索引
        max_ind = max(range(len(bbox_areas)), key=lambda i: bbox_areas[i])
        # 获取次大面积（需要排除 max_ind）
        other_indices = [i for i in range(len(bbox_areas)) if i != max_ind]
        second_max_area = max([bbox_areas[i] for i in other_indices])

        max_area = bbox_areas[max_ind]
        if max_area < 2 * second_max_area:
            return False
        else:
            return True


def check_multi_human_requirements(det_result):
    # filter results
    if len(det_result) < 2 or len(det_result) > 6:  # 太多了也不好
        return False
    else: # [3, 6]
        bbox_areas = [get_bbox_area(bbox) for bbox in det_result]
        # 获取最大 bbox 面积的索引
        max_ind = max(range(len(bbox_areas)), key=lambda i: bbox_areas[i])
        max_area = bbox_areas[max_ind]
        
        # 选择面积大于等于最大面积50%的bbox
        selected_indices = [i for i in range(len(bbox_areas)) if bbox_areas[i] >= 0.5 * max_area]   # 包含max_ind
        
        # 检查选中的bbox数量是否大于等于2
        if len(selected_indices) >= 2:
            return True
        else:
            return False
