import numpy as np

def largest_safe_bbox(avoid_bboxes):
    # 没有规避区域
    if not avoid_bboxes:
        return 0

    # 定义中心区域
    cx1, cy1, cx2, cy2 = 1/3, 1/3, 2/3, 2/3

    # 检查是否有规避框在中心区域
    for (x1, y1, x2, y2) in avoid_bboxes:
        if not (x2 < cx1 or x1 > cx2 or y2 < cy1 or y1 > cy2):
            return -1

    # 初始化最大可用bbox（默认整个图像）
    safe_x1, safe_y1, safe_x2, safe_y2 = 0.0, 0.0, 1.0, 1.0

    # 更新边界：假设避让框是需要避开的障碍物
    for (x1, y1, x2, y2) in avoid_bboxes:
        # 如果在左侧，右边界不能小于该框右边
        if x2 <= cx1:
            safe_x1 = max(safe_x1, x2)
        # 如果在右侧，左边界不能大于该框左边
        if x1 >= cx2:
            safe_x2 = min(safe_x2, x1)
        # 如果在上方
        if y2 <= cy1:
            safe_y1 = max(safe_y1, y2)
        # 如果在下方
        if y1 >= cy2:
            safe_y2 = min(safe_y2, y1)

    # 防止反转
    safe_x1, safe_x2 = min(safe_x1, safe_x2), max(safe_x1, safe_x2)
    safe_y1, safe_y2 = min(safe_y1, safe_y2), max(safe_y1, safe_y2)

    # 计算面积
    area = (safe_x2 - safe_x1) * (safe_y2 - safe_y1)

    if area > 9/10:
        return 0
    elif area >= 16/25:
        return (safe_x1, safe_y1, safe_x2, safe_y2)
    else:
        return -1





def largest_safe_bbox_multi(avoid_bboxes_list):
    cx1, cy1, cx2, cy2 = 1/3, 1/3, 2/3, 2/3

    per_frame_boxes = []
    for avoid_bboxes in avoid_bboxes_list:
        if not avoid_bboxes or len(avoid_bboxes) == 0:
            per_frame_boxes.append((0.0, 0.0, 1.0, 1.0))
            continue

        # 检查是否与中心区域重叠
        for (x1, y1, x2, y2) in avoid_bboxes:
            if not (x2 < cx1 or x1 > cx2 or y2 < cy1 or y1 > cy2):
                return -1  # 中心挡住

        # 计算该帧可用区域
        safe_x1, safe_y1, safe_x2, safe_y2 = 0.0, 0.0, 1.0, 1.0
        for (x1, y1, x2, y2) in avoid_bboxes:
            if x2 <= cx1:
                safe_x1 = max(safe_x1, x2)
            if x1 >= cx2:
                safe_x2 = min(safe_x2, x1)
            if y2 <= cy1:
                safe_y1 = max(safe_y1, y2)
            if y1 >= cy2:
                safe_y2 = min(safe_y2, y1)

        # 防反转
        safe_x1, safe_x2 = min(safe_x1, safe_x2), max(safe_x1, safe_x2)
        safe_y1, safe_y2 = min(safe_y1, safe_y2), max(safe_y1, safe_y2)
        per_frame_boxes.append((safe_x1, safe_y1, safe_x2, safe_y2))

    # 所有帧取交集
    x1s = [b[0] for b in per_frame_boxes]
    y1s = [b[1] for b in per_frame_boxes]
    x2s = [b[2] for b in per_frame_boxes]
    y2s = [b[3] for b in per_frame_boxes]

    stable_x1 = max(x1s)
    stable_y1 = max(y1s)
    stable_x2 = min(x2s)
    stable_y2 = min(y2s)

    if stable_x1 >= stable_x2 or stable_y1 >= stable_y2:
        return -1

    # 统一在最后判断面积
    area = (stable_x2 - stable_x1) * (stable_y2 - stable_y1)
    if area > 9/10:
        return 0
    elif area >= 16/25:
        return (stable_x1, stable_y1, stable_x2, stable_y2)
    else:
        return -1

