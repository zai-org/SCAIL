import time
import cv2
import numpy as np
from decord import VideoReader, cpu
from paddleocr import TextDetection
from OCR.safe_bbox import largest_safe_bbox, largest_safe_bbox_multi
import os
# 尝试导入 moviepy
try:
    import moviepy.editor as mpy
except ImportError:
    import moviepy as mpy



# 获取视频分辨率
def get_video_bboxes(model, np_frames, batch_size=8):
    L, H, W, _ = np_frames.shape
    frame_area = H * W
    min_x_ratio_thresh = 1 / 6
    min_y_ratio_thresh = 1 / 6
    min_area_ratio_thresh = 1 / 64
    max_area_ratio_thresh = 2 / 3   # poly 面积阈值比例
    all_bboxes = []  # 每帧的 bbox 列表

    # 每4帧取1帧
    frame_indices = list(range(0, L, 4))

    # 按 batch_size 分组处理这些索引
    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i:i + batch_size]
        frames = [np_frames[j] for j in batch_indices]

        # 批量预测
        results = model.predict(frames, batch_size=batch_size)

        # 处理每帧结果：poly → bbox
        for result in results:
            frame_bboxes = []
            if result and 'dt_polys' in result:
                for poly in result['dt_polys']:
                    poly = np.array(poly, dtype=np.int32)
                    area = cv2.contourArea(poly)
                    if area / frame_area > max_area_ratio_thresh or area / frame_area < min_area_ratio_thresh:
                        continue  # 跳过太小/太大的区域
                    x, y, w, h = cv2.boundingRect(poly)
                    if w / W > min_x_ratio_thresh or h / H > min_y_ratio_thresh:
                        frame_bboxes.append((x / W, y / H, (x + w) / W, (y + h) / H))
            all_bboxes.append(frame_bboxes)

    return all_bboxes



if __name__ == "__main__":
    model = TextDetection(limit_side_len=960, device="gpu", box_thresh=0.8, thresh=0.3)
    video_root = '/workspace/ywh_data/EvalSelf/Results/sc_14b_nlfx_1800iter_1017'
    for subdir in sorted(os.listdir(video_root)):
        video_path = os.path.join(video_root, subdir, f"{subdir}_output_000000.mp4")
        if not os.path.isfile(video_path):
            continue

        vr = VideoReader(video_path)
        np_frames = vr.get_batch(range(len(vr))).asnumpy()
        all_bboxes = get_video_bboxes(model, np_frames)
        largest_bbox_det_result = largest_safe_bbox_multi(all_bboxes)
        print(f"video_path: {video_path}, \nlargest_bbox_det_result: {largest_bbox_det_result}")

