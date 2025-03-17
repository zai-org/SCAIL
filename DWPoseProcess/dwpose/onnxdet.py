# https://github.com/IDEA-Research/DWPose
import cv2
import numpy as np
import onnxruntime


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    
    # 输出和原来尺寸一样
    return outputs


def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def inference_detector(session, oriImg):
    input_shape = (640, 640)
    img, ratio = preprocess(oriImg, input_shape)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)  
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        isscore = final_scores > 0.3
        iscat = final_cls_inds == 0
        isbbox = [i and j for (i, j) in zip(isscore, iscat)]
        final_boxes = final_boxes[isbbox]
    else:
        # print("no boxes detected")
        return []

    return final_boxes

def preprocess_batch(img_list, input_size, swap=(2, 0, 1)):
    """
    Preprocess a batch of images for input to the model.
    Args:
        img_list: List of input images (as numpy arrays).
        input_size: Tuple specifying the input size (height, width).
        swap: Tuple indicating the axis order for transpose.

    Returns:
        batched_imgs: A numpy array of preprocessed images with a batch dimension.
        ratios: List of resizing ratios for each image.
    """
    padded_imgs = []
    ratios = []

    for img in img_list:
        padded_img, r = preprocess(img, input_size, swap)
        # breakpoint()
        padded_imgs.append(padded_img)
        ratios.append(r)

    return np.stack(padded_imgs), ratios


def inference_detector_batch(session, img_list):
    """
    Perform inference on a batch of images.
    Args:
        session: ONNX runtime session.
        img_list: List of input images (as numpy arrays).

    Returns:
        results: A list of bounding box results for each image in the batch.
    """
    input_shape = (640, 640)
    batched_imgs, ratios = preprocess_batch(img_list, input_shape)
    # breakpoint()

    # Prepare input for ONNX model
    ort_inputs = {session.get_inputs()[0].name: batched_imgs}
    outputs = session.run(None, ort_inputs)

    # outputs[0].shape: (batch, num_boxes, 85)  -> num_boxes 检测框 + 80 坐标+置信度 + 5 类别信息
    predictions = demo_postprocess(outputs[0], input_shape)     # 输出不变

    results = []
    for i, prediction in enumerate(predictions):
        boxes = prediction[:, :4]
        scores = prediction[:, 4:5] * prediction[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratios[i]

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            isscore = final_scores > 0.3
            iscat = final_cls_inds == 0
            isbbox = [k and j for (k, j) in zip(isscore, iscat)]
            final_boxes = final_boxes[isbbox]   # array [num_box, 4]
        else:
            final_boxes = []

        results.append(final_boxes)

    return results  # list of ndarray batch_num * [num_box, 4]


# def test_inference_detector_batch():
#     """Test the batch inference with dummy data."""
#     # Initialize a fake ONNX session (replace with actual model path for real testing)
#     session = onnxruntime.InferenceSession("/workspace/yanwenhao/YOLOX/yolox_l.onnx")

#     # Generate random test images
#     img_list = [
#         (np.random.rand(480, 640, 3) * 255).astype(np.uint8) for _ in range(5)
#     ]

#     # Perform batch inference
#     results = inference_detector_batch(session, img_list)

#     # Print results
#     for i, result in enumerate(results):
#         print(f"Image {i + 1}: {len(result)} boxes detected.")


# if __name__ == "__main__":
#     test_inference_detector_batch()


