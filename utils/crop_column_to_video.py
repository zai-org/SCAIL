import decord
import av
import os
import numpy as np
from typing import Optional
import tempfile
from PIL import Image
import cv2 # Added missing import for cv2


def crop_rightmost_column(input_video_path: str, output_video_path: str, 
                         fps: Optional[float] = None) -> bool:
    """
    从四列叠放的MP4视频中提取最右侧一列，并保存为新的视频文件。
    使用decord读取视频，PyAV写入视频，输出为高质量H.264格式。
    
    Args:
        input_video_path (str): 输入视频文件路径
        output_video_path (str): 输出视频文件路径
        fps (float, optional): 输出视频帧率，如果为None则使用原视频帧率
        
    Returns:
        bool: 处理是否成功
    """
    try:
        vr = decord.VideoReader(input_video_path)
        frames = vr.get_batch(np.arange(len(vr))).asnumpy()  # t h w c
        
        height = frames.shape[1]  # 高度
        width = frames.shape[2]  # 宽度
        
        # 计算裁剪区域（最右侧四分之一）
        crop_width = width // 4
        crop_x = width - crop_width  # 从右侧开始
        
        print(f"裁剪区域: x={crop_x}, width={crop_width}")
        
        # 创建输出容器
        output_container = av.open(output_video_path, mode='w', format='mp4')
        
        stream = output_container.add_stream('libx264', rate=16)
        stream.height = height   # 裁剪后的宽度
        stream.width = crop_width        # 保持原高度
        stream.pix_fmt = 'yuv420p'
        stream.options = {
            'crf': '8',      # 接近无损
        }

        for frame_idx in range(len(frames)):
            frame = frames[frame_idx]
            # 裁剪最右侧一列
            cropped_frame = frame[ :, crop_x:crop_x + crop_width, :]   # h w c
            
            # 转换为PIL图像
            pil_image = Image.fromarray(cropped_frame)
            
            # 创建PyAV帧 - 使用与工作代码相同的方式
            av_frame = av.VideoFrame.from_image(pil_image)
            
            # 编码帧 - 使用与工作代码相同的方式
            output_container.mux(stream.encode(av_frame))
        
        output_container.mux(stream.encode())
        
        # 关闭容器
        output_container.close()
        return True
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def crop_rightmost_column_batch(input_dir: str, output_dir: str) -> None:
    """
    批量处理目录中的MP4文件，提取最右侧一列。
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        file_pattern (str): 文件匹配模式，默认为"*.mp4"
        fps (float, optional): 输出视频帧率，如果为None则使用原视频帧率
    """
    import glob
    
    mp4_list = sorted(glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True))
    for mp4_path in mp4_list:
        print(f"Processing {mp4_path}")
        real_path = os.path.relpath(mp4_path, input_dir)
        output_path = os.path.join(output_dir, real_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        crop_rightmost_column(mp4_path, output_path)


if __name__ == "__main__":
    input_directory = "/workspace/ywh_data/crossEval/ours_test_14B_12000iter"
    output_directory = "/workspace/ywh_data/crossEval/ours_test_14B_12000iter_single_col/"
    crop_rightmost_column_batch(input_directory, output_directory)
