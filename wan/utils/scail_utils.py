import decord
import numpy as np
from decord import VideoReader
import torch

from PIL import Image
import torchvision.transforms as TT

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize

def load_image_to_tensor_chw_normalized(image: Image.Image):
    # Open image using PIL
    # image = Image.open(image_data).convert('RGB')  # Convert to RGB in case it's a grayscale image or has an alpha channel
    # Define a transform to convert image to tensor
    transform = TT.Compose([TT.ToTensor()])
    # Apply the transform
    image_tensor = transform(image)
    # Scale the tensor back to [0, 255] and convert to uint8 (decord does this too)
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0)  # 1 C H W, -1-1
    return image_tensor

def load_video_for_pose_sample(video_data):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    indices = np.arange(0, len(vr))
    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    return tensor_frms


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=image_size[0], width=image_size[1]
    )
    return arr
