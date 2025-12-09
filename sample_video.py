import os
import sys
import math
import argparse
import json
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
from PIL import Image
import imageio
import time
import gc
import copy
import torch
import numpy as np
from einops import rearrange, repeat
from torchvision.utils import make_grid
import torchvision.transforms as TT

from sgm.util import get_obj_from_str, isheatmap, exists

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

import diffusion_video
from arguments import get_args, process_config_to_args
import decord
from decord import VideoReader
from torchvision import transforms
import shutil
import torch.nn.functional as F
from data_video import pad_last_frame, resize_for_rectangle_crop

def load_image_to_tensor_chw_normalized(image_data):
    # Open image using PIL
    image = Image.open(image_data).convert('RGB')  # Convert to RGB in case it's a grayscale image or has an alpha channel
    # Define a transform to convert image to tensor
    transform = TT.Compose([TT.ToTensor()])
    # Apply the transform
    image_tensor = transform(image)
    # Scale the tensor back to [0, 255] and convert to uint8 (decord does this too)
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0)  # 1 C H W, -1-1
    # C H W
    return image_tensor


def load_video_for_pose_sample(video_data):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    indices = np.arange(0, len(vr))
    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    return tensor_frms


import random
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
import cv2

def find_file_with_patterns(directory, patterns):
    """Find file matching any of the given patterns in the directory"""
    for pattern in patterns:
        file_path = os.path.join(directory, pattern)
        if os.path.exists(file_path):
            return file_path
    return None

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input('Please input in format like <prompt>@@<example_dir>, e.g. the girl is dancing@@examples/001 (Ctrl-D quit): ')
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass

def read_from_file(p, rank=0, world_size=1):
    with open(p, 'r') as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def save_multi_video_grid_and_mp4(
    video_batches: list, save_dir: str, fps: int = 5, args=None, key=None
):
    os.makedirs(save_dir, exist_ok=True)
    # base_count = len(glob(os.path.join(save_path, "*.mp4")))
    multi_video_batch = torch.stack(video_batches, dim=2)
    for i, multi_vid in enumerate(multi_video_batch):
        # save_image(vid, fp=os.path.join(save_path, f"{base_count:06d}.png"), nrow=4)
        # multi_vid: T, N, c, h, w 
        gif_frames = []
        for multi_frame in multi_vid:
            frame = rearrange(multi_frame, "n c h w -> h (n w) c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_dir, f"{key}_{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None
):
    os.makedirs(save_path, exist_ok=True)
    # base_count = len(glob(os.path.join(save_path, "*.mp4")))

    for i, vid in enumerate(video_batch):
        # save_image(vid, fp=os.path.join(save_path, f"{base_count:06d}.png"), nrow=4)
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
    if args.load is not None:
        load_checkpoint(model, args)
    model.eval()

    if args.input_type == 'cli':
        assert mpu.get_data_parallel_world_size() == 1, 'Only dp = 1 supported in cli mode.'
        data_iter = read_from_cli()
    elif args.input_type == 'txt':
        dp_rank, dp_world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        data_iter = read_from_file(args.input_file, rank=dp_rank, world_size=dp_world_size)
    else:
        raise NotImplementedError
    sample_func = model.sample
    
    # if not args.multi_cond_cfg:
    #     sample_func = model.sample
    # else:
    #     sample_func = model.sample_with_pose_cond

    num_samples = [1]
    force_uc_zero_embeddings = []

    vae_compress_size = args.vae_compress_size
    print('VAE_compress_size:', vae_compress_size)
    # if args.image2video:
    #     zero_pad_dict = torch.load('zero_pad_dict.pt', map_location='cpu')

    with torch.no_grad():
        torch.distributed.barrier(group=mpu.get_data_broadcast_group())
        while True:
            stopped = False
            if mpu.get_data_broadcast_rank() == 0:
                try:
                    text, cnt = next(data_iter)
                except StopIteration:
                    text = ''
                    stopped = True

                # text = 'FPS-%d. ' % args.sampling_fps + text

            else:
                text = ''
                cnt = 0

            broadcast_list = [text, cnt, stopped]
            # broadcast
            mp_size = mpu.get_model_parallel_world_size()
            sp_size = mpu.get_sequence_parallel_world_size()

            if mp_size > 1 or sp_size > 1:
                torch.distributed.broadcast_object_list(broadcast_list, src=mpu.get_data_broadcast_src_rank(), group=mpu.get_data_broadcast_group())

            text, cnt, stopped = broadcast_list
            if stopped:
                break

            if mpu.get_data_broadcast_rank() == 0:
                print(cnt, ': ', text)

            if args.image2video:    # i2v 输入是图片+prompt
                if args.use_pose:
                    text_parts = text.split('@@')
                    text = text_parts[0]
                    input_dir = text_parts[1]
                    
                    # Find reference image with multiple possible names
                    ref_image_patterns = ['ref.jpg', 'ref.png', 'ref_image.jpg', 'ref_image.png']
                    image_path = find_file_with_patterns(input_dir, ref_image_patterns)
                    if image_path is None:
                        raise FileNotFoundError(f"Reference image not found in {input_dir}. Tried: {ref_image_patterns}")
                    
                    # Find pose video with multiple possible names
                    pose_patterns = ['rendered_aligned.mp4', 'rendered.mp4']
                    pose_path = find_file_with_patterns(input_dir, pose_patterns)
                    if pose_path is None:
                        raise FileNotFoundError(f"Pose video not found in {input_dir}. Tried: {pose_patterns}")
                    
                    if text == "None":
                        text = ""
                    else:
                        text = text
                else:
                    text, image_path = text.split('@@')
                
                
                # ******获取动作序列******
                GT = None
                GT_patterns = ['GT.mp4']
                GT_path = find_file_with_patterns(input_dir, GT_patterns)
                if GT_path is not None:
                    GT = load_video_for_pose_sample(GT_path)
                    GT = GT.permute(0, 3, 1, 2) #
                    GT = (GT - 127.5) / 127.5   # color value: 0-255 -> -1-1

                if image_path != "firstframe":     
                    assert os.path.exists(image_path), "video should exist"
                    image_tensor = load_image_to_tensor_chw_normalized(image_path)
                else:                   # "firstframe" tag is for testing self-driven cases, directly using first frame of GT as reference image
                    assert GT is not None
                    image_tensor = GT[0].unsqueeze(0)    # C H W -> T C H W
                # 获取采样尺寸
                if image_tensor.shape[2] < image_tensor.shape[3]:
                    target_H, target_W = args.sampling_image_size
                else:
                    target_W, target_H = args.sampling_image_size

                
                # 获取驱动信号
                # Get fps from driving video
                decord.bridge.set_bridge("torch")
                vr_for_fps = VideoReader(uri=pose_path, height=-1, width=-1)
                driving_fps = vr_for_fps.get_avg_fps()
                print(f"Driving video fps: {driving_fps}")
                
                pose_video = load_video_for_pose_sample(pose_path)
                pose_video = pose_video.permute(0, 3, 1, 2) # T H W C -> T C H W 
                pose_video = resize_for_rectangle_crop(pose_video, [target_H, target_W], reshape_mode="center")
                pose_video = (pose_video - 127.5) / 127.5   # color value: 0-255 -> -1-1
                sampling_num_frames = pose_video.shape[0]
                
                # 其它的也都crop
                image_tensor = resize_for_rectangle_crop(image_tensor, [target_H, target_W], reshape_mode="center")
                if GT is not None:
                    GT = resize_for_rectangle_crop(GT, [target_H, target_W], reshape_mode="center")
                if "smpl" in args.representation:
                    smpl_render_video = pose_video
                    if "smpl_downsample" in args.representation:
                        smpl_render_video = F.interpolate(smpl_render_video, scale_factor=0.5, mode='bilinear', align_corners=False)  # t c h w
                        

                # VAE编码
                if model.i2v_encode_video:          # wan的模式,不需要再重复或者替换第一帧
                    assert args.use_pose, 'wan for not using pose has not been merged into this version'
                    pose_video = pose_video.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B T C H W
                    if "smpl" in args.representation:
                        smpl_render_video = smpl_render_video.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B T C H W
                    ori_image = image_tensor.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B 1 C H W, -1-1
                    image_to_save = ori_image.repeat(1, pose_video.shape[1], 1, 1, 1)
                    image = torch.concat([ori_image, torch.zeros_like(pose_video[:, 1:])], dim=1)
                    image = rearrange(image, 'b t c h w -> b c t h w').contiguous()
                    image = model.encode_first_stage(image, None, force_encode=True)
                    image = image.permute(0, 2, 1, 3, 4).contiguous() # BCTHW -> BTCHW
                    ref_concat = model.encode_first_stage(rearrange(ori_image,'b t c h w -> b c t h w').contiguous() , None, force_encode=True)
                    ref_concat = ref_concat.permute(0, 2, 1, 3, 4).contiguous()
                else:                               # 旧的cogvideo的模式，如果用到需要重写
                    T = int(sampling_num_frames / 4) + 1
                    pose_video = pose_video.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B T C H W, -1-1
                    ori_image = image_tensor.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B 1 C H W
                    image_to_save = ori_image.repeat(1, pose_video.shape[1], 1, 1, 1)
                    image = ori_image.clone()
                    image = rearrange(image, 'b t c h w -> b c t h w').contiguous()
                    image = model.encode_first_stage(image, None, force_encode=True)
                    image = image.permute(0, 2, 1, 3, 4).contiguous()   # BCTHW
                    ref_concat = image.clone()
                    pad_shape = (image.shape[0], T-1, image.shape[2], image.shape[3], image.shape[4])
                    if model.noised_image_all_concat:
                        image = image.repeat(1, T, 1, 1, 1)
                    else:
                        image = torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)


                if "smpl" in args.representation:
                    smpl_render_latent = model.encode_first_stage(rearrange(smpl_render_video, 'b t c h w -> b c t h w').contiguous(), None, force_encode=True)
                    smpl_render_latent = smpl_render_latent.permute(0, 2, 1, 3, 4).contiguous()   # B, T, C, H, W
                    pose_latent = smpl_render_latent
                else:
                    pose_latent = model.encode_first_stage(rearrange(pose_video, 'b t c h w -> b c t h w').contiguous(), None, force_encode=True)
                    pose_latent = pose_latent.permute(0, 2, 1, 3, 4).contiguous()   # B, T, C, H, W

                T = pose_latent.shape[1]
                C, H, W = image.shape[2], image.shape[3], image.shape[4]


                if model.use_i2v_clip:
                    model.i2v_clip.model.to('cuda')
                    image_clip_features = model.i2v_clip.visual(ori_image.permute(0, 2, 1, 3, 4))  # btchw -> bcthw
                    model.i2v_clip.model.cpu()
                
            else:
                raise NotImplementedError("image2video should be used")

            # TODO: broadcast image2video
            value_dict = {
                'prompt': text,
                # 'negative_prompt': "手部变形，脸部变形，低质量",
                'negative_prompt': "",
                'num_frames': torch.tensor(T).unsqueeze(0)
            }
            test_case_idx = os.path.basename(input_dir)  
            save_dir = os.path.join(args.output_dir, test_case_idx)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'text.txt'), 'w') as f:
                f.write(text)

            model.conditioner.embedders[0].to('cuda')
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            
            # 这里把batch加上text embedding包装成c和uc
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            model.conditioner.embedders[0].cpu()

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(
                        lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                    )

            if args.image2video:
                assert not args.multi_cond_cfg, "Multi Cond CFG does not work well"
                if args.use_pose:
                    c["concat_images"] = image
                    uc["concat_images"] = image
                    c["ref_concat"] = ref_concat
                    uc["ref_concat"] = ref_concat
                    c["concat_pose"] = pose_latent
                    uc["concat_pose"] = pose_latent
                    if "smpl" in args.representation:
                        c["concat_smpl_render"] = smpl_render_latent
                        uc["concat_smpl_render"] = smpl_render_latent
                        # c["concat_cheek_hands"] = cheek_hands_latent
                        # uc["concat_cheek_hands"] = cheek_hands_latent
                    # c['pose_downsample'] = downsample_pose_latent
                    # uc['pose_downsample'] = downsample_pose_latent
                else:
                    c["concat_images"] = image     # torch.Size([1, 32, 128, 32, 55])
                    uc["concat_images"] = image     # 如果为zeros_like t2v结果也不变
                if model.use_i2v_clip:
                    c["image_clip_features"] = image_clip_features
                    uc["image_clip_features"] = image_clip_features

                    


            for index in range(args.batch_size):
                if args.multi_cond_cfg:
                    raise NotImplementedError("Multi Cond CFG does't work well")
                else:
                    samples_z = sample_func(
                        c,
                        uc = uc,
                        batch_size = 1,
                        shape = (T, C, H, W),
                        ofs = torch.tensor([2.0]).to('cuda'),
                        fps = torch.tensor([args.sampling_fps]).to('cuda'),
                    )
                if mpu.get_sequence_parallel_rank() == 0:
                    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                    if args.only_save_latents:
                        if mpu.get_model_parallel_rank() == 0:
                            samples_z = 1.0 / model.scale_factor * samples_z
                            # torch.save(samples_z, save_path)
                    else:
                        samples_x = model.decode_first_stage(samples_z).to(torch.float32)
                        # samples_x = samples_x.view(b, t, *samples_x.shape[1:])
                        samples_x = samples_x.permute(0, 2, 1, 3, 4).contiguous()
                        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                        # grid = torch.stack([samples])
                        # grid = rearrange(grid, "n b c h w -> c (n h) (b w)")

                        if mpu.get_model_parallel_rank() == 0:
                            if GT is not None:
                                if "smpl_downsample" in args.representation:
                                    smpl_render_video = F.interpolate(smpl_render_video.squeeze(0), size=(GT.shape[2], GT.shape[3]), mode='bilinear', align_corners=False).unsqueeze(0)  # t c h w -> 1 t c h w
                                    save_list = [torch.clamp((smpl_render_video + 1.0) / 2.0, min=0.0, max=1.0).cpu(), torch.clamp((image_to_save + 1.0) / 2.0, min=0.0, max=1.0).cpu(), torch.clamp((GT.unsqueeze(0) + 1.0) / 2.0, min=0.0, max=1.0).cpu()]                          
                                elif "smpl" in args.representation:
                                    save_list = [torch.clamp((smpl_render_video + 1.0) / 2.0, min=0.0, max=1.0).cpu(), torch.clamp((image_to_save + 1.0) / 2.0, min=0.0, max=1.0).cpu(), torch.clamp((GT.unsqueeze(0) + 1.0) / 2.0, min=0.0, max=1.0).cpu()]
                                save_multi_video_grid_and_mp4(save_list + [samples], save_dir, fps=driving_fps, key=f"{test_case_idx}_concat") # 都要求是B T C H W
                            save_multi_video_grid_and_mp4([samples], save_dir, fps=driving_fps, key=f"{test_case_idx}_output")
                            

if __name__ == '__main__':
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    if "sigma_sampler_config" in args.model_config.loss_fn_config.params.keys() and hasattr(args.model_config.loss_fn_config.params.sigma_sampler_config.params, "uniform_sampling"):
        args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    if args.model_type == "dit":
        Engine = diffusion_video.SATVideoDiffusionEngine
    print(args.model_type)

    sampling_main(args, model_cls=Engine)
