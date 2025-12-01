import os
import sys
import argparse
from functools import partial
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed
import torchvision
from typing import Dict, List
from omegaconf import OmegaConf
import imageio

import torch
from torch import nn
import torch.nn.functional as F

from sat import mpu
from sat.training.deepspeed_training import training_main

from sgm.util import get_obj_from_str, isheatmap, exists

from diffusion_video_pnp import SATVideoDiffusionEngine
from data_video import data_collator
from arguments import get_args, process_config_to_args

from glob import glob
from einops import rearrange, repeat
try:
    import wandb
except ImportError:
    print("warning: wandb not installed")

def print_debug(args, s):
    if args.debug:
        s = f"RANK:[{torch.distributed.get_rank()}]:" + s
        print(s)

def save_texts(texts, save_dir, iterations):
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5, args=None, key=None
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
        if args is not None and args.wandb:
            wandb.log({key+f'_video_{i}': wandb.Video(now_save_path, fps=fps, format="mp4")}, step=args.iteration+1)


def log_video(batch, model, args, only_log_video_latents=False):
    # if not os.path.exists(os.path.join(args.save, 'training_config.yaml')):
    #     configs = [OmegaConf.load(cfg) for cfg in args.base]
    #     config = OmegaConf.merge(*configs)
    #     os.makedirs(args.save, exist_ok=True)
    #     OmegaConf.save(config=config, f=os.path.join(args.save, 'training_config.yaml'))

    texts = batch['txt']
    text_save_dir = os.path.join(args.save, "video_texts")
    os.makedirs(text_save_dir, exist_ok=True)
    save_texts(texts, text_save_dir, args.iteration)

    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),  # torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }
    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        videos = model.log_video(batch, only_log_video_latents=only_log_video_latents)

    if torch.distributed.get_rank() == 0:
        root = os.path.join(args.save, "video")

        num_frames = batch['num_frames'][0]
        fps = batch['fps'][0].cpu().item()
        if only_log_video_latents:
            root = os.path.join(root, "latents")
            filename = "{}_gs-{:06}".format(
                    'latents', args.iteration
                )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            os.makedirs(path, exist_ok=True)
            torch.save(videos['latents'], os.path.join(path, 'latent.pt'))
        else:
            for k in videos:
                N = videos[k].shape[0]
                if not isheatmap(videos[k]):
                    videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().float().cpu()
                    if not isheatmap(videos[k]):
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)
            for k in videos:
                samples = (videos[k] + 1.0) / 2.0
                filename = "{}_gs-{:06}".format(
                    k, args.iteration
                )

            num_frames = batch['num_frames'][0]
            fps = batch['fps'][0].cpu().item()
            if only_log_video_latents:
                root = os.path.join(root, "latents")
                filename = "{}_gs-{:06}".format(
                        'latents', args.iteration
                    )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                os.makedirs(path, exist_ok=True)
                torch.save(videos['latents'], os.path.join(path, 'latents.pt'))
            else:
                for k in videos:
                    samples = (videos[k] + 1.0) / 2.0
                    filename = "{}_gs-{:06}".format(
                        k, args.iteration
                    )

                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    save_video_as_grid_and_mp4(samples, path, num_frames // fps, fps, args, k)

def log_image(batch, model, args):

    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),  # torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        images = model.log_video(batch)

    for k in images:
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().float().cpu()
            images[k] = torch.clamp(images[k], -1.0, 1.0)

    if torch.distributed.get_rank() == 0:
        texts = batch['txt']
        text_save_dir = os.path.join(args.save, "texts")
        os.makedirs(text_save_dir, exist_ok=True)
        save_texts(texts, text_save_dir, args.iteration)

        root = os.path.join(args.save, "images")
        for k in images:
            if len(images[k].shape) == 5:
                images[k] = images[k].squeeze(1)
            grid = torchvision.utils.make_grid(images[k], nrow=2)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}.png".format(
                k, args.iteration
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            img = Image.fromarray(grid)
            img.save(path)
            if args is not None and args.wandb:
                wandb.log({k: wandb.Image(path)}, step=args.iteration+1)

def broad_cast_batch(batch):
    mp_size = mpu.get_model_parallel_world_size()
    global_rank = torch.distributed.get_rank() // mp_size
    src = global_rank * mp_size

    if batch['mp4'] is not None:
        broadcast_shape = [batch['mp4'].shape, batch['fps'].shape, batch['num_frames'].shape]
    else:
        broadcast_shape = None

    txt = [batch['txt'], broadcast_shape]
    torch.distributed.broadcast_object_list(txt, src=src, group=mpu.get_model_parallel_group())
    batch['txt'] = txt[0]

    # keys = ['mp4', 'fps', 'num_frames']
    # data_type = torch.float32
    # data_b = mpu.broadcast(keys, batch, data_type) # for cpu
    mp4_shape = txt[1][0]
    fps_shape = txt[1][1]
    num_frames_shape = txt[1][2]

    if mpu.get_model_parallel_rank() != 0:
        batch['mp4'] = torch.zeros(mp4_shape, device="cuda")
        batch['fps'] = torch.zeros(fps_shape, device='cuda', dtype=torch.long)
        batch['num_frames'] = torch.zeros(num_frames_shape, device='cuda', dtype=torch.long)

    torch.distributed.broadcast(batch['mp4'], src=src, group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(batch['fps'], src=src, group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(batch['num_frames'], src=src, group=mpu.get_model_parallel_group())
    return batch

def forward_step_eval(data_iterator, model, args, timers, data_class=None, only_log_video_latents=False):
    if mpu.get_model_parallel_rank() == 0:
        timers('data loader').start()
        batch_image = next(data_iterator)
        batch_video = next(data_iterator)
        timers('data loader').stop()
        if batch_video["fps"].sum() == 0:
            tmp = batch_video
            batch_video = batch_image
            batch_image = tmp
        for key in batch_image:
            if isinstance(batch_image[key], torch.Tensor):
                batch_image[key] = batch_image[key].cuda()
        for key in batch_video:
            if isinstance(batch_video[key], torch.Tensor):
                batch_video[key] = batch_video[key].cuda()
    else:
        batch_video = {'mp4': None, 'fps': None, 'num_frames': None, 'txt': None}
        batch_image = {'mp4': None, 'fps': None, 'num_frames': None, 'txt': None}
    broad_cast_batch(batch_video)
    broad_cast_batch(batch_image)
    # torch.distributed.broadcast_object_list(batch_video, src=src, group=mpu.get_model_parallel_group())
    # torch.distributed.broadcast_object_list(batch_image, src=src, group=mpu.get_model_parallel_group())
    if mpu.get_data_parallel_rank() == 0:
        log_video(batch_video, model, args, only_log_video_latents=only_log_video_latents)
        if not only_log_video_latents:
            log_image(batch_image, model, args)

    batch_video['global_step'] = args.iteration
    loss, loss_dict = model.shared_step(batch_video)
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)
    return loss, loss_dict

def forward_step(data_iterator, model, args, timers, data_class=None):

    if mpu.get_model_parallel_rank() == 0:
        timers('data loader').start()
        batch = next(data_iterator)
        timers('data loader').stop()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        # for image batch
        if batch['fps'][0] == 0:
            batch['mp4'] = rearrange(batch['mp4'], 'b t c h w -> (b t) c h w').unsqueeze(1)
            txt_all = []
            for item in batch['txt']:
                txt_all.extend(item.split('##$$##$'))
            batch['txt'] = txt_all
            batch['num_frames'] = batch['num_frames'].repeat(8)
            batch['fps'] = batch['fps'].repeat(8)

        if torch.distributed.get_rank() == 0:
            if not os.path.exists(os.path.join(args.save, 'training_config.yaml')):
                configs = [OmegaConf.load(cfg) for cfg in args.base]
                config = OmegaConf.merge(*configs)
                os.makedirs(args.save, exist_ok=True)
                OmegaConf.save(config=config, f=os.path.join(args.save, 'training_config.yaml'))
        # batch['num_frames'].to(torch.float32)
        # batch['fps'].to(torch.float32)
    else:
        batch = {'mp4': None, 'fps': None, 'num_frames': None, 'txt': None}

    batch['global_step'] = args.iteration

    # broad_cast_batch(batch)
    loss, loss_dict = model.shared_step(batch)
    return loss, loss_dict

if __name__ == '__main__':
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    data_class = get_obj_from_str(args.data_config["target"])
    create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])

    import yaml
    configs = []
    for config in args.base:
        with open(config, 'r') as f:
            base_config = yaml.safe_load(f)
        configs.append(base_config)
    args.log_config = configs

    training_main(args, model_cls=SATVideoDiffusionEngine,
        forward_step_function=partial(forward_step, data_class=data_class), 
        forward_step_eval=partial(forward_step_eval, data_class=data_class, only_log_video_latents=args.only_log_video_latents),
        create_dataset_function=create_dataset_function, collate_fn=data_collator)
    