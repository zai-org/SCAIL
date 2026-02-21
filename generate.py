# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import SCAIL_CONFIGS, SCAIL_CONFIG_PATHS
from wan.utils.utils import cache_video, str2bool
from wan.utils.scail_utils import load_image_to_tensor_chw_normalized, load_video_for_pose_sample, resize_for_rectangle_crop


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.pose is not None, "Please specify the pose video."
    assert args.image is not None, "Please specify the reference image."
    assert str(args.model).upper() in SCAIL_CONFIGS

    args.model = str(args.model).upper()

    if args.scail_config_path is None:
        args.scail_config_path = SCAIL_CONFIG_PATHS[args.model]

    if args.sample_steps is None:
        args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 3.0

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="SCAIL-14B",
        help="Type of SCAIL model. Choices: [SCAIL-14B, SCAIL-1.3B]")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./SCAIL-Preview/",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The reference image to generate the video from.")
    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help="The rendered pose video to generate the video from.")
    parser.add_argument(
        "--target_h",
        type=int,
        default=512,
        help="The target height of the generated video.")
    parser.add_argument(
        "--target_w",
        type=int,
        default=896,
        help="The target width of the generated video.")
    parser.add_argument(
        "--scail_path",
        type=str, 
        default=None,
        help="Path to converted SCAIL.safetensors")
    parser.add_argument(
        "--scail_config_path",
        type=str,
        default=None,
        help="Path to config.json of SCAIL")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", 
        type=int, 
        default=None, 
        help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to safetensors of LoRA."
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="Strength of LoRA. Default: 1.0"
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = SCAIL_CONFIGS[args.model]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if args.prompt is None:
        args.prompt = ""
    logging.info(f"Input prompt: {args.prompt}")
    logging.info(f"Input image: {args.image}")

    img = Image.open(args.image).convert("RGB")

    target_h = args.target_h
    target_w = args.target_w

    logging.info(f"Input pose video: {args.pose}")
    pose_video = load_video_for_pose_sample(args.pose) # t h w c
    pose_video = pose_video.permute(0, 3, 1, 2)  # t c h w
    pose_video = resize_for_rectangle_crop(pose_video, (target_h, target_w), reshape_mode="center")
    pose_video = (pose_video - 127.5) / 127.5  # -1 1

    img_uncropped = load_image_to_tensor_chw_normalized(img).to(device)  # 1 c h w, -1 to 1
    img = resize_for_rectangle_crop(img_uncropped, (target_h, target_w), reshape_mode="center")
    img = img.squeeze(0)  # c h w, -1, 1
    
    logging.info("Creating SCAIL pipeline.")
    scail_pipeline = wan.SCAILPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        scail_safetensors_path=args.scail_path,
        scail_config_path=args.scail_config_path,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        lora_path=args.lora_path,
        lora_alpha=args.lora_alpha,        
    )

    logging.info("Generating video ...")
    video = scail_pipeline.generate(
        args.prompt,
        img,
        pose_video=pose_video,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"SCAIL_{args.target_w}{'x' if sys.platform=='win32' else '*'}{args.target_h}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        cache_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
