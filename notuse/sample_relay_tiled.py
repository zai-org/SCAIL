import os
import sys
import math
import argparse
import json
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
from PIL import Image

import torch
import torch.nn.functional as functional
import numpy as np
from einops import rearrange, repeat
from torchvision.utils import make_grid
import torchvision.transforms as TT

from sgm.util import get_obj_from_str, isheatmap, exists

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint

from diffusion import SATDiffusionEngine
from arguments import get_args, process_config_to_args

from sample import read_from_file, get_batch, get_unique_embedder_keys_from_conditioner, perform_save_locally


def sampling_main_relay(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    # default input type txt
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)

    image_size = args.sampling_image_size
    final_size = args.final_size

    sample_func = model.sample_relay

    chained_trainsforms = []
    # chained_trainsforms.append(TT.Resize(size=image_size, interpolation=1))
    chained_trainsforms.append(TT.ToTensor())
    transform = TT.Compose(chained_trainsforms)

    assert args.input_dir is not None
    input_sample_dirs = os.listdir(args.input_dir)
    input_sample_dirs_and_rank = sorted([(int(name.split('_')[0]), name) for name in input_sample_dirs])
    input_sample_dirs = [os.path.join(args.input_dir, name) for _, name in input_sample_dirs_and_rank]

    H, W, C, F = final_size, final_size, 4, 8
    num_samples = [args.batch_size]
    force_uc_zero_embeddings = ['txt']
    with torch.no_grad():
        for text, cnt in tqdm(data_iter):
            input_sample_dir = input_sample_dirs[cnt]
            images = []
            for i in range(args.batch_size):
                filepath = os.path.join(input_sample_dir, f"{i:09}.png")
                image = Image.open(filepath).convert('RGB')
                image = transform(image) * 2 - 1
                images.append(image[None, ...])
            images = torch.cat(images, dim=0)
            # images = functional.interpolate(images, scale_factor=1/4, mode='bilinear', align_corners=False)
            images = functional.interpolate(images, scale_factor=4, mode='bilinear', align_corners=False)
            images = images.to(torch.float16).cuda()
            images = model.encode_first_stage(images)

            # text = "An extremely detailed 8k high resolution photo with best quality." + text
            text = 'masterpiece, best quality, highres, extremely detailed 8k wallpaper, very clear'
            negative_text = 'lowres, blurry, bad quality'
            value_dict = {
                'prompt': text,
                'negative_prompt': '',
                'original_size_as_tuple': (image_size, image_size),
                'target_size_as_tuple': (image_size, image_size),
                'orig_height': image_size,
                'orig_width': image_size,
                'target_height': image_size,
                'target_width': image_size,
                'crop_coords_top': 0,
                'crop_coords_left': 0
            }

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
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(
                        lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                    )

            samples_z = sample_func(
                images,
                c,
                uc = uc,
                batch_size = args.batch_size,
                shape = (C, H // F, W // F)
            )
            samples_x = model.decode_first_stage(samples_z).to(torch.float32)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            batch_size = samples.shape[0]
            assert (batch_size // args.grid_num_rows) * args.grid_num_rows == batch_size

            if args.batch_size == 1:
                grid = None
            else:
                grid = make_grid(samples, nrow=args.grid_num_rows)

            save_path = os.path.join(args.output_dir, str(cnt) + '_' + text.replace(' ', '_').replace('/', '')[:120])
            perform_save_locally(save_path, samples, grid)



if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    sampling_main_relay(args, model_cls=SATDiffusionEngine)