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
import numpy as np
from einops import rearrange, repeat
from torchvision.utils import make_grid
import torchvision.transforms as TT

from sgm.util import get_obj_from_str, isheatmap, exists

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args, process_config_to_args

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input('Please input English text (Ctrl-D quit): ')
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


def perform_save_locally(save_path, samples, grid):
    os.makedirs(save_path, exist_ok=True)
    base_count = len(os.listdir(save_path))
    if len(samples.shape) == 5:
        samples = samples.squeeze(1)
    for sample in samples:
        sample = 255.0 * rearrange(sample.numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1

    if grid is not None:
        grid_base_count = len([item for item in os.listdir(save_path) if 'grid' in item])
        grid = 255.0 * rearrange(grid.numpy(), "c h w -> h w c")
        Image.fromarray(grid.astype(np.uint8)).save(
            os.path.join(save_path, f"grid_{grid_base_count:09}.png")
        )
    
def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    if args.input_type == 'cli':
        data_iter = read_from_cli()
    elif args.input_type == 'txt':
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError
    
    image_size = args.sampling_image_size

    if args.sdedit:
        sample_func = model.sample_sdedit

        chained_trainsforms = []
        chained_trainsforms.append(TT.Resize(size=image_size, interpolation=1))
        chained_trainsforms.append(TT.ToTensor())
        transform = TT.Compose(chained_trainsforms)
    else:
        sample_func = model.sample
    
    T, H, W, C, F = 1, image_size[0], image_size[1], 4, 8
    num_samples = [args.batch_size]
    force_uc_zero_embeddings = ['txt']
    with torch.no_grad():
        for text, cnt in tqdm(data_iter):
            if args.sdedit:
                if text.count('@') == 2:
                    text, image_path, edit_ratio = text.split('@')
                    edit_ratio = float(edit_ratio)
                else:
                    text, image_path = text.split('@')
                    edit_ratio = None
                image = Image.open(image_path).convert('RGB')
                image = transform(image) * 2 - 1

                delta_h = image.shape[1] - image_size
                delta_w = image.shape[2] - image_size
                top = (delta_h + 1) // 2
                left = (delta_w + 1) // 2
                image = TT.functional.crop(
                    image, top=top, left=left, height=image_size, width=image_size
                )
                image = image[None, ...].to(torch.bfloat16).cuda()
                image = model.encode_first_stage(image)

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

            if args.sdedit:
                samples_z = sample_func(
                    image,
                    c,
                    uc = uc,
                    edit_ratio=edit_ratio,
                    batch_size = args.batch_size,
                    shape = (C, H // F, W // F)
                )
            else:
                samples_z = sample_func(
                    c,
                    uc = uc,
                    batch_size = args.batch_size,
                    shape = (T, C, H // F, W // F)
                )
            b, t = samples_z.shape[:2]
            samples_z = samples_z.view(-1, *samples_z.shape[2:])
            samples_x = model.decode_first_stage(samples_z).to(torch.float32)
            samples_x = samples_x.view(b, t, *samples_x.shape[1:])
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            batch_size = samples.shape[0]
            assert (batch_size // args.grid_num_rows) * args.grid_num_rows == batch_size

            # grid = torch.stack([samples])
            # grid = rearrange(grid, "n b c h w -> c (n h) (b w)")
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

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
