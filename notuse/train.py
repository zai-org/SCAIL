import os
import sys
import argparse
from functools import partial
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from typing import Dict, List
from omegaconf import OmegaConf

import torch
from torch import nn
import torch.nn.functional as F

from sat.training.deepspeed_training import training_main

from sgm.util import get_obj_from_str, isheatmap, exists

from diffusion import SATDiffusionEngine
from arguments import get_args, process_config_to_args

def save_texts(texts, save_dir, iterations):
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

def forward_step_eval(data_iterator, model, args, timers, data_class=None):
    timers('data loader').start()
    batch = next(data_iterator)
    timers('data loader').stop()

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()

    if torch.distributed.get_rank() == 0:
        if not os.path.exists(os.path.join(args.save, 'training_config.yaml')):
            configs = [OmegaConf.load(cfg) for cfg in args.base]
            config = OmegaConf.merge(*configs)
            os.makedirs(args.save, exist_ok=True)
            OmegaConf.save(config=config, f=os.path.join(args.save, 'training_config.yaml'))

        texts = batch['txt']
        text_save_dir = os.path.join(args.save, "texts")
        os.makedirs(text_save_dir, exist_ok=True)
        save_texts(texts, text_save_dir, args.iteration)

        gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),  # torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            images = model.log_images(batch)
        for k in images:
            N = images[k].shape[0]
            if not isheatmap(images[k]):
                images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().float().cpu()
                if not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)
        root = os.path.join(args.save, "images")
        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}.png".format(
                    k, args.iteration
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
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

    batch['global_step'] = args.iteration
    loss, loss_dict = model.shared_step(batch)
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)
    return loss, loss_dict

def forward_step(data_iterator, model, args, timers, data_class=None):
    timers('data loader').start()
    batch = next(data_iterator)
    timers('data loader').stop()

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()

    batch['global_step'] = args.iteration

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

    training_main(args, model_cls=SATDiffusionEngine,
        forward_step_function=partial(forward_step, data_class=data_class), 
        forward_step_eval=partial(forward_step_eval, data_class=data_class),
        create_dataset_function=create_dataset_function)
    