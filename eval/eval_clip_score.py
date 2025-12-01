import os
import tqdm
import pickle
import click
import json
import numpy as np
import scipy.linalg
from PIL import Image
import open_clip

import torch
from torch.utils.data.distributed import DistributedSampler

import dnnlib
from torch_utils import distributed as dist
import dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, txt_path, preprocess):
        self.images = []
        self.preprocess = preprocess
        with open(txt_path) as f:
            # self.text = json.load(f)
            self.text = f.readlines()

        for subdir in sorted(os.listdir(img_dir)):
            subdir_path = os.path.join(img_dir, subdir)
            if os.path.isdir(subdir_path):     
                images = sorted(os.listdir(subdir_path))
                for img in images:
                    if 'grid' in img:
                        continue
                    img_path = os.path.join(subdir_path, img)
                    if os.path.isfile(img_path):
                        self.images.append(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.preprocess(image)
        text = self.text[int((self.images[idx]).split('/')[-2].split('_')[0])]    
        return image, text
        
@click.command()
@click.option('--img_dir',                  help='Path to the images', metavar='PATH|ZIP', type=str, required=True)
@click.option('--txt_path',                  help='Path to the text', metavar='PATH|ZIP', type=str, required=True)
@click.option('--batch_size',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=64, show_default=True)

def main(img_dir, txt_path, batch_size, device=torch.device('cuda')):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    arch = "ViT-g-14"
    version = "/zhipu-data/tjy/CLIP-ViT-g-14-laion2B-s12B-b42K/open_clip_pytorch_model.bin"

    dist.print0('Loading CLIP model...')
    model, _, preprocess = open_clip.create_model_and_transforms(arch, device=rank, pretrained=version)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    dist.print0('Loading finished!')

    dataset = CustomDataset(img_dir, txt_path, preprocess)
    num_batches = ((len(dataset) - 1) // (batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=rank_batches, num_workers=3, prefetch_factor=2)

    dist.print0(f'Calculating statistics for {len(dataset)} images...')
    total_num = torch.tensor(0.0).to(device)
    total_clip_score = torch.tensor(0.0).to(device)

    for data in tqdm.tqdm(dataloader, disable=(dist.get_rank()!=0)):
        imgs, txts = data
        total_num += len(txts)

        with torch.no_grad():
            txt_tokens = tokenizer(txts).to(device)
            txt_features = model.encode_text(txt_tokens)
            txt_features /= txt_features.norm(dim=-1, keepdim=True)

            img_features = model.encode_image(imgs.to(device))
            img_features /= img_features.norm(dim=-1, keepdim=True)

            clip_score = (txt_features * img_features).sum()
            total_clip_score += clip_score

    torch.distributed.barrier()
    torch.distributed.reduce(total_clip_score, dst=0)
    torch.distributed.reduce(total_num, dst=0)
 
    if rank == 0:
        average_clip_score = total_clip_score / total_num
        print(f'Average CLIP Score: {average_clip_score.item()}')

if __name__ == "__main__":
    main()