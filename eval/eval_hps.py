import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TT
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
import io
import sys
import json
import time
import random
import requests
import click
from tqdm import tqdm
from functools import partial
from clint.textui import progress
from typing import Union
import webdataset as wds

environ_root = os.environ.get('HPS_ROOT')
root_path = os.path.expanduser('~/.cache/hpsv2') if environ_root == None else environ_root

model_dict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            '/zhipu-data/tjy/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val


@click.command()
@click.option('--img_dir',                  help='Path to the images', metavar='PATH|ZIP', type=str, required=True)
@click.option('--txt_path',                  help='Path to the text', metavar='PATH|ZIP', type=str, required=True)
@click.option('--batch_size',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=64, show_default=True)

def main(img_dir, txt_path, batch_size, cp="/zhipu-data/tjy/HPS_v2_compressed.pt", device=torch.device('cuda')):
    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if cp == os.path.join(root_path, 'HPS_v2_compressed.pt') and not os.path.exists(cp):
        print('Downloading HPS_v2_compressed.pt ...')
        url = 'https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
        r = requests.get(url, stream=True)
        with open(os.path.join(root_path,'HPS_v2_compressed.pt'), 'wb') as HPSv2:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    HPSv2.write(chunk)
                    HPSv2.flush()
        print('Download HPS_2_compressed.pt to {} sucessfully.'.format(root_path+'/'))

    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    dataset = CustomDataset(img_dir, txt_path, preprocess_val)
    it = iter(DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=16, prefetch_factor=6))
    total_num = 0
    total_hps = 0
    with torch.no_grad():
        for data in tqdm(it):
            images, texts = data
            total_num += len(texts)
            images = images.to(device=device, non_blocking=True)
            # Process the prompt
            texts = tokenizer(texts).to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images, texts)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                hps = (image_features * text_features).sum()
                total_hps += hps
    
    average_hps = total_hps / total_num
    print(f'Average HPS: {average_hps}')


if __name__ == "__main__":
    main()
