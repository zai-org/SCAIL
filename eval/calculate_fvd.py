import numpy as np
import torch
from tqdm.std import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
import os
import json
from decord import VideoReader, cpu
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from torchvision.transforms import Lambda, Compose
from torchvision.transforms._transforms_video import CenterCropVideo
from pytorchvideo.transforms import ShortSideScale
import scipy
from fvd.videogpt.fvd import load_i3d_pretrained as load1
from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats1
from fvd.styleganv.fvd import load_i3d_pretrained as load2
from fvd.styleganv.fvd import get_fvd_feats as get_fvd_feats2


VIDEOGPT_I3DPATH = "/workspace/ckpt/lsz/models/fvd/videogpt/i3d_pretrained_400.pt"
STYLEGANV_I3DPATH = "/workspace/ckpt/lsz/models/fvd/styleganv/i3d_torchscript.pt"


class VideoDataset(Dataset):
    def __init__(self,
                 video_dir,
                 num_frames,
                 sample_rate=1,
                 crop_size=None,
                 resolution=128,
                 ) -> None:
        super().__init__()
        self.video_files = self._combine_without_prefix(video_dir)
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        video_file = self.video_files[index]
        video_tensor = self._load_video(video_file)
        return video_tensor

    def _load_video(self, video_path):
        num_frames = self.num_frames
        sample_rate = self.sample_rate
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames >= sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            s = 0
            e = total_frames
            num_frames = int(total_frames / sample_frames_len * num_frames)
            print(f'sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}', video_path,
                  total_frames)

        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        video_data = _preprocess(video_data, short_size=self.short_size, crop_size=self.crop_size)
        return trans(video_data)

    def _combine_without_prefix(self, folder_path, prefix='.', sort_by_idx=False):
        folder = []
        for root, dirname, filename in os.walk(folder_path):
            for file in filename:
                if file.endswith('.mp4'):
                    file_path = os.path.join(root, file)
                    folder.append(file_path)
        folder.sort()
        return folder


def _preprocess(video_data, short_size=224, crop_size=224):
    transform = Compose(
        [
            Lambda(lambda x: x / 255.0),
            ShortSideScale(size=short_size),
            CenterCropVideo(crop_size=crop_size),
        ]
    )
    video_outputs = transform(video_data)
    # video_outputs = torch.unsqueeze(video_outputs, 0) # (bz,c,t,h,w)
    return video_outputs


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 3, 1, 1)

    # permute TCHW -> CTHW
    x = x.transpose(1, 0)

    return x


def extract_and_save_feat_stats(video_dir, save_path, num_frames=81):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    print('Create dataloader')
    dataset = VideoDataset(video_dir, num_frames=num_frames, crop_size=224, resolution=224)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    print('Load i3d models')
    model1 = load1(device, VIDEOGPT_I3DPATH)
    model2 = load2(device, STYLEGANV_I3DPATH)
    models = [model1, model2]
    feature_extract_funcs = [get_fvd_feats1, get_fvd_feats2]
    features = [[], []]
    for vids in tqdm(dataloader):
        for i3d_model, get_fvd_feats, feature_pool in zip(models, feature_extract_funcs, features):
            feats = get_fvd_feats(vids, i3d=i3d_model, device=device, bs=1)
            # 若包含多帧/时域, 池化成单向量
            if len(feats.shape) > 2:
                feats = feats.mean(dim=1)
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
            feature_pool.append(feats)
    features[0] = np.concatenate(features[0], 0)
    features[1] = np.concatenate(features[1], 0)

    mu_videogpt = np.mean(features[0], axis=0)
    sigma_videogpt = np.cov(features[0], rowvar=False)
    mu_stylegan = np.mean(features[1], axis=0)
    sigma_stylegan = np.cov(features[1], rowvar=False)
    np.savez(save_path, mu_videogpt=mu_videogpt, sigma_videogpt=sigma_videogpt, mu_stylegan=mu_stylegan, sigma_stylegan=sigma_stylegan)
    print(f'Feature stats saved to {save_path}')


def calculate_fvd_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fvd = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fvd))


def fvd_api(video_dir, stat_path, num_frames=81):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    print('Create dataloader')
    dataset = VideoDataset(video_dir, num_frames=num_frames, crop_size=224, resolution=224)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    print('Load i3d models')
    model1 = load1(device, VIDEOGPT_I3DPATH)
    model2 = load2(device, STYLEGANV_I3DPATH)
    models = [model1, model2]
    feature_extract_funcs = [get_fvd_feats1, get_fvd_feats2]
    features = [[], []]
    for vids in tqdm(dataloader):
        for i3d_model, get_fvd_feats, feature_pool in zip(models, feature_extract_funcs, features):
            feats = get_fvd_feats(vids, i3d=i3d_model, device=device, bs=1)
            # 若包含多帧/时域, 池化成单向量
            if len(feats.shape) > 2:
                feats = feats.mean(dim=1)
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
            feature_pool.append(feats)
    features[0] = np.concatenate(features[0], 0)
    features[1] = np.concatenate(features[1], 0)

    mu_videogpt = np.mean(features[0], axis=0)
    sigma_videogpt = np.cov(features[0], rowvar=False)
    mu_stylegan = np.mean(features[1], axis=0)
    sigma_stylegan = np.cov(features[1], rowvar=False)

    stat = np.load(stat_path)
    mu_videogpt_real, sigma_videogpt_real = stat['mu_videogpt'], stat['sigma_videogpt']
    mu_stylegan_real, sigma_stylegan_real = stat['mu_stylegan'], stat['sigma_stylegan']
    print('Calculate fvds...')
    fvd_videogpt = calculate_fvd_from_inception_stats(mu_videogpt, sigma_videogpt, mu_videogpt_real, sigma_videogpt_real)
    fvd_stylegan = calculate_fvd_from_inception_stats(mu_stylegan, sigma_stylegan, mu_stylegan_real, sigma_stylegan_real)

    print('Average fvd_videogpt: ', np.mean(fvd_videogpt))
    print('Average fvd_stylegan: ', np.mean(fvd_stylegan))

    # write into file
    save_json_file = video_dir + '/results.json'
    if os.path.exists(save_json_file):
        with open(save_json_file, 'r') as json_f:
            metrics = json.load(json_f)
    elif os.path.exists(save_json_file.replace('results.json', 'results_0.json')):
        result_jsons = [video_dir+'/'+f for f in os.listdir(video_dir) if f.endswith('.json')]
        result_jsons.sort()
        metrics = {"all_rewards": [], "average": {}}
        for json_path in result_jsons:
            with open(json_path, 'r') as json_f:
                metrics["all_rewards"].extend(json.load(json_f)["all_rewards"])
        metrics["average"]["average_reward"] = sum(metrics["all_rewards"]) / len(metrics["all_rewards"])
    else:
        metrics = {"average": {}}
    with open(save_json_file, 'w') as json_f:
        metrics["average"]["styleganv_fvd"] = round(fvd_stylegan, 4)
        metrics["average"]["videogpt_fvd"] = round(fvd_videogpt, 4)
        json.dump(metrics, json_f, indent=2, ensure_ascii=False)

    # remove reward results
    if os.path.exists(save_json_file.replace('results.json', 'results_0.json')):
        for json_f in result_jsons:
            os.remove(json_f)

    return fvd_videogpt, fvd_stylegan


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--precalc_real_stats', action='store_true', help='预先计算真实视频统计并保存npy')
    parser.add_argument('--real_video_dir', type=str,
                        help=('the path of real videos`'))
    parser.add_argument('--generated_video_dir', type=str,
                        help=('the path of generated videos`'))
    parser.add_argument('--real_stats_path', type=str, default='/workspace/ckpt/lsz/datasets/cogvideo_eval/i3d_stats.npz')
    parser.add_argument('--num_frames', type=int, default=81)
    args = parser.parse_args()

    if args.precalc_real_stats:
        extract_and_save_feat_stats(args.real_video_dir, args.real_stats_path, num_frames=args.num_frames)
    else:
        fvd_videogpt, fvd_stylegan = fvd_api(args.generated_video_dir, args.real_stats_path)

