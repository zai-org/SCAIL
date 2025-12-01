import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
import os
from decord import VideoReader, cpu
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from torchvision.transforms import Lambda, Compose
from torchvision.transforms._transforms_video import CenterCropVideo
from pytorchvideo.transforms import ShortSideScale
from multiprocessing import Process, Queue
import glob


class VideoDataset(Dataset):
    def __init__(self,
                 real_video_dir,
                 generated_video_dir,
                 num_frames,
                 sample_rate = 1,
                 crop_size=None,
                 resolution=128,
                 ) -> None:
        super().__init__()
        if isinstance(real_video_dir, list):
            self.real_video_files = real_video_dir
            self.generated_video_files = generated_video_dir
        else:
            self.real_video_files = self._combine_without_prefix(real_video_dir)
            self.generated_video_files = self._combine_without_prefix(generated_video_dir)
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution


    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_video_file = self.real_video_files[index]
        generated_video_file = self.generated_video_files[index]
        real_video_tensor  = self._load_video(real_video_file)
        generated_video_tensor  = self._load_video(generated_video_file)
        return {'real': real_video_tensor, 'generated':generated_video_tensor }


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
        video_data = video_data.permute(0, 3, 1, 2) # (T, H, W, C) -> (T, C, H, W)
        return _preprocess(video_data, short_size=self.short_size, crop_size = self.crop_size)

    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        os.makedirs(folder_path, exist_ok=True)
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            if osp.isfile(osp.join(folder_path, name)):
                folder.append(osp.join(folder_path, name))
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
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4)

    return x

def calculate_fvd(videos1, videos2, device, frames_number, method='styleganv', model=None):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from fvd.videogpt.fvd import frechet_distance

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    i3d = model
    # i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]
# 3.5的不要
    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}

    # for calculate FVD, each clip_timestamp must >= 10

    # for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):

    for clip_timestamp in frames_number:
            # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, : clip_timestamp]
        videos_clip2 = videos2[:, :, : clip_timestamp]

        # get FVD features

        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD when timestamps[:clip]
        # breakpoint()

    return result

# test code / using example

def calculate_common_metric(real_videos_path, generated_videos_path, args, rank, queue):

    dataset = VideoDataset(real_videos_path,
                           generated_videos_path,
                           num_frames=args.num_frames,
                           crop_size=224,
                           resolution=224)


    dataloader = DataLoader(dataset, args.batch_size,
                            num_workers=args.num_workers, drop_last=False)

    from fvd.videogpt.fvd import load_i3d_pretrained as load
    from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    device = f"cuda:{rank%8}"

    model = load(device=device)
    for batch_data in tqdm(dataloader): # {'real': real_video_tensor, 'generated':generated_video_tensor }
        real_videos = batch_data['real']
        generated_videos = batch_data['generated']
        real_videos.requires_grad = False
        generated_videos.requires_grad = False
        assert real_videos.shape[2] == generated_videos.shape[2]

        videos1 = trans(real_videos)
        videos2 = trans(generated_videos)
        videos_clip1 = videos1[:, :, : args.num_frames]
        videos_clip2 = videos2[:, :, : args.num_frames]

        # get FVD features

        feats1 = get_fvd_feats(videos_clip1, i3d=model, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=model, device=device)
        queue.put([feats1.cpu(), feats2.cpu()])


if __name__ == "__main__":
    # main()
    # exit(1)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size to use')
    parser.add_argument('--real_video_dir', type=str,
                    help=('the path of real videos`'))
    parser.add_argument('--generated_video_dir', type=str,
                    help=('the path of generated videos`'))
    parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--num_workers', type=int, default=0,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--resolution', type=int, default=336)
    parser.add_argument('--crop_size', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=100)

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    real_videos = glob.glob(args.real_video_dir + '/*.mp4')
    generated_videos = glob.glob(args.generated_video_dir + '/*.mp4')
    real_videos.sort()
    generated_videos.sort()
    n_process = 8
    queue = Queue()
    process_list = []
    for i in range(n_process):
        tmp_real_videos = real_videos[i::n_process]
        tmp_generated_videos = generated_videos[i::n_process]
        process = Process(target=calculate_common_metric, args=(tmp_real_videos, tmp_generated_videos, args, i, queue))
        process.start()
        process_list.append(process)

    for p in process_list:
        p.join()

    results_feats1 = []
    results_feats2 = []
    for i in range(n_process):
        feats = queue.get()
        results_feats1.append(feats[0])
        results_feats2.append(feats[1])

    results_feats1 = torch.cat(results_feats1, dim=0)
    results_feats2 = torch.cat(results_feats2, dim=0)
    from fvd.videogpt.fvd import frechet_distance
    fvd_results = {}
    fvd_results[args.num_frames] = frechet_distance(results_feats1, results_feats2)

    result = {
        "value": fvd_results,
        "video_setting_name": "batch_size, channel, time, heigth, width",
    }
    print(result)
    breakpoint()


    # metric_score_1, metric_score_2 = calculate_common_metric(args, dataloader, device, 1)
    # print('fvd_videogpt: ', metric_score_1)
    # print('fvd_styleganv: ', metric_score_2)

    # dataloader = DataLoader(dataset, args.batch_size,
    #                         num_workers=args.num_workers)
    # metric_score_2 = calculate_common_metric(args, dataloader, device, 2)
    # print('fvd_styleganv: ', metric_score_2)
