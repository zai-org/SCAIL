import io
import re
import os
import sys
import json
import numpy as np
from PIL import Image
from functools import partial
import math
import tarfile
from braceexpand import braceexpand
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import torchvision.transforms as TT

from sat import mpu
from sat.data_utils.webds import MetaDistributedWebDataset
import webdataset as wds
from webdataset import ResampledShards, DataPipeline
from webdataset.utils import pytorch_worker_seed
from webdataset.filters import pipelinefilter
from webdataset.tariterators import url_opener, group_by_keys
from webdataset.handlers import reraise_exception


def process_fn_sdxl(src, image_size, interpolation, transform):
    # while True:
    #     arr = torch.randn(3, image_size, image_size)
    #     h = w = image_size
    #     top = left = 0
    #     txt = ''

    #     item = {
    #         'jpg': arr,
    #         'txt': txt,
    #         'original_size_as_tuple': torch.tensor([h, w]),
    #         'crop_coords_top_left': torch.tensor([top, left]),
    #         'target_size_as_tuple': torch.tensor([image_size, image_size])
    #     }
    #     yield item

    for r in src:
        # read Image
        if ('png' not in r and 'jpg' not in r):
            continue

        img_bytes = r['png'] if 'png' in r else r['jpg']
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        h, w = img.size
        if h < image_size or w < image_size:
            continue

        # # resize
        # arr = TT.Resize(size=image_size, interpolation=interpolation)
        # arr = TT.ToTensor(arr)
        arr = transform(img)

        delta_h = arr.shape[1] - image_size
        delta_w = arr.shape[2] - image_size
        assert not all(
            [delta_h, delta_w]
        )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
        arr = TT.functional.crop(
            arr, top=top, left=left, height=image_size, width=image_size
        )

        # rescale
        arr = arr * 2 - 1

        txt = r['txt']
        if isinstance(txt, bytes):
            txt = txt.decode('utf-8')
        else:
            txt = str(txt)

        item = {
            'jpg': arr,
            'txt': txt,
            'original_size_as_tuple': torch.tensor([h, w]),
            'crop_coords_top_left': torch.tensor([top, left]),
            'target_size_as_tuple': torch.tensor([image_size, image_size])
        }
        yield item

class SDXLText2ImageWebDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        interpolation,
        nshards=sys.maxsize,
        shuffle_buffer=1000,
        include_dirs=None,
        **kwargs
    ):
        seed = int(os.environ.get("PL_GLOBAL_SEED", '0'))
        meta_names = []

        chained_trainsforms = []
        chained_trainsforms.append(TT.Resize(size=image_size, interpolation=interpolation))
        chained_trainsforms.append(TT.ToTensor())
        chained_trainsforms = TT.Compose(chained_trainsforms)

        super().__init__(
            path,
            partial(process_fn_sdxl, image_size=image_size, interpolation=interpolation, transform=chained_trainsforms),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs
        )

    @classmethod
    def create_dataset_function(cls, path, args, image_size, interpolation, **kwargs):
        path, include_dirs = path.split(';', 1)
        if len(include_dirs) == 0:
            include_dirs = None

        return cls(path, image_size=image_size, interpolation=interpolation, include_dirs=include_dirs)

def worker_seed_sat(group=None, seed=0):
    return pytorch_worker_seed(group=group) + seed * 23

class ConfiguredResampledShards(ResampledShards):
    def __init__(self, urls, seed, nshards=sys.maxsize, deterministic=True):
        from sat.mpu import get_data_parallel_group
        try:
            group = get_data_parallel_group()
        except AssertionError:
            group = None
        worker_seed_sat_this = partial(worker_seed_sat, group=group, seed=seed)
        super().__init__(urls, nshards, worker_seed_sat_this, deterministic)

def tar_file_iterator_with_recap(fileobj, recap_dir, skip_meta=r"__[^/]*__($|/)", suffix=None,handler=reraise_exception):
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    data_dir, filename = fileobj.name.rsplit('/', 1)

    recap_data = {}
    recap_name = filename.split('.')[0] + '_recap.jsonl'
    recap_path = os.path.join(recap_dir, recap_name)
    with open(recap_path, 'r') as recap_file:
        recap_list = []
        for lineno, line in enumerate(recap_file):
            try:
                recap_list.append(json.loads(line))
            except Exception as exn:
                from sat.helpers import print_rank0
                print_rank0(f'Error in loading jsonl {recap_name}, lineno {lineno}: {line}', level='DEBUG')
                continue
            
        for item in recap_list:
            recap_data[item['key']] = {
                'recaption': item['recaption']
            }

    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith("__")
                and fname.endswith("__")
            ):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            if fname.endswith('.txt') and suffix is not None:
                data = (stream.extractfile(tarinfo).read().decode() + suffix).encode()
            else:
                data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result

            if fname.endswith('.id'):
                fid = fname.split('.')[0]
                recap_fname = fid + '.recaption'
                recap = recap_data[fid]['recaption']
                yield dict(fname=recap_fname, data=recap)
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream

def tar_file_expander_with_recap(data, recap_dir, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator_with_recap(source["stream"], recap_dir):
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break

def tarfile_samples_with_recap(src, recap_dir, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander_with_recap(streams, recap_dir, handler)
    samples = group_by_keys(files, handler=handler)
    return samples  

def process_fn_recap(src, image_size, interpolation, transform, recap_ratio):
    for r in src:
        # read Image
        if ('png' not in r and 'jpg' not in r):
            continue

        img_bytes = r['png'] if 'png' in r else r['jpg']
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        h, w = img.size
        if h < image_size or w < image_size:
            continue

        # # resize
        # arr = TT.Resize(size=image_size, interpolation=interpolation)
        # arr = TT.ToTensor(arr)
        arr = transform(img)

        delta_h = arr.shape[1] - image_size
        delta_w = arr.shape[2] - image_size
        assert not all(
            [delta_h, delta_w]
        )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
        # top = np.random.randint(0, delta_h + 1)
        # left = np.random.randint(0, delta_w + 1)
        top, left = delta_h // 2, delta_w // 2 # try only centercrop
        arr = TT.functional.crop(
            arr, top=top, left=left, height=image_size, width=image_size
        )

        # rescale
        arr = arr * 2 - 1

        if np.random.random() > recap_ratio:
            txt = r['txt']
        else:
            txt = r['recaption']
        if isinstance(txt, bytes):
            txt = txt.decode('utf-8')
        else:
            txt = str(txt)

        item = {
            'jpg': arr,
            'txt': txt,
            # 'original_size_as_tuple': torch.tensor([h, w]),
            'original_size_as_tuple': torch.tensor([image_size, image_size]), # fit to only target size
            # 'crop_coords_top_left': torch.tensor([top, left]),
            'crop_coords_top_left': torch.tensor([0, 0]),
            'target_size_as_tuple': torch.tensor([image_size, image_size])
        }
        yield item

class RecapWebDataset(DataPipeline):
    def __init__(
        self, 
        path, 
        recap_dir, 
        image_size,
        interpolation,
        recap_ratio=0.95,
        nshards=sys.maxsize, 
        shuffle_buffer=1000, 
        include_dirs=None
    ):
        seed = int(os.environ.get("PL_GLOBAL_SEED", '0'))
        # os.environ['WDS_SHOW_SEED'] = '1'
        if include_dirs is not None: # /webdatasets/A,/webdatasets/C
            other_paths = []
            include_dirs = include_dirs.split(',')
            for include_dir in include_dirs:
                if '*' in include_dir:
                    include_dir, n = include_dir.split('*')
                    n = int(n)
                else:
                    n = 1
                for cur_dir, dirs, files in os.walk(include_dir):
                    for f in files:
                        if f.endswith('tar') and os.path.getsize(os.path.join(cur_dir,f)) > 0:
                            # other_paths.append(os.path.join(cur_dir,f))
                            other_paths.extend([os.path.join(cur_dir,f)]*n)
            # print(f'Adding dataset paths {",".join(other_paths)}')
            if len(path) > 0: # not "" 
                path = list(braceexpand(path)) + other_paths
            else:
                path = other_paths
        
        tarfile_samples = partial(tarfile_samples_with_recap, recap_dir=recap_dir)
        tarfile_to_samples = pipelinefilter(tarfile_samples)

        # if model parallel, shuffle_buffer should be 1 to disable shuffling
        try:
            from sat.mpu import get_model_parallel_world_size
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            pass

        chained_trainsforms = []
        chained_trainsforms.append(TT.Resize(size=image_size, interpolation=interpolation))
        chained_trainsforms.append(TT.ToTensor())
        chained_trainsforms = TT.Compose(chained_trainsforms)

        super().__init__(
            ConfiguredResampledShards(path, seed, nshards=nshards),
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            partial(process_fn_recap, image_size=image_size, interpolation=interpolation, transform=chained_trainsforms, recap_ratio=recap_ratio)
        )

    @classmethod
    def create_dataset_function(cls, path, args, recap_dir, image_size, interpolation, **kwargs):
        path, include_dirs = path.split(';', 1)
        if len(include_dirs) == 0:
            include_dirs = None

        return cls(path, recap_dir=recap_dir, image_size=image_size, interpolation=interpolation, include_dirs=include_dirs)
    


def tar_file_iterator_with_meta(fileobj, metas, skip_meta=r"__[^/]*__($|/)", suffix=None,handler=reraise_exception):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param meta_names: key of different items in meta file
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    data_dir, filename = fileobj.name.rsplit('/', 1)
    
    meta_names = []
    meta_data = {} # {id: {meta_name: meta_value, meta_name2: meta_value2, ...}}
    for meta in metas:
        meta_names.append(meta['key'])

        meta_dir = meta['dir']

        meta_dir_level = meta.get('dir_level', 1)
        sub_dirs = data_dir.split('/')[-meta_dir_level:][1:]

        meta_file_name = filename.split('.')[0] + meta['file_postfix']
        meta_path = os.path.join(meta_dir, *sub_dirs, meta_file_name)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as meta_file:
                if meta_path.endswith('.json'):
                    raw_meta_data = json.load(meta_file)
                elif meta_path.endswith('.jsonl'):
                    raw_meta_data = []
                    for line in meta_file:
                        try:
                            raw_meta_data.append(json.loads(line))
                        except:
                            continue
                else:
                    raise NotImplementedError
                
            for item in raw_meta_data:
                # item_key, label = item['key'], item[meta['key']]
                item_key = item['key']
                label = item.get(meta['key'], None)
                if not item_key in meta_data:
                    meta_data[item_key] = {}
                meta_data[item_key][meta['key']] = label
    
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith("__")
                and fname.endswith("__")
            ):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            if fname.endswith('.txt') and suffix is not None:
                data = (stream.extractfile(tarinfo).read().decode() + suffix).encode()
            else:
                data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            
            if fname.endswith('.id'):
                fid = fname.split('.')[0]
                meta_data_fid = meta_data.get(fid, {})
                for meta_name in meta_names:
                    meta_fname = fid + '.' + meta_name
                    meta = meta_data_fid.get(meta_name, None)
                    yield dict(fname=meta_fname, data=meta)
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream

def tar_file_expander_with_meta(data, metas, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator_with_meta(source["stream"], metas):
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break

def tarfile_samples_with_meta(src, metas, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander_with_meta(streams, metas, handler)
    samples = group_by_keys(files, handler=handler)
    return samples

class MetaWebDataset(DataPipeline):
    '''WebDataset with meta information files
    Extra Format:
        in webdataset (tar), for each sample there is a '.id'; 
        for each tar file, there is a '.meta.jsonl' file with the same name;
        The '.meta.jsonl' file contains lines of json objects, each with a 'key' field to match '.id'.
    '''
    def __init__(self, path, process_fn, seed, *, metas=[], nshards=sys.maxsize, shuffle_buffer=1000, include_dirs=None):
        if include_dirs is not None: # /webdatasets/A,/webdatasets/C
            other_paths = []
            include_dirs = include_dirs.split(',')
            for include_dir in include_dirs:
                if '*' in include_dir:
                    include_dir, n = include_dir.split('*')
                    n = int(n)
                else:
                    n = 1
                for cur_dir, dirs, files in os.walk(include_dir):
                    for f in files:
                        if f.endswith('tar') and os.path.getsize(os.path.join(cur_dir,f)) > 0:
                            # other_paths.append(os.path.join(cur_dir,f))
                            other_paths.extend([os.path.join(cur_dir,f)]*n)
            # print(f'Adding dataset paths {",".join(other_paths)}')
            if len(path) > 0: # not "" 
                path = list(braceexpand(path)) + other_paths
            else:
                path = other_paths

        tarfile_samples = partial(tarfile_samples_with_meta, metas=metas)
        tarfile_to_samples = pipelinefilter(tarfile_samples)

        # if model parallel, shuffle_buffer should be 1 to disable shuffling
        try:
            from sat.mpu import get_model_parallel_world_size
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            pass

        super().__init__(
            ConfiguredResampledShards(path, seed, nshards=nshards),
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            process_fn
        )

    
def process_fn_sft(src, image_size, interpolation, transform, filters, extra_texts):
    pass_num, total_num = 0, 0

    txt_keys, cum_sum, c_sum = [], [], 0
    for extra_text_item in extra_texts:
        key, prob = extra_text_item['key'], extra_text_item['prob']

        txt_keys.append(key)
        c_sum += prob
        cum_sum.append(c_sum)
    txt_keys.append('txt')
    cum_sum = np.array(cum_sum)

    for r in src:
        # read Image
        if ('png' not in r and 'jpg' not in r):
            continue

        # go through filters
        total_num += 1
        filter_flag = 0
        for filter in filters:
            key = filter['key']
            default_score = -float('inf') if filter["greater"] else float('inf')
            score = r.get(key, default_score) or default_score
            judge = (lambda a: a > filter["val"]) if filter["greater"] else (lambda a: a < filter["val"])
            if not judge(score):
                filter_flag = 1
                break
        if filter_flag:
            continue
        pass_num += 1
        # print(f'pick rate:', f"{int(pass_num/total_num*10000)/100}%", end='\r')

        img_bytes = r['png'] if 'png' in r else r['jpg']
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        w, h = img.size
        if w < image_size or h < image_size:
            continue

        arr = transform(img)
        delta_h = arr.shape[1] - image_size
        delta_w = arr.shape[2] - image_size
        assert not all(
            [delta_h, delta_w]
        )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
        top, left = delta_h // 2, delta_w // 2 # try only centercrop
        arr = TT.functional.crop(
            arr, top=top, left=left, height=image_size, width=image_size
        )

        # rescale
        arr = arr * 2 - 1

        # pick text
        idx = ((cum_sum - np.random.random()) < 0).sum()
        txt = r[txt_keys[idx]]
        if txt is None:
            txt = r['txt']
        if isinstance(txt, bytes):
            txt = txt.decode('utf-8')
        else:
            txt = str(txt)

        item = {
            'jpg': arr,
            'txt': txt,
            # 'original_size_as_tuple': torch.tensor([h, w]),
            'original_size_as_tuple': torch.tensor([image_size, image_size]), # fit to only target size
            # 'crop_coords_top_left': torch.tensor([top, left]),
            'crop_coords_top_left': torch.tensor([0, 0]),
            'target_size_as_tuple': torch.tensor([image_size, image_size])
        }
        yield item


class SFTDataset(MetaWebDataset):
    def __init__(
        self,
        path,
        image_size,
        interpolation,
        filters=[],
        extra_texts=[],
        nshards=sys.maxsize, 
        shuffle_buffer=1000, 
        include_dirs=None
    ):
        seed = int(os.environ.get("PL_GLOBAL_SEED", '0'))
        metas = filters + extra_texts

        chained_trainsforms = []
        chained_trainsforms.append(TT.Resize(size=image_size, interpolation=interpolation))
        chained_trainsforms.append(TT.ToTensor())
        chained_trainsforms = TT.Compose(chained_trainsforms)

        super().__init__(
            path,
            partial(process_fn_sft, image_size=image_size, interpolation=interpolation, transform=chained_trainsforms, filters=filters, extra_texts=extra_texts),
            seed,
            metas=metas,
            nshards=nshards,
            shuffle_buffer=shuffle_buffer,
            include_dirs=include_dirs
        )

    @classmethod
    def create_dataset_function(cls, path, args, image_size, interpolation, filters, extra_texts, **kwargs):
        path, include_dirs = path.split(';', 1)
        if len(include_dirs) == 0:
            include_dirs = None

        return cls(path, image_size= image_size, interpolation=interpolation, include_dirs=include_dirs, filters=filters, extra_texts=extra_texts)
    

def process_fn_origin(src, image_size, interpolation, transform, filters, extra_texts, shape_filter='all', reshape_mode='random', shape_cond='origin'):
    pass_num, total_num = 0, 0

    txt_keys, cum_sum, c_sum = [], [], 0
    for extra_text_item in extra_texts:
        key, prob = extra_text_item['key'], extra_text_item['prob']

        txt_keys.append(key)
        c_sum += prob
        cum_sum.append(c_sum)
    txt_keys.append('txt')
    cum_sum = np.array(cum_sum)

    assert shape_filter in ['all', 'bigger']
    assert reshape_mode in ['random', 'none', 'center']
    assert shape_cond in ['origin', 'resized']

    for r in src:
        # read Image
        if ('png' not in r and 'jpg' not in r):
            continue

        # go through filters
        total_num += 1
        filter_flag = 0
        for filter in filters:
            key = filter['key']
            default_score = -float('inf') if filter["greater"] else float('inf')
            score = r.get(key, default_score) or default_score
            judge = (lambda a: a > filter["val"]) if filter["greater"] else (lambda a: a < filter["val"])
            if not judge(score):
                filter_flag = 1
                break
        if filter_flag:
            continue
        pass_num += 1
        # print(f'pick rate:', f"{int(pass_num/total_num*10000)/100}%", end='\r')

        img_bytes = r['png'] if 'png' in r else r['jpg']
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        w, h = img.size
        if shape_filter == 'all':
            pass
        elif shape_filter == 'bigger':
            if w < image_size or h < image_size:
                continue
        else:
            raise NotImplementedError

        arr = transform(img)
        resized_h, resized_w = arr.shape[1], arr.shape[2]
        delta_h = arr.shape[1] - image_size
        delta_w = arr.shape[2] - image_size
        # assert not all(
        #     [delta_h, delta_w]
        # )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
        
        if reshape_mode == 'random' or reshape_mode == 'none':
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == 'center':
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(
            arr, top=top, left=left, height=image_size, width=image_size
        )

        # rescale
        arr = arr * 2 - 1

        # pick text
        idx = ((cum_sum - np.random.random()) < 0).sum()
        txt = r[txt_keys[idx]]
        if txt is None:
            txt = r['txt']
        if isinstance(txt, bytes):
            txt = txt.decode('utf-8')
        else:
            txt = str(txt)
        if txt.startswith('GeneratedText'):
            start_idx = len('GeneratedText(text="')
            txt = txt[start_idx:]
            end_idx = txt.find('generated_tokens=')
            txt = txt[:end_idx-3]

        if shape_cond == 'origin':
            size_h, size_w = h, w
        elif shape_cond == 'resized':
            size_h, size_w = resized_h, resized_w
        else:
            raise NotImplementedError

        item = {
            'jpg': arr,
            'txt': txt,
            'original_size_as_tuple': torch.tensor([size_h, size_w]),
            'crop_coords_top_left': torch.tensor([top, left]),
            'target_size_as_tuple': torch.tensor([image_size, image_size])
        }
        yield item


class MultiMetaWebDataset(MetaWebDataset):
    def __init__(
        self,
        path,
        image_size,
        interpolation,
        filters=[],
        extra_texts=[],
        nshards=sys.maxsize, 
        shuffle_buffer=1000, 
        include_dirs=None,
        shape_filter='all',
        reshape_mode='random',
        shape_cond='origin',
    ):
        seed = int(os.environ.get("PL_GLOBAL_SEED", '0'))
        metas = filters + extra_texts

        chained_trainsforms = []
        if reshape_mode != 'none':
            chained_trainsforms.append(TT.Resize(size=image_size, interpolation=interpolation))
        chained_trainsforms.append(TT.ToTensor())
        chained_trainsforms = TT.Compose(chained_trainsforms)

        super().__init__(
            path,
            partial(
                process_fn_origin, 
                image_size=image_size, 
                interpolation=interpolation, 
                transform=chained_trainsforms, 
                filters=filters, 
                extra_texts=extra_texts,
                shape_filter=shape_filter,
                reshape_mode=reshape_mode,
                shape_cond=shape_cond),
            seed,
            metas=metas,
            nshards=nshards,
            shuffle_buffer=shuffle_buffer,
            include_dirs=include_dirs
        )

    @classmethod
    def create_dataset_function(cls, idx, args, image_size, interpolation, ds_infos, **kwargs):
        ds_info = ds_infos[idx]
        path = ds_info["path"]
        include_dirs = ds_info["include_dirs"]
        filters = ds_info["filters"]
        extra_texts = ds_info["extra_texts"]

        if not include_dirs is None and "{" in include_dirs and "}" in include_dirs:
            include_dirs = list(braceexpand(include_dirs))
            include_dirs = ','.join(include_dirs)

        return cls(path, image_size=image_size, interpolation=interpolation, include_dirs=include_dirs, filters=filters, extra_texts=extra_texts, **kwargs)
