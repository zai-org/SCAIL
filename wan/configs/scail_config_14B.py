import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ SCAIL 14B ------------------------#

scail_14B = EasyDict(__name__='Config: SCAIL 14B')
scail_14B.update(wan_shared_cfg)
scail_14B.sample_neg_prompt = ""

scail_14B.t5_checkpoint = 'umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth'
scail_14B.t5_tokenizer = 'umt5-xxl'

# clip
scail_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
scail_14B.clip_dtype = torch.float16
scail_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth'
scail_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
scail_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
scail_14B.vae_stride = (4, 8, 8)

# transformer
scail_14B.patch_size = (1, 2, 2)
scail_14B.dim = 5120
scail_14B.ffn_dim = 13824
scail_14B.freq_dim = 256
scail_14B.num_heads = 40
scail_14B.num_layers = 40
scail_14B.window_size = (-1, -1)
scail_14B.qk_norm = True
scail_14B.cross_attn_norm = True
scail_14B.eps = 1e-6
