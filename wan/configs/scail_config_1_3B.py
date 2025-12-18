import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ SCAIL 1.3B ------------------------#

scail_1_3B = EasyDict(__name__='Config: SCAIL 1.3B')
scail_1_3B.update(wan_shared_cfg)
scail_1_3B.sample_neg_prompt = ""

scail_1_3B.t5_checkpoint = 'umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth'
scail_1_3B.t5_tokenizer = 'umt5-xxl'

# clip
scail_1_3B.clip_model = 'clip_xlm_roberta_vit_h_14'
scail_1_3B.clip_dtype = torch.float16
scail_1_3B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth'
scail_1_3B.clip_tokenizer = 'xlm-roberta-large'

# vae
scail_1_3B.vae_checkpoint = 'Wan2.1_VAE.pth'
scail_1_3B.vae_stride = (4, 8, 8)

# transformer
scail_1_3B.patch_size = (1, 2, 2)
scail_1_3B.dim = 1536
scail_1_3B.ffn_dim = 8960
scail_1_3B.freq_dim = 256
scail_1_3B.num_heads = 12
scail_1_3B.num_layers = 30
scail_1_3B.window_size = (-1, -1)
scail_1_3B.qk_norm = True
scail_1_3B.cross_attn_norm = True
scail_1_3B.eps = 1e-6
