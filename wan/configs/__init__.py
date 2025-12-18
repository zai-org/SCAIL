# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_14B import i2v_14B
from .wan_t2v_1_3B import t2v_1_3B
from .wan_t2v_14B import t2v_14B
from .scail_config_14B import scail_14B
from .scail_config_1_3B import scail_1_3B

# the config of t2i_14B is the same as t2v_14B
t2i_14B = copy.deepcopy(t2v_14B)
t2i_14B.__name__ = 'Config: Wan T2I 14B'

# the config of flf2v_14B is the same as i2v_14B
flf2v_14B = copy.deepcopy(i2v_14B)
flf2v_14B.__name__ = 'Config: Wan FLF2V 14B'
flf2v_14B.sample_neg_prompt = "镜头切换，" + flf2v_14B.sample_neg_prompt

WAN_CONFIGS = {
    't2v-14B': t2v_14B,
    't2v-1.3B': t2v_1_3B,
    'i2v-14B': i2v_14B,
    't2i-14B': t2i_14B,
    'flf2v-14B': flf2v_14B,
    'vace-1.3B': t2v_1_3B,
    'vace-14B': t2v_14B,
}

SCAIL_CONFIGS = {
    'SCAIL-14B': scail_14B,
    'SCAIL-1.3B': scail_1_3B,
}

SCAIL_CONFIG_PATHS = {
    'SCAIL-14B': 'configs/config-14b.json',
    'SCAIL-1.3B': 'configs/config-1.3b.json',
}