
import torch
from safetensors.torch import save_file
from typing import Dict
import os

class ModuleParser():
    def __init__(self, key: str):
        self.key = key
        self.modules = key.split('.')
        self.idx = 0
        
    def match(self, pattern: str) -> bool:
        patterns = pattern.split('.')
        for j, p in enumerate(patterns):
            if self.idx + j < len(self.modules) and self.modules[self.idx+j] == p:
                continue
            else:
                return False
        self.idx += len(patterns)
        return True
    
    def step(self) -> str:
        m = self.modules[self.idx]
        self.idx += 1
        return m
    
    def eof(self) -> bool:
        return self.idx == len(self.modules)

def get_new_mappings(key: str, param: torch.Tensor) -> Dict[str, torch.Tensor]:
    modules = []
    parser = ModuleParser(key)
    sat_prefix = "model.diffusion_model"
    assert parser.match(sat_prefix), key
    if parser.match("mixins"):
        if parser.match("adaln_layer"):
            if parser.match("adaLN_modulations"):
                modules.append("blocks")
                modules.append(parser.step())
                modules.append("modulation")
            elif parser.match("query_layernorm_list"):
                modules.append("blocks")
                modules.append(parser.step())
                modules.append("self_attn.norm_q")
                modules.append(parser.step())
            elif parser.match("key_layernorm_list"):
                modules.append("blocks")
                modules.append(parser.step())
                modules.append("self_attn.norm_k")
                modules.append(parser.step())
            elif parser.match("cross_query_layernorm_list"):
                modules.append("blocks")
                modules.append(parser.step())
                modules.append("cross_attn.norm_q")
                modules.append(parser.step())
            elif parser.match("cross_key_layernorm_list"):
                modules.append("blocks")
                modules.append(parser.step())
                modules.append("cross_attn.norm_k")
                modules.append(parser.step())
            elif parser.match("clip_feature_key_layernorm_list"):
                modules.append("blocks")
                modules.append(parser.step())
                modules.append("cross_attn.norm_k_img")
                modules.append(parser.step())
            elif parser.match("clip_feature_key_value_list"):
                modules.append("blocks")
                modules.append(parser.step())
                modules.append("cross_attn")
                prefix = '.'.join(modules)
                suffix = parser.step() # weight or bias
                key_param, value_param = param.chunk(2, dim=0)
                return {
                    ".".join([prefix, "k_img", suffix]): key_param, 
                    ".".join([prefix, "v_img", suffix]): value_param,
                }
            else:
                raise ValueError(key)
        elif parser.match("final_layer"):
            modules.append("head")
            if parser.match("adaLN_modulation"):
                modules.append("modulation")
            elif parser.match("linear"):
                modules.append("head")
                modules.append(parser.step()) # weight or bias
        elif parser.match("patch_embed"):
            if parser.match("proj"):
                modules.append("patch_embedding")
            elif parser.match("proj_pose"):
                modules.append("patch_embedding_pose")
            else:
                raise ValueError(key)
            modules.append(parser.step())
        else:
            raise ValueError(key)
    elif parser.match("transformer.layers"):
        modules.append("blocks")
        modules.append(parser.step())
        if parser.match("attention"):
            modules.append("self_attn")
            if parser.match("dense"):
                modules.append("o")
                modules.append(parser.step())
            elif parser.match("query_key_value"):
                prefix = '.'.join(modules)
                suffix = parser.step()
                query_param, key_param, value_param = param.chunk(3, dim=0)
                return {
                    ".".join([prefix, "q", suffix]): query_param,
                    ".".join([prefix, "k", suffix]): key_param, 
                    ".".join([prefix, "v", suffix]): value_param,
                }
            else:
                raise ValueError(key)
        elif parser.match("cross_attention"):
            modules.append("cross_attn")
            if parser.match("dense"):
                modules.append("o")
                modules.append(parser.step())
            elif parser.match("query"):
                modules.append("q")
                modules.append(parser.step())
            elif parser.match("key_value"):
                prefix = '.'.join(modules)
                suffix = parser.step()
                key_param, value_param = param.chunk(2, dim=0)
                return {
                    ".".join([prefix, "k", suffix]): key_param, 
                    ".".join([prefix, "v", suffix]): value_param,
                }
            else:
                raise ValueError(key)
        elif parser.match("post_cross_attention_layernorm"):
            modules.append("norm3")
            modules.append(parser.step()) # weight or bias
        elif parser.match("mlp"):
            modules.append("ffn")
            if parser.match("dense_h_to_4h"):
                modules.append("0")
            elif parser.match("dense_4h_to_h"):
                modules.append("2")
            else:
                raise ValueError(key)
            modules.append(parser.step()) # weight or bias
        else:
            raise ValueError(key)
    elif parser.match("time_embed"):
        modules.append("time_embedding")
        modules.append(parser.step())
        modules.append(parser.step())
    elif parser.match("adaln_projection"):
        modules.append("time_projection")
        modules.append(parser.step())
        modules.append(parser.step())
    elif parser.match("text_embedding"):
        modules.append("text_embedding")
        modules.append(parser.step())
        modules.append(parser.step())
    elif parser.match("clip_proj"):
        assert parser.match("proj"), key
        modules.append("img_emb.proj")
        modules.append(parser.step())
        modules.append(parser.step())
    else:
        raise ValueError(key)
    assert parser.eof(), key
    return {'.'.join(modules): param}

def get_new_state_dict(old: Dict[str, torch.Tensor]):
    new = dict()
    for key, value in old.items():
        map = get_new_mappings(key, value)
        for new_key, new_value in map.items():
            if new_key in new:
                print(f"Warning: duplicate new key {new_key} converted from {key}!")
            new[new_key] = new_value
    return new

def main(args):
   pt_file_path = os.path.join(args.scail_dir, args.sat_model_path)
   print(f"Loading from {pt_file_path}...")
   checkpoint = torch.load(pt_file_path)
   state_dict = checkpoint["module"]
   new_state_dict = get_new_state_dict(state_dict)
   print(f"Saving to {args.save_path}...")
   save_file(new_state_dict, args.save_path)
   print("Done.")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scail-dir", default="SCAIL-Preview/")
    parser.add_argument("--sat-model-path", default="model/1/mp_rank_00_model_states.pt")
    parser.add_argument("--save-path", default="Wan2.1-14B-SCAIL.safetensors")
    args = parser.parse_args()
    main(args)