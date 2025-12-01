import os
import torch


def get_latest_checkpoint(checkpoint_path):
        latest_path = os.path.join(checkpoint_path, "latest")
        iteration = open(latest_path).read().strip()
        return iteration

def main():
    original_checkpoint_path = "/workspace/yanwenhao/cogvideo_sat/ckpts/modified_20in_wan1b_fi2v/1/mp_rank_00_model_states.pt"  # 这个不管是不是2proj版本都不影响
    # original_checkpoint_path = "/workspace/yanwenhao/cogvideo_sat/ckpts/modified_36in_wan14b_fi2v/1/mp_rank_00_model_states.pt"
    ori_state_dict = torch.load(original_checkpoint_path)
    ori_module_dict = ori_state_dict["module"]

    # checkpoint_dir = "/workspace/yanwenhao/cogvideo_skeleton/ckpts/pose-wan-1bsc-v2-09-02-15-40"
    # checkpoint_dir = "/workspace/yanwenhao/cogvideo_skeleton/ckpts/pose-wan-1bsc-v0-09-03-02-26"
    checkpoint_dir = "/workspace/yanwenhao/cogvideo_skeleton/ckpts/pose-wan-1bsc-v0x-latent-09-23-11-45"
    # output_dir = checkpoint_dir + "_with_pose_crossattn_complete"
    output_dir = checkpoint_dir
    iteration = get_latest_checkpoint(checkpoint_dir)
    print(f"latest iteration: {iteration}")
    checkpoint_path = os.path.join(checkpoint_dir, iteration, "mp_rank_00_model_states.pt")
    print(f"checkpoint_path: {checkpoint_path}")
    new_state_dict = torch.load(checkpoint_path)
    new_module_dict = new_state_dict["module"]

    untrainable_keywords_crossattn = ["cross_attention.query.weight", "cross_attention.query.bias", "cross_attention.key_value.weight", "cross_attention.key_value.bias", "cross_attention.dense.weight", "cross_attention.dense.bias"]
    untrainable_keywords_finalproj = ["final_layer.linear"]
    untrainable_keywords = untrainable_keywords_crossattn + untrainable_keywords_finalproj

    for key in ori_module_dict.keys():
        for untrainable_keyword in untrainable_keywords:
            if untrainable_keyword in key:
                print(f"key: {key} is untrainable during training, so we use the original weights")
                new_module_dict[key] = ori_module_dict[key]
                break

    os.makedirs(os.path.join(output_dir, iteration), exist_ok=True)
    torch.save(new_state_dict, os.path.join(output_dir, iteration, "mp_rank_00_model_states.pt"))
    file_path = os.path.join(output_dir, "latest")
    with open(file_path, "w") as f:
        f.write(iteration)



if __name__ == "__main__":
    main()