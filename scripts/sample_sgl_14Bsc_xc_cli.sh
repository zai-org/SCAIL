#! /bin/bash

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

# run_cmd="$environs python sample_video.py --base configs/video_model/sd3-video-v2.yaml configs/sampling/base.yaml"

run_cmd="$environs python sample_video.py --base configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml configs/sampling/wan_pose_14Bsc_xc_cli.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"