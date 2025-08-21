# CONFIG_FILE=${1:-pexels_hengping.yaml}
# export PYTHONPATH=$(pwd)
# python DWPoseProcess/extract_dwpose.py --config DWPoseExtractConfig/${CONFIG_FILE} --gpu_ids 2

#! /bin/bash

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
CONFIG_FILE=${1:-stock_adobe_videos.yaml}
export PYTHONPATH=$(pwd)

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python DWPoseProcess/extract_dwpose.py --config DWPoseExtractConfig/${CONFIG_FILE}"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"