# CONFIG_FILE=${1:-pexels_hengping.yaml}
# export PYTHONPATH=$(pwd)
# python DWPoseProcess/extract_dwpose.py --config DWPoseExtractConfig/${CONFIG_FILE} --gpu_ids 2

#! /bin/bash

CONFIG_FILE=${1:-stock_adobe_videos.yaml}
export PYTHONPATH=$(pwd)
# run_cmd="python DWPoseProcess/extract_nlfpose.py --config DWPoseExtractConfig/${CONFIG_FILE} --local_rank 1 --world_size 8"

# echo ${run_cmd}
# eval ${run_cmd}
# 启动8个并行进程，rank从0到7
for rank in {0..7}; do
    run_cmd="python DWPoseProcess/extract_nlfpose.py --config DWPoseExtractConfig/${CONFIG_FILE} --local_rank ${rank} --world_size 8"
    echo "Starting rank ${rank}: ${run_cmd}"
    eval ${run_cmd} &
done
wait


echo "DONE on `hostname`"

