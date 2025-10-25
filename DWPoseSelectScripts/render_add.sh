#!/bin/bash

GPU_COUNT=8
MAX_PROCESSES=8

# 第一个参数是 yaml 文件名
YAML_NAME=$1

if [ -z "$YAML_NAME" ]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi

export PYTHONPATH=$(pwd)

for i in $(seq 0 $((MAX_PROCESSES-1))); do
    export CUDA_VISIBLE_DEVICES=$((i % GPU_COUNT))
    python DWPoseProcess/nlf_render.py \
        --config DWPoseExtractConfig/$YAML_NAME \
        --current_process $i \
        --input_root /workspace/ywh_data/pose_packed_wds_0929add_step3 \
        --max_processes $MAX_PROCESSES &

done
echo "All processes started"
wait