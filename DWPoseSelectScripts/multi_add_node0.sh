#!/bin/bash

GPU_COUNT=8
MAX_LOCAL_PROCESSES=32
ALL_PROCESSES=64

export PYTHONPATH=$(pwd)

for i in $(seq 0 $((MAX_LOCAL_PROCESSES-1))); do

    if (( i >= 8 && i % 8 == 0 )); then
        echo "Rank ${i} is sleeping for 120 seconds..."
        sleep 120
    fi

    export CUDA_VISIBLE_DEVICES=$((i % GPU_COUNT))
    python NLFPoseExtract/process_multinlf_mp4_batch.py \
        --local_rank $i \
        --world_size $ALL_PROCESSES &

done
echo "All processes started"
wait