#!/bin/bash
set -e

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=<your_mpi_port_number>     # e.g., 29500
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

mpirun -np 4 \
  --allow-run-as-root \
  -x MASTER_ADDR \
  -x MASTER_PORT \
  -x NCCL_DEBUG=VERSION \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  bash -c "
    python sample_video.py \
      --base configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml configs/sampling/wan_pose_14Bsc_xc_txt.yaml --seed $RANDOM
  " 2>&1 | tee single_node.log
