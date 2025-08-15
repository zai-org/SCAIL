#! /bin/bash

if [ -z "$MLP_WORKER_NUM" ]; then
    MLP_WORKER_NUM=1
    MLP_GPU=8
fi
# RANK=${MLP_ROLE_INDEX:-0}
# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$MLP_WORKER_0_HOST" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=35446
else
    MASTER_ADDR=$MLP_WORKER_0_HOST
    MASTER_PORT=$MLP_WORKER_0_PORT
fi

MP_SIZE=1

JOB_NAME="dwpose_run"

mkdir -p logs/${MLP_TASK_ID}_${JOB_NAME}
cat /root/mpi_hostfile
CONFIG_FILE=${1:-stock_adobe_videos.yaml}


mpi_cmd="MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT LOCAL_WORLD_SIZE=$MLP_GPU \
        mpirun -np $((MLP_WORKER_NUM * MLP_GPU)) \
        --allow-run-as-root -oversubscribe -map-by ppr:$MLP_GPU:node \
        --hostfile /root/mpi_hostfile \
        --mca oob_tcp_if_include eth8x \
        -mca pml ob1 -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include=eth8x \
        --output-filename logs/${MLP_TASK_ID}_${JOB_NAME} \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_RETRY_CNT=7 \
        -x NCCL_IB_TIME_OUT=32 \
        -x NCCL_DEBUG=INFO \
        -x MASTER_ADDR=$MLP_WORKER_0_HOST \
        -x MASTER_PORT=$MLP_WORKER_0_PORT \
        -x GLOO_SOCKET_IFNAME=eth8x \
        -x NCCL_SOCKET_IFNAME=eth8x \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        bash -c 'export PYTHONPATH=$(pwd); python DWPoseProcess/extract_dwpose.py --config DWPoseExtractConfig/${CONFIG_FILE}'"


echo ${mpi_cmd}
eval ${mpi_cmd} 2>&1 | tee logs/${MLP_TASK_ID}_${JOB_NAME}/${MLP_ROLE_INDEX}.log