# CONFIG_FILE=${1:-pexels_hengping.yaml}
# export PYTHONPATH=$(pwd)
# python DWPoseProcess/extract_dwpose.py --config DWPoseExtractConfig/${CONFIG_FILE} --gpu_ids 2

#! /bin/bash

export PYTHONPATH=$(pwd)
python DWPoseProcess/extract_mp4_dwpose.py