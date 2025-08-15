#! /bin/bash
CONFIG_FILE=${1:-DWPoseSelectConfig/bilibili_dance_videos.yaml}
export PYTHONPATH=$(pwd)
python DWPoseProcess/select_dwpose.py --config ${CONFIG_FILE}