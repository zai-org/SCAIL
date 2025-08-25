#! /bin/bash
CONFIG_FILE=${1:-stock_adobe_videos.yaml}
export PYTHONPATH=$(pwd)
python DWPoseProcess/select_dwpose.py --config DWPoseExtractConfig/${CONFIG_FILE}