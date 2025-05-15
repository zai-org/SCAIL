# 替代产生.pt后重新filter，直接对已经filter好的mp4进行处理，得到新的.pt

export PYTHONPATH=$(pwd)
python DWPoseProcess/extract_dwpose.py --config DWPoseProcessConfig/config_newdata_pexels.yaml