export PYTHONPATH=$(pwd)
python DWPoseProcess/extract_dwpose.py --config DWPoseProcess/config_newdata_bili.yaml
python DWPoseProcess/extract_dwpose.py --config DWPoseProcess/config_newdata_bili_multi.yaml