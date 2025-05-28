export PYTHONPATH=$(pwd)
python DWPoseProcess/extract_dwpose.py --config DWPoseProcessConfig/config_newdata_pexels_multi.yaml
python DWPoseProcess/extract_dwpose.py --config DWPoseProcessConfig/config_newdata_pexels.yaml