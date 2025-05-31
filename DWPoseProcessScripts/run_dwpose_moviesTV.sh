export PYTHONPATH=$(pwd)
python DWPoseProcess/extract_dwpose.py --config DWPoseProcessConfig/config_newdata_moviesTV_multi.yaml
python DWPoseProcess/extract_dwpose.py --config DWPoseProcessConfig/config_newdata_moviesTV.yaml