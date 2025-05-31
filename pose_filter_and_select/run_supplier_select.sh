export PYTHONPATH=$(pwd)
cd pose_filter_and_select
python select_pose.py --config config_supplier_multi.yaml 
python select_pose.py --config config_supplier_single.yaml 