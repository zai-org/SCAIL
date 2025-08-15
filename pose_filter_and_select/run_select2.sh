export PYTHONPATH=$(pwd)
cd pose_filter_and_select
# python select_pose.py --config config_newsft_multi.yaml # 分到select3中
# python select_pose.py --config config_newsft_single.yaml # 分到select3中
python select_pose.py --config config_moviesTV_multi.yaml 
python select_pose.py --config config_moviesTV_single.yaml 