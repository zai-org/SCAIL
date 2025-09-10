export PYTHONPATH=$(pwd)
# bash /workspace/yanwenhao/dwpose_draw/DWPoseSelectScripts/run_sgl.sh --config DWPoseExtractConfig/pexels1k.yaml --output_root /workspace/ywh_data/pose_packed_wds_0903_pretrain --max_processes 2 --force_no_filter
bash /workspace/yanwenhao/dwpose_draw/DWPoseSelectScripts/run_sgl.sh --config DWPoseExtractConfig/bili_dance_hengping_250328.yaml --output_root /workspace/ywh_data/pose_packed_wds_0903_pretrain --max_processes 8 --force_no_filter
bash /workspace/yanwenhao/dwpose_draw/DWPoseSelectScripts/run_sgl.sh --config DWPoseExtractConfig/bili_dance_shuping_250328.yaml --output_root /workspace/ywh_data/pose_packed_wds_0903_pretrain --max_processes 8 --force_no_filter
bash /workspace/yanwenhao/dwpose_draw/DWPoseSelectScripts/run_sgl.sh --config DWPoseExtractConfig/dongman.yaml --output_root /workspace/ywh_data/pose_packed_wds_0903_pretrain --max_processes 8 --force_no_filter
bash /workspace/yanwenhao/dwpose_draw/DWPoseSelectScripts/run_sgl.sh --config DWPoseExtractConfig/pexels_hengping.yaml --output_root /workspace/ywh_data/pose_packed_wds_0903_pretrain --max_processes 8 --force_no_filter

