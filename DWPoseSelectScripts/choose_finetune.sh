export PYTHONPATH=$(pwd)

export CUDA_VISIBLE_DEVICES=0 && python DWPoseProcess/finetune_filter.py --config DWPoseExtractConfig/bili_dance_hengping_250328.yaml &
export CUDA_VISIBLE_DEVICES=1 && python DWPoseProcess/finetune_filter.py --config DWPoseExtractConfig/bili_dance_shuping_250328.yaml &
# export CUDA_VISIBLE_DEVICES=3 && python DWPoseProcess/finetune_filter.py --config DWPoseExtractConfig/motionarray_nowat_2.yaml &
# export CUDA_VISIBLE_DEVICES=4 && python DWPoseProcess/finetune_filter.py --config DWPoseExtractConfig/motionarray_nowat_3.yaml &
export CUDA_VISIBLE_DEVICES=2 && python DWPoseProcess/finetune_filter.py --config DWPoseExtractConfig/dongman.yaml &
export CUDA_VISIBLE_DEVICES=5 && python DWPoseProcess/finetune_filter.py --config DWPoseExtractConfig/panda70m.yaml &
export CUDA_VISIBLE_DEVICES=6 && python DWPoseProcess/finetune_filter.py --config DWPoseExtractConfig/pexels_hengping.yaml &
export CUDA_VISIBLE_DEVICES=7 && python DWPoseProcess/finetune_filter.py --config DWPoseExtractConfig/pexels1k.yaml &

# 等待所有后台进程完成
wait