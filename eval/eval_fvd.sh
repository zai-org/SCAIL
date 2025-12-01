python calculate_fvd.py \
    --real_video_dir /workspace/data2/processed_videos_wds/validation/webvid/selected_gt \
    --generated_video_dir  /workspace/ckpt/yzy/snapbatch_dir/1713506689568/samples/ablation/webvid/noise-schedule-shift-3.0-blockscale015 \
    --batch_size 496 \
    --num_frames 20 \
    --device 'cuda:0'