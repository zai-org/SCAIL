torchrun --standalone --nproc_per_node=8 eval_clip_score.py \
    --img_dir=/zhipu-data/home/tjy/sat_sdxl/coco/256-mistral-lora \
    --txt_path=/zhipu-data/zwd/coco_text_5000