# python eval_fid.py ref --data=/zhipu-data/tjy/coco_5000 --dest=fid-refs/coco.npz

torchrun --standalone --nproc_per_node=8 eval_fid.py calc \
    --images=/zhipu-data/home/tjy/sat_sdxl/coco/256-mistral-lora \
    --ref=/zhipu-data/home/tjy/edm/fid-refs/coco-256x256.npz --num 5000