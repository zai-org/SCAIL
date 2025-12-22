# SCAIL

## Quickstart

### Installation
Clone the repo:
```sh
git clone https://github.com/zai-org/SCAIL.git
cd SCAIL
```

Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```

### Download the Checkpoint

| ckpts       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| SCAIL-Preview(14B) | [🤗 Hugging Face](https://huggingface.co/zai-org/SCAIL-Preview)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/SCAIL-Preview)     | Supports  512P

Use the following commands to download the model weights
(We have integrated both Wan VAE and T5 modules into this checkpoint for convenience).

```bash
# Download the repository (skip automatic LFS file downloads)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/zai-org/SCAIL-Preview
```
The files should be organized like:
```
SCAIL-Preview/
├── Wan2.1_VAE.pth
├── model
│   ├── 1
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── umt5-xxl
    ├── ...
```

### Convert to Safetensors
Use the following commands to convert our [SAT](https://github.com/THUDM/SwissArmyTransformer) checkpoint into the `.safetensors` supported by the Wan2.1 version. 

```sh
# --scail-dir: path to SCAIL repository folder, default: SCAIL-Preview/
# --save-path: path to save the converted .safetensors file
python convert.py --scail-dir /path/to/SCAIL-Preview/ --save-path /path/to/SCAIL.safetensors 
```

### Render the Pose Video
Please refer to the guidance in our [SCAIL-Pose](https://github.com/zai-org/SCAIL-Pose/) repository to render a pose video (like [rendered.mp4](./examples/rendered.mp4)) from a driving video (like [driving.mp4](./examples/driving.mp4)). 


### Run SCAIL
Use the following commands to run the SCAIL pipeline with our [examples](./examples/). 
```sh
# --target_w: width of the output video, default: 896
# --target_h: height of the output video, default: 512
python generate.py --model SCAIL-14B --ckpt_dir /path/to/SCAIL-Preview --scail_path /path/to/SCAIL.safetensors --image examples/SCAIL/ref.jpg --pose examples/SCAIL/rendered.mp4 --prompt "the girl is dancing" --target_w 896 --target_h 512 
```
> Note: `--pose` should be the path to the **rendered pose video**. 

## License Agreement
The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generated contents, granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. For a complete list of restrictions and details regarding your rights, please refer to the full text of the [license](LICENSE.txt).


## Acknowledgements

This repository is derived from [Wan2.1](https://github.com/Wan-Video/Wan2.1), which is licensed under the Apache 2.0 License.
