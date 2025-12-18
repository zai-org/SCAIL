 <h1>Pose Extraction & Rendering Instruction for SCAIL</h1>

## 🚀 Getting Started

Use git submodule to download the `scail_pose` module:
```
git submodule update --init --recursive
```
After that, the project structure should be like this:
```
SCAIL/
├── examples
├── sat
├── configs
├── ...
├── scail_pose
```

Change dir to this pose extraction & rendering folder:

```
cd scail_pose/
```

### Environment Setup

We recommend using [mmpose](https://github.com/open-mmlab) for the environment setup. You can refer to the official
mmpose [installation guide](https://mmpose.readthedocs.io/en/latest/installation.html). Note that the example in the guide uses python 3.8, however we recommend using python>=3.10 for compatibility with [SAMURAI](https://github.com/yangchris11/samurai).
The following commands are used to install the required packages once you have setup the environment.

```bash
conda activate openmmlab
pip install -r requirements.txt

# [optional] sam2 is only for multi-human extraction purposes, you can skip this step if you only need single human extraction
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd ..
```



### Weights Download

First, download pretrained weights for pose extraction & rendering. The script below
downloads [NLFPose](https://github.com/isarandi/nlf) (torchscript), [DWPose](https://github.com/IDEA-Research/DWPose) (
onnx) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (onnx) weights. You can also download the weights
manually and put them into the `pretrained_weights` folder.

```bash
mkdir pretrained_weights && cd pretrained_weights
# download NLFPose Model Weights
wget https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript
# download DWPose Model Weights & Detection Model Weights
mkdir DWPose
wget -O DWPose/dw-ll_ucoco_384.onnx \
  https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx
wget -O DWPose/yolox_l.onnx \
  https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx
cd ..
```

The weights should be formatted as follows:

```
pretrained_weights/
├── nlf_l_multi_0.3.2.torchscript
└── DWPose/
    ├── dw-ll_ucoco_384.onnx
    └── yolox_l.onnx
```


[Optional] Then download SAM2 weights for segmentation if you need to use multi-human extraction & rendering. Run the following commands:
```bash
cd sam2/checkpoints && \
./download_ckpts.sh && \
cd ../..
```


## 🦾 Usage

Default Extraction & Rendering:

```
python NLFPoseExtract/process_pose.py --subdir <path_to_the_example_pair> --resolution [512, 896]
```

Extraction & Rendering using 3D Retarget:

```
python NLFPoseExtract/process_pose.py --subdir <path_to_the_example_pair> --use_align --resolution [512, 896]
```

Multi-Human Extraction & Rendering:

```
python NLFPoseExtract/process_pose_multi.py --subdir <path_to_the_example_pair> --resolution [512, 896]
```

Note that the examples are in the main repo folder, you can also use your own images or videos. After the extraction and rendering, the results will be saved in the example folder and you can continue to use that folder to generate character animations in the main folder.