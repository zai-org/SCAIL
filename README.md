 <h1>Pose Extraction & Rendering Code for SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations</h1>

This repository contains the 3D pose extraction & rendering code for **SCAIL (Studio-Grade Character Animation via
In-Context Learning)**, a framework that enables high-fidelity character animation under diverse and challenging
conditions, including large motion variations, stylized characters, and multi-character interactions.
<p align="center">
  <img src="resources/pose_teaser.png" alt="teaser" width="90%">
</p>


## 📋 Methods

When processing multi-person data, we segment each person, extract their poses, and then render them together to achieve multi-person pose extraction.
<p align="center">
  <img src="resources/data.png" alt="data" width="90%">
</p>

Our multi-stage pose extraction pipeline provides robust estimations under multi-person interactions:
<p align="center">
  <img src='resources/pose_result.png' alt='Teaser' width='95%'>
</p>

By applying 3D pose instead of 2D key-point based methods, our model is able to recognize occlusion relationships and preserve motion characteristics during augmentation and retarget.
<p align="center">
  <img src="resources/pose_comp.png" alt="comp" width="90%">
</p>




## 📋 TODOs

- [x] **Inference Code for 3D Pose Extraction & Rendering**

- [x] **Inference Code for 3D Pose Retarget**

- [ ] **Inference Code for Multi-Human Pose Extraction & Rendering**

## 🚀 Getting Started

Make sure you have already clone the main repo, this repo should be cloned under the main repo folder:
```
SCAIL/
├── examples
├── sat
├── configs
├── ...
├── SCAIL-Pose
```

Change dir to this pose extraction & rendering folder:

```
cd SCAIL-Pose/
```

### Weights Download

Download Pretrained Weights for pose extraction & rendering. The script below
downloads [NLFPose](https://github.com/isarandi/nlf) (torchscript), [DWPose](https://github.com/IDEA-Research/DWPose) (
onnx) and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (onnx) weights. You can also download the weights
manually and put them into the `pretrained_weights` folder.

```
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

### Environment Setup

We recommend using [mmpose](https://github.com/open-mmlab) for the environment setup. You can refer to the official
mmpose [installation guide](https://mmpose.readthedocs.io/en/latest/installation.html). Note that the example in the guide uses python 3.8, however we recommend using python>=3.10 for compatibility with [SAMURAI](https://github.com/yangchris11/samurai).
The following commands are used to install the required packages once you have setup the environment.

```shell
conda activate openmmlab
pip install -r requirements.txt
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

Note that the examples are in the main repo folder, you can also use your own images or videos. After the extraction and rendering, the results will be saved in the example folder and you can continue to use that folder to generate character animations in the main repo.
