 <h1>Pose Extraction & Rendering Code for SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations</h1>
  <div align="center">
  <a href='https://arxiv.org/abs/2512.05905'><img src='https://img.shields.io/badge/📖 arXiv-2512.05905-red'></a>
  <a href='https://teal024.github.io/SCAIL/'><img src='https://img.shields.io/badge/🌐 Project Page-green'></a>
  <a href="https://github.com/zai-org/SCAIL">
    <img src="https://img.shields.io/badge/%20Main GitHub Repo-181717?logo=github">
  </a>
</div>


This repository contains the 3D pose extraction & rendering code for **SCAIL (Studio-Grade Character Animation via
In-Context Learning)**, a framework that enables high-fidelity character animation under diverse and challenging
conditions, including large motion variations, stylized characters, and multi-character interactions. The main repo is at [zai-org/SCAIL](https://github.com/zai-org/SCAIL).
<p align="center">
  <img src="resources/pose_teaser.png" alt="teaser" width="90%">
</p>


## 📋 Methods
We connect estimated 3D human keypoints according to skeletal topology and represent bones as spatial cylinders. The resulting 3D skeleton is rasterized to obtain 2D motion guidance signals.

When processing multi-character data, we segment each character, extract their poses, and then render them together to achieve multi-character pose extraction.
<p align="center">
  <img src="resources/data.png" alt="data" width="90%">
</p>

Our multi-stage pose extraction pipeline provides robust estimations under multi-character interactions, benefiting from [NLFPose](https://github.com/isarandi/nlf)’s reliable depth estimation:
<p align="center">
  <img src='resources/pose_result.png' alt='Teaser' width='95%'>
</p>

Utilizing such representation, our framework further resolves the challenge that pose representations cannot simultaneously prevent identity leakage and preserve rich motion information.
<p align="center">
  <img src="resources/pose_comp.png" alt="comp" width="90%">
</p>

## 🗞️ Update and News
* 2025.12.16: ❤️ Huge thanks to KJ for the work done on adaptation — the pose extraction & rendering has also been partly adapted to ComfyUI in [ComfyUI-SCAIL-Pose](https://github.com/kijai/ComfyUI-SCAIL-Pose)!! We are looking forward for the ComfyUI support of multi-character tracking to avoid character color exchange and the support of multi-character facial keypoints.
* 2025.12.16: ❕NOTE that our code should able to support as many characters as the segmentation node can handle, by default we use 2, but you can use more if you want. We recommend to use different colors for overlapping characters to alleviate ID exchange.






## 📋 TODOs

- [x] **Inference Code for 3D Pose Extraction & Rendering**

- [x] **Inference Code for 3D Pose Retarget**

- [x] **Inference Code for Multi-Human Pose Extraction & Rendering**


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

Note that the examples are in the main repo folder, you can also use your own images or videos. After the extraction and rendering, the results will be saved in the example folder and you can continue to use that folder to generate character animations in the main repo.

## 📄 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{yan2025scail,
  title={SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations},
  author={Yan, Wenhao and Ye, Sheng and Yang, Zhuoyi and Teng, Jiayan and Dong, ZhenHui and Wen, Kairui and Gu, Xiaotao and Liu, Yong-Jin and Tang, Jie},
  journal={arXiv preprint arXiv:2512.05905},
  year={2025}
}
```
