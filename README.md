 <h1>SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations</h1>


 <div align="center">
  <a href='https://arxiv.org/abs/2512.05905'><img src='https://img.shields.io/badge/📖 arXiv-2512.05905-red'></a>
  <a href='https://huggingface.co/zai-org/SCAIL-Preview/tree/main'><img src='https://img.shields.io/badge/🤗 HuggingFace-yellow'></a>
  <a href='https://modelscope.cn/models/ZhipuAI/SCAIL-Preview'><img src='https://img.shields.io/badge/🤖 ModelScope-blue'></a>
  <a href='https://teal024.github.io/SCAIL/'><img src='https://img.shields.io/badge/🌐 Project Page-green'></a>
</div>



This repository contains the official implementation code for **SCAIL (Studio-Grade Character Animation via In-Context Learning)**, a framework that enables high-fidelity character animation under diverse and challenging conditions, including large motion variations, stylized characters, and multi-character interactions.


<p align="center">
  <img src='resources/teaser.png' alt='Teaser' width='90%'>
</p>

## 🔎 Motivation and Results
SCAIL identifies the key bottlenecks that hinder character animation towards production level: limited generalization towards characters and incoherent motion under complex scenarios (e.g., the long-standing challenge of multi-character interactions, as well as common failures in basic motions like flipping and turning). We revisit the core components of character animation -- how to represent the pose condition and how to inject the pose condition. Our framework resolves the challenge that pose representations cannot simultaneously prevent identity leakage and preserve rich motion information, and compels the model to perform spatiotemporal reasoning over the entire motion sequence for more natural and coherent movements. Check our methods, results gallery, as well as comparisons against other baselines at our [project page](https://teal024.github.io/SCAIL/).
<p align="center">
  <img src='resources/1.gif' width='66%'>
  <img src='resources/2.gif' width='66%'>
  <img src='resources/3.gif' width='66%'>
</p>

## 🌱 Community Works
❤️ A heartfelt thanks to friends in the community for their creativity! All results below are shared with their gracious consent. We were surprised to see the emergent abilities our model exhibited — understanding the 3D spatial relationships of 2D characters, driving hand-drawn artwork, and even controlling quadrupeds despite having no animal training data at all. 

<table align="center" border="0" cellspacing="0" cellpadding="6">
  <!-- 第一行 -->
  <tr>
    <td align="center">
      <img src="resources/community1.gif" width="220"><br>
      <em>Chibi Gotham Battle</em>
    </td>
    <td align="center">
      <img src="resources/community2.gif" width="250"><br>
      <em>Homer in Slowmo (w/ Uni3c)</em>
    </td>
    <td align="center" rowspan="2">
      <img src="resources/community4.gif" width="150"><br>
      <em>Anime Art Animation</em>
    </td>
  </tr>

  <!-- 第二行 -->
  <tr>
    <td align="center" colspan="2">
      <img src="resources/community3.gif" width="400"><br>
      <em>Street Fighter 6 Motion Mimic</em>
    </td>
  </tr>

  <!-- 第三行 -->
  <tr>
    <td align="center">
      <img src="resources/community6.gif" width="150"><br>
      <em>Doodle Art Animation</em>
    </td>
    <td align="center">
      <img src="resources/community8.gif" width="150"><br>
      <em>Dual Dance</em>
    </td>
    <td align="center" colspan="2">
      <img src="resources/community5.gif" width="150"><br>
      <em>Group Dance</em>
      <img src="resources/community7.gif" width="200"><br>
      <em>Quadrupeds Animation (w/ ViTPose)</em>
    </td>
  </tr>
</table>


## 🗞️ Updates and Plans
* 2025.12.19: 📣 We offer the Wan Official Framework of SCAIL instead of [SAT](https://github.com/THUDM/SwissArmyTransformer) for more convenient inference. Check the [wan branch](https://github.com/zai-org/SCAIL/tree/wan) of SCAIL. We will update the training code of SCAIL on SAT for reproducibility.
* 2025.12.11: 💥 The preview version of SCAIL is now opensourced on [HuggingFace](https://huggingface.co/zai-org/SCAIL-Preview) and [ModelScope](https://modelscope.cn/models/ZhipuAI/SCAIL-Preview).
* 2025.12.08: 🔥 We release the inference code of SCAIL on [SAT](https://github.com/THUDM/SwissArmyTransformer).

### TODOs
- [x] **SCAIL-14B-Preview Model Weights**(512p, 5s) and Inference Config
- [x] Prompt Optimization Snippets
- [x] **Implementation on Wan Official Framework**
- [ ] **SCAIL-Official(1.3B/14B) Model Weights**(Improved Stability and Clarity, Innate Long Video Generation Capability) and Inference Config

## 📰 News
* 2025.12.19: [ComfyUI-SCAIL-Pose](https://github.com/kijai/ComfyUI-SCAIL-Pose) now supports saving NLF mesh as 3D glb animation and 3D previewing of the SCAIL-Pose skeleton.
* 2025.12.19: Thanks to [deepbeepmeep](https://github.com/deepbeepmeep) for Low VRAM SCAIL Preview Support in [WanGP](https://github.com/deepbeepmeep/Wan2GP)! WanGP version has the following perks: 3D pose Preprocessing fully integrated, speed optimized, and compatible with any pytorch version.
* 2025.12.17: Thanks to [VantageWithAI](https://hf-mirror.com/vantagewithai), GGUF version is now available at [SCAIL-Preview-GGUF](https://hf-mirror.com/vantagewithai/SCAIL-Preview-GGUF/tree/main)!
* 2025.12.16: ❤️ Huge thanks to KJ for the work done on adaptation — SCAIL is now available in [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)!!! Meanwhile, the pose extraction & rendering has also been partly adapted to ComfyUI in [ComfyUI-SCAIL-Pose](https://github.com/kijai/ComfyUI-SCAIL-Pose), currently without multi-character tracking.
* 2025.12.14: 🥳 Thanks to friends in the community for testing the work! Despite the fact that only 1.5% of SCAIL’s training samples are anime data, and that we did not intentionally collect any multi-character anime data, the model can generalize towards many complex anime characters. The release of SCAIL-Preview is intended to demonstrate the soundness of our proposed pose representation and model architecture, with clear potential for further scaling and enhancement.


## 🚀 Getting Started
### Checkpoints Download

| ckpts       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| SCAIL-Preview(14B) | [🤗 Hugging Face](https://huggingface.co/zai-org/SCAIL-Preview)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/SCAIL-Preview)     | Trained with resolutions under 512p.<br> H and W should be both divisible by 32<br> (e.g. 704*1280) if using other resolutions. |

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


### Environment Setup
Please make sure your Python version is between 3.10 and 3.12, inclusive of both 3.10 and 3.12.
```
pip install -r requirements.txt
```

## 🦾 Usage
### Input preparation
The input data should be organized as follows, we have provided some example data in `examples/`:
```
examples/
├── 001
│   ├── driving.mp4
│   ├── ref.jpg
└── 002
    ├── driving.mp4
    └── ref.jpg
...
```
### Pose Extraction & Rendering
Use git submodule to download the `scail_pose` module and then follow the [POSE_INSTRUCTION.md](POSE_INSTRUCTION.md) to extract and render the pose from the driving video. 

```shell
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
Change dir into the subdir and follow instructions:
```shell
cd scail_pose
# follow instructions in POSE_INSTRUCTION.md
```
After pose extraction and rendering, the input data should be organized as follows:
```
examples/
├── 001
│   ├── driving.mp4
│   ├── ref.jpg
│   └── rendered.mp4 (or rendered_aligned.mp4)
└── 002
...
```

### Model Inference
For inference in Wan Official Framework, please refer to the [wan branch](https://github.com/zai-org/SCAIL/tree/wan) of SCAIL.

For inference in SAT, run the following command to start the inference with CLI input:
```
bash scripts/sample_sgl_14Bsc_xc_cli.sh
```

The CLI will ask you to input in format like `<prompt>@@<example_dir>`, e.g. `the girl is dancing@@examples/001`. The `example_dir` should contain rendered.mp4 or rendered_aligned.mp4 after pose extraction and rendering. Results will be save to `samples/`.

We support direct txt input too, change `input_file` in [sample_sgl_14Bsc_xc_txt.yaml](configs/sampling/sample_sgl_14Bsc_xc_txt.yaml) to path of your input file, and fill in the input file with format like `<prompt>@@<example_dir>`, then run the following command:
```
bash scripts/sample_sgl_14Bsc_xc_txt.sh
```

Note that our model is trained with **long detailed prompts**, even though a short or even null prompt can be used, the result may not be as good as the long prompt. We will provide our prompt generation snippets, using Google [Gemini](https://deepmind.google/models/gemini/) to read from the reference image and the driving motion and generate a detailed prompt like `A woman with curly hair is joyfully dancing along a rocky shoreline, wearing a sleek blue two-piece outfit. She performs various dance moves, including twirling, raising her hands, and embracing the lively seaside atmosphere, her tattoos and confident demeanor adding to her dynamic presence.` 

You can further choose sampling configurations like resolution in the yaml file under `configs/sampling/` or directly modify `sample_video.py` for customized sampling logic.

## ✨ Acknowledgements
Our implementation is built upon the foundation of [Wan 2.1](https://github.com/Wan-Video/Wan2.1) and the overall project architecture is built using [SAT](https://github.com/THUDM/SwissArmyTransformer). We utilized [NLFPose](https://github.com/isarandi/nlf) for reliable pose extraction. Thanks for their remarkable contribution and released code.

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

## 🗝️ License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
