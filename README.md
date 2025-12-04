# Official Implementation of SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations
This repository contains the official implementation code for **SCAIL (Studio-Grade Character Animation via In-Context Learning)**, a framework that enables high-fidelity character animation under diverse and challenging conditions, including large motion variations, stylized characters, and multi-character interactions.

## 🔎 Project Page
Check our demo and gallery at [this link](https://teal024.github.io/SCAIL/), more examples will be added soon.


## 📋 TODOs

- [x] **Inference Code for SAT**
- [x] **Config for Preview 14B SCAIL Model & Model Weights(512p)**
- [ ] **Config for Official 1.3B/14B Model & Model Weights(720p with history support)**
- [ ] **Inference Code for Diffusers**

## 🚀 Getting Started
### Weights Download
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
We provide our pose extraction and rendering code in another repo [SCAIL-Pose](https://github.com/teal024/SCAIL-Pose), which can be used to extract the pose from the driving video and render them. We recommand using another environment for pose extraction due to dependency issues. Clone that repo to `SCAIL-Pose` folder and follow instructions in it.
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
Run the following command to start the inference:
```
bash scripts/sample_sgl_1Bsc_xc_cli.sh
```

The CLI will ask you to input in format like `<prompt>@@<example_dir>, `e.g. `the girl is dancing@@examples/001`. The `example_dir` should contain rendered.mp4 or rendered_aligned.mp4 after pose extraction and rendering. Results will be save to `samples/`.

You can further choose sampling configurations like resolution in the yaml file under `configs/sampling/` or directly modify `sample_video.py` for customized sampling logic.

Though our model prioritize pose control and is robust to text prompts, we still suggest using VLMs like Gemini or GPT-4o to optimize the prompt especially on character features and gesture description. This is because the model is trained with long prompts and good prompts can improve generation quality. Example codes for prompt optimization will be provided soon.


## 📄 Citation

If you find this work useful in your research, please cite:

*Coming soon*