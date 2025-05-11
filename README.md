<div align="center">
  <a href="https://fotographer.ai/zen-control">
    <img src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/banner_1.png" alt="ZenCtrl Banner" />
  </a>
  <h1>ZenCtrl</h1>
</div>

**An all-in-one, control framework for unified visual content creation using GenAI.**  
Generate multi-view, diverse-scene, and task-specific high-resolution images from a single subject image—without fine-tuning.

<div align="center">
  <a href="https://huggingface.co/fotographerai/zenctrl_tools/tree/main/weights" name="huggingface_model_link"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace Model"></a>
  <a href="https://huggingface.co/spaces/fotographerai/ZenCtrl" name="huggingface_space_link"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace Space"></a>
  <a href="https://discord.com/invite/b9RuYQ3F8k" name="discord_link"><img src="https://img.shields.io/badge/Discord-Join-7289da.svg?logo=discord" alt="Discord"></a>
  <a href="https://fotographer.ai/zen-control" name="lp_link"><img src="https://img.shields.io/badge/Website-Landing_Page-blue" alt="LP"></a>
  <a href="https://x.com/FotographerAI" name="twitter_link"><img src="https://img.shields.io/twitter/follow/FotographerAI?style=social" alt="X"></a>
</div>

---

## 🧠 Overview

**ZenCtrl** is a comprehensive toolkit built to tackle core challenges in image generation:

- No fine-tuning needed – works from **a single subject image**
- Maintains **control over shape, pose, camera angle, context**
- Supports **high-resolution**, multi-scene generation
- Modular toolkit for preprocessing, control, editing, and post-processing tasks

ZenCtrl is based on OminiControl but enhanced with more fine-grained control, consistent subject preservation, and more improved and ready-to-use models. Our goal is to build an **agentic visual generation system** that can orchestrate image/video creation from **LLM-driven recipes.**

<div align="center">
  <a href="https://fotographer.ai/zen-control">
    <img src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/banner_3.png" alt="ZenCtrl Collage" style="width:100%; max-width:960px;" />
  </a>
</div>

---

## 🛠 Toolkit Components (coming soon)

### 🧹 Preprocessing

- Background removal
- Matting
- Reshaping
- Segmentation

### 🎮 Control Models

- Shape (HED, Scribble, Depth)
- Pose (OpenPose, DensePose)
- Mask control
- Camera/View control

### 🎨 Post-processing

- Deblurring
- Color fixing
- Natural blending

### ✏️ Editing Models

- Inpainting (removal, masked editing, replacement)
- Outpainting
- Transformation / Motion
- Relighting

---

## 🎯 Supported Tasks

- Background generation
- Controlled background generation
- Subject-consistent context-aware generation
- Object and subject placement (coming soon)
- In-context image/video generation (coming soon)
- Multi-object/subject merging & blending (coming soon)
- Video generation (coming soon)

---

## 📦 Target Use Cases

- Product photography
- Fashion & accessory try-on
- Virtual try-on (shoes, hats, glasses, etc.)
- People & portrait control
- Illustration, animation, and ad creatives

All of these tasks can be **mixed and layered** — ZenCtrl is designed to support real-world visual workflows with **agentic task composition**.

---

## 📢 News

- **2025-03-24**: 🧠 First release — model weights available on Hugging Face!
- **2025-05-06**: 📢 Update — ource code release, latest model weights available on Hugging Face!
- **Coming Soon**: Quick Start guide, Upscaling source code, Example notebooks
- **Next**: Controlled fine-grain version on our platform and API (Pro version)
- **Future**: Video generation toolkit release

---

## 🚀 Quick Start

Before running the Gradio code, please install the requirements and download the weights from our HuggingFace repository:  
👉 [https://huggingface.co/fotographerai/zenctrl_tools](https://huggingface.co/fotographerai/zenctrl_tools)

We matched our original code with the Omnicontrol structure. Our model takes two inputs instead, but we are going to release the original code soon with the LLaMA task driver — so stay tuned. We will also update the tasks for specific verticals (e.g., virtual try-on, ad creatives, etc.).

---

### Quick Setup (CMD)

You can follow the step-by-step setup instructions below:

```cmd
*** Cloning and setting up ZenCtrl
git clone https://github.com/FotographerAI/ZenCtrl.git
cd ZenCtrl

*** Creating virtual environment
python -m venv venv
call venv\Scripts\activate.bat

***Installing PyTorch and requirements
pip install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

***Downloading model weights
curl --create-dirs -L https://huggingface.co/fotographerai/zenctrl_tools/resolve/main/weights/zen2con_1440_17000/pytorch_lora_weights.safetensors -o weights\zen2con_1440_17000\pytorch_lora_weights.safetensors

***All set! Launching Gradio app
python app/gradio_app.py
```

---

## 🎨 Demo

#### Examples

<div align="center">
  <div>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/car.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/car.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/car.png"  width="49%"/>
    </picture>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/model.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/model.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/model.png"  width="49%"/>
    </picture>
  </div>
  <div>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/yellow_chair.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/yellow_chair.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/yellow_chair.png"  width="49%"/>
    </picture>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/assets/github/im4_1.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/assets/github/im4_1.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/assets/github/im4_1.png"  width="49%" />
    </picture>
  </div>
</div>

### 🧪 Try it now on [Hugging Face Space](https://huggingface.co/spaces/fotographerai/ZenCtrl)

<!--
| Input Image | Generated Image |
|-------------|------------------|
| ![](./assets/) | ![](./assets/) |
| ![](./assets/) | ![](./assets/d) |
-->

---

## 🔧 Models (Updated Weights Released)

| Type                  | Name                  | Base         | Resolution | Description                       | links                                                      |
| --------------------- | --------------------- | ------------ | ---------- | --------------------------------- | ---------------------------------------------------------- |
| Subject Generation    | `zen2con_1440_17000`   | FLUX.1       | 1024x1024  | Core model for subject-driven gen | [link](https://huggingface.co/fotographerai/zenctrl_tools/tree/main/weights/zen2con_1440_17000) |
| Bg generation + Canny | `bg_canny_58000_1024` | FLUX.1       | 1024x1024  | Enhanced background control       | [link](https://huggingface.co/fotographerai/zenctrl_tools) |
| Deblurring Model      | `deblurr_1024_10000`  | OminiControl | 1024x1024  | Quality recovery post-generation  | [link](https://huggingface.co/fotographerai/zenctrl_tools) |

---

## 🚧 Limitations

1. Models currently perform best with **objects**, and to some extent **humans**.
2. Resolution support is currently capped at **1024x1024** (higher quality coming soon).
3. Performance with **illustrations** is currently limited.
4. The models were **not trained on large-scale or highly diverse datasets** yet — we plan to improve quality and variation by training on larger and more diverse datasets, especially for **illustration and stylized content**.
5. Video support and the full **agentic task pipeline** are still under development.

---

## 📋 To-do

- [x] Release early pretrained model weights for defined tasks
- [x] Release additional task-specific models and modes
- [x] Release open source code
- [x] Launch API access via Baseten for easier deployment
- [ ] Release Quick Start guide and example notebooks
- [ ] Launch API access via our app for easier deployment
- [ ] Release high-resolution models (1500×1500+)
- [ ] Enable full toolkit integration with agent API
- [ ] Add video generation module

---

## 🤝 Join the Community

- 💬 [Discord](https://discord.com/invite/b9RuYQ3F8k) – share ideas and feedback
- 🌐 [Landing Page](https://fotographer.ai/zen-control)
- 🧪 [Try it now on Hugging Face Space](https://huggingface.co/fotographerai/zenctrl_tools/tree/main/weights)
<!-- - 🧠 [Blog]() -->

---

## 🤝 Community Collaboration

We hope to collaborate closely with the open-source community to make **ZenCtrl** a powerful and extensible toolkit for visual content creation.  
Once the source code is released, we welcome contributions in training, expanding supported use cases, and developing new task-specific modules.  
Our vision is to make ZenCtrl the **standard framework** for agentic, high-quality image and video generation — built together, for everyone.
