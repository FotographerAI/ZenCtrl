# ZenCtrl

**An all-in-one, control framework for unified visual content creation using GenAI.**  
Generate multi-view, diverse-scene, and task-specific high-resolution images from a single subject image—without fine-tuning.

<div align="center">
  <a href="https://huggingface.co/YOUR_ORG/ZenCtrl" name="huggingface_model_link"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace Model"></a>
  <a href="https://huggingface.co/spaces/YOUR_ORG/ZenCtrl" name="huggingface_space_link"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace Space"></a>
  <a href="https://discord.com/invite/b9RuYQ3F8k" name="discord_link"><img src="https://img.shields.io/badge/Discord-Join-7289da.svg?logo=discord" alt="Discord"></a>
  <a href="https://fotographer.ai/" name="lp_link"><img src="https://img.shields.io/badge/Website-Landing_Page-blue" alt="LP"></a>
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

---

## 🛠 Toolkit Components (coming soon)

### 🧹 Preprocessing

- Background removal
- Matting
- Reshaping
- Segmentation

### 🎮 Control Models 

- Shape (Canny, HED, Scribble, Depth)
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
- **Coming Soon**: Source code release, Quick Start guide, Example notebooks
- **Next**: Controlled fine-grain version on our platform and API (Pro version)
- **Future**: Video generation toolkit release

---

## 🚀 Quick Start

> 🚧 Coming Soon: Setup instructions & example notebooks.

---

## 🎨 Demo

> 🧪 Try it now on [Hugging Face Space](https://huggingface.co/spaces/YOUR_ORG/ZenCtrl)

<!--
| Input Image | Generated Image |
|-------------|------------------|
| ![](./assets/) | ![](./assets/) |
| ![](./assets/) | ![](./assets/d) |
-->

---

## 🔧 Models (Initial Weights Released)

<!--
## 🔧 Models (Initial Weights Released)

| Type                | Name            | Base        | Resolution | Description                         |
|---------------------|-----------------|-------------|------------|-------------------------------------|
| Bg generation Model       | `tbd`   | FLUX.1      | 1024x1024   | Core model for subject-driven gen   |
| Bg generation + Canny     | `tbd`   | FLUX.1      | 1024x1024   | Enhanced background control         |
| Deblurring Model    | `tbd`     | OminiControl    | 1024x1024   | Quality recovery post-generation    |

---
-->

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
- [ ] Release additional task-specific models and modes
- [ ] Release open source code (coming soon)
- [ ] Release Quick Start guide and example notebooks
- [ ] Launch API access via our app and Baseten for easier deployment
- [ ] Release high-resolution models (1500×1500+)
- [ ] Enable full toolkit integration with agent API
- [ ] Add video generation module

---

## 🤝 Join the Community

- 💬 [Discord](https://discord.com/invite/b9RuYQ3F8k) – share ideas and feedback
- 🌐 [Landing Page](https://fotographer.ai)
- 🧪 [Try on Hugging Face](https://huggingface.co/spaces/YOUR_ORG/ZenCtrl)
<!-- - 🧠 [Blog]() -->

---

## 🤝 Community Collaboration

We hope to collaborate closely with the open-source community to make **ZenCtrl** a powerful and extensible toolkit for visual content creation.  
Once the source code is released, we welcome contributions in training, expanding supported use cases, and developing new task-specific modules.  
Our vision is to make ZenCtrl the **standard framework** for agentic, high-quality image and video generation — built together, for everyone.
