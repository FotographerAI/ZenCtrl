<div align="center">
  <a href="https://fotographer.ai/zen-control">
    <picture>
      <source media="(max-width: 424px" srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/banner_sm.avif" type="image/avif">
      <source media="(min-width: 425px" srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/banner.avif" type="image/avif">
      <source media="(max-width: 424px" srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/banner_sm.webp" type="image/webp">
      <source media="(min-width: 425px" srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/banner.webp" type="image/webp">
      <img alt="ZenCtrl Banner" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/banner.png" />
    </picture>
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
  <div>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/bottle_1.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/bottle_1.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/bottle_1.png"  width="24%"/>
    </picture>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/bottle_2.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/bottle_2.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/bottle_2.png"  width="24%"/>
    </picture>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/speaker_1.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/speaker_1.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/speaker_1.png"  width="24%"/>
    </picture>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/speaker_2.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/speaker_2.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/speaker_2.png"  width="24%"/>
    </picture>
  </div>
  <div>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/chair_1.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/chair_1.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/chair_1.png"  width="24%"/>
    </picture>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/chair_2.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/chair_2.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/chair_2.png"  width="24%" />
    </picture>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/handcream_1.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/handcream_1.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/handcream_1.png"  width="24%"/>
    </picture>
    <picture>
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/handcream_2.avif" type="image/avif" />
      <source srcset="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/handcream_2.webp" type="image/webp" />
      <img alt="bottle on top of a rock" src="https://storage.googleapis.com/fotographer-cdn/app-static-assets/zen_ctrl/handcream_2.png"  width="24%" />
    </picture>
  </div>
</div>

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

## 🔧 Models (Initial Weights Released)

| Type                  | Name                  | Base         | Resolution | Description                       | links                                                      |
| --------------------- | --------------------- | ------------ | ---------- | --------------------------------- | ---------------------------------------------------------- |
| Subject Generation    | `subject_99000_512`   | FLUX.1       | 1024x1024  | Core model for subject-driven gen | [link](https://huggingface.co/fotographerai/zenctrl_tools) |
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
- 🌐 [Landing Page](https://fotographer.ai/zen-control)
- 🧪 [Try it now on Hugging Face Space](https://huggingface.co/fotographerai/zenctrl_tools/tree/main/weights)
<!-- - 🧠 [Blog]() -->

---

## 🤝 Community Collaboration

We hope to collaborate closely with the open-source community to make **ZenCtrl** a powerful and extensible toolkit for visual content creation.  
Once the source code is released, we welcome contributions in training, expanding supported use cases, and developing new task-specific modules.  
Our vision is to make ZenCtrl the **standard framework** for agentic, high-quality image and video generation — built together, for everyone.
