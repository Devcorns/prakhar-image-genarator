# Text-to-Image Generation System

> **Open-source, local-first image generation powered by Stable Diffusion — optimized for consumer GPUs with 2–4 GB VRAM.**

---

## Table of Contents

1. [Recommended Architecture](#1-recommended-architecture)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Project Structure](#3-project-structure)
4. [Setup Instructions](#4-setup-instructions)
5. [Usage](#5-usage)
6. [Hardware Optimization Strategies](#6-hardware-optimization-strategies)
7. [Fine-Tuning with LoRA](#7-fine-tuning-with-lora)
8. [Dataset Structure](#8-dataset-structure)
9. [Performance Tips](#9-performance-tips)

---

## 1. Recommended Architecture

### **Stable Diffusion v1.5** (via Hugging Face `diffusers`)

| Criterion             | Why SD v1.5                                               |
| --------------------- | --------------------------------------------------------- |
| **VRAM footprint**    | ~3.2 GB in fp16 (fits 4 GB GPUs)                         |
| **Ecosystem**         | Largest community, most LoRA/checkpoints available        |
| **Licensing**         | CreativeML Open RAIL-M — fully open-source, free          |
| **Quality**           | Production-quality 512×512, good upscale to 768           |
| **Fine-tuning**       | LoRA adapters train in ~4 GB VRAM                         |
| **Tooling**           | First-class support in `diffusers`, `peft`, `accelerate`  |

**Alternative models for extreme VRAM constraints:**

| Model                                  | VRAM (fp16) | Notes                         |
| -------------------------------------- | ----------- | ----------------------------- |
| `segmind/small-sd`                     | ~1.8 GB     | Knowledge-distilled SD        |
| `OFA-Sys/small-stable-diffusion-v0`    | ~1.5 GB     | Smallest Stable Diffusion     |
| `stabilityai/stable-diffusion-2-1-base`| ~3.4 GB     | Better anatomy, 512×512       |

---

## 2. Architecture Deep Dive

Stable Diffusion is a **Latent Diffusion Model (LDM)** with four main components:

```
                    ┌──────────────┐
  "a cat on a       │  CLIP Text   │   text embeddings
   rainbow cloud" ──│  Encoder     │──────────┐
                    └──────────────┘          │
                                              ▼
 Random Noise ──► ┌──────────────────────────────┐
  (latent space)  │      UNet  (Denoiser)         │ ◄── timestep t
                  │  Iterative noise prediction   │
                  └──────────┬───────────────────┘
                             │  denoised latents
                             ▼
                    ┌──────────────┐
                    │  VAE Decoder │   pixel-space image
                    │  (Autoencoder)│──► 512×512 RGB
                    └──────────────┘
```

### 2.1 Text Encoder (CLIP ViT-L/14)

- **What:** A frozen CLIP transformer that converts text → 77×768 token embeddings.
- **Parameters:** ~123 M
- **Role:** Provides semantic understanding of the prompt. The embeddings
  condition the UNet at each denoising step via cross-attention.

### 2.2 VAE (Variational Autoencoder)

- **What:** A convolutional encoder-decoder that compresses images 8× spatially.
- **Parameters:** ~83 M
- **Role:**
  - **Encoder:** 512×512×3 image → 64×64×4 latent (used during training / img2img).
  - **Decoder:** 64×64×4 latent → 512×512×3 image (used during generation).
- **Why latent space?** Operating at 64×64 instead of 512×512 reduces compute by **64×**,
  making diffusion tractable on consumer GPUs.

### 2.3 UNet (Noise Predictor)

- **What:** A time-conditional U-Net with residual blocks, self-attention, and
  cross-attention layers.
- **Parameters:** ~860 M (the largest component)
- **Architecture:**
  - **Encoder path:** Downsamples latents through ResNet blocks + attention.
  - **Middle block:** Bottleneck with self-attention + cross-attention to text.
  - **Decoder path:** Upsamples with skip connections from the encoder.
  - **Cross-attention:** At each resolution, attends to CLIP text embeddings —
    this is how text controls the image content.
- **Input:** noisy latent + timestep embedding + text embeddings.
- **Output:** predicted noise ε (or v-prediction, depending on scheduler).

### 2.4 Diffusion Process

**Forward process (training):**
1. Take a clean image, encode it to latent space with VAE.
2. Add Gaussian noise at a random timestep `t ∈ [0, T]`.
3. Train the UNet to predict the added noise.

**Reverse process (generation):**
1. Start from pure Gaussian noise in latent space (64×64×4).
2. For each timestep `T → 0` (guided by a scheduler like DPM++ 2M):
   - The UNet predicts the noise component.
   - The scheduler removes a portion of the predicted noise.
   - **Classifier-Free Guidance (CFG):** Run UNet twice — once with the text
     prompt, once unconditionally — and blend:
     `ε_guided = ε_uncond + scale × (ε_cond − ε_uncond)`
3. Decode the clean latent through the VAE decoder → final image.

---

## 3. Project Structure

```
text-to-image/
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── __init__.py
│   ├── config.py               # All configuration dataclasses
│   ├── generator.py            # TextToImageGenerator class
│   ├── optimizations.py        # VRAM/speed optimization utilities
│   ├── schedulers.py           # Scheduler factory
│   ├── train_lora.py           # LoRA fine-tuning script
│   └── prepare_dataset.py      # Dataset preparation utilities
│
├── data/
│   ├── train/                  # Training images + captions
│   │   ├── 0000.png
│   │   ├── 0000.txt
│   │   └── ...
│   └── raw/                    # Unprocessed source images
│
├── models/
│   ├── cache/                  # HuggingFace model cache
│   └── lora/                   # Saved LoRA adapter weights
│       ├── checkpoint-250/
│       ├── checkpoint-500/
│       └── final/
│
└── outputs/                    # Generated images
    ├── gen_20260227_143021_000.png
    └── ...
```

---

## 4. Setup Instructions

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with 2+ GB VRAM (or CPU — very slow)
- **CUDA 11.8 or 12.x** toolkit installed

### Step-by-Step

```bash
# 1. Clone / navigate to the project
cd "text to image"

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install PyTorch with CUDA support
#    (visit https://pytorch.org/get-started/locally/ for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt

# 5. (Optional) Install xformers for memory-efficient attention
pip install xformers

# 6. Verify GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

> **First run:** The model (~4 GB) will be downloaded automatically from
> Hugging Face and cached in `./models/cache/`.

---

## 5. Usage

### Basic Generation

```bash
# Simple prompt
python main.py "a majestic castle on a floating island, fantasy art, 4k"

# With parameters
python main.py "cyberpunk cityscape at night, neon lights, rain" \
    --steps 40 --cfg 8.0 --seed 42 --width 768 --height 512

# Multiple images
python main.py "portrait of a medieval knight" --num 4 --seed 123

# Estimate VRAM before generating
python main.py "test" --estimate-vram --width 768 --height 768
```

### Python API

```python
from src.config import AppConfig
from src.generator import TextToImageGenerator

# Create with default config (optimized for low VRAM)
config = AppConfig()
gen = TextToImageGenerator(config)
gen.load_model()

# Generate
images = gen.generate(
    prompt="a serene Japanese garden in autumn, watercolor style",
    seed=42,
    num_images=2,
)

# Save
gen.save_images(images, prefix="garden")

# Load LoRA for custom style
gen.load_lora_weights("./models/lora/final", scale=0.8)
images = gen.generate("a portrait in my custom style")
gen.save_images(images, prefix="lora")

# Cleanup
gen.unload()
```

### Using Alternative Models (Ultra-Low VRAM)

```python
config = AppConfig()
config.model.model_id = "segmind/small-sd"             # Only ~1.8 GB VRAM
config.optimization.enable_sequential_cpu_offload = True # Even less VRAM
config.optimization.enable_model_cpu_offload = False

gen = TextToImageGenerator(config)
gen.load_model()
```

---

## 6. Hardware Optimization Strategies

### VRAM Budget Guide

| VRAM Available | Recommended Configuration                                       |
| -------------- | --------------------------------------------------------------- |
| **2 GB**       | `segmind/small-sd` + sequential CPU offload + fp16              |
| **3 GB**       | SD v1.5 + model CPU offload + attention slicing + fp16          |
| **4 GB**       | SD v1.5 + model CPU offload + fp16 (comfortable)               |
| **6 GB+**      | SD v1.5 on GPU directly + xformers + fp16                       |
| **8 GB+**      | SD v2.1 768px or SDXL with fp16                                 |

### Optimization Breakdown

| Optimization              | VRAM Saved | Speed Impact   | How to Enable                    |
| ------------------------- | ---------- | -------------- | -------------------------------- |
| fp16 precision            | ~50%       | ±0%            | `config.model.precision = "fp16"`|
| Attention slicing         | ~25%       | −5%            | Enabled by default               |
| VAE slicing               | ~200 MB    | ±0%            | Enabled by default               |
| Disable safety checker    | ~300 MB    | +5%            | Enabled by default               |
| Model CPU offload         | ~50%       | −30%           | Enabled by default               |
| Sequential CPU offload    | ~75%       | −300%          | For 2 GB GPUs only               |
| xformers attention        | ~20%       | +15%           | Requires `pip install xformers`  |
| torch.compile             | ±0%        | +15%           | First run slow, then faster      |
| Channels-last format      | ±0%        | +5-10%         | Enabled by default               |
| Fewer inference steps     | ±0%        | Linear         | Use 20-25 steps with DPM++      |
| Smaller resolution        | Quadratic  | Quadratic      | Use 384×384 or 448×448           |

### Emergency: Not Enough VRAM

```python
# Nuclear option: everything on CPU, only active layer on GPU
config.optimization.enable_sequential_cpu_offload = True
config.optimization.enable_model_cpu_offload = False
config.optimization.enable_attention_slicing = True
config.optimization.enable_vae_slicing = True

# Use smallest model
config.model.model_id = "OFA-Sys/small-stable-diffusion-v0"

# Reduce resolution
config.generation.width = 384
config.generation.height = 384
config.generation.num_inference_steps = 20
```

---

## 7. Fine-Tuning with LoRA

**LoRA (Low-Rank Adaptation)** lets you teach the model new concepts using only
**4-8 training images** and **~4 GB VRAM**.

### How LoRA Works

Instead of fine-tuning the full 860M-parameter UNet, LoRA:
1. Freezes all original weights.
2. Injects small trainable matrices (rank 4 = only ~0.1% of parameters).
3. The adapter learns `ΔW = A × B` where A and B are low-rank.
4. Final weight = `W_original + α × (A × B)`.

This keeps the adapter file tiny (~5 MB) and training fast (~10 min on a 4 GB GPU).

### Training Steps

```bash
# 1. Prepare your dataset (see Section 8)
python -m src.prepare_dataset --src ./data/raw --dst ./data/train --size 512

# 2. Train LoRA adapter
python -m src.train_lora --dataset_dir ./data/train --max_steps 1000 --rank 4 --lr 1e-4

# 3. Generate with your trained adapter
python main.py "a photo of sks dog in a spacesuit" --lora ./models/lora/final
```

### LoRA Training Tips

| Parameter         | Low Quality/Fast | Balanced       | High Quality    |
| ----------------- | ---------------- | -------------- | --------------- |
| `rank`            | 2                | 4              | 8-16            |
| `max_steps`       | 200              | 500-1000       | 2000+           |
| `learning_rate`   | 5e-4             | 1e-4           | 5e-5            |
| `train_images`    | 4-8              | 10-20          | 30-100          |
| `resolution`      | 384              | 512            | 512             |
| Adapter file size | ~2 MB            | ~5 MB          | ~15 MB          |

---

## 8. Dataset Structure

### For LoRA Fine-Tuning

```
data/
├── raw/                        # Your original images (any size)
│   ├── photo_001.jpg
│   ├── photo_001.txt           # Optional: caption
│   ├── photo_002.png
│   └── ...
│
└── train/                      # Processed by prepare_dataset.py
    ├── 0000.png                # 512×512, center-cropped
    ├── 0000.txt                # "a photo of sks dog playing in the park"
    ├── 0001.png
    ├── 0001.txt
    └── ...
```

### Caption Guidelines

| Training Goal      | Caption Strategy                                              |
| ------------------- | ------------------------------------------------------------ |
| **Specific object** | Use a unique token: `"a photo of sks dog in [scene]"`       |
| **Art style**       | Describe the style: `"a painting in the style of [artist]"` |
| **Person/character**| Use a trigger word: `"photo of ohwx person, [description]"` |
| **Texture/material**| `"close-up of xyz texture on [surface]"`                     |

### Captioning Tips

- **Unique trigger word:** Use a rare token like `sks`, `ohwx`, or `xyz` to
  represent your concept. This prevents the model from confusing it with
  existing knowledge.
- **Variety:** Vary backgrounds, lighting, and angles in your images.
- **Quality:** Remove blurry, occluded, or watermarked images.
- **Minimum:** 4-8 images for an object, 10-20 for a style, 20+ for a face.

---

## 9. Performance Tips

### Generation Speed

1. **Use DPM++ 2M Karras scheduler** — converges in 20-30 steps (vs. 50 for DDIM).
2. **Reduce steps:** 20 steps is often sufficient for DPM++ schedulers.
3. **Use xformers** if on NVIDIA: 15-20% faster + less VRAM.
4. **torch.compile:** 10-20% faster after warmup (PyTorch 2.0+).
5. **Batch generation:** Multiple images in one call is faster than sequential.

### Quality Tips

1. **Guidance scale 7-9** is the sweet spot. Too high → oversaturated.
2. **Negative prompts matter:** Always include quality-related negatives.
3. **Seeds for iteration:** Fix a seed, then tweak the prompt to refine.
4. **Higher steps for detail:** Use 40-50 steps for final renders.
5. **Prompt engineering:**
   - Put important elements first.
   - Add quality boosters: `"masterpiece, best quality, highly detailed"`.
   - Specify medium: `"oil painting"`, `"photograph"`, `"digital art"`.
   - Specify lighting: `"dramatic lighting"`, `"soft diffused light"`.

### System-Level Optimizations

1. **Close other GPU apps** (browsers, games) to free VRAM.
2. **Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** for better memory allocation.
3. **Use an SSD** — model loading is I/O-bound on first load.
4. **Pre-download models** to avoid network latency during generation.

---

## License

This project uses only open-source components:

| Component          | License                    |
| ------------------ | -------------------------- |
| Stable Diffusion   | CreativeML Open RAIL-M     |
| diffusers          | Apache 2.0                 |
| transformers       | Apache 2.0                 |
| PyTorch            | BSD                        |
| CLIP               | MIT                        |
| This project       | MIT                        |
