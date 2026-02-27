"""
Configuration module for the Text-to-Image generation system.
Centralizes all settings for model loading, generation, and optimization.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class Scheduler(Enum):
    EULER_A = "euler_a"
    EULER = "euler"
    DPM_PP_2M = "dpm++_2m"
    DPM_PP_2M_KARRAS = "dpm++_2m_karras"
    DDIM = "ddim"
    LMS = "lms"
    PNDM = "pndm"


@dataclass
class ModelConfig:
    """Configuration for model selection and loading."""

    # ── Model Selection ─────────────────────────────────────────────
    # Recommended model for low-VRAM GPUs (2-4 GB):
    #   "stabilityai/stable-diffusion-2-1-base"    ~3.4 GB VRAM (fp16)
    #   "runwayml/stable-diffusion-v1-5"            ~3.2 GB VRAM (fp16)
    #   "segmind/small-sd"                          ~1.8 GB VRAM (fp16, distilled)
    #   "OFA-Sys/small-stable-diffusion-v0"         ~1.5 GB VRAM (smallest)
    model_id: str = "runwayml/stable-diffusion-v1-5"

    # Precision: fp16 is essential for low-VRAM GPUs
    precision: Precision = Precision.FP16

    # Load safety checker (uses ~300 MB extra VRAM)
    enable_safety_checker: bool = False

    # Cache directory for downloaded models
    cache_dir: Optional[str] = "./models/cache"


@dataclass
class GenerationConfig:
    """Configuration for image generation parameters."""

    # Image dimensions (must be multiples of 8)
    width: int = 512
    height: int = 512

    # Diffusion parameters
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = (
        "blurry, bad anatomy, bad hands, cropped, worst quality, "
        "low quality, normal quality, jpeg artifacts, watermark"
    )

    # Reproducibility
    seed: Optional[int] = None

    # Batch settings
    num_images_per_prompt: int = 1

    # Scheduler
    scheduler: Scheduler = Scheduler.DPM_PP_2M_KARRAS


@dataclass
class OptimizationConfig:
    """Hardware optimization settings for low-VRAM GPUs (2-4 GB)."""

    # ── Memory Optimization ─────────────────────────────────────────
    # Attention slicing: splits attention computation into chunks
    # Reduces VRAM by ~25% with minimal speed impact
    enable_attention_slicing: bool = True

    # VAE slicing: processes VAE in slices (saves ~200 MB)
    enable_vae_slicing: bool = True

    # VAE tiling: tile-based VAE decode for very large images
    enable_vae_tiling: bool = False

    # Sequential CPU offload: moves each layer to GPU only when needed
    # Dramatically reduces VRAM (~1.5 GB total) but 3-5x slower
    enable_sequential_cpu_offload: bool = False

    # Model CPU offload: keeps full model on CPU, copies submodules to GPU
    # Good balance: ~2.5 GB VRAM, ~1.5x slower
    enable_model_cpu_offload: bool = True

    # xformers memory-efficient attention (requires xformers installed)
    enable_xformers: bool = False

    # Torch compile (PyTorch 2.0+): JIT compilation for 10-20% speedup
    enable_torch_compile: bool = False

    # Channels-last memory format: 5-10% speedup on modern GPUs
    enable_channels_last: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    # LoRA hyperparameters
    rank: int = 4
    alpha: int = 4
    dropout: float = 0.0
    target_modules: list = field(
        default_factory=lambda: ["to_q", "to_v", "to_k", "to_out.0"]
    )

    # Training settings
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    save_steps: int = 250

    # Dataset
    dataset_dir: str = "./data/train"
    output_dir: str = "./models/lora"
    resolution: int = 512

    # Mixed precision training
    mixed_precision: str = "fp16"

    # Gradient checkpointing (saves VRAM during training)
    gradient_checkpointing: bool = True

    # 8-bit Adam optimizer (saves ~2 GB VRAM)
    use_8bit_adam: bool = True


@dataclass
class AppConfig:
    """Top-level application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    output_dir: str = "./outputs"
