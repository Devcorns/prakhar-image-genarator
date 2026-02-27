"""
Optimization utilities for low-VRAM GPU image generation.
Applies memory and speed optimizations to Stable Diffusion pipelines.
"""

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"GPU detected: {name} ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS backend detected")
        return torch.device("mps")
    else:
        logger.warning("No GPU detected — falling back to CPU (very slow)")
        return torch.device("cpu")


def get_torch_dtype(precision: str, device: torch.device) -> torch.dtype:
    """Return the appropriate torch dtype for the given precision and device."""
    if device.type == "cpu":
        return torch.float32
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(precision, torch.float16)


def apply_pipeline_optimizations(pipe, config) -> None:
    """
    Apply all configured memory/speed optimizations to a diffusion pipeline.

    Optimization priority for 2-4 GB VRAM GPUs:
      1. fp16 precision                  (loaded at init)
      2. Attention slicing               (~25% VRAM savings)
      3. VAE slicing                     (~200 MB savings)
      4. Model CPU offload               (~50% VRAM savings, slight slowdown)
      5. Sequential CPU offload          (~75% VRAM savings, significant slowdown)
      6. xformers memory-efficient attn  (~20% VRAM savings + speed boost)
      7. torch.compile                   (10-20% speed boost, slow first run)
      8. Channels-last memory format     (5-10% speed boost)
    """
    opt = config.optimization

    # ── Attention Slicing ───────────────────────────────────────────
    if opt.enable_attention_slicing:
        pipe.enable_attention_slicing(slice_size="auto")
        logger.info("Enabled attention slicing")

    # ── VAE Slicing ─────────────────────────────────────────────────
    if opt.enable_vae_slicing:
        pipe.enable_vae_slicing()
        logger.info("Enabled VAE slicing")

    # ── VAE Tiling ──────────────────────────────────────────────────
    if opt.enable_vae_tiling:
        pipe.enable_vae_tiling()
        logger.info("Enabled VAE tiling")

    # ── CPU Offload (mutually exclusive — pick one) ─────────────────
    if opt.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        logger.info("Enabled sequential CPU offload (lowest VRAM, slowest)")
    elif opt.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        logger.info("Enabled model CPU offload (balanced)")

    # ── xformers ────────────────────────────────────────────────────
    if opt.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory-efficient attention")
        except Exception as e:
            logger.warning(f"xformers not available: {e}")

    # ── Torch Compile (PyTorch 2.0+) ───────────────────────────────
    if opt.enable_torch_compile:
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            logger.info("Compiled UNet with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # ── Channels-Last Memory Format ─────────────────────────────────
    if opt.enable_channels_last:
        if hasattr(pipe, "unet"):
            pipe.unet.to(memory_format=torch.channels_last)
            logger.info("Set UNet to channels-last memory format")


def flush_memory() -> None:
    """Aggressively free GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.debug("Flushed CUDA memory cache")


def log_memory_usage() -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(
            f"GPU memory — Allocated: {allocated:.2f} GB | "
            f"Reserved: {reserved:.2f} GB | Total: {total:.2f} GB"
        )


def estimate_vram_usage(
    width: int = 512,
    height: int = 512,
    precision: str = "fp16",
    cpu_offload: bool = True,
) -> dict:
    """Estimate VRAM usage for a given configuration (approximate)."""
    # Base model sizes (approximate for SD v1.5)
    bytes_per_param = 2 if precision == "fp16" else 4
    unet_params = 860_000_000
    vae_params = 83_000_000
    text_encoder_params = 123_000_000

    unet_gb = (unet_params * bytes_per_param) / (1024 ** 3)
    vae_gb = (vae_params * bytes_per_param) / (1024 ** 3)
    text_gb = (text_encoder_params * bytes_per_param) / (1024 ** 3)

    # Activation memory scales with resolution
    pixels = width * height
    base_pixels = 512 * 512
    activation_scale = pixels / base_pixels
    activation_gb = 0.5 * activation_scale * (2 if precision == "fp16" else 4) / 2

    if cpu_offload:
        # Only one sub-model on GPU at a time
        peak_gb = max(unet_gb, vae_gb, text_gb) + activation_gb
    else:
        peak_gb = unet_gb + vae_gb + text_gb + activation_gb

    return {
        "unet_gb": round(unet_gb, 2),
        "vae_gb": round(vae_gb, 2),
        "text_encoder_gb": round(text_gb, 2),
        "activation_gb": round(activation_gb, 2),
        "estimated_peak_gb": round(peak_gb, 2),
    }
