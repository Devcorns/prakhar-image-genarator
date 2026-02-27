"""
Text-to-Image Generator — core pipeline wrapper.

Loads a Stable Diffusion model, applies low-VRAM optimizations,
and exposes a simple `generate()` API.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from PIL import Image

from src.config import AppConfig, Precision
from src.optimizations import (
    apply_pipeline_optimizations,
    flush_memory,
    get_device,
    get_torch_dtype,
    log_memory_usage,
)
from src.schedulers import build_scheduler

logger = logging.getLogger(__name__)


class TextToImageGenerator:
    """
    High-level wrapper around Stable Diffusion for text-to-image generation,
    optimized for 2-4 GB VRAM consumer GPUs.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.device = get_device()
        self.dtype = get_torch_dtype(self.config.model.precision.value, self.device)
        self.pipe: Optional[StableDiffusionPipeline] = None

    # ── Model Loading ───────────────────────────────────────────────

    def load_model(self) -> "TextToImageGenerator":
        """Download (if needed) and load the Stable Diffusion pipeline."""
        model_id = self.config.model.model_id
        logger.info(f"Loading model: {model_id}  (dtype={self.dtype})")

        kwargs = {
            "torch_dtype": self.dtype,
            "cache_dir": self.config.model.cache_dir,
        }

        # Disable safety checker to save ~300 MB VRAM
        if not self.config.model.enable_safety_checker:
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

        # Swap scheduler
        sched_name = self.config.generation.scheduler.value
        self.pipe.scheduler = build_scheduler(sched_name, self.pipe.scheduler.config)
        logger.info(f"Scheduler: {sched_name}")

        # Auto-disable GPU-only optimizations when running on CPU
        opt = self.config.optimization
        if self.device.type != "cuda":
            opt.enable_model_cpu_offload = False
            opt.enable_sequential_cpu_offload = False
            opt.enable_xformers = False
            opt.enable_channels_last = False
            logger.info("CPU mode: disabled GPU-only optimizations")

        # Apply all VRAM / speed optimizations
        apply_pipeline_optimizations(self.pipe, self.config)

        # Move to device only if no CPU-offload strategy is active
        if not (opt.enable_model_cpu_offload or opt.enable_sequential_cpu_offload):
            self.pipe.to(self.device)

        log_memory_usage()
        return self

    def load_lora_weights(self, lora_path: str, scale: float = 1.0) -> None:
        """Load LoRA adapter weights into the pipeline."""
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")
        logger.info(f"Loading LoRA weights from {lora_path}")
        self.pipe.load_lora_weights(lora_path)
        try:
            self.pipe.fuse_lora(lora_scale=scale)
        except TypeError:
            # Older diffusers versions use adapter_name
            self.pipe.fuse_lora()

    # ── Image Generation ────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> List[Image.Image]:
        """
        Generate images from a text prompt.

        Parameters override config defaults when provided.
        Returns a list of PIL Images.
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")

        gen_cfg = self.config.generation
        _width = width or gen_cfg.width
        _height = height or gen_cfg.height
        _steps = num_inference_steps or gen_cfg.num_inference_steps
        _cfg = guidance_scale if guidance_scale is not None else gen_cfg.guidance_scale
        _neg = negative_prompt or gen_cfg.negative_prompt
        _seed = seed if seed is not None else gen_cfg.seed

        # Deterministic generation
        generator = None
        if _seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(_seed)
            logger.info(f"Using seed: {_seed}")

        logger.info(
            f"Generating {num_images} image(s) | "
            f"{_width}x{_height} | steps={_steps} | cfg={_cfg}"
        )
        logger.info(f"Prompt: {prompt}")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=_neg,
            width=_width,
            height=_height,
            num_inference_steps=_steps,
            guidance_scale=_cfg,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        flush_memory()
        log_memory_usage()

        return result.images

    # ── Saving ──────────────────────────────────────────────────────

    def save_images(
        self,
        images: List[Image.Image],
        output_dir: Optional[str] = None,
        prefix: str = "gen",
    ) -> List[str]:
        """Save a list of PIL images to disk. Returns saved file paths."""
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = []
        for idx, img in enumerate(images):
            fname = f"{prefix}_{timestamp}_{idx:03d}.png"
            path = out / fname
            img.save(path)
            saved.append(str(path))
            logger.info(f"Saved: {path}")

        return saved

    # ── Cleanup ─────────────────────────────────────────────────────

    def unload(self) -> None:
        """Release the pipeline and free GPU memory."""
        del self.pipe
        self.pipe = None
        flush_memory()
        logger.info("Pipeline unloaded, memory freed.")
