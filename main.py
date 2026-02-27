"""
main.py — CLI entry point for text-to-image generation.

Usage:
    python main.py "a cat sitting on a rainbow cloud"
    python main.py "cyberpunk cityscape at night" --steps 40 --seed 42 --width 768 --height 512
    python main.py "portrait of a knight" --lora ./models/lora/final --num 4
"""

import argparse
import logging
import sys

from src.config import AppConfig, Precision, Scheduler
from src.generator import TextToImageGenerator
from src.optimizations import estimate_vram_usage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate images from text prompts using Stable Diffusion"
    )
    p.add_argument("prompt", type=str, help="Text prompt for image generation")
    p.add_argument("--negative", type=str, default=None, help="Negative prompt")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--cfg", type=float, default=7.5, help="Guidance scale")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--num", type=int, default=1, help="Number of images")
    p.add_argument("--output", type=str, default="./outputs", help="Output directory")
    p.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="dpm++_2m_karras",
        choices=[s.value for s in Scheduler],
    )
    p.add_argument("--lora", type=str, default=None, help="Path to LoRA weights")
    p.add_argument("--lora_scale", type=float, default=1.0, help="LoRA strength")
    p.add_argument(
        "--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    p.add_argument(
        "--cpu-offload",
        action="store_true",
        default=True,
        help="Enable model CPU offload (default: on)",
    )
    p.add_argument(
        "--sequential-offload",
        action="store_true",
        default=False,
        help="Enable aggressive sequential CPU offload (slowest, lowest VRAM)",
    )
    p.add_argument("--estimate-vram", action="store_true", help="Only print VRAM estimates, don't generate")
    return p.parse_args()


def main():
    args = parse_args()

    # ── VRAM estimation mode ────────────────────────────────────────
    if args.estimate_vram:
        est = estimate_vram_usage(
            width=args.width,
            height=args.height,
            precision=args.precision,
            cpu_offload=args.cpu_offload or args.sequential_offload,
        )
        print("\n  VRAM Estimate")
        print("  " + "─" * 40)
        for k, v in est.items():
            print(f"  {k:<25} {v:.2f} GB")
        print()
        return

    # ── Build config ────────────────────────────────────────────────
    config = AppConfig()
    config.model.model_id = args.model
    config.model.precision = Precision(args.precision)
    config.generation.width = args.width
    config.generation.height = args.height
    config.generation.num_inference_steps = args.steps
    config.generation.guidance_scale = args.cfg
    config.generation.scheduler = Scheduler(args.scheduler)
    config.generation.seed = args.seed
    config.optimization.enable_model_cpu_offload = args.cpu_offload
    config.optimization.enable_sequential_cpu_offload = args.sequential_offload
    config.output_dir = args.output

    # ── Generate ────────────────────────────────────────────────────
    generator = TextToImageGenerator(config)
    generator.load_model()

    if args.lora:
        generator.load_lora_weights(args.lora, scale=args.lora_scale)

    images = generator.generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        seed=args.seed,
        num_images=args.num,
    )

    saved = generator.save_images(images)
    print(f"\n  Generated {len(saved)} image(s):")
    for path in saved:
        print(f"    → {path}")

    generator.unload()


if __name__ == "__main__":
    main()
