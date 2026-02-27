"""
LoRA Fine-Tuning Script for Stable Diffusion.

Train a lightweight LoRA adapter on your own images so the model
learns a new concept (style, object, character, etc.) while using
minimal VRAM (~4 GB with all optimizations enabled).

Usage:
    python -m src.train_lora
    python -m src.train_lora --dataset_dir ./data/train --max_steps 800
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

from src.config import AppConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════════

class TextImageDataset(Dataset):
    """
    Expects a folder structured as:

        dataset_dir/
        ├── image_001.png
        ├── image_001.txt      # caption for image_001
        ├── image_002.jpg
        ├── image_002.txt
        └── ...

    If a .txt caption file is missing, the filename (without extension)
    is used as the caption.
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, dataset_dir: str, resolution: int = 512):
        self.root = Path(dataset_dir)
        self.images = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {dataset_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load caption
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()
        else:
            caption = img_path.stem.replace("_", " ").replace("-", " ")

        return {"pixel_values": image, "caption": caption}


# ════════════════════════════════════════════════════════════════════
#  Training Loop
# ════════════════════════════════════════════════════════════════════

def train(config: AppConfig):
    lora_cfg = config.lora
    model_cfg = config.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if lora_cfg.mixed_precision == "fp16" else torch.float32

    logger.info(f"Device: {device} | dtype: {weight_dtype}")
    logger.info(f"Base model: {model_cfg.model_id}")

    # ── Load components ─────────────────────────────────────────────
    tokenizer = CLIPTokenizer.from_pretrained(
        model_cfg.model_id, subfolder="tokenizer", cache_dir=model_cfg.cache_dir
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_cfg.model_id, subfolder="text_encoder", cache_dir=model_cfg.cache_dir
    )
    vae = AutoencoderKL.from_pretrained(
        model_cfg.model_id, subfolder="vae", cache_dir=model_cfg.cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_cfg.model_id, subfolder="unet", cache_dir=model_cfg.cache_dir
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_cfg.model_id, subfolder="scheduler", cache_dir=model_cfg.cache_dir
    )

    # Freeze everything except UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Move frozen models to device in low-precision
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # ── Apply LoRA to UNet ──────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
    )
    unet = get_peft_model(unet, lora_config)
    unet.to(device, dtype=weight_dtype)
    unet.print_trainable_parameters()

    # Gradient checkpointing to save VRAM
    if lora_cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ── Optimizer ───────────────────────────────────────────────────
    if lora_cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            logger.warning("bitsandbytes not installed — falling back to standard AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = optimizer_cls(trainable_params, lr=lora_cfg.learning_rate)

    # ── Dataset & DataLoader ────────────────────────────────────────
    dataset = TextImageDataset(lora_cfg.dataset_dir, lora_cfg.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=lora_cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    logger.info(f"Dataset: {len(dataset)} images from {lora_cfg.dataset_dir}")

    # ── LR Scheduler ────────────────────────────────────────────────
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=lora_cfg.max_train_steps,
    )

    # ── Training ────────────────────────────────────────────────────
    output_dir = Path(lora_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    unet.train()

    progress = tqdm(total=lora_cfg.max_train_steps, desc="Training LoRA")

    while global_step < lora_cfg.max_train_steps:
        for batch in dataloader:
            if global_step >= lora_cfg.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            captions = batch["caption"]

            # Tokenize captions
            tokens = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            # Encode images → latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Encode text
                encoder_hidden_states = text_encoder(tokens)[0]

            # Sample noise & timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()

            # Add noise to latents (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise with UNet
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # MSE loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            loss.backward()

            # Gradient accumulation
            if (global_step + 1) % lora_cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
            global_step += 1

            # Periodic checkpoint
            if global_step % lora_cfg.save_steps == 0:
                ckpt_path = output_dir / f"checkpoint-{global_step}"
                unet.save_pretrained(str(ckpt_path))
                logger.info(f"Checkpoint saved: {ckpt_path}")

    progress.close()

    # ── Save final weights ──────────────────────────────────────────
    final_path = output_dir / "final"
    unet.save_pretrained(str(final_path))
    logger.info(f"LoRA training complete. Weights saved to {final_path}")


# ════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Stable Diffusion")
    parser.add_argument("--model_id", type=str, default=None, help="HuggingFace model ID")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Training images folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save LoRA weights")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=None, help="Training image resolution")
    args = parser.parse_args()

    config = AppConfig()

    if args.model_id:
        config.model.model_id = args.model_id
    if args.dataset_dir:
        config.lora.dataset_dir = args.dataset_dir
    if args.output_dir:
        config.lora.output_dir = args.output_dir
    if args.max_steps:
        config.lora.max_train_steps = args.max_steps
    if args.rank:
        config.lora.rank = args.rank
    if args.lr:
        config.lora.learning_rate = args.lr
    if args.resolution:
        config.lora.resolution = args.resolution

    train(config)


if __name__ == "__main__":
    main()
