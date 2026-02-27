"""
Dataset Preparation Utilities.

Helps you organize, resize, and caption images for LoRA fine-tuning.

Usage:
    python -m src.prepare_dataset --src ./raw_images --dst ./data/train --size 512
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def resize_and_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize to `size` on the short edge, then center-crop to square."""
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def prepare_dataset(
    src_dir: str,
    dst_dir: str,
    size: int = 512,
    caption_prefix: str = "",
    default_caption: Optional[str] = None,
    format: str = "png",
):
    """
    Process raw images into a training-ready dataset.

    For each image:
      1. Resize + center-crop to `size` x `size`
      2. Save as .png
      3. Create a .txt caption file (copies existing or generates from filename)

    Args:
        src_dir:  Folder with raw images (and optional .txt captions).
        dst_dir:  Output folder for processed images + captions.
        size:     Target resolution (default 512).
        caption_prefix: Optional prefix added to every caption (e.g. "a photo of sks ").
        default_caption: If set, ALL images get this caption (useful for style training).
        format:   Output image format (png/jpg).
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in src.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        logger.error(f"No images found in {src_dir}")
        return

    logger.info(f"Processing {len(images)} images → {dst_dir} ({size}x{size})")

    for idx, img_path in enumerate(tqdm(images, desc="Preparing")):
        # ── Process image ───────────────────────────────────────────
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Skipping {img_path.name}: {e}")
            continue

        img = resize_and_crop(img, size)

        out_name = f"{idx:04d}.{format}"
        img.save(dst / out_name)

        # ── Create caption ──────────────────────────────────────────
        caption_path = dst / f"{idx:04d}.txt"

        if default_caption:
            caption = default_caption
        else:
            # Check for existing caption file alongside source image
            src_caption = img_path.with_suffix(".txt")
            if src_caption.exists():
                caption = src_caption.read_text(encoding="utf-8").strip()
            else:
                caption = img_path.stem.replace("_", " ").replace("-", " ")

        if caption_prefix:
            caption = caption_prefix + caption

        caption_path.write_text(caption, encoding="utf-8")

    logger.info(f"Done. {len(images)} images saved to {dst}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA fine-tuning")
    parser.add_argument("--src", required=True, help="Source folder with raw images")
    parser.add_argument("--dst", required=True, help="Destination folder for processed dataset")
    parser.add_argument("--size", type=int, default=512, help="Target resolution (default: 512)")
    parser.add_argument("--caption_prefix", type=str, default="", help="Prefix for all captions")
    parser.add_argument("--default_caption", type=str, default=None, help="Override all captions")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg"])
    args = parser.parse_args()

    prepare_dataset(
        src_dir=args.src,
        dst_dir=args.dst,
        size=args.size,
        caption_prefix=args.caption_prefix,
        default_caption=args.default_caption,
        format=args.format,
    )


if __name__ == "__main__":
    main()
