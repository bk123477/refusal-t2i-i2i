#!/usr/bin/env python3
"""
Resize All Images to 512x512
Converts all images in final/ directory to 512x512 JPG format.

Usage:
    python tools/resize_all_to_512.py              # Dry run
    python tools/resize_all_to_512.py --execute   # Actually resize
"""

import argparse
from pathlib import Path
from PIL import Image

TARGET_SIZE = (512, 512)
JPEG_QUALITY = 95

def resize_image(img_path: Path, dry_run: bool = True) -> dict:
    """Resize image to 512x512."""
    img = Image.open(img_path)
    original_size = f"{img.width}x{img.height}"

    result = {
        "path": str(img_path),
        "original_size": original_size,
        "target_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
        "needs_resize": img.width != TARGET_SIZE[0] or img.height != TARGET_SIZE[1]
    }

    if not result["needs_resize"]:
        result["status"] = "skip"
        return result

    if dry_run:
        result["status"] = "dry_run"
        print(f"[DRY RUN] {img_path.name}: {original_size} → 512x512")
        return result

    # Convert to RGB if needed
    if img.mode != 'RGB':
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert('RGB')

    # Resize
    img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    # Save as JPG (overwrite)
    jpg_path = img_path.with_suffix('.jpg')
    img_resized.save(jpg_path, 'JPEG', quality=JPEG_QUALITY)

    # Remove original if it was PNG
    if img_path.suffix.lower() == '.png' and img_path != jpg_path:
        img_path.unlink()

    result["status"] = "resized"
    print(f"[RESIZED] {img_path.name}: {original_size} → 512x512")
    return result

def main():
    parser = argparse.ArgumentParser(description="Resize all images to 512x512")
    parser.add_argument("--execute", action="store_true", help="Actually resize")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    final_dir = script_dir.parent / "data" / "source_images" / "fairface" / "final"

    print("=" * 60)
    print("Resize All Images to 512x512")
    print("=" * 60)
    print(f"Directory: {final_dir}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print("=" * 60)
    print()

    # Find all images
    images = []
    for race_dir in final_dir.iterdir():
        if race_dir.is_dir() and not race_dir.name.startswith('.'):
            for img in race_dir.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images.append(img)

    print(f"Found {len(images)} images")
    print()

    # Process
    needs_resize = 0
    resized = 0

    for img_path in sorted(images):
        result = resize_image(img_path, dry_run=not args.execute)
        if result["needs_resize"]:
            needs_resize += 1
            if result["status"] == "resized":
                resized += 1

    print()
    print("=" * 60)
    print(f"Images needing resize: {needs_resize}/{len(images)}")

    if args.execute:
        print(f"Resized: {resized}")
        print("Done!")
    else:
        print()
        print("Run with --execute to resize:")
        print("  python tools/resize_all_to_512.py --execute")

if __name__ == "__main__":
    main()
