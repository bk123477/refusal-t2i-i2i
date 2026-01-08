#!/usr/bin/env python3
"""
PNG to JPG Converter for Final Images
Converts PNG files in final/ directory to 1024x1024 JPG format.
Required for Step1X model which needs exactly 1024x1024 input.

Usage:
    python tools/convert_png_to_jpg.py              # Dry run (preview)
    python tools/convert_png_to_jpg.py --execute   # Actually convert
"""

import argparse
from pathlib import Path
from PIL import Image

# Target size for Step1X model (requires exactly 1024x1024)
TARGET_SIZE = (1024, 1024)
JPEG_QUALITY = 95  # High quality

def find_png_files(final_dir: Path) -> list[Path]:
    """Find all PNG files in final directory."""
    png_files = []
    for race_dir in final_dir.iterdir():
        if race_dir.is_dir() and not race_dir.name.startswith('.'):
            for png_file in race_dir.glob('*.png'):
                png_files.append(png_file)
    return png_files

def convert_png_to_jpg(png_path: Path, dry_run: bool = True) -> dict:
    """
    Convert PNG to 1024x1024 JPG.

    Returns info dict with conversion details.
    """
    jpg_path = png_path.with_suffix('.jpg')

    # Open and get original info
    img = Image.open(png_path)
    original_size = f"{img.width}x{img.height}"
    original_mode = img.mode

    result = {
        "png_path": str(png_path),
        "jpg_path": str(jpg_path),
        "original_size": original_size,
        "target_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
        "original_mode": original_mode,
    }

    if dry_run:
        result["status"] = "dry_run"
        print(f"[DRY RUN] {png_path.name}")
        print(f"  Original: {original_size} {original_mode}")
        print(f"  Target:   {TARGET_SIZE[0]}x{TARGET_SIZE[1]} RGB JPG")
        print(f"  Output:   {jpg_path.name}")
        print()
        return result

    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to target size (high quality)
    img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    # Save as JPG
    img_resized.save(jpg_path, 'JPEG', quality=JPEG_QUALITY)

    # Remove original PNG
    png_path.unlink()

    result["status"] = "converted"
    print(f"[CONVERTED] {png_path.name} → {jpg_path.name}")
    print(f"  {original_size} → {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Convert PNG to 448x448 JPG")
    parser.add_argument("--execute", action="store_true", help="Actually convert (default: dry run)")
    parser.add_argument("--final-dir", type=str, default=None, help="Path to final directory")
    args = parser.parse_args()

    # Find final directory
    if args.final_dir:
        final_dir = Path(args.final_dir)
    else:
        # Auto-detect from script location
        script_dir = Path(__file__).parent
        final_dir = script_dir.parent / "data" / "source_images" / "final"

    if not final_dir.exists():
        print(f"Error: Directory not found: {final_dir}")
        return 1

    print("=" * 60)
    print("PNG to JPG Converter")
    print("=" * 60)
    print(f"Directory: {final_dir}")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print("=" * 60)
    print()

    # Find PNG files
    png_files = find_png_files(final_dir)

    if not png_files:
        print("No PNG files found. All images are already in JPG format.")
        return 0

    print(f"Found {len(png_files)} PNG file(s):")
    print()

    # Convert each
    results = []
    for png_path in sorted(png_files):
        result = convert_png_to_jpg(png_path, dry_run=not args.execute)
        results.append(result)

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total PNG files: {len(png_files)}")

    if args.execute:
        converted = sum(1 for r in results if r["status"] == "converted")
        print(f"Converted: {converted}")
        print()
        print("Done! All images are now 448x448 JPG.")
    else:
        print()
        print("This was a DRY RUN. No files were modified.")
        print("Run with --execute to actually convert files:")
        print()
        print("  python tools/convert_png_to_jpg.py --execute")

    return 0

if __name__ == "__main__":
    exit(main())
