"""
Verification script for Milestone M2.
Checks KGRUN_DATA dataset mount point and reads an image to inspect dimensions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from PIL import Image


def main() -> int:
    """
    Main verification routine.
    """
    print("=== [kgrun M2] Verifying Dataset Mount ===", flush=True)

    data_dir_env = os.environ.get("KGRUN_DATA")
    print(f"KGRUN_DATA environment variable: {data_dir_env}", flush=True)

    if not data_dir_env or not Path(data_dir_env).is_dir():
        print(f"Error: KGRUN_DATA directory '{data_dir_env}' does not exist!", file=sys.stderr)
        return 1

    data_dir = Path(data_dir_env)
    images = sorted(list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png")))
    print(f"Found {len(images)} image file(s) in {data_dir}", flush=True)

    if not images:
        print(f"Error: No image files found in {data_dir}", file=sys.stderr)
        return 1

    sample_image_path = images[0]
    print(f"Reading sample image: {sample_image_path.name}", flush=True)

    with Image.open(sample_image_path) as img:
        width, height = img.size
        img_format = img.format
        img_mode = img.mode

    print(f"Sample image details: Format={img_format}, Mode={img_mode}, Dimensions={width}x{height}", flush=True)
    file_size_bytes = sample_image_path.stat().st_size
    print(f"Sample image file size: {file_size_bytes} bytes ({file_size_bytes / 1024:.2f} KB)", flush=True)

    out_dir = Path(os.environ.get("KGRUN_OUT", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "success",
        "dataset_mount": str(data_dir),
        "total_images": len(images),
        "sample_image": sample_image_path.name,
        "format": img_format,
        "mode": img_mode,
        "width": width,
        "height": height,
        "file_size_bytes": file_size_bytes,
    }

    result_path = out_dir / "dataset_verification.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved verification result to {result_path}", flush=True)
    print("=== [kgrun M2] Verification Completed Successfully ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

