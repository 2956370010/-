"""扫描图片目录，生成 data/images.json 的宽高索引。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from common import DATA_DIR, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="生成图片尺寸索引 images.json")
    parser.add_argument("--image-dir", type=Path, default=DATA_DIR / "images")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "images.json")
    args = parser.parse_args()

    result = {}
    for p in sorted(args.image_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        with Image.open(p) as im:
            result[p.stem] = [im.width, im.height]
    save_json(args.out, result)
    print(f"[images_json] 已写入 {len(result)} 个条目到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
