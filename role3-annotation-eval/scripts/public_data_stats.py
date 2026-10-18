"""统计下载的公开数据集：类别分布、框数、尺寸、负样本数量，并生成统计表。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from common import ROOT, save_csv


def parse_label(path: Path) -> list[int]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append(int(float(parts[0])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="统计公开数据集")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "public_data")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.data_dir

    images_dir = args.data_dir / "images"
    labels_dir = args.data_dir / "labels"
    neg_images_dir = args.data_dir / "images_negative"
    neg_labels_dir = args.data_dir / "labels_negative"

    rows = []
    class_image_counts = {0: 0, 1: 0}
    class_box_counts = {0: 0, 1: 0}
    positive_images = 0
    sizes = []

    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem
        boxes = parse_label(labels_dir / f"{stem}.txt")
        try:
            with Image.open(img_path) as im:
                sizes.append((im.width, im.height))
        except Exception:
            sizes.append((0, 0))
        present = set(boxes)
        if present:
            positive_images += 1
        for c in present:
            class_image_counts[c] += 1
        for c in boxes:
            class_box_counts[c] += 1
        rows.append([stem, "positive", len(boxes), "fire" if 0 in present else "", "smoke" if 1 in present else ""])

    negative_images = 0
    if neg_images_dir.exists():
        for img_path in sorted(neg_images_dir.glob("*.jpg")):
            negative_images += 1
            try:
                with Image.open(img_path) as im:
                    sizes.append((im.width, im.height))
            except Exception:
                sizes.append((0, 0))
            rows.append([img_path.stem, "negative", 0, "", ""])

    fire_images = class_image_counts[0]
    smoke_images = class_image_counts[1]
    both = sum(1 for r in rows if r[1] == "positive" and r[3] == "fire" and r[4] == "smoke")
    save_csv(
        out_dir / "statistics.csv",
        ["image", "kind", "box_count", "has_fire", "has_smoke"],
        rows,
    )
    manifest = []
    for r in rows:
        if r[1] == "positive":
            manifest.append({"kind": "positive", "image": f"images/{r[0]}.jpg", "label": f"labels/{r[0]}.txt"})
        else:
            manifest.append({"kind": "negative", "image": f"images_negative/{r[0]}.jpg", "label": f"labels_negative/{r[0]}.txt"})
    (out_dir / "manifest.json").write_text(
        json.dumps({"source": "liuzifan/fire_smoke_datasets", "class_names": ["fire", "smoke"], "files": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    width_counts = {}
    for w, h in sizes:
        width_counts[f"{w}x{h}"] = width_counts.get(f"{w}x{h}", 0) + 1
    top_sizes = sorted(width_counts.items(), key=lambda kv: -kv[1])[:5]

    lines = [
        "# 公开数据集统计表",
        "",
        "| 项目 | 数量 |",
        "| --- | --- |",
        f"| 正样本图像（火/烟） | {positive_images} |",
        f"| 负样本图像（无火无烟） | {negative_images} |",
        f"| 含 fire 的图像 | {fire_images} |",
        f"| 含 smoke 的图像 | {smoke_images} |",
        f"| 同时含 fire 和 smoke 的图像 | {both} |",
        f"| fire 标注框总数 | {class_box_counts[0]} |",
        f"| smoke 标注框总数 | {class_box_counts[1]} |",
        f"| 图像总量 | {positive_images + negative_images} |",
        "",
        "## 常见分辨率",
        "",
    ]
    for size, cnt in top_sizes:
        lines.append(f"- {size}：{cnt} 张")
    lines += [
        "",
        "说明：正样本标签为 YOLO 归一化格式，0=fire、1=smoke；负样本为空标签（无目标）。",
    ]
    (out_dir / "dataset_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stats] 正样本 {positive_images}，负样本 {negative_images}，fire 框 {class_box_counts[0]}，smoke 框 {class_box_counts[1]}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
