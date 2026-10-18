"""从公开数据中切出独立测试集，并初始化工程包 data/ 目录。

作用：
1. 把 public_data 里的图片复制到 data/images/；
2. 把公开标签复制到 data/gt/（仅作为初始参考，最终金标准需人工校正）；
3. 生成 data/images.json 与 data/test_set.csv；
4. 初始化 annotator_a / annotator_b / auto / preds_yolo / preds_open / masks_gt / masks_pred 目录。

测试集必须与训练集隔离：角色一训练时要把 data/images 下的这些图排除掉。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

from common import DATA_DIR, ROOT, save_json


def read_boxes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append(int(float(parts[0])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="从公开数据构建独立测试集")
    parser.add_argument("--public-data", type=Path, default=ROOT.parent / "public_data")
    parser.add_argument("--out-data", type=Path, default=DATA_DIR)
    parser.add_argument("--count", type=int, default=0, help="目标测试集数量，0 表示全部可用")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--reset", action="store_true", help="清空 data/ 下评估目录后重建（旧合成样例会被移除）")
    args = parser.parse_args()

    public_images = args.public_data / "images"
    public_labels = args.public_data / "labels"
    if not public_images.exists():
        raise SystemExit(f"找不到公开数据目录: {public_images}")

    eval_dirs = ["annotator_a", "annotator_b", "auto", "preds_yolo", "preds_open", "masks_gt", "masks_pred"]
    if args.reset:
        for name in eval_dirs:
            target = args.out_data / name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
        for name in ["images", "gt"]:
            target = args.out_data / name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
    else:
        for name in eval_dirs + ["images", "gt"]:
            (args.out_data / name).mkdir(parents=True, exist_ok=True)

    records = []
    for img_path in sorted(public_images.glob("*.jpg")) + sorted(public_images.glob("*.png")):
        stem = img_path.stem
        label = public_labels / f"{stem}.txt"
        boxes = read_boxes(label)
        records.append({
            "stem": stem,
            "boxes": boxes,
            "has_fire": 0 in boxes,
            "has_smoke": 1 in boxes,
            "negative": not boxes,
        })

    if args.count and args.count < len(records):
        rng = random.Random(args.seed)
        # 按类别分组后分层抽样，保证 fire/smoke/负样本尽量齐全
        groups = {"both": [], "fire": [], "smoke": [], "negative": []}
        for r in records:
            if r["has_fire"] and r["has_smoke"]:
                groups["both"].append(r)
            elif r["has_fire"]:
                groups["fire"].append(r)
            elif r["has_smoke"]:
                groups["smoke"].append(r)
            else:
                groups["negative"].append(r)
        selected: list[dict] = []
        for key in ["both", "fire", "smoke", "negative"]:
            rng.shuffle(groups[key])
            take = min(len(groups[key]), args.count - len(selected))
            selected.extend(groups[key][:take])
        records = selected

    images_meta = {}
    test_rows = [["image", "has_fire", "has_smoke", "negative", "box_count"]]
    for r in records:
        src_img = public_images / f"{r['stem']}.jpg"
        if not src_img.exists():
            src_img = public_images / f"{r['stem']}.png"
        if not src_img.exists():
            continue
        shutil.copy2(src_img, args.out_data / "images" / src_img.name)
        with Image.open(src_img) as im:
            images_meta[r["stem"]] = [im.width, im.height]
        label_src = public_labels / f"{r['stem']}.txt"
        if label_src.exists():
            shutil.copy2(label_src, args.out_data / "gt" / f"{r['stem']}.txt")
        else:
            (args.out_data / "gt" / f"{r['stem']}.txt").write_text("", encoding="utf-8")
        test_rows.append([r["stem"], "1" if r["has_fire"] else "0", "1" if r["has_smoke"] else "0", "1" if r["negative"] else "0", len(r["boxes"])])

    save_json(args.out_data / "images.json", images_meta)
    with (args.out_data / "test_set.csv").open("w", encoding="utf-8-sig", newline="") as f:
        f.write("\n".join(",".join(str(v) for v in row) for row in test_rows) + "\n")

    print(f"[build_test_set] 测试集已生成：{len(images_meta)} 幅图片 -> {args.out_data}")
    print(f"[build_test_set] 图片目录 data/images，初始标签 data/gt，尺寸表 data/images.json，清单 data/test_set.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
