"""GroundingDINO + SAM 自动预标注流水线。

真实模式：用开放域视觉大模型按文本提示词输出候选框，再以候选框为 box prompt
调用 SAM 生成分割掩码，最终输出 YOLO 格式标签、掩码与完整运行配置快照。

离线演示模式（--demo）：无模型、无网络也可生成确定性样例，用于跑通下游脚本。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import DATA_DIR, OUTPUT_DIR, ROOT, load_json, save_csv, save_json, write_yolo_label


def demo_preannotate(image_stems: list[str], images: dict, cfg: dict, out_labels: Path, out_masks: Path):
    out_labels.mkdir(parents=True, exist_ok=True)
    if cfg.get("save_masks", True):
        out_masks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261017)
    counts = {}
    for i, stem in enumerate(image_stems):
        w, h = images.get(stem, [640, 640])
        objs = []
        for cls in range(2):
            # 每图每类 0~2 个框，位置随机但由种子保证可复现
            for _ in range(rng.integers(0, 3)):
                cw, ch = rng.uniform(0.08, 0.35), rng.uniform(0.08, 0.35)
                cx = rng.uniform(cw / 2, 1 - cw / 2)
                cy = rng.uniform(ch / 2, 1 - ch / 2)
                objs.append({"class_id": cls, "box": [float(cx), float(cy), float(cw), float(ch)], "conf": float(rng.uniform(0.5, 0.98))})
        write_yolo_label(out_labels / f"{stem}.txt", objs)
        counts[stem] = len(objs)
        if cfg.get("save_masks", True):
            mask = np.zeros((int(h), int(w)), dtype=np.uint8)
            yy, xx = np.mgrid[0 : int(h), 0 : int(w)]
            for obj in objs:
                cx, cy, cw, ch = obj["box"]
                x1, y1 = int((cx - cw / 2) * w), int((cy - ch / 2) * h)
                x2, y2 = int((cx + cw / 2) * w), int((cy + ch / 2) * h)
                mask[y1:y2, x1:x2] = 1
            Image.fromarray(mask * 255).save(out_masks / f"{stem}.png")
    return counts


def real_preannotate(image_dir: Path, cfg: dict, out_labels: Path, out_masks: Path, images: dict):
    try:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )
    except ImportError as exc:
        print("[ERROR] 真实预标注需要 transformers 与 torch。请在训练机安装后重试：")
        print("  pip install transformers torch accelerate pillow")
        raise SystemExit(2) from exc

    device = cfg.get("device", "cuda")
    grounding_name = cfg["grounding_model"]
    sam_name = cfg["sam_model"]
    print(f"[preannotate] 加载 GroundingDINO: {grounding_name}")
    gd_processor = AutoProcessor.from_pretrained(grounding_name)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_name).to(device)
    print(f"[preannotate] 加载 SAM: {sam_name}")
    sam_processor = SamProcessor.from_pretrained(sam_name)
    sam_model = SamModel.from_pretrained(sam_name).to(device)
    gd_model.eval()
    sam_model.eval()

    out_labels.mkdir(parents=True, exist_ok=True)
    if cfg.get("save_masks", True):
        out_masks.mkdir(parents=True, exist_ok=True)

    prompts_fire = set(cfg["prompts_fire"])
    counts: dict[str, int] = {}
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    text_prompt = cfg["text_prompt"]
    for img_path in image_paths:
        stem = img_path.stem
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        inputs = gd_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gd_model(**inputs)
        results = gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=cfg["box_threshold"],
            text_threshold=cfg["text_threshold"],
            target_sizes=[image.size[::-1]],
        )[0]
        objs = []
        mask_full = np.zeros((h, w), dtype=np.uint8)
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            label_l = label.lower()
            cls = 0 if any(p in label_l for p in prompts_fire) else 1
            # SAM 以候选框为 box prompt 生成掩码
            if cfg.get("save_masks", True):
                sam_inputs = sam_processor(
                    image, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    sam_outputs = sam_model(**sam_inputs)
                masks = sam_processor.image_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu(),
                )[0]
                mask = masks[0][0] > sam_processor.image_processor.mask_threshold
                mask_full = np.maximum(mask_full, mask.numpy().astype(np.uint8))
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            objs.append({"class_id": cls, "box": [cx, cy, bw, bh], "conf": float(score)})
        write_yolo_label(out_labels / f"{stem}.txt", objs)
        counts[stem] = len(objs)
        if cfg.get("save_masks", True):
            Image.fromarray(mask_full * 255).save(out_masks / f"{stem}.png")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="GroundingDINO + SAM 自动预标注")
    parser.add_argument("--image-dir", type=Path, default=DATA_DIR / "images")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "pipeline.json")
    parser.add_argument("--out-labels", type=Path, default=DATA_DIR / "auto")
    parser.add_argument("--out-masks", type=Path, default=DATA_DIR / "masks_pred")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "preannotation")
    parser.add_argument("--demo", action="store_true", help="离线合成演示模式")
    args = parser.parse_args()

    cfg_all = load_json(args.config)
    cfg = cfg_all["preannotation"]
    images = load_json(DATA_DIR / "images.json") if (DATA_DIR / "images.json").exists() else {}

    if args.demo:
        stems = sorted(images.keys()) or [f"img_{i:03d}" for i in range(10)]
        counts = demo_preannotate(stems, images, cfg, args.out_labels, args.out_masks)
    else:
        counts = real_preannotate(args.image_dir, cfg, args.out_labels, args.out_masks, images)

    record = {
        "mode": "demo" if args.demo else "real",
        "config_snapshot": cfg,
        "image_count": len(counts),
        "box_count_total": int(sum(counts.values())),
        "per_image_counts": counts,
    }
    save_json(args.out_dir / "run_record.json", record)
    rows = [[stem, counts[stem]] for stem in sorted(counts)]
    save_csv(args.out_dir / "manifest.csv", ["image", "box_count"], rows)
    print(f"[preannotate] 完成：{len(counts)} 幅图像，共 {sum(counts.values())} 个框。")
    print(f"[preannotate] 标签目录: {args.out_labels}")
    print(f"[preannotate] 运行记录: {args.out_dir / 'run_record.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
