"""生成可复现的合成样例数据，用于在没有真实数据时端到端跑通角色三链路。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import DATA_DIR, save_json, write_yolo_label


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成样例数据")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.default_rng(20261018)
    data = args.data_dir
    (data / "masks_gt").mkdir(parents=True, exist_ok=True)
    (data / "masks_pred").mkdir(parents=True, exist_ok=True)
    stems = [f"img_{i:03d}" for i in range(args.count)]
    images = {s: [640, 640] for s in stems}
    save_json(data / "images.json", images)

    meta_rows = [["image", "time"]]
    for i, stem in enumerate(stems):
        meta_rows.append([stem, float(i * 2.0)])
    (data / "meta.csv").write_text("\n".join(",".join(str(v) for v in r) for r in meta_rows) + "\n", encoding="utf-8")

    for stem in stems:
        # 金标准：1~2 个目标
        gt = []
        n_gt = int(rng.integers(1, 3))
        for _ in range(n_gt):
            cls = int(rng.integers(0, 2))
            cw, ch = float(rng.uniform(0.12, 0.32)), float(rng.uniform(0.12, 0.32))
            cx = float(rng.uniform(cw / 2, 1 - cw / 2))
            cy = float(rng.uniform(ch / 2, 1 - ch / 2))
            gt.append({"class_id": cls, "box": [cx, cy, cw, ch], "conf": None})
        write_yolo_label(data / "gt" / f"{stem}.txt", gt)

        # 标注者 A：金标准小扰动 + 少量漏检
        a = []
        for obj in gt:
            if rng.random() < 0.08:
                continue
            bx = [v + float(rng.normal(0, 0.015)) for v in obj["box"]]
            a.append({"class_id": obj["class_id"], "box": bx, "conf": None})
        write_yolo_label(data / "annotator_a" / f"{stem}.txt", a)

        # 标注者 B：金标准另一种扰动
        b = []
        for obj in gt:
            if rng.random() < 0.1:
                continue
            bx = [v + float(rng.normal(0, 0.018)) for v in obj["box"]]
            b.append({"class_id": obj["class_id"], "box": bx, "conf": None})
        write_yolo_label(data / "annotator_b" / f"{stem}.txt", b)

        # 自动标注：相对人工标注更粗糙，且存在少量误检
        auto = []
        for obj in gt:
            if rng.random() < 0.12:
                continue
            bx = [v + float(rng.normal(0, 0.025)) for v in obj["box"]]
            auto.append({"class_id": obj["class_id"], "box": bx, "conf": float(rng.uniform(0.55, 0.95))})
        if rng.random() < 0.2:
            auto.append({"class_id": int(rng.integers(0, 2)), "box": [float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8)), 0.15, 0.15], "conf": float(rng.uniform(0.4, 0.6))})
        write_yolo_label(data / "auto" / f"{stem}.txt", auto)

        # YOLO 预测：与金标准较接近
        yolo = []
        for obj in gt:
            if rng.random() < 0.04:
                continue
            bx = [v + float(rng.normal(0, 0.012)) for v in obj["box"]]
            yolo.append({"class_id": obj["class_id"], "box": bx, "conf": float(rng.uniform(0.6, 0.99))})
        write_yolo_label(data / "preds_yolo" / f"{stem}.txt", yolo)

        # 开放域预测：零样本，性能略低
        open_preds = []
        for obj in gt:
            if rng.random() < 0.12:
                continue
            bx = [v + float(rng.normal(0, 0.03)) for v in obj["box"]]
            open_preds.append({"class_id": obj["class_id"], "box": bx, "conf": float(rng.uniform(0.45, 0.9))})
        write_yolo_label(data / "preds_open" / f"{stem}.txt", open_preds)

        # 掩码：金标准与预测均为框内椭圆，预测含噪声
        w, h = 640, 640
        yy, xx = np.mgrid[0:h, 0:w]
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        for obj in gt:
            cx, cy, cw, ch = obj["box"]
            x1, y1 = (cx - cw / 2) * w, (cy - ch / 2) * h
            x2, y2 = (cx + cw / 2) * w, (cy + ch / 2) * h
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
            gt_mask[((xx - mx) / (rx or 1)) ** 2 + ((yy - my) / (ry or 1)) ** 2 <= 1] = 1
            rx2, ry2 = rx * float(rng.uniform(0.85, 1.1)), ry * float(rng.uniform(0.85, 1.1))
            pred_mask[((xx - mx) / (rx2 or 1)) ** 2 + ((yy - my) / (ry2 or 1)) ** 2 <= 1] = 1
        np.save(data / "masks_gt" / f"{stem}.npy", gt_mask)
        np.save(data / "masks_pred" / f"{stem}.npy", pred_mask)
        Image.fromarray(gt_mask * 255).save(data / "masks_gt" / f"{stem}.png")
        Image.fromarray(pred_mask * 255).save(data / "masks_pred" / f"{stem}.png")

    print(f"[sample] 已生成 {args.count} 幅合成样例数据到 {data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
