"""YOLO 与开放域模型在相同测试集、相同 IoU、相同输入分辨率下的性能对比。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    DATA_DIR,
    OUTPUT_DIR,
    ROOT,
    evaluate_detections,
    load_json,
    read_images,
    read_label_dir,
    save_csv,
    save_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO 与开放域模型同集对比")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "pipeline.json")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "compare")
    args = parser.parse_args()

    cfg_all = load_json(args.config)
    cfg = cfg_all["comparison"]
    class_names = cfg_all["project"]["class_names"]
    iou_thr = float(cfg.get("iou_threshold", 0.5))
    images = read_images(args.data_dir / "images.json")
    gt = read_label_dir(args.data_dir / "gt")
    preds_yolo = read_label_dir(args.data_dir / "preds_yolo")
    preds_open = read_label_dir(args.data_dir / "preds_open")

    yolo_metrics = evaluate_detections(
        preds_yolo, gt, class_names, images, iou_thr, float(cfg.get("conf_threshold_yolo", 0.25))
    )
    open_metrics = evaluate_detections(
        preds_open, gt, class_names, images, iou_thr, float(cfg.get("conf_threshold_open", 0.25))
    )
    yolo_best = evaluate_detections(preds_yolo, gt, class_names, images, iou_thr, None)
    open_best = evaluate_detections(preds_open, gt, class_names, images, iou_thr, None)

    speed_yolo = cfg["speed"]["yolo"]
    speed_open = cfg["speed"]["open_domain"]
    fps_yolo = speed_yolo.get("fps")
    fps_open = speed_open.get("fps")

    def fmt_fps(v):
        return str(v) if v is not None else "待填写"

    result = {
        "protocol": {
            "test_set_images": len(gt),
            "iou_threshold": iou_thr,
            "input_resolution": cfg["input_resolution"],
            "conf_threshold_yolo": cfg["conf_threshold_yolo"],
            "conf_threshold_open": cfg["conf_threshold_open"],
            "note": "开放域模型是否微调及微调样本量需在报告中明确说明；本脚本只计算指标。",
        },
        "yolo": {"at_configured_threshold": yolo_metrics, "at_best_f1_threshold": yolo_best, "speed": speed_yolo},
        "open_domain": {"at_configured_threshold": open_metrics, "at_best_f1_threshold": open_best, "speed": speed_open},
    }
    save_json(args.out_dir / "comparison_metrics.json", result)

    header = ["metric", "yolo", "open_domain"]
    rows = [
        ["precision", yolo_metrics["precision"], open_metrics["precision"]],
        ["recall", yolo_metrics["recall"], open_metrics["recall"]],
        ["f1", yolo_metrics["f1"], open_metrics["f1"]],
        ["map50", yolo_metrics["map50"], open_metrics["map50"]],
        ["mean_iou", yolo_metrics["mean_iou"], open_metrics["mean_iou"]],
        ["fps", fmt_fps(fps_yolo), fmt_fps(fps_open)],
        ["hardware", speed_yolo.get("hardware", "待填写"), speed_open.get("hardware", "待填写")],
    ]
    for cls in class_names:
        rows.append([f"ap_{cls}", yolo_metrics["ap_per_class"].get(cls, 0.0), open_metrics["ap_per_class"].get(cls, 0.0)])
    save_csv(args.out_dir / "comparison_table.csv", header, rows)

    print("[compare] YOLO 与开放域模型对比完成。")
    print(f"[compare] YOLO      P={yolo_metrics['precision']} R={yolo_metrics['recall']} F1={yolo_metrics['f1']} mAP@0.5={yolo_metrics['map50']}")
    print(f"[compare] OpenDomain P={open_metrics['precision']} R={open_metrics['recall']} F1={open_metrics['f1']} mAP@0.5={open_metrics['map50']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
