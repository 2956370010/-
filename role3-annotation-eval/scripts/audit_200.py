"""200 幅测试集双人独立标注审计：一致性、自动标注指标、按类别统计与工时估算。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    DATA_DIR,
    OUTPUT_DIR,
    ROOT,
    load_json,
    pairwise_match_metrics,
    per_class_match,
    read_images,
    read_label_dir,
    save_csv,
    save_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="200 幅双人标注一致性与自动标注审计")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "pipeline.json")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "audit")
    args = parser.parse_args()

    cfg_all = load_json(args.config)
    cfg = cfg_all["audit"]
    class_names = cfg_all["project"]["class_names"]
    iou_thr = float(cfg.get("consistency_iou_threshold", 0.5))

    images = read_images(args.data_dir / "images.json")
    auto = read_label_dir(args.data_dir / "auto")
    annotator_a = read_label_dir(args.data_dir / "annotator_a")
    annotator_b = read_label_dir(args.data_dir / "annotator_b")

    auto_vs_a = pairwise_match_metrics(auto, annotator_a, class_names, images, iou_thr)
    auto_vs_b = pairwise_match_metrics(auto, annotator_b, class_names, images, iou_thr)
    a_vs_b = pairwise_match_metrics(annotator_a, annotator_b, class_names, images, iou_thr)

    auto_vs_a_per = per_class_match(auto, annotator_a, class_names, images, iou_thr)
    auto_vs_b_per = per_class_match(auto, annotator_b, class_names, images, iou_thr)
    a_vs_b_per = per_class_match(annotator_a, annotator_b, class_names, images, iou_thr)

    n_images = len(set(auto) | set(annotator_a) | set(annotator_b))
    manual_sec = float(cfg["manual_annotation_seconds_per_image"])
    review_sec = float(cfg["review_seconds_per_image_after_auto"])
    manual_hours = n_images * manual_sec / 3600
    review_hours = n_images * review_sec / 3600
    saved_hours = manual_hours - review_hours
    saved_pct = (manual_sec - review_sec) / manual_sec * 100 if manual_sec else 0.0

    metrics = {
        "evaluated_images": n_images,
        "iou_threshold": iou_thr,
        "auto_vs_annotator_a": auto_vs_a,
        "auto_vs_annotator_b": auto_vs_b,
        "annotator_a_vs_b": a_vs_b,
        "per_class": {
            "auto_vs_annotator_a": auto_vs_a_per,
            "auto_vs_annotator_b": auto_vs_b_per,
            "annotator_a_vs_b": a_vs_b_per,
        },
        "labor_estimate": {
            "images": n_images,
            "manual_seconds_per_image": manual_sec,
            "review_seconds_per_image": review_sec,
            "manual_hours": round(manual_hours, 2),
            "review_hours": round(review_hours, 2),
            "saved_hours": round(saved_hours, 2),
            "saved_percent": round(saved_pct, 2),
        },
    }
    save_json(args.out_dir / "audit_metrics.json", metrics)

    per_class_rows = []
    for cls in class_names:
        per_class_rows.append([cls, "auto-vs-A", auto_vs_a_per[cls]["precision"], auto_vs_a_per[cls]["recall"], auto_vs_a_per[cls]["f1"], auto_vs_a_per[cls]["mean_iou"], auto_vs_a_per[cls]["tp"], auto_vs_a_per[cls]["fp"], auto_vs_a_per[cls]["fn"]])
        per_class_rows.append([cls, "auto-vs-B", auto_vs_b_per[cls]["precision"], auto_vs_b_per[cls]["recall"], auto_vs_b_per[cls]["f1"], auto_vs_b_per[cls]["mean_iou"], auto_vs_b_per[cls]["tp"], auto_vs_b_per[cls]["fp"], auto_vs_b_per[cls]["fn"]])
        per_class_rows.append([cls, "A-vs-B", a_vs_b_per[cls]["precision"], a_vs_b_per[cls]["recall"], a_vs_b_per[cls]["f1"], a_vs_b_per[cls]["mean_iou"], a_vs_b_per[cls]["tp"], a_vs_b_per[cls]["fp"], a_vs_b_per[cls]["fn"]])
    save_csv(args.out_dir / "per_class.csv", ["class", "comparison", "precision", "recall", "f1", "mean_iou", "tp", "fp", "fn"], per_class_rows)

    per_image_rows = []
    for stem in sorted(set(auto) | set(annotator_a) | set(annotator_b)):
        a = len(annotator_a.get(stem, []))
        b = len(annotator_b.get(stem, []))
        au = len(auto.get(stem, []))
        per_image_rows.append([stem, a, b, au])
    save_csv(args.out_dir / "per_image.csv", ["image", "annotator_a_boxes", "annotator_b_boxes", "auto_boxes"], per_image_rows)

    summary = [
        "# 角色三：200 幅双人标注审计结果",
        "",
        f"- 参与审计图像数：{n_images}（任务书要求 200 幅，本文件按实际输入统计）",
        f"- 匹配 IoU 阈值：{iou_thr}",
        "",
        "## 总体指标",
        "",
        "| 对比 | Precision | Recall | F1 | 平均 IoU |",
        "| --- | --- | --- | --- | --- |",
        f"| 自动标注 vs 标注者A | {auto_vs_a['precision']} | {auto_vs_a['recall']} | {auto_vs_a['f1']} | {auto_vs_a['mean_iou']} |",
        f"| 自动标注 vs 标注者B | {auto_vs_b['precision']} | {auto_vs_b['recall']} | {auto_vs_b['f1']} | {auto_vs_b['mean_iou']} |",
        f"| 标注者A vs 标注者B | {a_vs_b['precision']} | {a_vs_b['recall']} | {a_vs_b['f1']} | {a_vs_b['mean_iou']} |",
        "",
        "## 工时估算",
        "",
        f"- 人工标注单幅耗时：{manual_sec} 秒；自动预标注后复核单幅耗时：{review_sec} 秒",
        f"- 预计节省工时：{saved_hours:.2f} 小时（{saved_pct:.1f}%）",
        "",
        "## 说明",
        "",
        "标注者一致性（A vs B）反映人工标注协议本身的稳定性；自动标注指标反映预标注相对人工标注的质量。",
    ]
    (args.out_dir / "audit_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"[audit] 完成：{n_images} 幅图像。")
    print(f"[audit] auto-vs-A F1={auto_vs_a['f1']} mIoU={auto_vs_a['mean_iou']}")
    print(f"[audit] A-vs-B   F1={a_vs_b['f1']} mIoU={a_vs_b['mean_iou']}")
    print(f"[audit] 预计节省工时 {saved_hours:.2f} 小时")
    return 0


if __name__ == "__main__":
    sys.exit(main())
