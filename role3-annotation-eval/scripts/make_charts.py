"""根据 output/ 里已有的指标 JSON 生成报告图表（真实数据与样例数据通用）。"""
from __future__ import annotations

import sys
from pathlib import Path

from charts import bar_chart, line_chart
from common import OUTPUT_DIR, load_json


def main() -> int:
    charts_dir = OUTPUT_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    audit_path = OUTPUT_DIR / "audit" / "audit_metrics.json"
    if audit_path.exists():
        audit = load_json(audit_path)
        ava = audit["auto_vs_annotator_a"]
        avb = audit["auto_vs_annotator_b"]
        ab = audit["annotator_a_vs_b"]
        bar_chart(
            ["Precision", "Recall", "F1", "Mean IoU"],
            [
                {"label": "Auto vs A", "values": [ava["precision"], ava["recall"], ava["f1"], ava["mean_iou"]]},
                {"label": "Auto vs B", "values": [avb["precision"], avb["recall"], avb["f1"], avb["mean_iou"]]},
                {"label": "A vs B", "values": [ab["precision"], ab["recall"], ab["f1"], ab["mean_iou"]]},
            ],
            charts_dir / "audit_metrics.png",
            "Auto-annotation audit",
            "Score",
        )

    compare_path = OUTPUT_DIR / "compare" / "comparison_metrics.json"
    if compare_path.exists():
        cmp = load_json(compare_path)
        yolo = cmp["yolo"]["at_configured_threshold"]
        opend = cmp["open_domain"]["at_configured_threshold"]
        bar_chart(
            ["Precision", "Recall", "F1", "mAP@0.5", "Mean IoU"],
            [
                {"label": "YOLO", "values": [yolo["precision"], yolo["recall"], yolo["f1"], yolo["map50"], yolo["mean_iou"]]},
                {"label": "Open-domain", "values": [opend["precision"], opend["recall"], opend["f1"], opend["map50"], opend["mean_iou"]]},
            ],
            charts_dir / "comparison_metrics.png",
            "YOLO vs open-domain on same test set",
            "Score",
        )

    segment_path = OUTPUT_DIR / "segment" / "segment_area.json"
    if segment_path.exists():
        seg = load_json(segment_path)
        times = [r["time"] for r in seg["rows"]]
        line_chart(
            times,
            [
                {"label": "Predicted area ratio", "values": [r["pred_area_ratio"] for r in seg["rows"]]},
                {"label": "GT area ratio", "values": [r["gt_area_ratio"] for r in seg["rows"]]},
            ],
            charts_dir / "area_trend.png",
            "Fire area ratio trend",
            "Time",
            "Area ratio",
        )
        bar_chart(
            [r["image"] for r in seg["rows"]],
            [{"label": "Area error", "values": [r["area_error"] for r in seg["rows"]]}],
            charts_dir / "area_error.png",
            "Per-image area error",
            "Area error",
        )

    print("[charts] 已根据 output/ 现有指标生成图表到", charts_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
