"""50 幅测试图像的 SAM 分割面积、蔓延趋势与面积误差分析。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import (
    DATA_DIR,
    OUTPUT_DIR,
    ROOT,
    load_json,
    save_csv,
    save_json,
)


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return (arr > 0.5).astype(np.uint8)
    else:
        arr = np.array(Image.open(path).convert("L"))
        return (arr > 127).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 分割面积与误差分析")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "pipeline.json")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "segment")
    args = parser.parse_args()

    cfg_all = load_json(args.config)
    cfg = cfg_all["segment"]
    meta = {}
    meta_path = args.data_dir / "meta.csv"
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8-sig").splitlines()[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                meta[parts[0]] = float(parts[1])

    gt_dir = args.data_dir / "masks_gt"
    pred_dir = args.data_dir / "masks_pred"
    stems = sorted({p.stem for p in pred_dir.glob("*")} & {p.stem for p in gt_dir.glob("*")})

    rows = []
    trend = []
    for stem in stems:
        gt = load_mask(next(gt_dir.glob(stem + ".*")))
        pred = load_mask(next(pred_dir.glob(stem + ".*")))
        if gt.shape != pred.shape:
            print(f"[segment] 跳过 {stem}：掩码尺寸不一致")
            continue
        total = gt.size
        gt_ratio = float(gt.sum()) / total
        pred_ratio = float(pred.sum()) / total
        area_error = abs(pred_ratio - gt_ratio)
        rel_error = area_error / gt_ratio if gt_ratio > 0 else (area_error if pred_ratio > 0 else 0.0)
        inter = float((gt & pred).sum())
        union = float((gt | pred).sum())
        iou = inter / union if union > 0 else 0.0
        t = meta.get(stem, len(trend))
        rows.append([stem, round(pred_ratio, 6), round(gt_ratio, 6), round(area_error, 6), round(rel_error, 4), round(iou, 4), t])
        trend.append({"image": stem, "time": t, "pred_area_ratio": round(pred_ratio, 6), "gt_area_ratio": round(gt_ratio, 6), "area_error": round(area_error, 6), "mask_iou": round(iou, 4)})

    rows.sort(key=lambda r: r[-1])
    trend.sort(key=lambda x: x["time"])
    save_csv(args.out_dir / "segment_area.csv", ["image", "pred_area_ratio", "gt_area_ratio", "area_error", "relative_error", "mask_iou", "time"], rows)

    # 蔓延速率：相邻帧面积占比差 / 时间差
    spread = []
    for prev, curr in zip(trend, trend[1:]):
        dt = curr["time"] - prev["time"]
        if dt > 0:
            spread.append({
                "from": prev["image"],
                "to": curr["image"],
                "delta_area_ratio": round(curr["pred_area_ratio"] - prev["pred_area_ratio"], 6),
                "delta_time": round(dt, 3),
                "spread_rate": round((curr["pred_area_ratio"] - prev["pred_area_ratio"]) / dt, 6),
            })
    save_csv(args.out_dir / "area_trend.csv", ["image", "time", "pred_area_ratio", "gt_area_ratio", "area_error", "mask_iou"], [[r["image"], r["time"], r["pred_area_ratio"], r["gt_area_ratio"], r["area_error"], r["mask_iou"]] for r in trend])

    # 失败案例：取 IoU 最低的若干张
    n_fail = int(cfg.get("failure_case_count", 3))
    worst = sorted(trend, key=lambda x: x["mask_iou"])[:n_fail]
    failures = []
    for r in worst:
        if r["pred_area_ratio"] < r["gt_area_ratio"] * 0.85:
            reason = "漏检或欠分割：预测面积显著小于金标准，目标边缘未完整覆盖。"
        elif r["pred_area_ratio"] > r["gt_area_ratio"] * 1.15:
            reason = "过分割或误检：预测面积显著大于金标准，可能混入背景或相邻目标。"
        else:
            reason = "边缘粗糙：面积接近但边界未对齐，导致 IoU 偏低。"
        failures.append({"image": r["image"], "mask_iou": r["mask_iou"], "pred_area_ratio": r["pred_area_ratio"], "gt_area_ratio": r["gt_area_ratio"], "reason": reason})

    save_json(args.out_dir / "segment_area.json", {"sample_count": len(trend), "rows": trend, "spread_rate": spread, "failure_cases": failures})

    # 角色二前端所需标准时间序列（面积占比 + 时间戳）
    frontend = [{"time": r["time"], "image": r["image"], "area_ratio": r["pred_area_ratio"], "mask_iou": r["mask_iou"]} for r in trend]
    save_json(args.out_dir / "frontend_timeseries.json", {"series": frontend, "time_unit": cfg.get("time_unit", "second")})
    save_csv(args.out_dir / "frontend_timeseries.csv", ["time", "image", "area_ratio", "mask_iou"], [[r["time"], r["image"], r["area_ratio"], r["mask_iou"]] for r in frontend])

    lines = ["# 角色三：50 幅分割面积与误差分析", ""]
    lines.append(f"- 参与分析图像数：{len(trend)}（任务书要求 50 幅，本文件按实际输入统计）")
    if trend:
        mean_err = float(np.mean([r["area_error"] for r in trend]))
        mean_iou = float(np.mean([r["mask_iou"] for r in trend]))
        lines.append(f"- 平均面积误差：{mean_err:.6f}")
        lines.append(f"- 平均掩码 IoU：{mean_iou:.4f}")
    lines += ["", "## 失败案例", ""]
    for f in failures:
        lines.append(f"- {f['image']}（IoU={f['mask_iou']}）：{f['reason']}")
    (args.out_dir / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[segment] 完成：{len(trend)} 幅掩码，{len(spread)} 个蔓延区间，{len(failures)} 个失败案例。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
