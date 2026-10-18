"""角色三公共工具：YOLO 标签读写、IoU、AP、P/R/F1 与文件 IO。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def read_images(path: Path | None = None) -> dict[str, list[int]]:
    """读取图片尺寸表 {stem: [w, h]}。缺失时返回空，IoU 将按归一化坐标计算。"""
    if path is None:
        path = DATA_DIR / "images.json"
    if not path.exists():
        return {}
    return load_json(path)


def read_yolo_label(path: Path) -> list[dict]:
    """读取单张图 YOLO 标签。返回 [{'class_id': int, 'box': [cx,cy,w,h], 'conf': float|None}]。"""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        box = [float(x) for x in parts[1:5]]
        conf = float(parts[5]) if len(parts) >= 6 else None
        out.append({"class_id": cls, "box": box, "conf": conf})
    return out


def write_yolo_label(path: Path, objects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for obj in objects:
        box = obj["box"]
        parts = [str(int(obj["class_id"]))]
        parts += [f"{v:.6f}" for v in box]
        if obj.get("conf") is not None:
            parts.append(f"{obj['conf']:.6f}")
        lines.append(" ".join(parts))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_label_dir(directory: Path) -> dict[str, list[dict]]:
    """读取一个目录下所有 .txt 标签，返回 {stem: [objects]}。"""
    if not directory.exists():
        return {}
    out = {}
    for p in sorted(directory.glob("*.txt")):
        out[p.stem] = read_yolo_label(p)
    return out


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def box_iou(a: list[float], b: list[float], size: list[int] | None = None) -> float:
    """计算两个 [cx,cy,w,h] 归一化框的 IoU；给出图片尺寸时按像素计算，更准确。"""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    if size is not None:
        w, h = int(size[0]), int(size[1])
        ax1, ax2 = ax1 * w, ax2 * w
        bx1, bx2 = bx1 * w, bx2 * w
        ay1, ay2 = ay1 * h, ay2 * h
        by1, by2 = by1 * h, by2 * h
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _match_predictions(
    preds: list[dict], gts: list[dict], size: list[int] | None, iou_threshold: float
) -> tuple[int, int, int, list[float]]:
    """按类别贪婪匹配。返回 (TP, FP, FN, matched_ious)。"""
    preds = sorted(preds, key=lambda p: -(p.get("conf") if p.get("conf") is not None else 1.0))
    matched_gt = [False] * len(gts)
    tp = fp = 0
    matched_ious: list[float] = []
    for pred in preds:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched_gt[j]:
                continue
            if gt["class_id"] != pred["class_id"]:
                continue
            iou = box_iou(pred["box"], gt["box"], size)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched_gt[best_j] = True
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1
    fn = sum(1 for m in matched_gt if not m)
    return tp, fp, fn, matched_ious


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compute_ap(
    preds: list[dict], gts: list[dict], size: list[int] | None, iou_threshold: float
) -> float:
    """VOC 11 点插值法计算单类别 AP（阈值无关，preds 已按 conf 降序）。"""
    preds = sorted(preds, key=lambda x: -(x["conf"] or 0.0))
    n_gt = len(gts)
    if n_gt == 0:
        return 0.0
    matched_gt = [False] * n_gt
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, pred in enumerate(preds):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched_gt[j] or gt["class_id"] != pred["class_id"]:
                continue
            iou = box_iou(pred["box"], gt["box"], size)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched_gt[best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = 0.0
    for thresh in np.linspace(0.0, 1.0, 11):
        if np.any(recalls >= thresh):
            ap += float(np.max(precisions[recalls >= thresh]))
    return ap / 11.0


def evaluate_detections(
    preds_by_image: dict[str, list[dict]],
    gts_by_image: dict[str, list[dict]],
    class_names: list[str],
    images: dict[str, list[int]] | None = None,
    iou_threshold: float = 0.5,
    conf_threshold: float | None = None,
) -> dict:
    """在相同测试集上评估检测结果。

    返回 overall P/R/F1（按 conf_threshold 或自动取最优 F1 阈值）、
    每类 AP、mAP@0.5、平均 IoU、最佳置信度阈值。
    """
    images = images or {}
    all_preds: list[tuple[list[int] | None, dict]] = []
    for stem, preds in preds_by_image.items():
        size = images.get(stem)
        for p in preds:
            all_preds.append((size, p))

    # 选最佳置信度阈值（在所有预测置信度上扫描，最大化整体 F1）
    if conf_threshold is None:
        candidate = sorted({(p["conf"] or 0.0) for _, p in all_preds}, reverse=True)
        best_thr, best_f1 = 0.0, -1.0
        for thr in candidate + [0.0]:
            tp = fp = fn = 0
            for stem, gts in gts_by_image.items():
                size = images.get(stem)
                preds = [p for p in preds_by_image.get(stem, []) if (p["conf"] or 0.0) >= thr]
                t, f, fneg, _ = _match_predictions(preds, gts, size, iou_threshold)
                tp += t
                fp += f
                fn += fneg
            prec = _safe_div(tp, tp + fp)
            rec = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * prec * rec, prec + rec)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        conf_threshold = best_thr

    tp = fp = fn = 0
    matched_ious: list[float] = []
    for stem, gts in gts_by_image.items():
        size = images.get(stem)
        preds = [p for p in preds_by_image.get(stem, []) if (p["conf"] or 0.0) >= conf_threshold]
        t, f, fneg, ious = _match_predictions(preds, gts, size, iou_threshold)
        tp += t
        fp += f
        fn += fneg
        matched_ious.extend(ious)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # AP 使用各图尺寸；题目要求同输入分辨率，故测试集所有图同尺寸。
    # 以第一张测试图的尺寸作为 IoU 计算基准；若 images 缺失则按归一化坐标。
    representative_size = None
    for stem in gts_by_image:
        if stem in images:
            representative_size = images[stem]
            break
    ap_per_class = {}
    for i, name in enumerate(class_names):
        cp: list[dict] = []
        for stem, preds in preds_by_image.items():
            cp += [p for p in preds if p["class_id"] == i]
        cg: list[dict] = []
        for stem, gts in gts_by_image.items():
            cg += [g for g in gts if g["class_id"] == i]
        ap_per_class[name] = compute_ap(cp, cg, representative_size, iou_threshold)

    map50 = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": round(map50, 4),
        "mean_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        "conf_threshold": round(float(conf_threshold), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "ap_per_class": {k: round(v, 4) for k, v in ap_per_class.items()},
    }


def pairwise_match_metrics(
    preds_by_image: dict[str, list[dict]],
    refs_by_image: dict[str, list[dict]],
    class_names: list[str],
    images: dict[str, list[int]] | None = None,
    iou_threshold: float = 0.5,
) -> dict:
    """把 preds 与 refs 做配对匹配（用于双人一致性 / 自动标注 vs 人工）。"""
    images = images or {}
    tp = fp = fn = 0
    matched_ious: list[float] = []
    for stem in set(preds_by_image) | set(refs_by_image):
        preds = preds_by_image.get(stem, [])
        refs = refs_by_image.get(stem, [])
        size = images.get(stem)
        t, f, fneg, ious = _match_predictions(preds, refs, size, iou_threshold)
        tp += t
        fp += f
        fn += fneg
        matched_ious.extend(ious)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def per_class_match(
    preds_by_image: dict[str, list[dict]],
    refs_by_image: dict[str, list[dict]],
    class_names: list[str],
    images: dict[str, list[int]] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, dict]:
    """逐类配对匹配，返回每个类别的 P/R/F1/mIoU 与 TP/FP/FN。"""
    images = images or {}
    result = {}
    for i, name in enumerate(class_names):
        tp = fp = fn = 0
        ious: list[float] = []
        for stem in set(preds_by_image) | set(refs_by_image):
            preds = [p for p in preds_by_image.get(stem, []) if p["class_id"] == i]
            refs = [g for g in refs_by_image.get(stem, []) if g["class_id"] == i]
            size = images.get(stem)
            t, f, fneg, mi = _match_predictions(preds, refs, size, iou_threshold)
            tp += t
            fp += f
            fn += fneg
            ious.extend(mi)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        result[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
    return result
