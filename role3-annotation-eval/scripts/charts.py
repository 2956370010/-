"""轻量 SVG 图表生成，并用项目自带 Node + sharp 渲染为 PNG（不依赖 matplotlib）。"""
from __future__ import annotations

import html
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_EXE = r"C:\Users\saini\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
NODE_MODULES = r"C:\Users\saini\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"

COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def svg_to_png(svg_text: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_png.with_suffix(".svg")
    svg_path.write_text(svg_text, encoding="utf-8")
    script = (
        "const sharp=require('sharp');"
        "sharp(process.argv[1]).resize({width:1600}).png().toFile(process.argv[2])"
        ".then(()=>process.exit(0)).catch(e=>{console.error(e);process.exit(1)});"
    )
    env = dict(__import__("os").environ)
    env["NODE_PATH"] = NODE_MODULES
    proc = subprocess.run(
        [NODE_EXE, "-e", script, str(svg_path), str(out_png)],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sharp 渲染失败: {proc.stderr}")
    svg_path.unlink(missing_ok=True)


def bar_chart(categories: list[str], series: list[dict], out_png: Path, title: str, ylabel: str) -> None:
    """分组柱状图。series: [{'label': str, 'values': [float, ...]}, ...]"""
    width, height = 900, 480
    left, right, top, bottom = 70, 30, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max([1e-9] + [v for s in series for v in s["values"]])
    n_cat = len(categories)
    n_series = len(series)
    group_w = plot_w / max(1, n_cat)
    bar_w = group_w * 0.72 / max(1, n_series)

    def x(cat_i, ser_i):
        center = left + group_w * cat_i + group_w / 2
        offset = (ser_i - (n_series - 1) / 2) * bar_w
        return center + offset

    def y(v):
        return top + plot_h - (v / max_val) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="{left}" y="30" font-size="20" font-family="Arial, sans-serif" fill="#111827">{_esc(title)}</text>')
    # y 轴刻度
    for i in range(5):
        val = max_val * i / 4
        yy = y(val)
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{width-right}" y2="{yy}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#6b7280">{val:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#9ca3af"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#9ca3af"/>')
    for ci, cat in enumerate(categories):
        for si, s in enumerate(series):
            v = s["values"][ci]
            bx = x(ci, si) - bar_w / 2
            by = y(v)
            bh = (top + plot_h) - by
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{COLORS[si % len(COLORS)]}" rx="1"/>')
        cx = left + group_w * ci + group_w / 2
        parts.append(f'<text x="{cx}" y="{top+plot_h+18}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#374151">{_esc(cat)}</text>')
    # y 轴标题
    parts.append(f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#374151" transform="rotate(-90 18 {top+plot_h/2})">{_esc(ylabel)}</text>')
    # 图例
    lx = left
    ly = height - 18
    for si, s in enumerate(series):
        parts.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{COLORS[si % len(COLORS)]}"/>')
        parts.append(f'<text x="{lx+16}" y="{ly}" font-size="12" font-family="Arial, sans-serif" fill="#374151">{_esc(s["label"])}</text>')
        lx += 24 + 8 * len(s["label"])
    parts.append("</svg>")
    svg_to_png("\n".join(parts), out_png)


def line_chart(x_values: list[float], series: list[dict], out_png: Path, title: str, xlabel: str, ylabel: str) -> None:
    """折线图。series: [{'label': str, 'values': [float, ...]}, ...]"""
    width, height = 900, 480
    left, right, top, bottom = 70, 30, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_vals = [v for s in series for v in s["values"]]
    max_val = max([1e-9] + all_vals)
    min_val = min([0.0] + all_vals)
    x_min, x_max = min(x_values), max(x_values)
    x_range = (x_max - x_min) or 1.0
    val_range = (max_val - min_val) or 1.0

    def sx(v):
        return left + (v - x_min) / x_range * plot_w

    def sy(v):
        return top + plot_h - (v - min_val) / val_range * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="{left}" y="30" font-size="20" font-family="Arial, sans-serif" fill="#111827">{_esc(title)}</text>')
    for i in range(5):
        val = min_val + val_range * i / 4
        yy = sy(val)
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{width-right}" y2="{yy}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#6b7280">{val:.2f}</text>')
    for i in range(5):
        xv = x_min + x_range * i / 4
        xx = sx(xv)
        parts.append(f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="{top+plot_h}" stroke="#f3f4f6"/>')
        parts.append(f'<text x="{xx}" y="{top+plot_h+18}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#374151">{xv:.1f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#9ca3af"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#9ca3af"/>')
    for si, s in enumerate(series):
        pts = " ".join(f"{sx(xv):.1f},{sy(v):.1f}" for xv, v in zip(x_values, s["values"]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[si % len(COLORS)]}" stroke-width="2"/>')
    parts.append(f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#374151" transform="rotate(-90 18 {top+plot_h/2})">{_esc(ylabel)}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-12}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#374151">{_esc(xlabel)}</text>')
    lx = left
    ly = 32
    for si, s in enumerate(series):
        parts.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{COLORS[si % len(COLORS)]}"/>')
        parts.append(f'<text x="{lx+16}" y="{ly}" font-size="12" font-family="Arial, sans-serif" fill="#374151">{_esc(s["label"])}</text>')
        lx += 24 + 8 * len(s["label"])
    parts.append("</svg>")
    svg_to_png("\n".join(parts), out_png)
