"""从公开数据集下载火/烟图片与 YOLO 标签，并整理成可直接使用的目录。

默认数据源为 liuzifan/fire_smoke_datasets（GitHub，0=fire、1=smoke）。
在你自己的电脑上运行即可，不需要这里受限的网络环境。

用法示例：
  python fetch_public_data.py --out ../public_data --fire 650 --smoke 650
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


API = "https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def http_get_json(url: str, retries: int = 3) -> dict:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "role3-fetch"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1 + i)
    raise RuntimeError(f"请求失败: {url}: {last}")


def http_download(url: str, out: Path, retries: int = 3) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "role3-fetch"})
            with urllib.request.urlopen(req, timeout=120) as resp, out.open("wb") as f:
                f.write(resp.read())
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1 + i)
    raise RuntimeError(f"下载失败: {url}: {last}")


def stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="下载并整理公开火/烟数据集")
    parser.add_argument("--repo", default="liuzifan/fire_smoke_datasets")
    parser.add_argument("--branch", default="master")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / ".." / "public_data")
    parser.add_argument("--fire", type=int, default=650, help="fire 类目标数量")
    parser.add_argument("--smoke", type=int, default=650, help="smoke 类目标数量")
    parser.add_argument("--mixed", type=int, default=0, help="fire_smoke 混合高分辨率图数量（质量优先时使用）")
    args = parser.parse_args()

    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    tree = http_get_json(API.format(repo=args.repo, branch=args.branch))
    if tree.get("truncated"):
        print("[warn] 文件树被截断，部分文件可能缺失。")
    entries = tree.get("tree", [])

    jpgs = [e for e in entries if e["path"].lower().endswith(".jpg")]
    txts = {stem(e["path"]): e for e in entries if e["path"].lower().endswith(".txt")}

    def pick(prefix: str, n: int):
        return sorted(
            [e for e in jpgs if e["path"].rsplit("/", 1)[-1].startswith(prefix)],
            key=lambda e: e["path"],
        )[:n]

    selected = pick("fire-image_", args.fire) + pick("smoke-image_", args.smoke) + pick("fire_smoke_", args.mixed)
    manifest = []
    for img in selected:
        img_stem = stem(img["path"])
        label = txts.get(img_stem)
        if label is None:
            print(f"[skip] 无标签: {img['path']}")
            continue
        img_url = RAW.format(repo=args.repo, branch=args.branch, path=img["path"])
        lab_url = RAW.format(repo=args.repo, branch=args.branch, path=label["path"])
        http_download(img_url, out / "images" / f"{img_stem}.jpg")
        http_download(lab_url, out / "labels" / f"{img_stem}.txt")
        manifest.append({"image": f"images/{img_stem}.jpg", "label": f"labels/{img_stem}.txt", "size_bytes": img.get("size", 0)})

    (out / "manifest.json").write_text(
        json.dumps({"source": args.repo, "branch": args.branch, "class_names": ["fire", "smoke"], "files": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[fetch] 完成：{len(manifest)} 张图片与标签已下载到 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
