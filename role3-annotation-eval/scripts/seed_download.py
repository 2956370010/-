"""本环境专用：绕过 DNS 劫持，从 liuzifan/fire_smoke_datasets 下载一份高质量种子子集。

仅用于当前受限环境。在正常网络下请改用 fetch_public_data.py。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT.parent / "_tree_full.json"
OUT = ROOT.parent / "public_data"
RAW_IPS = ["185.199.109.133", "185.199.111.133", "185.199.108.133"]
RAW_HOST = "raw.githubusercontent.com"
RAW = f"https://{RAW_HOST}/liuzifan/fire_smoke_datasets/master/{{path}}"
API_IP = "140.82.112.6"
API_HOST = "api.github.com"
TREE_URL = f"https://{API_HOST}/repos/liuzifan/fire_smoke_datasets/git/trees/master?recursive=1"


def curl_get(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    last_error = None
    for ip in RAW_IPS:
        try:
            subprocess.run(
                [
                    "curl.exe",
                    "--noproxy", "*",
                    "--resolve", f"{RAW_HOST}:443:{ip}",
                    "-sL", "--retry", "2", "--retry-delay", "1", "--max-time", "180",
                    "-o", str(out),
                    url,
                ],
                check=True,
                timeout=200,
            )
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
    raise RuntimeError(f"下载失败: {url}: {last_error}")


def main() -> int:
    if not TREE.exists():
        subprocess.run(
            [
                "curl.exe",
                "--noproxy", "*",
                "--resolve", f"{API_HOST}:443:{API_IP}",
                "-sL", "--max-time", "180",
                "-o", str(TREE),
                TREE_URL,
            ],
            check=True,
        )
    tree = json.loads(TREE.read_text(encoding="utf-8"))
    entries = tree["tree"]
    jpgs = [e for e in entries if e["path"].lower().endswith(".jpg")]
    txts = {p["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]: p for p in entries if p["path"].lower().endswith(".txt")}

    def pick(prefix: str, n: int):
        return sorted([e for e in jpgs if e["path"].rsplit("/", 1)[-1].startswith(prefix)], key=lambda e: e["path"])[:n]

    # 大图 fire_smoke_* 在当前受限网络下不稳定，先下载稳定的小图；大图请用 fetch_public_data.py 在本机全速获取。
    selected = pick("fire-image_", 80) + pick("smoke-image_", 80)
    negatives = [e for e in jpgs if e["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0] not in txts][:19]

    manifest = []
    for img in selected:
        stem = img["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        label = txts.get(stem)
        if label is None:
            continue
        curl_get(RAW.format(path=img["path"]), OUT / "images" / f"{stem}.jpg")
        curl_get(RAW.format(path=label["path"]), OUT / "labels" / f"{stem}.txt")
        manifest.append({"kind": "positive", "image": f"images/{stem}.jpg", "label": f"labels/{stem}.txt", "size_bytes": img.get("size", 0)})

    for img in negatives:
        stem = img["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        curl_get(RAW.format(path=img["path"]), OUT / "images_negative" / f"{stem}.jpg")
        (OUT / "labels_negative" / f"{stem}.txt").write_text("", encoding="utf-8")
        manifest.append({"kind": "negative", "image": f"images_negative/{stem}.jpg", "label": f"labels_negative/{stem}.txt", "size_bytes": img.get("size", 0)})

    (OUT / "manifest.json").write_text(
        json.dumps(
            {"source": "liuzifan/fire_smoke_datasets", "branch": "master", "class_names": ["fire", "smoke"], "files": manifest},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[seed] 完成：{len(manifest)} 个文件（正样本 {sum(1 for m in manifest if m['kind']=='positive')}，负样本 {sum(1 for m in manifest if m['kind']=='negative')}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
