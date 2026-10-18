"""一键端到端演示：合成数据 -> 审计 -> 模型对比 -> 分割分析 -> 图表。"""
from __future__ import annotations

import subprocess
import sys

from common import ROOT


PYTHON = r"C:\Users\saini\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str) -> None:
    cmd = [PYTHON, str(SCRIPTS / script), *args]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=SCRIPTS)


def main() -> int:
    run("make_sample_data.py", "--count", "10")
    run("audit_200.py")
    run("compare_models.py")
    run("segment_area.py")
    run("make_charts.py")

    print("\n[run_all_demo] 全部完成。查看 output/ 目录：audit / compare / segment / charts。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
