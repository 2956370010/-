# 智能火灾报警系统 - 角色三：标注与评估工程包

本目录是智能火灾报警系统项目中"角色三：标注与评估负责人"的完整工作包，覆盖任务书中的角色三主线：

1. GroundingDINO + SAM 自动预标注流程（提示词、模型版本、阈值、输入尺寸与后处理参数全程记录）
2. 200 幅测试集的双人独立标注审计（自动标注准确率、召回率、平均 IoU、按类别统计、工时估算）
3. YOLO 与开放域模型在相同测试集、相同 IoU、相同输入分辨率下的对比（Precision / Recall / F1 / mAP@0.5 / FPS）
4. 50 幅测试图像的 SAM 分割面积、蔓延趋势与面积误差分析（输出角色二前端所需的标准 CSV / JSON）

## 目录结构

```text
role3-annotation-eval/
  config/pipeline.json        # 预标注与评估的固定协议（提示词、阈值、输入尺寸、IoU、工时参数）
  scripts/
    common.py                 # YOLO 格式读写、IoU、AP、P/R/F1 等公共指标
    preannotate.py            # GroundingDINO + SAM 自动预标注（含离线合成演示模式）
    make_images_json.py       # 扫描图片目录生成宽高索引 images.json
    build_test_set.py         # 从公开数据切出独立测试集并初始化 data/ 目录
    fetch_public_data.py      # 在正常网络下批量下载公开火/烟数据并整理
    public_data_stats.py      # 统计公开数据集类别、框数与分辨率
    audit_200.py              # 200 幅双人标注一致性与自动标注指标审计
    compare_models.py         # YOLO 与开放域模型同集对比
    segment_area.py           # 50 幅分割面积、蔓延趋势与误差分析
    make_charts.py            # 根据 output/ 现有指标生成报告图表
    charts.py                 # SVG 图表生成与 PNG 渲染
    make_sample_data.py       # 生成可复现的样例数据（用于无真实数据时跑通全流程）
    run_all_demo.py           # 一键端到端演示
  data/                       # 数据目录约定（真实数据按此结构放置）
  output/                     # 所有脚本的输出（指标 JSON/CSV、图表 PNG）
  docs/                       # 角色三评估协议、报告章节、交付物清单、周报模板
```

## 快速开始（先跑通样例）

样例数据是合成数据，只用于验证脚本链路，不代表真实指标。真实数据请直接放入 `data/` 并运行对应脚本。

```powershell
$py = 'C:\Users\saini\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py scripts/run_all_demo.py
```

运行结束后查看 `output/`：

- `audit/audit_metrics.json`、`audit/per_class.csv`：200 幅双人标注一致性审计（样例为 10 幅演示）
- `compare/comparison_table.csv`、`compare/comparison_metrics.json`：YOLO 与开放域模型同集对比
- `segment/segment_area.csv`、`segment/segment_area.json`、`segment/area_trend.csv`：分割面积与趋势
- `charts/*.png`：自动标注指标、模型对比、面积趋势、面积误差四类图

## 接入真实数据的目录约定

`data/` 下按如下结构放置真实数据（示例见 `data/README.md`）：

```text
data/
  images.json                # {"图片stem": [宽, 高]}，用于归一化坐标转像素 IoU
  meta.csv                   # 可选，时间/帧序信息，供面积蔓延趋势计算
  gt/                        # 测试集人工金标准 YOLO 标签（子任务3/4共用）
  annotator_a/               # 200 幅审计中标注者 A 的 YOLO 标签
  annotator_b/               # 200 幅审计中标注者 B 的 YOLO 标签
  auto/                      # 自动预标注输出（或由 preannotate.py 生成）
  preds_yolo/                # YOLO 模型在测试集上的预测（YOLO 格式，含置信度）
  preds_open/                # 开放域模型在测试集上的预测（YOLO 格式，含置信度）
  masks_gt/                  # 50 幅金标准掩码（PNG 或 .npy）
  masks_pred/                # SAM 分割掩码（PNG 或 .npy）
```

YOLO 标签格式：每行 `class_id cx cy w h`（归一化）。预测文件额外带置信度：`class_id cx cy w h conf`。
掩码格式：与图像同尺寸的二值图（0 背景 / 255 前景的 PNG，或 0/1 的 `.npy`）。

把真实图片放进 `data/images/` 后，先生成尺寸索引：

```powershell
& $py scripts/make_images_json.py --image-dir data/images
```

## 真实数据运行步骤

按顺序执行（在训练机完成 GroundingDINO 与 SAM 的批量推理，其余为纯指标计算）：

```powershell
$py = 'C:\Users\saini\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

# 1) 自动预标注（真实模式需要 transformers + torch，见 requirements.txt）
& $py scripts/preannotate.py --image-dir data/images --config config/pipeline.json

# 2) 200 幅双人标注审计
& $py scripts/audit_200.py

# 3) YOLO 与开放域模型同集对比
& $py scripts/compare_models.py

# 4) 50 幅分割面积与误差分析
& $py scripts/segment_area.py
```

各脚本均会写出 JSON/CSV/Markdown 证据；图表由 `run_all_demo.py` 或单独调用 `charts.py` 生成。

## 真实模型运行说明

自动预标注脚本在真实数据上运行时需要安装 `transformers`、`torch`、`Pillow`（以及可选的 `segment-anything`）。
请在你的训练机环境中安装，并把模型缓存与权重路径写入 `config/pipeline.json`。脚本会：

- 用 GroundingDINO 依据文本提示词输出候选框；
- 用候选框作为 SAM 的 box prompt 生成分割掩码；
- 按 `box_threshold` / `text_threshold` 过滤并做 NMS；
- 输出 YOLO 格式标签、掩码，以及完整运行配置快照，保证标注可追溯。

开放域对比与分割分析是纯指标计算，不需要 GPU，可在训练机批量推理后直接运行。

给角色二的轻量分割方案、模型路径与调用说明见 `docs/轻量分割方案.md`。

## 公开数据下载

公开火/烟数据（含 YOLO 标签）用 `scripts/fetch_public_data.py` 在你的电脑上全速下载：

```powershell
& $py scripts/fetch_public_data.py --out ..\public_data --fire 650 --smoke 650 --mixed 0
& $py scripts/public_data_stats.py --data-dir ..\public_data
```

数据源、许可与负样本来源说明见 `docs/公开数据来源与许可.md`。

## 真实数据审计与评估流程

公开数据落地后，按 `docs/真实数据审计与评估手册.md` 的顺序执行：切测试集、自动预标注、双人标注、审计、模型对比、分割面积、出图。核心命令概览：

```powershell
$py = 'C:\Users\saini\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py scripts/build_test_set.py --public-data ..\public_data --out-data data --count 200 --reset
& $py scripts/preannotate.py --image-dir data/images
& $py scripts/audit_200.py
& $py scripts/compare_models.py
& $py scripts/segment_area.py
& $py scripts/make_charts.py
```

