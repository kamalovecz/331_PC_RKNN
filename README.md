# 331_PC_RKNN — 港机表面缺陷检测 RKNN 部署全链路

[![Platform](https://img.shields.io/badge/Platform-RK3588-orange)](https://www.rock-chips.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

跨平台 RKNN 模型部署与对齐工程：从 PyTorch 权重到 RK3588 板端推理评估的完整链路。

---

## 目录

- [环境要求](#环境要求)
- [快速开始（5 分钟跑通）](#快速开始5-分钟跑通)
- [详细步骤](#详细步骤)
  - [Step 1: PC 端环境配置](#step-1-pc-端环境配置)
  - [Step 2: 板端环境配置](#step-2-板端环境配置)
  - [Step 3: PT 模型转换](#step-3-pt-模型转换)
  - [Step 4: 板端推理与评估](#step-4-板端推理与评估)
  - [Step 5: PT-RKNN 对齐验证](#step-5-pt-rknn-对齐验证)
- [仓库结构](#仓库结构)
- [核心指标](#核心指标)
- [常见问题](#常见问题)
- [板端速查](#板端速查)

---

## 环境要求

### PC 端（WSL2 Ubuntu）

| 软件 | 版本 | 用途 |
|---|---|---|
| WSL2 Ubuntu | 22.04+ | 运行环境 |
| Miniconda | 最新 | 环境管理 |
| Python | 3.8 | rknn-toolkit2 要求 |
| rknn-toolkit2 | 2.3.2 | ONNX → RKNN 转换 |
| ultralytics | 8.x | YOLO PT → ONNX 导出 |
| onnxruntime | 1.x | ONNX 推理验证 |

### 板端（Orange Pi 5 RK3588）

| 软件 | 版本 | 用途 |
|---|---|---|
| Orange Pi 5 | RK3588 | 推理硬件 |
| Ubuntu | 官方镜像 | 系统 |
| Python | 3.12 (miniforge3) | 运行环境 |
| rknn-toolkit-lite2 | 2.3.2 | RKNN 推理 |

---

## 快速开始（5 分钟跑通）

```bash
# ===== 1. 克隆仓库 =====
git clone https://github.com/kamalovecz/331_PC_RKNN.git
cd 331_PC_RKNN

# ===== 2. PC 端: 转换模型 (WSL) =====
conda activate rknntools
cp /path/to/your_model.pt PC/models/pt/
cd PC && bash run_convert.sh full models/pt/your_model.pt
# 模型自动 scp 到板端

# ===== 3. 板端: 推理评估 (Orange Pi 5) =====
ssh orangepi@<你的板端IP>
cd ~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite
./run_pipeline.sh models/your_model.rknn full
# 输出在 outputs/ 目录
```

---

## 详细步骤

### Step 1: PC 端环境配置

#### 1.1 安装 WSL2 + Ubuntu

在 Windows PowerShell（管理员）中执行：

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

重启电脑，按提示创建 Ubuntu 用户名和密码。

#### 1.2 安装 Miniconda

在 WSL Ubuntu 终端中：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 一路回车，遇到 "Do you accept?" 输入 yes
# 最后问 "Do you wish the installer to initialize?" 输入 yes
# 关闭终端重新打开使 conda 生效
```

#### 1.3 创建环境并安装依赖

```bash
# 创建 Python 3.8 环境（rknn-toolkit2 要求）
conda create -n rknntools python=3.8 -y
conda activate rknntools

# 安装 RKNN Toolkit2 (仓库自带 wheel)
pip install tools/rknn_packages/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 安装其他依赖
pip install ultralytics onnxruntime opencv-python numpy pycocotools
```

#### 1.4 验证安装

```bash
python -c "from rknn.api import RKNN; print('rknn-toolkit2 OK')"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
python -c "import onnxruntime; print('onnxruntime OK')"
```

三条命令都打印 "OK" 即环境就绪。

---

### Step 2: 板端环境配置

#### 2.1 烧录系统

1. 从 [Orange Pi 官网](http://www.orangepi.org/) 下载 Orange Pi 5 的 Ubuntu 镜像
2. 使用 [balenaEtcher](https://www.balena.io/etcher/) 烧录到 SD 卡
3. 插入 SD 卡，接通电源启动

#### 2.2 连接网络并 SSH

```bash
# 查看板端 IP（在板端接显示器/键盘）
ip addr show

# 从 PC SSH 连接
ssh orangepi@<你的板端IP>
# 默认密码通常是 orangepi
```

#### 2.3 安装 miniforge3

```bash
# 下载 ARM64 版本
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
# 全部 yes 即可

# 安装 rknn-toolkit-lite2 (仓库自带 wheel)
pip install tools/rknn_packages/rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 安装依赖
pip install opencv-python numpy pycocotools
```

#### 2.4 部署项目代码

```bash
# 在板端创建项目目录
mkdir -p ~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite

# 方式 A: 从 GitHub clone
git clone https://github.com/kamalovecz/331_PC_RKNN.git /tmp/repo
cp -r /tmp/repo/RKNN/* ~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite/

# 方式 B: 从 PC scp
# 在 PC 端执行:
# scp -r RKNN/* orangepi@<你的板端IP>:~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite/
```

#### 2.5 准备测试数据

```bash
# 创建数据目录
mkdir -p ~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite/data/{images,labels}

# 放入测试图片 (JPEG/PNG) 和 YOLO 格式标注 (TXT)
# 标注格式: <class_id> <cx> <cy> <w> <h>   (全部归一化到 [0,1])
# 示例: 2 0.7944 0.6870 0.4111 0.1143
```

---

### Step 3: PT 模型转换

#### 3.1 了解转换流程

```
PT 权重 (.pt)
    │
    └── [pt_export.py]  导出 ONNX
            │
            └── [rknn_convert.py]  转换为 RKNN
                    │
                    └── [scp]  传输到板端
```

#### 3.2 放置模型

```bash
cd 331_PC_RKNN/PC
mkdir -p models/pt

# 将你的 YOLOv8 .pt 文件放到这里
cp /path/to/your_model.pt models/pt/
```

#### 3.3 分步执行（推荐首次使用）

```bash
# 第 1 步: PT → ONNX 导出 (约 2 秒)
python scripts/pt_export.py --pt models/pt/your_model.pt --imgsz 640
# 输出: models/onnx/your_model.onnx

# 第 2 步: ONNX → RKNN 转换 (约 5-10 秒)
python scripts/rknn_convert.py --onnx models/onnx/your_model.onnx
# 输出: models/rknn/your_model.rknn + meta.json

# 第 3 步: 传输到板端
scp models/rknn/your_model.rknn \
  orangepi@<你的板端IP>:~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite/models/
```

#### 3.4 一键执行

```bash
bash run_convert.sh full models/pt/your_model.pt
# 自动完成以上三步 + ONNX 基准推理
```

#### 3.5 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--imgsz` | 640 | 必须与训练时的输入尺寸一致 |
| `--mean` | 0,0,0 | RKNN 归一化均值 |
| `--std` | 255,255,255 | RKNN 归一化标准差 |
| `--opset` | 12 | ONNX opset 版本 |

> ⚠️ `--std 255,255,255` 表示 RKNN 内部会执行 `x/255` 归一化。板端推理时必须使用 `--norm_mode none`！详见[常见问题 Q1](#q1-rknn-推理-modeonedet0)。

---

### Step 4: 板端推理与评估

#### 4.1 单张图片推理 (冒烟测试)

```bash
ssh orangepi@<你的板端IP>
cd ~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite

/home/orangepi/miniforge3/bin/python3 scripts/infer.py \
  --model_path models/your_model.rknn \
  --input data/images/test_1.jpg \
  --conf 0.25
```

预期输出类似：`[1/1] test_1.jpg | mode=flat | det=1 | pre=4ms infer=45ms post=3ms`

#### 4.2 批量图片推理

```bash
# 推理整个文件夹
/home/orangepi/miniforge3/bin/python3 scripts/infer.py \
  --model_path models/your_model.rknn \
  --input data/images/ \
  --conf 0.25
```

#### 4.3 完整评估 (mAP 计算)

```bash
/home/orangepi/miniforge3/bin/python3 scripts/evaluate.py \
  --model_dir models/ \
  --image_dir data/images/ \
  --label_path data/labels/ \
  --out_dir outputs/eval/my_test \
  --input_size 640x640 \
  --auto_best_conf_scan \
  --conf_scan_values 0.35,0.25,0.15,0.10,0.05 \
  --conf_select_metric map50
```

评估过程：
1. 自动检测标注格式 → 打印识别结果
2. 对每个模型/阈值组合跑 350 张推理
3. 计算 Precision / Recall / mAP@50 / mAP@75 / mAP@50-95
4. 输出 Markdown 表格 + CSV + JSON

#### 4.4 查看结果

```bash
# 汇总报告
cat outputs/eval/my_test/paper_ready_summary.md

# 详细 JSON
cat outputs/eval/my_test/best_working_point/summary.json
```

#### 4.5 一键全链路

```bash
./run_pipeline.sh models/your_model.rknn full
# 等价于:
#   align_dump (对齐 dump)
#   → layerwise_diag (逐层诊断)
#   → evaluate (mAP 评估)
```

---

### Step 5: PT-RKNN 对齐验证

当板端 RKNN 推理结果与 PT 端不一致时，用对齐工具定位差异。

#### 5.1 导出 RKNN 中间张量

```bash
# 板端
/home/orangepi/miniforge3/bin/python3 scripts/align_dump.py \
  --model-path models/your_model.rknn \
  --input data/images/test_1.jpg \
  --out-dir outputs/align_dump/test \
  --save-npy
```

导出内容：
- 预处理后输入张量
- RKNN 原始输出 (dequant 前)
- dequant 后输出
- decode 后中间结果
- 最终检测框

#### 5.2 对比 ONNX 基准

```bash
# PC 端
cd 331_PC_RKNN/PC
python scripts/onnx_infer.py \
  --onnx models/onnx/your_model.onnx \
  --input data/images/ \
  --limit 5
# 输出: outputs/onnx/<model>/*.npz (ONNX 中间张量)
```

将 PC 端 ONNX 输出与板端 RKNN dump 的对应张量对比，定位差异层。

#### 5.3 逐层诊断

```bash
# 板端 - 导出每层 topK 统计
/home/orangepi/miniforge3/bin/python3 scripts/layerwise_diag.py \
  --model-path models/your_model.rknn \
  --images data/images/ \
  --out-dir outputs/layerwise_diag/ \
  --limit 5
# 输出 JSON: 每层每个通道的 topK 值、统计信息
```

---

## 仓库结构

```
331_PC_RKNN/
│
├── PC/                              # 🔵 PC 端: PT → ONNX → RKNN
│   ├── models/{pt,onnx,rknn}/       #    模型三态
│   ├── scripts/
│   │   ├── config.py                #    配置加载
│   │   ├── pt_export.py             #    ① PT → ONNX 导出
│   │   ├── rknn_convert.py          #    ② ONNX → RKNN 转换
│   │   ├── onnx_infer.py            #    ③ ONNX 推理 (对齐基准)
│   │   └── pt_infer.py              #    ④ PT 推理 (真值)
│   ├── run_convert.sh               #    一键: ①②③ → scp 板端
│   └── pipeline_config.json
│
├── RKNN/                            # 🟠 板端: 推理 + 对齐 + 评估
│   ├── scripts/
│   │   ├── paths.py                 #    配置加载
│   │   ├── infer.py                 #    推理引擎 (DFL + flat)
│   │   ├── align_dump.py            #    对齐 Dump
│   │   ├── layerwise_diag.py        #    逐层诊断
│   │   ├── evaluate.py              #    mAP 评估
│   │   └── paper_eval.py            #    Paper 评估
│   ├── run_pipeline.sh              #    一键全链路
│   └── pipeline_config.json
│
├── eval_results/                    # 📊 9 模型评估数据
├── figures/                         # 📈 论文图表
└── README.md                        # 📖 本文件
```

---

## 核心指标

RK3588 Orange Pi 5, 640x640, Port Defect 数据集 (350 张):

| 模型 | 输出头 | Precision | Recall | **mAP@50** | 推理延迟 | FPS |
|---|---|---|---|---|---|---|
| **baseline_test0625** | flat | 0.667 | 0.420 | **0.627** | 48ms | **21.0** |
| PROCESSED_FULL_MODEL_s42 | DFL | — | — | 0.662 | 85ms | 11.2 |
| B3-Lite_V2 | DFL | — | — | 0.602 | 87ms | 9.6 |
| B3-Llite-V3 | DFL | — | — | — | 62ms | 12.9 |

> 💡 **baseline YOLOv8n 以 3M 参数实现 0.627 mAP@50，FPS 比 DFL 模型高近 2 倍。**

---

## 常见问题

### Q1: RKNN 推理 `mode=none, det=0`

这是最常见的问题。**原因**: 归一化配置不匹配。

**PC 端转换** `--std 255,255,255` 意味着 RKNN 模型内部会执行 `x/255` 归一化。
**板端推理** 必须使用 `--norm_mode none`，让 RKNN 自己归一化。

如果板端使用了 `--norm_mode div255`，会导致**双归一化**：图片先被 Python 代码 `x/255`，RKNN 内部再 `x/255`，输入变成 `[0, 1/255]` ≈ 全黑。

```bash
# 正确用法:
/home/orangepi/miniforge3/bin/python3 scripts/infer.py \
  --model_path models/your_model.rknn \
  --input data/images/test_1.jpg \
  --norm_mode none      # ← 关键!
```

### Q2: `ModuleNotFoundError: No module named 'rknnlite'`

板端必须使用 miniforge3 的 Python：

```bash
/home/orangepi/miniforge3/bin/python3 scripts/infer.py [args]
```

### Q3: RKNN 转换报错 `input_size_list should be set`

已修复。`PC/scripts/rknn_convert.py` 已移除 `load_onnx(inputs=...)` 参数。

### Q4: mAP 极低 (< 0.01)

检查清单：
1. ✅ 类别数量/顺序是否与训练一致
2. ✅ `--norm_mode none`（板端）
3. ✅ `--input_layout nhwc`（板端，推荐）
4. ✅ 模型输出格式（flat vs DFL）与解码匹配

### Q5: 如何生成 INT8 量化模型

```bash
python scripts/rknn_convert.py \
  --onnx models/onnx/model.onnx \
  --quantize \
  --calib-dir data/images/ \
  --calib-num 200
```

### Q6: 添加新模型到评估

编辑 `RKNN/scripts/evaluate.py`，找到 `DEFAULT_MODEL_ORDER` 列表，添加模型文件名。

### Q7: 板端磁盘空间不足

```bash
# 查看磁盘
df -h /

# 清理 APT 缓存
sudo apt clean

# 清理 miniforge3 缓存
conda clean --all -y

# 清理回收站
rm -rf ~/.local/share/Trash/*
```

---

## 板端速查

| 项目 | 值 |
|---|---|
| IP | <你的板端IP> |
| 用户 | orangepi |
| 项目路径 | `~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite/` |
| Python 路径 | `/home/orangepi/miniforge3/bin/python3` |
| 模型目录 | `models/` |
| 数据目录 | `data/{images,labels}/` |
| 输出目录 | `outputs/` |
