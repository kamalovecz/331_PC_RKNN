# 331_PC_RKNN — RKNN-PT Model Alignment Pipeline

Cross-platform pipeline for Port Defect Detection on RK3588 (Orange Pi 5).

## Structure

```
├── PC/                         # PC-side (WSL): PT -> ONNX -> RKNN conversion
│   ├── scripts/
│   │   ├── pt_export.py        # PT -> ONNX export
│   │   ├── rknn_convert.py     # ONNX -> RKNN conversion
│   │   ├── onnx_infer.py       # ONNX batch inference (alignment reference)
│   │   └── pt_infer.py         # PT batch inference (ground truth)
│   ├── run_convert.sh          # One-click: PT->ONNX->RKNN->scp to board
│   ├── pipeline_config.json    # Global config
│   └── README.md
│
├── RKNN/                       # Board-side (Orange Pi 5): inference + evaluation
│   ├── scripts/
│   │   ├── infer.py            # Main RKNN inference engine (DFL + flat head)
│   │   ├── align_dump.py       # PT-RKNN alignment dump
│   │   ├── layerwise_diag.py   # Layer-wise diagnosis
│   │   ├── evaluate.py         # Full mAP evaluation + conf scan
│   │   └── paper_eval.py       # Paper-quality evaluation
│   ├── run_pipeline.sh         # One-click: align->diag->eval
│   ├── pipeline_config.json    # Global config
│   └── README.md
│
├── eval_results/               # Paper evaluation results (9 models)
│   ├── PROCESSED_FULL_MODEL_s42/
│   ├── PROCESSED_YOLOV8N_BASELINE_s42/
│   └── ... (7 more)
│
├── figures/                    # Paper figures (SVG + data tables)
├── global_summary.*            # Cross-model comparison
└── README.md                   # This file
```

## Full Pipeline

```
PC (WSL)                            Board (Orange Pi 5)
========                            ===================

PT model (.pt)
  │
  ├── PC/scripts/pt_export.py
  │       └── ONNX (.onnx)
  │
  ├── PC/scripts/rknn_convert.py
  │       └── RKNN (.rknn)
  │               │
  │         scp models/*.rknn  ───→  RKNN/models/
  │                                      │
  │                                RKNN/run_pipeline.sh full
  │                                      │
  │                                ├── align_dump (PT-RKNN tensor diff)
  │                                ├── layerwise_diag (topK stats)
  │                                └── evaluate (mAP metrics)
  │
  ├── PC/scripts/onnx_infer.py
  │       └── ONNX reference outputs  ←→  alignment comparison
  │
  └── PC/scripts/pt_infer.py
          └── PT ground truth        ←→  evaluation comparison
```

## Key Results

| Model | head | mAP@50 | FPS | Params |
|---|---|---|---|---|
| baseline_test0625 (YOLOv8n) | flat | 0.627 | 21.0 | 3.0M |
| PROCESSED_FULL_MODEL_s42 | DFL | 0.662 | ~11 | — |
| B3-Lite | DFL | 0.602 | ~10 | — |

All evaluations on RK3588 Orange Pi 5, 640x640 input, Port Defect dataset (350 images).

## Quick Start

### PC side (WSL)
```bash
cd PC
bash run_convert.sh full models/pt/your_model.pt
```

### Board side (Orange Pi 5)
```bash
cd RKNN
./run_pipeline.sh models/your_model.rknn full
```

## Board Info
- Device: Orange Pi 5 (RK3588)
- IP: 192.168.137.250
- Python: miniforge3 Python 3.12
- RKNN: rknn-toolkit-lite2 v2.3.2
