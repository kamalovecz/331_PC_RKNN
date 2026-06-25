# RKNN-PT Model Alignment Pipeline

Board-side RKNN model vs PC-side PyTorch model layer-wise alignment & inference evaluation.

## Directory Structure

    yolov8_rknn-toolkit2-lite/
    ├── models/                  # Unified model directory (drop PC-converted .rknn here)
    ├── data/
    │   ├── images/              # Test images (350)
    │   └── labels/              # YOLO format labels (350)
    ├── scripts/                 # All core scripts
    │   ├── paths.py             # Config loader (auto-reads pipeline_config.json)
    │   ├── infer.py             # Main inference engine
    │   ├── align_dump.py        # Alignment dump (export intermediate tensors)
    │   ├── layerwise_diag.py    # Layer-wise diagnosis (topK export)
    │   ├── evaluate.py          # Batch evaluation (mAP metrics)
    │   ├── paper_eval.py        # Paper-quality evaluation
    │   ├── count_detections.py  # Batch detection count
    │   ├── debug_model.py       # Model output diagnostic
    │   ├── input_probe.py       # Input format probe
    │   └── coco_utils.py        # COCO utilities
    ├── outputs/                 # All output results
    │   ├── align_dump/          # Alignment dump data
    │   ├── layerwise_diag/      # Layer-wise diagnosis JSON
    │   └── eval/                # Evaluation results
    ├── run_pipeline.sh          # One-click pipeline script
    ├── pipeline_config.json     # Global configuration
    └── README.md

## Quick Start

### 1. Drop your PC-converted RKNN model

    scp your_model.rknn orangepi@192.168.137.250:~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite/models/

### 2. Run the full pipeline

    cd ~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite

    # Full pipeline: align -> diag -> eval
    ./run_pipeline.sh models/PROCESSED_FULL_MODEL_s42_640_fp.rknn full

    # Alignment dump only
    ./run_pipeline.sh models/your_model.rknn align --limit 10

    # Layer-wise diagnosis only
    ./run_pipeline.sh models/your_model.rknn diag --limit 5

    # Evaluation only
    ./run_pipeline.sh models/your_model.rknn eval

    # Single image inference
    ./run_pipeline.sh models/your_model.rknn infer data/images/test_1.jpg

### 3. Manual script execution

    cd scripts/

    # Single image inference
    python3 infer.py --model_path ../models/PROCESSED_FULL_MODEL_s42_640_fp.rknn --input ../data/images/test_1.jpg

    # Folder batch inference
    python3 infer.py --model_path ../models/PROCESSED_FULL_MODEL_s42_640_fp.rknn --input ../data/images/

    # Alignment dump (save intermediate tensors for PT comparison)
    python3 align_dump.py --model-path ../models/PROCESSED_FULL_MODEL_s42_640_fp.rknn --input ../data/images/ --out-dir ../outputs/align_dump/my_test --save-npy --limit 5

    # Layer-wise diagnosis
    python3 layerwise_diag.py --model-path ../models/PROCESSED_FULL_MODEL_s42_640_fp.rknn --images ../data/images/ --out-dir ../outputs/layerwise_diag/ --limit 5

    # Full evaluation
    python3 evaluate.py --model_dir ../models/ --image_dir ../data/images/ --label_path ../data/labels/ --out_dir ../outputs/eval/my_test

## Python Environment

Must use miniforge3 Python (system Python lacks cv2/rknnlite):

    /home/orangepi/miniforge3/bin/python3 scripts/infer.py [args]

The pipeline_config.json "python" field is pre-configured. run_pipeline.sh uses it automatically.

## Configuration

pipeline_config.json manages all paths and defaults globally:

    {
      "paths": {
        "models_dir": "./models",
        "data_images_dir": "./data/images",
        "data_labels_dir": "./data/labels",
        "outputs_dir": "./outputs"
      },
      "inference": {
        "input_size": "640x640",
        "default_conf": 0.35,
        "default_iou": 0.50
      },
      "model": {
        "classes": ["Rust", "Cracks", "patches", "Scratches", "Pitting"],
        "reg_max": 15
      }
    }

Edit this one file to change paths globally — no need to touch individual scripts.

## Pipeline Flow

    PC-side PT Model
         │
         ├── rknn-toolkit2 convert (on PC)
         │
         ▼
    models/*.rknn  <-- Drop point
         │
         ├── [1] align_dump.py     -> outputs/align_dump/
         │     Export preprocess/dequant/decode intermediate tensors
         │     Compare layer-by-layer with PC-side PT model
         │
         ├── [2] layerwise_diag.py -> outputs/layerwise_diag/
         │     Per-layer topK statistics JSON
         │
         └── [3] evaluate.py       -> outputs/eval/
               COCO mAP / per-class metrics

    Compare outputs/ dump data with PT-side outputs -> locate precision gaps

## Models Available

| Model | Description |
|---|---|
| PROCESSED_FULL_MODEL_s42_640_fp.rknn | Full model FP |
| PROCESSED_YOLOV8N_BASELINE_s42_640_fp.rknn | YOLOv8n baseline |
| PROCESSED_A_GFPN_s42_640_fp.rknn | A_GFPN variant |
| PROCESSED_A_GFPN_REPHFE_s42_640_fp.rknn | A_GFPN + REPHFE |
| PROCESSED_REPHFE_s42_640_fp.rknn | REPHFE variant |
| PROCESSED_SADH_s42_640_fp.rknn | SADH variant |
| B3_Lite_640_fp.rknn | B3-Lite |
| NEU_Pretrain_pt_Full_model_Port_defect_640_fp.rknn | NEU pretrained |
| Port_RuleLoss_Full_model_640_fp.rknn | RuleLoss |
| yolov8n_baseline_fp.rknn | YOLOv8n baseline (legacy) |
| yolov8n_port_RULELOSS_fp.rknn | RuleLoss (legacy) |
| B3-Lite_V2_fp.rknn | B3-Lite V2 |
| B3-Llite-V3_fp.rknn | B3-Lite V3 |

## Board Info

- Device: Orange Pi 5 (RK3588)
- IP: 192.168.137.250
- OS: Linux 5.10.160-rockchip-rk3588
- Python: miniforge3 Python 3.12
- NPU: RK3588 (3 cores)
