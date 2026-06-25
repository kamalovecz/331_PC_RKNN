# PC-side Pipeline: PT -> ONNX -> RKNN Conversion

WSL-based conversion pipeline for RK3588 deployment.
One-click from PyTorch weights to board-ready RKNN model.

## Directory Structure

    0422_Paper/pipeline/
    |-- models/
    |   |-- pt/              # Source .pt weights (drop here)
    |   |-- onnx/            # Exported .onnx files
    |   |-- rknn/            # Converted .rknn (scp to board)
    |-- data/
    |   |-- images/          # Calibration + test images
    |   -- pt_infer.py      # PT inference (ground truth)
    |-- outputs/
    |   |-- onnx/            # ONNX inference results
    |   -- README.md

## Quick Start

### 1. Drop your PT model

    cp /path/to/your_model.pt 0422_Paper/pipeline/models/pt/

### 2. Full pipeline (PT -> ONNX -> RKNN -> scp to board)

    cd 0422_Paper/pipeline
    bash run_convert.sh full models/pt/your_model.pt

### 3. Step by step

    # Step 1: Export PT -> ONNX
    bash run_convert.sh export models/pt/your_model.pt

    # Step 2: Convert ONNX -> RKNN
    bash run_convert.sh convert models/onnx/your_model.onnx

    # Step 3: ONNX reference inference (for alignment)
    bash run_convert.sh onnx-ref models/onnx/your_model.onnx

    # Step 4: PT ground truth inference
    bash run_convert.sh pt-ref models/pt/your_model.pt

### 4. Manual script execution

    # Activate conda env first
    conda activate rknntools
    cd 0422_Paper/pipeline

    # Export
    python scripts/pt_export.py --pt models/pt/model.pt --imgsz 640

    # Convert FP mode (no quantization, fast alignment)
    python scripts/rknn_convert.py --onnx models/onnx/model.onnx

    # Convert INT8 mode (quantized, production)
    python scripts/rknn_convert.py --onnx models/onnx/model.onnx --quantize --calib-dir data/images/

    # ONNX inference
    python scripts/onnx_infer.py --onnx models/onnx/model.onnx --input data/images/ --limit 10

    # PT inference
    python scripts/pt_infer.py --pt models/pt/model.pt --input data/images/ --limit 10

## Environment

- WSL Ubuntu with conda
- Conda env: rknntools
- rknn-toolkit2 (PC simulator)
- ultralytics (YOLO export)
- onnxruntime (ONNX inference)

## Full Cross-Device Pipeline

    PC (WSL)                              Board (Orange Pi 5)
    ========                              ===================

    models/pt/model.pt
        |
        +-- pt_export.py
        |       |
        |       v
        +-- models/onnx/model.onnx
        |       |
        |       +-- rknn_convert.py
        |       |       |
        |       |       v
        |       +-- models/rknn/model.rknn
        |       |       |
        |       |       +-- scp ------------> models/model.rknn
        |       |                                  |
        |       |                            run_pipeline.sh full
        |       |                                  |
        |       |                            +-- align_dump
        |       |                            +-- layerwise_diag
        |       |                            +-- evaluate
        |       |
        |       +-- onnx_infer.py
        |               |
        |               v
        |         outputs/onnx/   <-- alignment ref --> board align_dump
        |
        +-- pt_infer.py
                |
                v
          outputs/pt/      <-- ground truth --> board evaluate

## Configuration

Edit pipeline_config.json to change:

- Model paths
- Export parameters (imgsz, opset)
- RKNN conversion params (mean, std, input_name)
- Board connection (IP, user, remote paths)

## Board Connection

    IP: 192.168.137.250
    User: orangepi
    Remote pipeline: ~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite/

## Key Conversion Parameters

For RK3588 / Orange Pi 5 deployment with YOLOv8 models:

    --imgsz 640          # Input size must match training
    --mean 0,0,0         # RGB input mean
    --std 255,255,255    # Normalize to [0,1] (div255)
    --input_name images  # ONNX input node name
    --target rk3588      # Target NPU platform

These defaults are pre-configured in pipeline_config.json.
