#!/bin/bash
# ============================================================
# RKNN-PT Alignment Pipeline — One-Click Entry
# ============================================================
# Usage:
#   ./run_pipeline.sh <model.rknn> <mode>
#
# Modes:
#   full      Full pipeline: align -> diag -> eval
#   align     Alignment dump only (compare with PT intermediates)
#   diag      Layer-wise diagnosis only (topK export)
#   eval      Evaluation only (mAP metrics on test set)
#   infer     Single-image inference
#
# Examples:
#   ./run_pipeline.sh models/PROCESSED_FULL_MODEL_s42_640_fp.rknn full
#   ./run_pipeline.sh models/yolov8n_baseline_fp.rknn align
#   ./run_pipeline.sh models/B3-Lite_V2_fp.rknn eval
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$(python3 -c 'import json; print(json.load(open("pipeline_config.json"))["python"])')"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model.rknn> <mode> [extra_args...]"
    echo "Modes: full | align | diag | eval | infer"
    exit 1
fi

MODEL="$1"
MODE="$2"
shift 2

if [ ! -f "$MODEL" ]; then
    # Try models/ prefix
    if [ -f "models/$MODEL" ]; then
        MODEL="models/$MODEL"
    else
        echo "ERROR: Model not found: $MODEL"
        echo "Available models:"
        ls models/*.rknn 2>/dev/null | while read m; do echo "  $m"; done
        exit 1
    fi
fi

MODEL_NAME="$(basename "$MODEL" .rknn)"
echo "=========================================="
echo "Pipeline: mode=$MODE  model=$MODEL_NAME"
echo "=========================================="

case "$MODE" in
    full)
        echo "\n[1/3] Alignment Dump..."
        $PYTHON scripts/align_dump.py --model-path "$MODEL" --input data/images/             --out-dir "outputs/align_dump/$MODEL_NAME" --limit 10 "$@"

        echo "\n[2/3] Layer-wise Diagnosis..."
        $PYTHON scripts/layerwise_diag.py --model-path "$MODEL"             --images data/images/ --out-dir "outputs/layerwise_diag/" --limit 5 "$@"

        echo "\n[3/3] Evaluation..."
        $PYTHON scripts/evaluate.py --model_dir models/ --image_dir data/images/             --label_path data/labels/ --out_dir "outputs/eval/$MODEL_NAME" "$@"

        echo "\n=========================================="
        echo "Full pipeline complete!"
        echo "Results: outputs/"
        echo "=========================================="
        ;;
    align)
        $PYTHON scripts/align_dump.py --model-path "$MODEL" --input data/images/             --out-dir "outputs/align_dump/$MODEL_NAME" "$@"
        ;;
    diag)
        $PYTHON scripts/layerwise_diag.py --model-path "$MODEL"             --images data/images/ --out-dir "outputs/layerwise_diag/" "$@"
        ;;
    eval)
        $PYTHON scripts/evaluate.py --model_dir "$(dirname "$MODEL")"             --image_dir data/images/ --label_path data/labels/             --out_dir "outputs/eval/$MODEL_NAME" "$@"
        ;;
    infer)
        IMG="${1:-data/images/}"
        $PYTHON scripts/infer.py --model_path "$MODEL" --input "$IMG"             --output_dir "outputs/inference" "${@:2}"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: full | align | diag | eval | infer"
        exit 1
        ;;
esac
