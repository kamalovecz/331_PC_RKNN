#!/bin/bash
# ============================================================
# PC-side Pipeline: PT -> ONNX -> RKNN Conversion
# Run with:  bash run_convert.sh <mode> [args...]
#
# Modes:
#   export    PT -> ONNX export
#   convert   ONNX -> RKNN conversion
#   full      PT -> ONNX -> RKNN -> scp to board -> board pipeline
#   onnx-ref  ONNX inference (alignment reference)
#   pt-ref    PT inference (ground truth baseline)
#
# Examples:
#   bash run_convert.sh export models/pt/model.pt
#   bash run_convert.sh convert models/onnx/model.onnx
#   bash run_convert.sh full models/pt/model.pt
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Read config
CONDA_ENV=$(python3 -c "import json; print(json.load(open('pipeline_config.json'))['wsl']['conda_env'])" 2>/dev/null || echo "rknntools")
PYTHON=$(python3 -c "import json; print(json.load(open('pipeline_config.json'))['wsl']['python'])" 2>/dev/null || echo "python3")

# We run inside WSL - all commands use conda
run_py() {
    if command -v conda &>/dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
        conda activate "$CONDA_ENV" 2>/dev/null
    fi
    python "$@"
}

if [ $# -lt 1 ]; then
    echo "Usage: bash run_convert.sh <mode> [args...]"
    echo ""
    echo "Modes:"
    echo "  export    PT -> ONNX export"
    echo "  convert   ONNX -> RKNN conversion"
    echo "  full      PT -> ONNX -> RKNN -> scp to board"
    echo "  onnx-ref  ONNX inference (reference for alignment)"
    echo "  pt-ref    PT inference (ground truth)"
    echo ""
    echo "Examples:"
    echo "  bash run_convert.sh export models/pt/model.pt"
    echo "  bash run_convert.sh full models/pt/model.pt"
    exit 1
fi

MODE="$1"
shift

case "$MODE" in
    export)
        run_py scripts/pt_export.py --pt "$@"
        ;;
    convert)
        run_py scripts/rknn_convert.py --onnx "$@"
        ;;
    full)
        PT_PATH="$1"
        if [ ! -f "$PT_PATH" ]; then
            echo "ERROR: PT model not found: $PT_PATH"
            echo "Available .pt files:"
            find models/pt -name "*.pt" 2>/dev/null || echo "  (none in models/pt/)"
            exit 1
        fi
        MODEL_NAME="$(basename "$PT_PATH" .pt)"

        echo "============================================"
        echo "FULL PIPELINE: $MODEL_NAME"
        echo "============================================"

        echo ""
        echo "[1/4] Exporting PT -> ONNX..."
        run_py scripts/pt_export.py --pt "$PT_PATH" --onnx-out "models/onnx/${MODEL_NAME}.onnx"

        echo ""
        echo "[2/4] Converting ONNX -> RKNN..."
        run_py scripts/rknn_convert.py --onnx "models/onnx/${MODEL_NAME}.onnx" --out "models/rknn/${MODEL_NAME}.rknn"

        echo ""
        echo "[3/4] ONNX reference inference (5 images for alignment)..."
        run_py scripts/onnx_infer.py --onnx "models/onnx/${MODEL_NAME}.onnx" \
            --input data/images/ --out-dir "outputs/onnx/${MODEL_NAME}" --limit 5

        echo ""
        echo "[4/4] Copying RKNN to board..."
        BOARD_IP=$(python3 -c "import json; print(json.load(open('pipeline_config.json'))['board']['ip'])")
        BOARD_USER=$(python3 -c "import json; print(json.load(open('pipeline_config.json'))['board']['user'])")
        BOARD_DIR=$(python3 -c "import json; print(json.load(open('pipeline_config.json'))['board']['remote_models_dir'])")
        echo "  scp models/rknn/${MODEL_NAME}.rknn ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/"
        scp "models/rknn/${MODEL_NAME}.rknn" "${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/" 2>&1 || echo "  (scp failed - copy manually)"

        echo ""
        echo "============================================"
        echo "PIPELINE COMPLETE"
        echo "============================================"
        echo "Next on board:"
        echo "  ssh ${BOARD_USER}@${BOARD_IP}"
        echo "  cd ~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite"
        echo "  ./run_pipeline.sh models/${MODEL_NAME}.rknn full"
        ;;
    onnx-ref)
        run_py scripts/onnx_infer.py --onnx "$@"
        ;;
    pt-ref)
        run_py scripts/pt_infer.py --pt "$@"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: export | convert | full | onnx-ref | pt-ref"
        exit 1
        ;;
esac
