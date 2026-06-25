#!/usr/bin/env python3
"""PC-side pipeline config loader. Import this first in all scripts."""

import json
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINE_ROOT = _SCRIPT_DIR.parent
_config_path = _PIPELINE_ROOT / "pipeline_config.json"

with open(_config_path, "r", encoding="utf-8") as f:
    _cfg = json.load(f)

def _resolve(p):
    return str((_PIPELINE_ROOT / p).resolve()) if p.startswith("./") else p

_p = _cfg["paths"]
MODELS_PT_DIR   = _resolve(_p["models_pt_dir"])
MODELS_ONNX_DIR = _resolve(_p["models_onnx_dir"])
MODELS_RKNN_DIR = _resolve(_p["models_rknn_dir"])
DATA_DIR        = _resolve(_p["data_dir"])
CALIB_LIST      = _resolve(_p["calib_list"])
OUTPUTS_DIR     = _resolve(_p["outputs_dir"])

_exp = _cfg["export"]
DEFAULT_IMGSZ   = _exp["imgsz"]
DEFAULT_OPSET   = _exp["opset"]

_rk = _cfg["rknn"]
RKNN_TARGET     = _rk["target"]
RKNN_MEAN       = _rk["mean"]
RKNN_STD        = _rk["std"]
RKNN_INPUT_NAME = _rk["input_name"]

_inf = _cfg["inference"]
CLASSES         = tuple(_inf["classes"])

PIPELINE_ROOT   = str(_PIPELINE_ROOT)
CONDA_ENV       = _cfg["wsl"]["conda_env"]
PYTHON_BIN      = _cfg["wsl"]["python"]

_board = _cfg["board"]
BOARD_IP        = _board["ip"]
BOARD_USER      = _board["user"]
BOARD_MODELS_DIR = _board["remote_models_dir"]
BOARD_PIPELINE   = _board["remote_pipeline"]

if __name__ == "__main__":
    print(f"PIPELINE_ROOT: {PIPELINE_ROOT}")
    print(f"MODELS_PT:     {MODELS_PT_DIR}")
    print(f"MODELS_ONNX:   {MODELS_ONNX_DIR}")
    print(f"MODELS_RKNN:   {MODELS_RKNN_DIR}")
    print(f"DATA_DIR:      {DATA_DIR}")
    print(f"OUTPUTS_DIR:   {OUTPUTS_DIR}")
    print(f"CONDA_ENV:     {CONDA_ENV}")
