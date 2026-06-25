#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-detect project paths from pipeline_config.json. Import this first in all scripts."""

import json
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_config_path = _PROJECT_ROOT / "pipeline_config.json"

with open(_config_path, "r", encoding="utf-8") as f:
    _cfg = json.load(f)

# Resolve relative paths against project root
def _resolve(p):
    return str((_PROJECT_ROOT / p).resolve()) if p.startswith("./") else p

_p = _cfg["paths"]
MODELS_DIR       = _resolve(_p["models_dir"])
DATA_IMAGES_DIR  = _resolve(_p["data_images_dir"])
DATA_LABELS_DIR  = _resolve(_p["data_labels_dir"])
OUTPUTS_DIR      = _resolve(_p["outputs_dir"])
ALIGN_DUMP_DIR   = _resolve(_p["align_dump_dir"])
LAYERWISE_DIAG_DIR = _resolve(_p["layerwise_diag_dir"])
EVAL_DIR         = _resolve(_p["eval_dir"])

_inf = _cfg["inference"]
DEFAULT_INPUT_SIZE = _inf["input_size"]
DEFAULT_CONF       = _inf["default_conf"]
DEFAULT_IOU        = _inf["default_iou"]
DEFAULT_CORE       = _inf["core"]

_m = _cfg["model"]
CLASSES = tuple(_m["classes"])
REG_MAX = _m["reg_max"]
REG_CH  = 4 * (REG_MAX + 1)

PYTHON_BIN = _cfg.get("python", "python3")
PROJECT_ROOT = str(_PROJECT_ROOT)

# For backward compat
FP_MODEL_DIR = MODELS_DIR
NEU_CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

print("paths.py: project_root=", PROJECT_ROOT)
print("paths.py: models_dir=", MODELS_DIR)
