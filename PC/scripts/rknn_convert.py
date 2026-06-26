#!/usr/bin/env python3
"""
ONNX -> RKNN Conversion for RK3588 (FP mode by default).

Usage:
    python rknn_convert.py --onnx models/onnx/model.onnx [--quantize] [--calib-dir data/images]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_ONNX_DIR, MODELS_RKNN_DIR, DATA_DIR, CALIB_LIST,
    RKNN_TARGET, RKNN_MEAN, RKNN_STD, RKNN_INPUT_NAME,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX -> RKNN converter (RK3588)")
    p.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model")
    p.add_argument("--out", type=Path, default=None,
                   help="Output .rknn path (default: models/rknn/<name>.rknn)")
    p.add_argument("--target", type=str, default=RKNN_TARGET)
    p.add_argument("--quantize", action="store_true", help="Enable INT8 quantization")
    p.add_argument("--calib-dir", type=Path, default=Path(DATA_DIR),
                   help="Calibration image directory")
    p.add_argument("--calib-list", type=Path, default=None,
                   help="Text file with calib image paths")
    p.add_argument("--calib-num", type=int, default=200, help="Max calibration images")
    p.add_argument("--mean", type=str, default=",".join(map(str, RKNN_MEAN)))
    p.add_argument("--std", type=str, default=",".join(map(str, RKNN_STD)))
    p.add_argument("--input-name", type=str, default=RKNN_INPUT_NAME)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_calib_list(image_dir: Path, output_txt: Path, max_images: int) -> Path:
    image_paths = []
    for p in sorted(image_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            image_paths.append(p.resolve())
    image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        for p in image_paths:
            f.write(str(p) + "\n")
    print(f"[Calib] {len(image_paths)} images -> {output_txt}")
    return output_txt


def parse_triplet(text: str, name: str) -> List[float]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must have 3 comma-separated values, got: {text}")
    return [float(x) for x in parts]


def convert_onnx_to_rknn(onnx_path: Path, rknn_path: Path, target: str,
                         quantize: bool, calib_list: Path | None,
                         mean: list, std: list, input_name: str,
                         verbose: bool = False) -> Path:
    """Convert ONNX model to RKNN format."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("ERROR: rknn-toolkit2 not installed.")
        print("  Install in WSL: pip install rknn-toolkit2")
        sys.exit(1)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=verbose)

    # --- Config ---
    print(f"[RKNN] Config: mean={mean}  std={std}  target={target}")
    rknn.config(
        mean_values=[mean],
        std_values=[std],
        target_platform=target,
    )

    # --- Load ONNX ---
    print(f"[RKNN] Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"load_onnx failed: ret={ret}")

    # --- Build ---
    if quantize:
        if not calib_list or not calib_list.exists():
            calib_list = build_calib_list(
                Path(DATA_DIR),
                Path(CALIB_LIST),
                max_images=200,
            )
        print(f"[RKNN] Building INT8 with dataset: {calib_list}")
        ret = rknn.build(do_quantization=True, dataset=str(calib_list))
    else:
        print("[RKNN] Building FP model (no quantization)")
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        raise RuntimeError(f"build failed: ret={ret}")

    # --- Export ---
    print(f"[RKNN] Exporting: {rknn_path}")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"export_rknn failed: ret={ret}")

    # --- Save meta ---
    meta_path = rknn_path.with_suffix(".meta.json")
    meta = {
        "onnx_source": str(onnx_path),
        "rknn_output": str(rknn_path),
        "target": target,
        "quantize": quantize,
        "mean": mean,
        "std": std,
        "input_name": input_name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[RKNN] Meta saved: {meta_path}")

    rknn.release()
    print(f"[RKNN] Done: {rknn_path}  ({rknn_path.stat().st_size / 1e6:.1f} MB)")
    return rknn_path


def main():
    args = parse_args()
    onnx_path = args.onnx.resolve()
    if args.out:
        rknn_path = args.out.resolve()
    else:
        rknn_path = Path(MODELS_RKNN_DIR) / f"{onnx_path.stem}.rknn"

    calib_list = args.calib_list.resolve() if args.calib_list else None
    mean = parse_triplet(args.mean, "--mean")
    std = parse_triplet(args.std, "--std")

    convert_onnx_to_rknn(
        onnx_path=onnx_path,
        rknn_path=rknn_path,
        target=args.target,
        quantize=args.quantize,
        calib_list=calib_list,
        mean=mean,
        std=std,
        input_name=args.input_name,
        verbose=args.verbose,
    )

    # --- Print scp command for convenience ---
    print(f"\n[Next] Copy to board:")
    print(f"  scp {rknn_path} orangepi@<你的板端IP>:~/Paper_pass_Projects/yolov8_rknn-toolkit2-lite/models/")


if __name__ == "__main__":
    main()
