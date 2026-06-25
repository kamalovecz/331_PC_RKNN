#!/usr/bin/env python3
"""
PT -> ONNX Export for RK3588 YOLOv8 Models.

Usage:
    python pt_export.py --pt models/pt/model.pt [--imgsz 640] [--opset 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure we can import config from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_PT_DIR, MODELS_ONNX_DIR, DEFAULT_IMGSZ, DEFAULT_OPSET


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PT -> ONNX export for RK3588")
    p.add_argument("--pt", type=Path, required=True, help="Path to .pt model")
    p.add_argument("--onnx-out", type=Path, default=None,
                   help="Output ONNX path (default: models/onnx/<name>.onnx)")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Input size")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    p.add_argument("--half", action="store_true", help="Export FP16")
    p.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX (default: True)")
    p.add_argument("--no-simplify", action="store_false", dest="simplify", help="Skip ONNX simplify")
    return p.parse_args()


def export_pt_to_onnx(pt_path: Path, onnx_path: Path, imgsz: int, opset: int,
                      half: bool = False, simplify: bool = True):
    """Export a YOLO .pt model to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("  pip install ultralytics")
        sys.exit(1)

    print(f"[Export] Loading PT: {pt_path}")
    model = YOLO(str(pt_path))

    print(f"[Export] imgsz={imgsz}  opset={opset}  half={half}  simplify={simplify}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        half=half,
        simplify=simplify,
    )

    # Ultralytics puts the ONNX next to the PT file by default
    default_onnx = pt_path.with_suffix(".onnx")
    if default_onnx.exists() and default_onnx.resolve() != onnx_path.resolve():
        import shutil
        shutil.move(str(default_onnx), str(onnx_path))

    print(f"[Export] Done: {onnx_path}")
    print(f"[Export] Size: {onnx_path.stat().st_size / 1e6:.1f} MB")
    return onnx_path


def main():
    args = parse_args()
    pt_path = args.pt.resolve()
    if not pt_path.exists():
        print(f"ERROR: PT model not found: {pt_path}")
        sys.exit(1)

    if args.onnx_out:
        onnx_path = args.onnx_out.resolve()
    else:
        onnx_path = Path(MODELS_ONNX_DIR) / f"{pt_path.stem}.onnx"

    export_pt_to_onnx(pt_path, onnx_path, args.imgsz, args.opset,
                      half=args.half, simplify=args.simplify)


if __name__ == "__main__":
    main()
