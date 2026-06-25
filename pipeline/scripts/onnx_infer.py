#!/usr/bin/env python3
"""
ONNX Batch Inference — produces reference outputs for RKNN alignment.

Usage:
    python onnx_infer.py --onnx models/onnx/model.onnx --input data/images/ [--out-dir outputs/onnx/model_name]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_ONNX_DIR, DATA_DIR, OUTPUTS_DIR, CLASSES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="ONNX batch inference for alignment reference")
    p.add_argument("--onnx", type=Path, required=True, help="ONNX model path")
    p.add_argument("--input", type=str, required=True, help="Image dir or single image")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--limit", type=int, default=0, help="Max images (0=all)")
    return p.parse_args()


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def collect_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    items = sorted(p.iterdir())
    return [str(x) for x in items if x.suffix.lower() in IMG_EXTS]


def run_onnx_inference(onnx_path, image_path, imgsz=640):
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Failed to load: {image_path}")
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_lb, ratio, pad = letterbox(im_rgb, (imgsz, imgsz))
    inp = im_lb.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))  # HWC -> CHW
    inp = np.expand_dims(inp, axis=0)    # -> NCHW

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})
    return [np.array(o) for o in outputs], ratio, pad, im.shape


def main():
    args = parse_args()
    onnx_path = args.onnx.resolve()
    if not onnx_path.exists():
        print(f"ERROR: ONNX not found: {onnx_path}")
        sys.exit(1)

    model_name = onnx_path.stem
    out_dir = args.out_dir or Path(OUTPUTS_DIR) / "onnx" / model_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.input)
    if args.limit > 0:
        images = images[:args.limit]

    print(f"[ONNX Infer] model={model_name}  images={len(images)}")
    summary = []

    for i, img_path in enumerate(images):
        img_name = Path(img_path).name
        try:
            outputs, ratio, pad, orig_shape = run_onnx_inference(onnx_path, img_path, args.imgsz)
            # Save raw outputs as .npy for alignment comparison
            np_path = out_dir / f"{Path(img_name).stem}_onnx_outputs.npz"
            np.savez(np_path, *outputs)
            summary.append({
                "image": img_name,
                "output_shapes": [list(o.shape) for o in outputs],
                "ratio": ratio,
                "pad": list(pad),
                "orig_shape": list(orig_shape),
            })
            print(f"  [{i+1}/{len(images)}] {img_name}  shapes={[o.shape for o in outputs]}")
        except Exception as e:
            print(f"  [{i+1}/{len(images)}] {img_name}  ERROR: {e}")

    # Save summary
    summary_path = out_dir / "onnx_inference_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[ONNX Infer] Summary: {summary_path}")
    print(f"[ONNX Infer] Outputs: {out_dir}/")


if __name__ == "__main__":
    main()
