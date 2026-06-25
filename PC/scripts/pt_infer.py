#!/usr/bin/env python3
"""
PyTorch Batch Inference — ground truth baseline for RKNN alignment.

Usage:
    python pt_infer.py --pt models/pt/model.pt --input data/images/ [--out-dir outputs/pt/model_name]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_PT_DIR, DATA_DIR, OUTPUTS_DIR, CLASSES, DEFAULT_IMGSZ

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="PT batch inference for ground truth")
    p.add_argument("--pt", type=Path, required=True, help="PyTorch .pt model path")
    p.add_argument("--input", type=str, required=True, help="Image dir or single image")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--limit", type=int, default=0, help="Max images (0=all)")
    return p.parse_args()


def collect_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    items = sorted(p.iterdir())
    return [str(x) for x in items if x.suffix.lower() in IMG_EXTS]


def extract_features(model, image_path, imgsz=640):
    """Run PT inference and return intermediate feature maps (for alignment)."""
    import torch

    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Failed to load: {image_path}")
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Letterbox
    shape = im_rgb.shape[:2]
    r = min(imgsz / shape[0], imgsz / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (imgsz - new_unpad[0]) / 2
    dh = (imgsz - new_unpad[1]) / 2
    im_lb = cv2.resize(im_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_lb = cv2.copyMakeBorder(im_lb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    inp = im_lb.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))
    inp = torch.from_numpy(inp).unsqueeze(0)

    with torch.no_grad():
        results = model(inp, verbose=False)

    # Extract detection results from ultralytics YOLO
    dets = []
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(xyxy, conf, cls_ids):
                # Scale back to original coordinates
                b[0] = (b[0] - dw) / r
                b[1] = (b[1] - dh) / r
                b[2] = (b[2] - dw) / r
                b[3] = (b[3] - dh) / r
                dets.append({
                    "bbox": b.tolist(),
                    "score": float(s),
                    "class": int(c),
                    "class_name": CLASSES[c] if c < len(CLASSES) else str(c),
                })

    return dets, (r, (dw, dh)), im.shape


def main():
    args = parse_args()
    pt_path = args.pt.resolve()
    if not pt_path.exists():
        print(f"ERROR: PT model not found: {pt_path}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. pip install ultralytics")
        sys.exit(1)

    print(f"[PT Infer] Loading model: {pt_path}")
    model = YOLO(str(pt_path))

    model_name = pt_path.stem
    out_dir = args.out_dir or Path(OUTPUTS_DIR) / "pt" / model_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.input)
    if args.limit > 0:
        images = images[:args.limit]

    print(f"[PT Infer] images={len(images)}  conf={args.conf}  iou={args.iou}")
    all_dets = []

    results = model.predict(
        source=images,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    total_dets = 0
    for i, r in enumerate(results):
        img_name = Path(images[i]).name if i < len(images) else f"img_{i}"
        boxes = r.boxes
        dets = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
            cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, 'cpu') else boxes.cls
            for b, s, c in zip(xyxy, conf, cls_ids):
                dets.append({
                    "bbox": [round(float(x), 2) for x in b],
                    "score": round(float(s), 4),
                    "class": int(c),
                    "class_name": CLASSES[int(c)] if int(c) < len(CLASSES) else str(c),
                })
        all_dets.append({
            "image": img_name,
            "detections": dets,
            "count": len(dets),
        })
        total_dets += len(dets)
        print(f"  [{i+1}/{len(images)}] {img_name}  det={len(dets)}")

    # Save
    result_path = out_dir / "pt_inference_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_dets, f, indent=2, ensure_ascii=False)
    print(f"\n[PT Infer] Results: {result_path}")
    print(f"[PT Infer] Total: {total_dets} detections across {len(images)} images")


if __name__ == "__main__":
    main()
