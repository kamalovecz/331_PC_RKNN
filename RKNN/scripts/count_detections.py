#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from paths import MODELS_DIR, DATA_IMAGES_DIR, OUTPUTS_DIR
from infer import (
    filter_edge_boxes,
    letterbox,
    post_process,
    scale_boxes_to_original,
    try_get_output_details,
)


DEFAULT_MODELS = [
    os.path.join(MODELS_DIR, "PROCESSED_FULL_MODEL_s42_640_fp.rknn"),
    os.path.join(MODELS_DIR, "PROCESSED_YOLOV8N_BASELINE_s42_640_fp.rknn"),
    os.path.join(MODELS_DIR, "yolov8n_baseline_fp.rknn"),
    os.path.join(MODELS_DIR, "yolov8n_port_RULELOSS_fp.rknn"),
    os.path.join(MODELS_DIR, "B3-Lite_V2_fp.rknn"),
    os.path.join(MODELS_DIR, "B3-Llite-V3_fp.rknn"),
]


def img_check(path):
    return os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folder):
    names = sorted(os.listdir(folder))
    return [
        os.path.join(folder, name)
        for name in names
        if os.path.isfile(os.path.join(folder, name)) and img_check(os.path.join(folder, name))
    ]


def parse_model_metadata(model_path):
    info = {
        "input_layout": "NCHW",
        "input_dtype": "uint8",
    }

    try:
        out = subprocess.check_output(
            ["strings", "-a", model_path],
            text=True,
            errors="ignore",
        )
    except Exception:
        return info

    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{") or '"dtype"' not in line or '"layout"' not in line:
            continue

        try:
            obj = json.loads(line)
        except Exception:
            try:
                obj = ast.literal_eval(line)
            except Exception:
                continue

        for key in ("images", "input"):
            val = obj.get(key)
            if isinstance(val, dict):
                layout = str(val.get("layout", info["input_layout"])).upper()
                dtype = str(val.get("dtype", info["input_dtype"])).lower()
                if layout in {"NHWC", "NCHW"}:
                    info["input_layout"] = layout
                info["input_dtype"] = dtype
                return info

    return info


def infer_num_classes(outputs):
    dfl_candidates = []

    for out in outputs:
        arr = np.array(out)
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] > 64:
            nc = int(arr.shape[1] - 64)
            if 1 <= nc <= 256:
                dfl_candidates.append(nc)

    if dfl_candidates:
        return Counter(dfl_candidates).most_common(1)[0][0]

    for out in outputs:
        arr = np.array(out)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            continue

        small_dim = min(arr.shape)
        large_dim = max(arr.shape)
        if 5 <= small_dim <= 260 and large_dim >= 100:
            nc = int(small_dim - 4)
            if 1 <= nc <= 256:
                return nc

    return None


def prepare_input(im0, input_size, input_layout, input_dtype):
    im_lb, ratio, pad = letterbox(im0, new_shape=input_size, color=(114, 114, 114))
    im_rgb = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)

    if input_layout == "NHWC":
        inp = np.expand_dims(im_rgb, 0)
    else:
        inp = np.expand_dims(im_rgb.transpose(2, 0, 1), 0)

    if "float" in input_dtype:
        inp = inp.astype(np.float32) / 255.0
    else:
        inp = inp.astype(np.uint8)

    return inp, ratio, pad


def build_model_summary(model_path, rows, meta, status, error, elapsed_s):
    det_counts = [row["det_count"] for row in rows]
    nonzero = sum(1 for n in det_counts if n > 0)
    modes = Counter(row["decode_mode"] for row in rows if row["decode_mode"])
    return {
        "model_name": os.path.basename(model_path),
        "model_path": model_path,
        "status": status,
        "error": error,
        "image_count": len(rows),
        "total_detections": int(sum(det_counts)),
        "nonzero_images": int(nonzero),
        "max_det_single_image": int(max(det_counts) if det_counts else 0),
        "avg_det_per_image": float(sum(det_counts) / len(det_counts)) if det_counts else 0.0,
        "decode_modes": dict(modes),
        "input_layout": meta["input_layout"],
        "input_dtype": meta["input_dtype"],
        "elapsed_seconds": round(elapsed_s, 3),
    }


def run_model(model_path, images, input_size, conf, iou, max_det, core_mask):
    meta = parse_model_metadata(model_path)
    start_t = time.perf_counter()
    rows = []
    status = "ok"
    error = ""

    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: ret={ret}")

        ret = rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: ret={ret}")

        output_details = try_get_output_details(rknn)
        num_classes = None

        for idx, img_path in enumerate(images, 1):
            im0 = cv2.imread(img_path)
            if im0 is None:
                rows.append({
                    "model_name": os.path.basename(model_path),
                    "model_path": model_path,
                    "image_name": os.path.basename(img_path),
                    "image_path": img_path,
                    "decode_mode": "read_failed",
                    "det_count": 0,
                    "num_classes": num_classes,
                })
                continue

            inp, ratio, pad = prepare_input(
                im0,
                input_size=input_size,
                input_layout=meta["input_layout"],
                input_dtype=meta["input_dtype"],
            )
            outputs = rknn.inference(inputs=[inp])

            if num_classes is None:
                num_classes = infer_num_classes(outputs)
                if num_classes is None:
                    raise RuntimeError("cannot infer num_classes from model outputs")

            boxes, cids, scores, mode = post_process(
                outputs=outputs,
                output_details=output_details,
                input_size=input_size,
                num_classes=num_classes,
                conf_thres=conf,
                iou_thres=iou,
                max_det=max_det,
                decode_mode="auto",
                min_wh=10.0,
                min_area=120.0,
                max_aspect_ratio=8.0,
                merge_gap=12,
            )

            if boxes is None or len(boxes) == 0:
                det_count = 0
            else:
                boxes0 = scale_boxes_to_original(boxes, ratio, pad, im0.shape)
                boxes0, cids, scores = filter_edge_boxes(
                    boxes0,
                    cids,
                    scores,
                    im0.shape,
                    edge_margin=4,
                    min_edge_box=20,
                )
                det_count = 0 if boxes0 is None else int(len(boxes0))

            rows.append({
                "model_name": os.path.basename(model_path),
                "model_path": model_path,
                "image_name": os.path.basename(img_path),
                "image_path": img_path,
                "decode_mode": mode,
                "det_count": det_count,
                "num_classes": num_classes,
            })

            if idx == 1 or idx % 25 == 0 or idx == len(images):
                print(
                    f"[{os.path.basename(model_path)}] "
                    f"{idx}/{len(images)} images processed"
                )

    except Exception as exc:
        status = "failed"
        error = str(exc)
        if not rows:
            for img_path in images:
                rows.append({
                    "model_name": os.path.basename(model_path),
                    "model_path": model_path,
                    "image_name": os.path.basename(img_path),
                    "image_path": img_path,
                    "decode_mode": "model_failed",
                    "det_count": 0,
                    "num_classes": None,
                })
    finally:
        try:
            rknn.release()
        except Exception:
            pass

    elapsed_s = time.perf_counter() - start_t
    summary = build_model_summary(model_path, rows, meta, status, error, elapsed_s)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description="Count RKNN detections without saving output images")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=DATA_IMAGES_DIR,
        help="Image folder to run inference on",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(Path(OUTPUTS_DIR) / "detection_counts.csv"),
        help="Per-image count CSV output path",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(Path(OUTPUTS_DIR) / "detection_counts.json"),
        help="Model summary JSON output path",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Explicit model list",
    )
    parser.add_argument("--input_size", type=str, default="640x640")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument(
        "--core",
        type=str,
        default="0",
        choices=["0", "1", "2", "all"],
    )
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"image_dir not found: {args.image_dir}")

    images = collect_images(args.image_dir)
    if not images:
        raise RuntimeError(f"no valid images found in: {args.image_dir}")

    in_w, in_h = [int(x) for x in args.input_size.lower().split("x")]
    input_size = (in_w, in_h)

    core_map = {
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
        "all": RKNNLite.NPU_CORE_0_1_2,
    }
    core_mask = core_map[args.core]

    model_summaries = []
    all_rows = []

    print(f"Images: {len(images)}")
    print(f"Models: {len(args.models)}")
    print(f"Conf: {args.conf}")
    print(f"IoU: {args.iou}")

    for model_path in args.models:
        print(f"Running model: {model_path}")
        summary, rows = run_model(
            model_path=model_path,
            images=images,
            input_size=input_size,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            core_mask=core_mask,
        )
        model_summaries.append(summary)
        all_rows.extend(rows)
        print(
            f"Completed: {summary['model_name']} | "
            f"status={summary['status']} | "
            f"total_detections={summary['total_detections']} | "
            f"nonzero_images={summary['nonzero_images']}"
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "model_path",
                "image_name",
                "image_path",
                "decode_mode",
                "det_count",
                "num_classes",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    payload = {
        "image_dir": args.image_dir,
        "image_count": len(images),
        "input_size": list(input_size),
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "models": model_summaries,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
