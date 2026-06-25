#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 实现不同的RKNN模型在环境下面的推理效果评测，输出预测结果和评测指标。
"""
Paper-oriented RKNN evaluation for the port defect dataset.

What it does:
1. Auto-detect annotation format (YOLO txt or COCO json) and print the rationale.
2. Reuse the existing RKNN inference pipeline without changing decode behavior.
3. Run two experiments required by the paper:
   - unified_conf_0.10
   - best_working_point
4. Export:
   - prediction JSON
   - Markdown tables
   - CSV tables
   - JSON summaries
   - academic-style report text
5. Validate on the first N images before the full 209-image evaluation.
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from paths import MODELS_DIR, DATA_IMAGES_DIR, DATA_LABELS_DIR, OUTPUTS_DIR, EVAL_DIR
import infer as infer_core


DEFAULT_MODEL_ORDER = [
    "baseline_test0625.rknn",
    "B3-Lite_V2_fp.rknn",
    "B3-Llite-V3_fp.rknn",
    "yolov8n_baseline_fp.rknn",
    "yolov8n_port_RULELOSS_fp.rknn",
]

DEFAULT_BEST_CONF = {
    "baseline_test0625.rknn": 0.05,
    "B3-Lite_V2_fp.rknn": 0.10,
    "B3-Llite-V3_fp.rknn": 0.10,
    "yolov8n_baseline_fp.rknn": 0.01,
    "yolov8n_port_RULELOSS_fp.rknn": 0.001,
}
DEFAULT_CONF_SCAN_VALUES = "0.35,0.20,0.10,0.05,0.03,0.01,0.005,0.001"


def resolve_model_specs(model_dir):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    default_specs = [(name, str(model_dir / name)) for name in DEFAULT_MODEL_ORDER]
    if all(Path(path).exists() for _, path in default_specs):
        return default_specs

    discovered = sorted(model_dir.glob("*.rknn"))
    if not discovered:
        raise FileNotFoundError(f"No .rknn models found under: {model_dir}")

    return [(path.name, str(path)) for path in discovered]


def build_experiments(model_names):
    unified = {name: 0.10 for name in model_names}
    best = {name: DEFAULT_BEST_CONF.get(name, 0.10) for name in model_names}

    description = "Best validated operating point for each model."
    if any(name not in DEFAULT_BEST_CONF for name in model_names):
        description += " Models without an explicit tuned threshold fall back to conf=0.10."

    return {
        "unified_conf_0.10": {
            "description": "Fair comparison using the unified confidence threshold conf=0.10 for all models.",
            "conf_by_model": unified,
        },
        "best_working_point": {
            "description": description,
            "conf_by_model": best,
        },
    }


def parse_conf_scan_values(spec):
    values = []
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0:
            raise ValueError(f"Confidence threshold must be >= 0, got {value}")
        values.append(value)

    if not values:
        raise ValueError("No confidence thresholds were provided for conf scan")

    return sorted(set(values), reverse=True)


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def f4(x):
    return "N/A" if x is None else f"{x:.4f}"


def f2(x):
    return "N/A" if x is None else f"{x:.2f}"


def pct(x):
    return "N/A" if x is None else f"{x * 100.0:.2f}%"


def markdown_table(headers, rows):
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body)


def write_csv(path, headers, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def detect_label_format(label_path, image_dir):
    label_path = Path(label_path)
    image_dir = Path(image_dir)

    if label_path.is_file() and label_path.suffix.lower() == ".json":
        rationale = f"Detected COCO json because label path itself is a json file: {label_path.name}"
        return {"format": "coco_json", "source": label_path, "rationale": rationale}

    json_files = sorted(label_path.glob("*.json")) if label_path.is_dir() else []
    txt_files = sorted(label_path.glob("*.txt")) if label_path.is_dir() else []

    if json_files:
        rationale = (
            f"Detected COCO json because {label_path} contains {len(json_files)} json file(s); "
            f"using {json_files[0].name} as the annotation manifest"
        )
        return {"format": "coco_json", "source": json_files[0], "rationale": rationale}

    if txt_files:
        sample_lines = []
        for txt in txt_files[:5]:
            for raw in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if line:
                    sample_lines.append(line)
                    break
        yolo_like = True
        for line in sample_lines:
            parts = line.split()
            if len(parts) != 5:
                yolo_like = False
                break
            try:
                int(float(parts[0]))
                [float(v) for v in parts[1:]]
            except ValueError:
                yolo_like = False
                break
        if yolo_like:
            rationale = (
                f"Detected YOLO txt because {label_path} contains {len(txt_files)} txt annotation files, "
                f"no json manifest, and sampled lines follow the 5-column '<cls> <cx> <cy> <w> <h>' pattern"
            )
            return {"format": "yolo_txt", "source": label_path, "rationale": rationale}

    raise RuntimeError(
        f"Unable to determine label format under '{label_path}'. Expected YOLO txt files or a COCO json file."
    )


def yolo_line_to_xyxy(parts, width, height):
    cls_id = int(float(parts[0]))
    cx = float(parts[1]) * float(width)
    cy = float(parts[2]) * float(height)
    bw = float(parts[3]) * float(width)
    bh = float(parts[4]) * float(height)
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(float(width - 1), cx + bw / 2.0)
    y2 = min(float(height - 1), cy + bh / 2.0)
    return cls_id, [x1, y1, x2, y2]


def load_yolo_gt(label_file, width, height):
    boxes = []
    class_ids = []
    if not label_file.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    for raw in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, box = yolo_line_to_xyxy(parts, width, height)
        class_ids.append(cls_id)
        boxes.append(box)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return np.asarray(boxes, dtype=np.float32), np.asarray(class_ids, dtype=np.int32)


def build_coco_index(coco_json):
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    categories = sorted(data.get("categories", []), key=lambda x: x["id"])
    category_ids = [cat["id"] for cat in categories]
    cat_to_contig = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    return images, anns_by_image, categories, cat_to_contig


def load_coco_gt(image_name, coco_index, image_dir):
    images, anns_by_image, _, cat_to_contig = coco_index
    image_id = None
    image_info = None

    for img_id, info in images.items():
        if info.get("file_name") == image_name:
            image_id = img_id
            image_info = info
            break

    if image_id is None:
        img_path = Path(image_dir) / image_name
        im = cv2.imread(str(img_path))
        if im is None:
            raise RuntimeError(f"Failed to read image for COCO lookup: {img_path}")
        h, w = im.shape[:2]
        return w, h, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    width = int(image_info["width"])
    height = int(image_info["height"])
    boxes = []
    class_ids = []
    for ann in anns_by_image.get(image_id, []):
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        class_ids.append(cat_to_contig[ann["category_id"]])

    if not boxes:
        return width, height, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return width, height, np.asarray(boxes, dtype=np.float32), np.asarray(class_ids, dtype=np.int32)


def build_dataset_records(image_dir, label_info, limit=None):
    image_paths = [Path(p) for p in infer_core.collect_images(str(image_dir))]
    if limit is not None:
        image_paths = image_paths[:limit]

    records = []
    coco_index = None
    if label_info["format"] == "coco_json":
        coco_index = build_coco_index(label_info["source"])

    for img_path in image_paths:
        if label_info["format"] == "yolo_txt":
            im = cv2.imread(str(img_path))
            if im is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            height, width = im.shape[:2]
            gt_boxes, gt_classes = load_yolo_gt(Path(label_info["source"]) / f"{img_path.stem}.txt", width, height)
        else:
            width, height, gt_boxes, gt_classes = load_coco_gt(img_path.name, coco_index, image_dir)

        records.append(
            {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "width": int(width),
                "height": int(height),
                "gt_boxes": gt_boxes,
                "gt_classes": gt_classes,
            }
        )

    return records


def summarize_label_mapping(records):
    label_counts = Counter()
    unknown_ids = []
    for record in records:
        for cid in record["gt_classes"]:
            cid_int = int(cid)
            label_counts[cid_int] += 1
            if cid_int < 0 or cid_int >= len(infer_core.CLASSES):
                unknown_ids.append(cid_int)

    mapping_rows = []
    for cid in sorted(label_counts):
        mapping_rows.append(
            [
                cid,
                infer_core.CLASSES[cid] if 0 <= cid < len(infer_core.CLASSES) else "UNKNOWN",
                label_counts[cid],
            ]
        )

    return {
        "label_counts": dict(sorted(label_counts.items())),
        "mapping_rows": mapping_rows,
        "mapping_basis": (
            "YOLO txt labels store only integer class ids. "
            "The current id->name mapping is taken from infer_core.CLASSES "
            f"{infer_core.CLASSES} and is consistent with yolov8_NEU.py."
        ),
        "unknown_ids": sorted(set(unknown_ids)),
    }


def check_image_label_pairs(image_dir, label_info, sample_count=5):
    image_paths = [Path(p) for p in infer_core.collect_images(str(image_dir))]
    image_stems = {p.stem for p in image_paths}

    if label_info["format"] == "yolo_txt":
        label_paths = sorted(Path(label_info["source"]).glob("*.txt"))
        label_stems = {p.stem for p in label_paths}
        sample_pairs = []
        for img_path in image_paths[:sample_count]:
            label_path = Path(label_info["source"]) / f"{img_path.stem}.txt"
            sample_pairs.append(
                {
                    "image": img_path.name,
                    "label": label_path.name,
                    "exists": label_path.exists(),
                }
            )
        return {
            "missing_label_images": sorted(image_stems - label_stems),
            "extra_label_files": sorted(label_stems - image_stems),
            "sample_pairs": sample_pairs,
        }

    coco_index = build_coco_index(label_info["source"])
    images, _, _, _ = coco_index
    coco_names = {info.get("file_name") for info in images.values()}
    sample_pairs = []
    for img_path in image_paths[:sample_count]:
        sample_pairs.append(
            {
                "image": img_path.name,
                "label": img_path.name,
                "exists": img_path.name in coco_names,
            }
        )
    return {
        "missing_label_images": sorted([p.name for p in image_paths if p.name not in coco_names]),
        "extra_label_files": sorted([name for name in coco_names if name not in {p.name for p in image_paths}]),
        "sample_pairs": sample_pairs,
    }


def inspect_yolo_coordinate_system(image_dir, label_info, sample_count=5, seed=42):
    if label_info["format"] != "yolo_txt":
        return {
            "format_ok": False,
            "message": "Coordinate inspection is only implemented for YOLO txt labels.",
            "sample_conversions": [],
        }

    rng = random.Random(seed)
    image_paths = [Path(p) for p in infer_core.collect_images(str(image_dir))]
    sampled_images = rng.sample(image_paths, min(sample_count, len(image_paths)))
    image_size_counts = Counter()
    normalized_out_of_range = []
    pixel_out_of_range = []
    sample_conversions = []
    edge_touch_boxes = 0
    total_boxes = 0

    for img_path in image_paths:
        im = cv2.imread(str(img_path))
        if im is None:
            raise RuntimeError(f"Failed to read image during coordinate inspection: {img_path}")
        height, width = im.shape[:2]
        image_size_counts[f"{width}x{height}"] += 1
        label_path = Path(label_info["source"]) / f"{img_path.stem}.txt"

        for line_no, raw in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = [float(v) for v in parts[1:]]

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                normalized_out_of_range.append(
                    {
                        "image": img_path.name,
                        "line": line_no,
                        "raw": line,
                    }
                )

            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height
            total_boxes += 1

            if x1 < -1e-3 or y1 < -1e-3 or x2 > width + 1e-3 or y2 > height + 1e-3:
                pixel_out_of_range.append(
                    {
                        "image": img_path.name,
                        "line": line_no,
                        "raw": line,
                        "xyxy": [x1, y1, x2, y2],
                        "image_size": [width, height],
                    }
                )

            if x1 <= 1.0 or y1 <= 1.0 or x2 >= (width - 1.0) or y2 >= (height - 1.0):
                edge_touch_boxes += 1

            if img_path in sampled_images and len(sample_conversions) < sample_count:
                sample_conversions.append(
                    {
                        "image": img_path.name,
                        "label": f"{img_path.stem}.txt",
                        "image_size": [width, height],
                        "class_id": cls_id,
                        "class_name": infer_core.CLASSES[cls_id] if 0 <= cls_id < len(infer_core.CLASSES) else "UNKNOWN",
                        "raw": line,
                        "xyxy": [
                            round(max(0.0, x1), 2),
                            round(max(0.0, y1), 2),
                            round(min(width - 1.0, x2), 2),
                            round(min(height - 1.0, y2), 2),
                        ],
                    }
                )

    return {
        "format_ok": True,
        "message": "Confirmed 5-column YOLO format '<class> <x_center> <y_center> <width> <height>'.",
        "image_size_counts": dict(image_size_counts),
        "normalized_out_of_range": normalized_out_of_range,
        "pixel_out_of_range": pixel_out_of_range,
        "sample_conversions": sample_conversions,
        "edge_touch_boxes": edge_touch_boxes,
        "total_boxes": total_boxes,
    }


def box_iou_xyxy(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-9)


def match_counts_for_image(preds, gt_boxes, gt_classes, num_classes, iou_thr=0.5):
    per_class = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
        for cid in range(num_classes)
    }

    if gt_boxes.size == 0:
        for pred in preds:
            cid = int(pred["class_id"])
            per_class[cid]["fp"] += 1
            per_class[cid]["pred"] += 1
        return per_class

    for cid in range(num_classes):
        gt_idx = np.where(gt_classes == cid)[0]
        pred_list = [pred for pred in preds if int(pred["class_id"]) == cid]
        pred_list.sort(key=lambda x: x["score"], reverse=True)
        matched = np.zeros(len(gt_idx), dtype=bool)

        per_class[cid]["gt"] += int(len(gt_idx))
        per_class[cid]["pred"] += int(len(pred_list))

        if len(gt_idx) == 0:
            per_class[cid]["fp"] += int(len(pred_list))
            continue

        gt_cls_boxes = gt_boxes[gt_idx]
        for pred in pred_list:
            ious = box_iou_xyxy(np.asarray(pred["bbox_xyxy"], dtype=np.float32), gt_cls_boxes)
            best_local = int(np.argmax(ious)) if len(ious) else -1
            if best_local >= 0 and ious[best_local] >= iou_thr and not matched[best_local]:
                matched[best_local] = True
                per_class[cid]["tp"] += 1
            else:
                per_class[cid]["fp"] += 1

        per_class[cid]["fn"] += int(np.sum(~matched))

    return per_class


def compute_ap50(records, predictions_by_image, num_classes, iou_thr=0.5):
    ap_by_class = {}
    gt_count_by_class = {cid: 0 for cid in range(num_classes)}

    for record in records:
        for cid in record["gt_classes"]:
            gt_count_by_class[int(cid)] += 1

    for cid in range(num_classes):
        npos = gt_count_by_class[cid]
        if npos == 0:
            ap_by_class[cid] = None
            continue

        gt_state = {}
        pred_pool = []
        for record in records:
            mask = record["gt_classes"] == cid
            gt_boxes = record["gt_boxes"][mask]
            gt_state[record["image_name"]] = {
                "boxes": gt_boxes,
                "matched": np.zeros(len(gt_boxes), dtype=bool),
            }
            for pred in predictions_by_image.get(record["image_name"], []):
                if int(pred["class_id"]) == cid:
                    pred_pool.append(
                        {
                            "image_name": record["image_name"],
                            "score": float(pred["score"]),
                            "bbox_xyxy": np.asarray(pred["bbox_xyxy"], dtype=np.float32),
                        }
                    )

        pred_pool.sort(key=lambda x: x["score"], reverse=True)
        if not pred_pool:
            ap_by_class[cid] = 0.0
            continue

        tp = np.zeros(len(pred_pool), dtype=np.float32)
        fp = np.zeros(len(pred_pool), dtype=np.float32)

        for idx, pred in enumerate(pred_pool):
            state = gt_state[pred["image_name"]]
            gt_boxes = state["boxes"]
            matched = state["matched"]
            if len(gt_boxes) == 0:
                fp[idx] = 1.0
                continue

            ious = box_iou_xyxy(pred["bbox_xyxy"], gt_boxes)
            best = int(np.argmax(ious)) if len(ious) else -1
            if best >= 0 and ious[best] >= iou_thr and not matched[best]:
                matched[best] = True
                tp[idx] = 1.0
            else:
                fp[idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / max(float(npos), 1e-9)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ap = 0.0
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        ap_by_class[cid] = float(ap)

    valid_aps = [v for v in ap_by_class.values() if v is not None]
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    return ap_by_class, map50


def summarize_predictions(records, predictions_by_image, num_classes):
    overall = {"tp": 0, "fp": 0, "fn": 0}
    per_class = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
        for cid in range(num_classes)
    }

    for record in records:
        preds = predictions_by_image.get(record["image_name"], [])
        counts = match_counts_for_image(preds, record["gt_boxes"], record["gt_classes"], num_classes, iou_thr=0.5)
        for cid in range(num_classes):
            for key in ("tp", "fp", "fn", "gt", "pred"):
                per_class[cid][key] += counts[cid][key]

    for cid in range(num_classes):
        overall["tp"] += per_class[cid]["tp"]
        overall["fp"] += per_class[cid]["fp"]
        overall["fn"] += per_class[cid]["fn"]

    ap_by_class, map50 = compute_ap50(records, predictions_by_image, num_classes, iou_thr=0.5)

    overall_precision = safe_div(overall["tp"], overall["tp"] + overall["fp"])
    overall_recall = safe_div(overall["tp"], overall["tp"] + overall["fn"])
    overall_f1 = safe_div(2.0 * overall_precision * overall_recall, overall_precision + overall_recall)

    per_class_metrics = []
    for cid in range(num_classes):
        stats = per_class[cid]
        precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
        recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        per_class_metrics.append(
            {
                "class_id": cid,
                "class_name": infer_core.CLASSES[cid] if cid < len(infer_core.CLASSES) else f"class_{cid}",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50": ap_by_class[cid],
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "gt_count": stats["gt"],
                "pred_count": stats["pred"],
            }
        )

    return {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "map50": map50,
        "tp": overall["tp"],
        "fp": overall["fp"],
        "fn": overall["fn"],
        "per_class": per_class_metrics,
    }


def compute_alignment_gap(records, prediction_run, iou_thr=0.5, example_limit=8):
    gt_total = 0
    same_class_hits = 0
    any_class_hits = 0
    best_same_ious = []
    best_any_ious = []
    informative_examples = []

    for record in records:
        preds = prediction_run["predictions_by_image"].get(record["image_name"], [])
        for gt_box, gt_cid in zip(record["gt_boxes"], record["gt_classes"]):
            gt_total += 1
            gt_cid = int(gt_cid)
            best_any_iou = 0.0
            best_same_iou = 0.0
            best_any_pred = None

            for pred in preds:
                pred_box = np.asarray(pred["bbox_xyxy"], dtype=np.float32)
                cur_iou = float(box_iou_xyxy(np.asarray(gt_box, dtype=np.float32), pred_box[None, :])[0])
                if cur_iou > best_any_iou:
                    best_any_iou = cur_iou
                    best_any_pred = pred
                if int(pred["class_id"]) == gt_cid and cur_iou > best_same_iou:
                    best_same_iou = cur_iou

            best_any_ious.append(best_any_iou)
            best_same_ious.append(best_same_iou)
            if best_any_iou >= iou_thr:
                any_class_hits += 1
            if best_same_iou >= iou_thr:
                same_class_hits += 1

            if best_any_pred is not None and best_any_iou >= 0.20 and len(informative_examples) < example_limit:
                informative_examples.append(
                    {
                        "image": record["image_name"],
                        "gt_class_id": gt_cid,
                        "gt_class_name": infer_core.CLASSES[gt_cid] if 0 <= gt_cid < len(infer_core.CLASSES) else "UNKNOWN",
                        "gt_box": [round(float(v), 2) for v in gt_box.tolist()],
                        "best_any_iou": round(best_any_iou, 4),
                        "best_same_iou": round(best_same_iou, 4),
                        "best_pred_class_id": int(best_any_pred["class_id"]),
                        "best_pred_class_name": best_any_pred["class_name"],
                        "best_pred_score": round(float(best_any_pred["score"]), 4),
                        "best_pred_box": [round(float(v), 2) for v in best_any_pred["bbox_xyxy"]],
                    }
                )

    return {
        "gt_total": gt_total,
        "same_class_hits_iou50": same_class_hits,
        "any_class_hits_iou50": any_class_hits,
        "same_class_match_rate": safe_div(same_class_hits, gt_total),
        "any_class_match_rate": safe_div(any_class_hits, gt_total),
        "mean_best_same_iou": float(np.mean(best_same_ious)) if best_same_ious else 0.0,
        "mean_best_any_iou": float(np.mean(best_any_ious)) if best_any_ious else 0.0,
        "max_best_same_iou": float(np.max(best_same_ious)) if best_same_ious else 0.0,
        "max_best_any_iou": float(np.max(best_any_ious)) if best_any_ious else 0.0,
        "informative_examples": informative_examples,
    }


def draw_labeled_box(image, box, text, color, thickness=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text_y = y1 - 6 if y1 > 18 else y1 + 16
    cv2.putText(
        image,
        text,
        (max(2, x1), max(12, text_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def write_alignment_visualizations(records, prediction_run, out_dir, max_pred_draw=20):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        im = cv2.imread(record["image_path"])
        if im is None:
            raise RuntimeError(f"Failed to read image for visualization: {record['image_path']}")

        canvas = im.copy()
        detections = list(prediction_run["predictions_by_image"].get(record["image_name"], []))
        detections.sort(key=lambda x: x["score"], reverse=True)

        for gt_box, gt_cid in zip(record["gt_boxes"], record["gt_classes"]):
            gt_cid = int(gt_cid)
            gt_name = infer_core.CLASSES[gt_cid] if 0 <= gt_cid < len(infer_core.CLASSES) else f"class_{gt_cid}"
            draw_labeled_box(canvas, gt_box, f"GT:{gt_name}", (0, 220, 0), thickness=2)

        for det in detections[:max_pred_draw]:
            draw_labeled_box(
                canvas,
                det["bbox_xyxy"],
                f"PD:{det['class_name']} {det['score']:.3f}",
                (0, 0, 255),
                thickness=2,
            )

        cv2.putText(
            canvas,
            "Green=GT  Red=Prediction",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Green=GT  Red=Prediction",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / record["image_name"]), canvas)


def build_alignment_conclusion(pair_check, coord_check, model_diags):
    if pair_check["missing_label_images"] or pair_check["extra_label_files"]:
        return (
            "图片与标签未一一对应是当前低 mAP 的主因。",
            "存在缺失标签图像或多余标签文件，需要先修复文件对应关系。",
        )

    if coord_check.get("normalized_out_of_range") or coord_check.get("pixel_out_of_range"):
        return (
            "YOLO 坐标转换或标签坐标体系存在异常，是当前低 mAP 的主因。",
            "发现了越界或非归一化标注，需要先修复标签格式或坐标变换。",
        )

    best_any = max((diag["alignment"]["any_class_match_rate"] for diag in model_diags), default=0.0)
    best_gap = max(
        (diag["alignment"]["any_class_match_rate"] - diag["alignment"]["same_class_match_rate"] for diag in model_diags),
        default=0.0,
    )

    if best_any < 0.10:
        return (
            "更接近“模型预测确实与 GT 差距很大”，而不是类别映射错误或 patch 坐标体系错误。",
            "在忽略类别后，IoU>=0.5 的可匹配 GT 仍然很少，说明框本身大多没有贴到 GT 上。",
        )

    if best_gap >= 0.20:
        return (
            "类别映射可能存在明显问题，但不是唯一问题。",
            "忽略类别后的匹配率明显高于同类匹配率，建议进一步核对训练标签顺序与导出模型类别顺序。",
        )

    return (
        "低 mAP 主要来自预测框与 GT 的几何对齐不足，类别映射问题最多只占次要因素。",
        "文件对应关系和 YOLO 坐标变换未见异常，下一步应重点核对模型输出框位置、预处理和标签生成流程。",
    )


def write_alignment_diagnosis_report(
    out_dir,
    label_mapping,
    pair_check,
    coord_check,
    model_diags,
    diagnostic_records,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_root = out_dir / "visualizations"

    mapping_table = markdown_table(
        ["标签ID", "类别名", "数量"],
        [[row[0], row[1], row[2]] for row in label_mapping["mapping_rows"]],
    )

    pair_table = markdown_table(
        ["图片", "标签", "存在"],
        [[row["image"], row["label"], row["exists"]] for row in pair_check["sample_pairs"]],
    )

    conversion_rows = []
    for item in coord_check.get("sample_conversions", []):
        conversion_rows.append(
            [
                item["image"],
                item["raw"],
                item["xyxy"],
            ]
        )
    conversion_table = markdown_table(
        ["样例图片", "原始YOLO行", "转换后xyxy"],
        conversion_rows if conversion_rows else [["N/A", "N/A", "N/A"]],
    )

    model_rows = []
    for diag in model_diags:
        align = diag["alignment"]
        model_rows.append(
            [
                diag["model_name"],
                f"{diag['conf']:.4f}",
                align["gt_total"],
                align["same_class_hits_iou50"],
                align["any_class_hits_iou50"],
                f4(align["mean_best_same_iou"]),
                f4(align["mean_best_any_iou"]),
                f"{diag['vis_dir']}",
            ]
        )
    model_table = markdown_table(
        [
            "模型名",
            "conf",
            "GT数",
            "同类IoU>=0.5",
            "忽略类别IoU>=0.5",
            "mean_best_same_iou",
            "mean_best_any_iou",
            "可视化目录",
        ],
        model_rows,
    )

    headline, explanation = build_alignment_conclusion(pair_check, coord_check, model_diags)

    lines = [
        "# Alignment Diagnosis Report",
        "",
        "## 1. 类别映射检查",
        label_mapping["mapping_basis"],
        "",
        mapping_table,
        "",
        f"未知类别ID数量: {len(label_mapping['unknown_ids'])}",
        "",
        "## 2. 图片与标签一一对应检查",
        f"缺失标签图片数: {len(pair_check['missing_label_images'])}",
        f"多余标签文件数: {len(pair_check['extra_label_files'])}",
        "",
        pair_table,
        "",
        "## 3. Patch/YOLO 坐标体系检查",
        coord_check.get("message", "N/A"),
        f"图像尺寸分布: {coord_check.get('image_size_counts', {})}",
        f"归一化越界标注数: {len(coord_check.get('normalized_out_of_range', []))}",
        f"像素坐标越界标注数: {len(coord_check.get('pixel_out_of_range', []))}",
        f"贴边框数量: {coord_check.get('edge_touch_boxes', 0)} / {coord_check.get('total_boxes', 0)}",
        "",
        conversion_table,
        "",
        "## 4. 10张图预测/标签对齐诊断",
        f"诊断样本数: {len(diagnostic_records)}",
        f"可视化根目录: {vis_root}",
        "",
        model_table,
        "",
        "## 5. 诊断结论",
        headline,
        explanation,
        "",
    ]

    for diag in model_diags:
        lines.append(f"### {diag['model_name']}")
        lines.append(f"可视化目录: {diag['vis_dir']}")
        lines.append("代表性样例:")
        examples = diag["alignment"]["informative_examples"]
        if examples:
            for item in examples:
                lines.append(
                    f"- {item['image']} | GT={item['gt_class_name']} {item['gt_box']} | "
                    f"BestPred={item['best_pred_class_name']} {item['best_pred_box']} "
                    f"score={item['best_pred_score']:.4f} | "
                    f"best_any_iou={item['best_any_iou']:.4f} best_same_iou={item['best_same_iou']:.4f}"
                )
        else:
            lines.append("- 没有达到 best_any_iou>=0.20 的代表性样例。")
        lines.append("")

    report_path = out_dir / "alignment_diagnosis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(out_dir / "alignment_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_mapping": label_mapping,
                "pair_check": pair_check,
                "coord_check": coord_check,
                "models": model_diags,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[Alignment] Mapping")
    print(mapping_table)
    print("\n[Alignment] Image/Label Pairs")
    print(pair_table)
    print("\n[Alignment] Coordinate Samples")
    print(conversion_table)
    print("\n[Alignment] Model Summary")
    print(model_table)
    print(f"\n[Alignment] Conclusion: {headline}")
    print(f"[Alignment] Detail: {explanation}")

    return report_path


def run_alignment_diagnosis(
    all_records,
    diagnostic_records,
    label_info,
    model_specs,
    input_size,
    core,
    out_root,
    image_dir,
    experiments,
    input_format="rgb",
    letterbox_color=114,
):
    label_mapping = summarize_label_mapping(all_records)
    pair_check = check_image_label_pairs(image_dir, label_info, sample_count=5)
    coord_check = inspect_yolo_coordinate_system(image_dir, label_info, sample_count=5, seed=42)
    diag_root = Path(out_root) / "alignment_diagnosis"
    vis_root = diag_root / "visualizations"
    model_diags = []

    for model_name, model_path in model_specs:
        conf = experiments["best_working_point"]["conf_by_model"][model_name]
        prediction_run = run_predictions_for_model(
            model_path,
            diagnostic_records,
            conf,
            input_size,
            core,
            input_format=input_format,
            letterbox_color=letterbox_color,
        )
        alignment = compute_alignment_gap(diagnostic_records, prediction_run, iou_thr=0.5, example_limit=8)
        vis_dir = vis_root / model_name.replace(".rknn", "")
        write_alignment_visualizations(diagnostic_records, prediction_run, vis_dir)
        model_diags.append(
            {
                "model_name": prediction_run["model_name"],
                "model_path": prediction_run["model_path"],
                "conf": prediction_run["actual_conf"],
                "conf_source": prediction_run["conf_source"],
                "norm_mode": prediction_run["actual_norm_mode"],
                "alignment": alignment,
                "vis_dir": str(vis_dir),
            }
        )

    return write_alignment_diagnosis_report(
        out_dir=diag_root,
        label_mapping=label_mapping,
        pair_check=pair_check,
        coord_check=coord_check,
        model_diags=model_diags,
        diagnostic_records=diagnostic_records,
    )


def run_predictions_for_model(
    model_path,
    records,
    conf,
    input_size,
    core,
    decode_mode="auto",
    input_format="rgb",
    letterbox_color=114,
):
    resolved_model = infer_core.resolve_model_path(str(model_path), prefer_fp=False)
    profile = infer_core.resolve_model_runtime_profile(resolved_model, conf, "auto", debug=False)
    actual_conf = profile["conf"]
    actual_norm_mode = profile["norm_mode"]
    actual_input_layout = profile.get("input_layout", "auto")
    bbox_expand_scale = profile.get("bbox_expand_scale", 1.0)
    bbox_expand_pad = profile.get("bbox_expand_pad", 0.0)
    model_name = profile["model_name"]
    num_classes = len(infer_core.CLASSES)

    rknn = infer_core.RKNNLite()
    ret = rknn.load_rknn(resolved_model)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: ret={ret}, model={resolved_model}")
    ret = rknn.init_runtime(core_mask=infer_core.resolve_core_mask(core))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"init_runtime failed: ret={ret}, model={resolved_model}")

    output_details = infer_core.try_get_output_details(rknn)
    try:
        dummy = infer_core.make_dummy_input(
            rknn,
            input_size,
            input_format=input_format,
            input_layout=actual_input_layout,
            norm_mode=actual_norm_mode,
            letterbox_color=letterbox_color,
        )
        for _ in range(2):
            _ = rknn.inference(inputs=[dummy])
    except Exception:
        pass

    entries = []
    predictions_by_image = {}

    try:
        total = len(records)
        print(
            f"[Eval] model={model_name} running={resolved_model} conf={actual_conf:.6f} "
            f"norm_mode={actual_norm_mode} images={total} "
            f"bbox_expand_scale={bbox_expand_scale:.2f} bbox_expand_pad={bbox_expand_pad:.2f} "
            f"input_format={input_format} letterbox_color={int(letterbox_color)}"
        )
        for idx, record in enumerate(records, 1):
            im0 = cv2.imread(record["image_path"])
            if im0 is None:
                raise RuntimeError(f"Failed to read image: {record['image_path']}")

            t0 = time.perf_counter()
            inp, ratio, pad = infer_core.prepare_input(
                im0,
                rknn,
                input_size,
                input_format=input_format,
                input_layout=actual_input_layout,
                norm_mode=actual_norm_mode,
                letterbox_color=letterbox_color,
                debug=False,
            )
            t1 = time.perf_counter()
            outputs = rknn.inference(inputs=[inp])
            t2 = time.perf_counter()
            boxes, cids, scores, mode = infer_core.post_process(
                outputs=outputs,
                output_details=output_details,
                input_size=input_size,
                num_classes=num_classes,
                conf_thres=actual_conf,
                iou_thres=0.50,
                max_det=300,
                decode_mode=decode_mode,
                min_wh=10.0,
                min_area=120.0,
                max_aspect_ratio=8.0,
                merge_gap=12,
                model_name=model_name,
                flat_head_debug=False,
                conf_scan_values=None,
                flat_force_sigmoid=False,
                mode_diagnostic=False,
            )
            t3 = time.perf_counter()

            detections = []
            det_count = 0
            if boxes is not None and len(boxes) > 0:
                boxes = infer_core.expand_boxes_xyxy(
                    boxes,
                    input_size,
                    scale=bbox_expand_scale,
                    pad=bbox_expand_pad,
                )
                boxes0 = infer_core.scale_boxes_to_original(boxes, ratio, pad, im0.shape)
                boxes0, cids, scores = infer_core.filter_edge_boxes(
                    boxes0,
                    cids,
                    scores,
                    im0.shape,
                    edge_margin=4,
                    min_edge_box=20,
                )
                if boxes0 is not None and len(boxes0) > 0:
                    det_count = int(len(boxes0))
                    for box, cid, score in zip(boxes0, cids, scores):
                        detections.append(
                            {
                                "class_id": int(cid),
                                "class_name": infer_core.CLASSES[int(cid)],
                                "score": float(score),
                                "bbox_xyxy": [float(v) for v in box.tolist()],
                            }
                        )
                else:
                    det_count = 0

            pre_ms = (t1 - t0) * 1000.0
            infer_ms = (t2 - t1) * 1000.0
            post_ms = (t3 - t2) * 1000.0

            predictions_by_image[record["image_name"]] = detections
            entries.append(
                {
                    "image_name": record["image_name"],
                    "image_path": record["image_path"],
                    "mode": mode,
                    "det_count": det_count,
                    "pre_ms": pre_ms,
                    "infer_ms": infer_ms,
                    "post_ms": post_ms,
                    "detections": detections,
                }
            )

            if idx == total or idx % 25 == 0:
                print(
                    f"[Eval] {model_name} progress {idx}/{total} "
                    f"last_mode={mode} last_det={det_count}"
                )
    finally:
        rknn.release()

    return {
        "model_name": model_name,
        "model_path": resolved_model,
        "actual_conf": actual_conf,
        "conf_source": profile["conf_source"],
        "actual_norm_mode": actual_norm_mode,
        "bbox_expand_scale": bbox_expand_scale,
        "bbox_expand_pad": bbox_expand_pad,
        "input_format": input_format,
        "letterbox_color": int(letterbox_color),
        "entries": entries,
        "predictions_by_image": predictions_by_image,
    }


def build_model_summary(records, prediction_run):
    entries = prediction_run["entries"]
    metrics = summarize_predictions(records, prediction_run["predictions_by_image"], len(infer_core.CLASSES))

    mode_counts = Counter(entry["mode"] for entry in entries)
    empty_count = sum(entry["det_count"] == 0 for entry in entries)
    mean_det = float(np.mean([entry["det_count"] for entry in entries])) if entries else 0.0
    mean_pre = float(np.mean([entry["pre_ms"] for entry in entries])) if entries else 0.0
    mean_infer = float(np.mean([entry["infer_ms"] for entry in entries])) if entries else 0.0
    mean_post = float(np.mean([entry["post_ms"] for entry in entries])) if entries else 0.0
    mean_e2e = mean_pre + mean_infer + mean_post
    fps = safe_div(1000.0, mean_e2e)

    return {
        "model_name": prediction_run["model_name"],
        "model_path": prediction_run["model_path"],
        "conf": prediction_run["actual_conf"],
        "conf_source": prediction_run["conf_source"],
        "norm_mode": prediction_run["actual_norm_mode"],
        "bbox_expand_scale": prediction_run.get("bbox_expand_scale", 1.0),
        "bbox_expand_pad": prediction_run.get("bbox_expand_pad", 0.0),
        "input_format": prediction_run.get("input_format", "rgb"),
        "letterbox_color": int(prediction_run.get("letterbox_color", 114)),
        "num_images": len(entries),
        "empty_count": empty_count,
        "empty_rate": safe_div(empty_count, len(entries)),
        "mean_det_per_image": mean_det,
        "mode_counts": {
            "flat": int(mode_counts.get("flat", 0)),
            "dfl_concat": int(mode_counts.get("dfl_concat", 0)),
            "none": int(mode_counts.get("none", 0)),
        },
        "latency_ms": {
            "pre_avg": mean_pre,
            "infer_avg": mean_infer,
            "post_avg": mean_post,
            "pre_min": float(np.min([entry["pre_ms"] for entry in entries])) if entries else 0.0,
            "infer_min": float(np.min([entry["infer_ms"] for entry in entries])) if entries else 0.0,
            "post_min": float(np.min([entry["post_ms"] for entry in entries])) if entries else 0.0,
            "pre_max": float(np.max([entry["pre_ms"] for entry in entries])) if entries else 0.0,
            "infer_max": float(np.max([entry["infer_ms"] for entry in entries])) if entries else 0.0,
            "post_max": float(np.max([entry["post_ms"] for entry in entries])) if entries else 0.0,
            "fps_e2e": fps,
        },
        "metrics": metrics,
        "entries": entries,
    }


def conf_selection_key(summary, metric_name):
    metrics = summary["metrics"]
    primary = metrics["map50"] if metric_name == "map50" else metrics["f1"]
    return (
        primary,
        metrics["f1"],
        metrics["recall"],
        metrics["precision"],
        -summary["empty_rate"],
        summary["conf"],
    )


def scan_confidence_thresholds(
    records,
    model_specs,
    conf_values,
    input_size,
    core,
    out_root,
    dataset_meta,
    select_metric="map50",
    input_format="rgb",
    letterbox_color=114,
):
    scan_root = Path(out_root) / "conf_scan"
    scan_root.mkdir(parents=True, exist_ok=True)
    metric_label = "mAP@0.5" if select_metric == "map50" else "F1"

    scan_rows = []
    best_conf_by_model = {}
    best_rows = []

    for model_name, model_path in model_specs:
        print(f"[ConfScan] model={model_name} values={conf_values}")
        model_scan_items = []

        for conf in conf_values:
            prediction_run = run_predictions_for_model(
                model_path,
                records,
                conf,
                input_size,
                core,
                input_format=input_format,
                letterbox_color=letterbox_color,
            )
            summary = build_model_summary(records, prediction_run)
            model_scan_items.append(summary)
            scan_rows.append(
                [
                    summary["model_name"],
                    f"{summary['conf']:.6f}",
                    f4(summary["metrics"]["precision"]),
                    f4(summary["metrics"]["recall"]),
                    f4(summary["metrics"]["f1"]),
                    f4(summary["metrics"]["map50"]),
                    pct(summary["empty_rate"]),
                    f2(summary["mean_det_per_image"]),
                    f2(summary["latency_ms"]["infer_avg"]),
                    f2(summary["latency_ms"]["post_avg"]),
                    f2(summary["latency_ms"]["fps_e2e"]),
                ]
            )

        best_summary = max(model_scan_items, key=lambda item: conf_selection_key(item, select_metric))
        best_conf_by_model[model_name] = best_summary["conf"]
        best_rows.append(
            [
                best_summary["model_name"],
                f"{best_summary['conf']:.6f}",
                select_metric,
                f4(best_summary["metrics"]["precision"]),
                f4(best_summary["metrics"]["recall"]),
                f4(best_summary["metrics"]["f1"]),
                f4(best_summary["metrics"]["map50"]),
                pct(best_summary["empty_rate"]),
                f2(best_summary["mean_det_per_image"]),
                f2(best_summary["latency_ms"]["infer_avg"]),
                f2(best_summary["latency_ms"]["post_avg"]),
                f2(best_summary["latency_ms"]["fps_e2e"]),
            ]
        )
        print(
            f"[ConfScan] selected model={model_name} conf={best_summary['conf']:.6f} "
            f"map50={best_summary['metrics']['map50']:.4f} "
            f"f1={best_summary['metrics']['f1']:.4f} "
            f"fps={best_summary['latency_ms']['fps_e2e']:.2f}"
        )

    scan_headers = [
        "模型名",
        "conf",
        "Precision",
        "Recall",
        "F1",
        "mAP@0.5",
        "空检率",
        "平均det",
        "平均infer(ms)",
        "平均post(ms)",
        "FPS",
    ]
    best_headers = [
        "模型名",
        "最佳conf",
        "选择依据",
        "Precision",
        "Recall",
        "F1",
        "mAP@0.5",
        "空检率",
        "平均det",
        "平均infer(ms)",
        "平均post(ms)",
        "FPS",
    ]

    write_csv(scan_root / "conf_scan_rows.csv", scan_headers, scan_rows)
    (scan_root / "conf_scan_rows.md").write_text(markdown_table(scan_headers, scan_rows) + "\n", encoding="utf-8")
    write_csv(scan_root / "best_conf_table.csv", best_headers, best_rows)
    (scan_root / "best_conf_table.md").write_text(markdown_table(best_headers, best_rows) + "\n", encoding="utf-8")

    with open(scan_root / "best_conf_by_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "select_metric": select_metric,
                "conf_values": conf_values,
                "best_conf_by_model": best_conf_by_model,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    report_lines = [
        "# Confidence Scan Report",
        "",
        f"Selection metric: `{select_metric}`",
        f"Scanned confidence thresholds: {', '.join(f'{v:.6f}' for v in conf_values)}",
        f"The best threshold for each model was selected by ranking {metric_label} first, then F1, Recall, Precision, lower empty rate, and finally higher confidence threshold.",
        "",
        "## Best Conf Table",
        markdown_table(best_headers, best_rows),
        "",
        "## All Scan Rows",
        markdown_table(scan_headers, scan_rows),
        "",
    ]
    (scan_root / "conf_scan_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return best_conf_by_model


def make_tables(model_summaries):
    table1_headers = ["模型名", "Precision", "Recall", "F1", "mAP@0.5"]
    table1_rows = []
    table2_headers = ["模型名", "平均infer(ms)", "平均post(ms)", "FPS"]
    table2_rows = []
    table3_headers = ["模型名", "空检率", "平均det", "mode分布(flat/dfl/none)"]
    table3_rows = []
    per_class_headers = ["模型名", "类别", "Precision", "Recall", "F1", "AP@0.5", "GT数", "Pred数"]
    per_class_rows = []

    for summary in model_summaries:
        m = summary["metrics"]
        lat = summary["latency_ms"]
        mode_counts = summary["mode_counts"]

        table1_rows.append(
            [
                summary["model_name"],
                f4(m["precision"]),
                f4(m["recall"]),
                f4(m["f1"]),
                f4(m["map50"]),
            ]
        )
        table2_rows.append(
            [
                summary["model_name"],
                f2(lat["infer_avg"]),
                f2(lat["post_avg"]),
                f2(lat["fps_e2e"]),
            ]
        )
        table3_rows.append(
            [
                summary["model_name"],
                pct(summary["empty_rate"]),
                f2(summary["mean_det_per_image"]),
                f"{mode_counts['flat']}/{mode_counts['dfl_concat']}/{mode_counts['none']}",
            ]
        )

        for cls_metric in m["per_class"]:
            per_class_rows.append(
                [
                    summary["model_name"],
                    cls_metric["class_name"],
                    f4(cls_metric["precision"]),
                    f4(cls_metric["recall"]),
                    f4(cls_metric["f1"]),
                    f4(cls_metric["ap50"]),
                    cls_metric["gt_count"],
                    cls_metric["pred_count"],
                ]
            )

    return {
        "table1": {"headers": table1_headers, "rows": table1_rows},
        "table2": {"headers": table2_headers, "rows": table2_rows},
        "table3": {"headers": table3_headers, "rows": table3_rows},
        "per_class": {"headers": per_class_headers, "rows": per_class_rows},
    }


def validate_stage(stage_name, model_summaries):
    for summary in model_summaries:
        metrics = summary["metrics"]
        for key in ("precision", "recall", "f1", "map50"):
            value = metrics[key]
            if not (0.0 <= value <= 1.0):
                raise RuntimeError(f"{stage_name} invalid {key} for {summary['model_name']}: {value}")
        if summary["empty_count"] > summary["num_images"]:
            raise RuntimeError(f"{stage_name} invalid empty count for {summary['model_name']}")


def write_experiment_outputs(out_dir, experiment_name, experiment_desc, dataset_meta, model_summaries):
    exp_dir = Path(out_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    tables = make_tables(model_summaries)

    for table_name, table in tables.items():
        md = markdown_table(table["headers"], table["rows"])
        (exp_dir / f"{table_name}.md").write_text(md + "\n", encoding="utf-8")
        write_csv(exp_dir / f"{table_name}.csv", table["headers"], table["rows"])

    prediction_export = {
        "experiment": experiment_name,
        "description": experiment_desc,
        "dataset": dataset_meta,
        "models": [],
    }
    for summary in model_summaries:
        prediction_export["models"].append(
            {
                "model_name": summary["model_name"],
                "model_path": summary["model_path"],
                "conf": summary["conf"],
                "conf_source": summary["conf_source"],
                "norm_mode": summary["norm_mode"],
                "bbox_expand_scale": summary.get("bbox_expand_scale", 1.0),
                "bbox_expand_pad": summary.get("bbox_expand_pad", 0.0),
                "input_format": summary.get("input_format", "rgb"),
                "letterbox_color": int(summary.get("letterbox_color", 114)),
                "num_images": summary["num_images"],
                "entries": summary["entries"],
            }
        )

    with open(exp_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(prediction_export, f, ensure_ascii=False, indent=2)

    compact_summary = {
        "experiment": experiment_name,
        "description": experiment_desc,
        "dataset": dataset_meta,
        "models": [],
    }
    for summary in model_summaries:
        compact_summary["models"].append(
            {
                "model_name": summary["model_name"],
                "model_path": summary["model_path"],
                "conf": summary["conf"],
                "conf_source": summary["conf_source"],
                "norm_mode": summary["norm_mode"],
                "bbox_expand_scale": summary.get("bbox_expand_scale", 1.0),
                "bbox_expand_pad": summary.get("bbox_expand_pad", 0.0),
                "input_format": summary.get("input_format", "rgb"),
                "letterbox_color": int(summary.get("letterbox_color", 114)),
                "num_images": summary["num_images"],
                "empty_count": summary["empty_count"],
                "empty_rate": summary["empty_rate"],
                "mean_det_per_image": summary["mean_det_per_image"],
                "mode_counts": summary["mode_counts"],
                "latency_ms": summary["latency_ms"],
                "metrics": summary["metrics"],
            }
        )
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(compact_summary, f, ensure_ascii=False, indent=2)

    best_map_model = max(model_summaries, key=lambda x: x["metrics"]["map50"])
    best_f1_model = max(model_summaries, key=lambda x: x["metrics"]["f1"])
    fastest_model = min(model_summaries, key=lambda x: x["latency_ms"]["infer_avg"])

    report_lines = [
        f"## {experiment_name}",
        "",
        "### Experimental Setup",
        f"The experiment was conducted on {dataset_meta['num_images']} test images under the '{experiment_name}' protocol.",
        f"Annotations were parsed as {dataset_meta['label_format']} because {dataset_meta['label_rationale']}.",
        "All predictions were produced by the existing `port_defect_rknn_infer_from_zooV2.py` inference pipeline without modifying the DFL or flat-head post-processing logic.",
        f"Input preprocessing used `input_format={dataset_meta.get('input_format', 'rgb')}` and `letterbox_color={int(dataset_meta.get('letterbox_color', 114))}`.",
        "For all models, true positives and false positives were determined by one-to-one matching with IoU=0.5.",
        "",
        "### Metric Definition",
        "Precision was computed as TP / (TP + FP), Recall as TP / (TP + FN), and F1-score as the harmonic mean of Precision and Recall.",
        "mAP@0.5 was calculated as the mean of per-class AP values obtained from score-ranked precision-recall curves at IoU=0.5.",
        "Runtime FPS was derived from the mean end-to-end latency defined as preprocess + infer + postprocess.",
        "",
        "### Result Analysis",
        f"Under this protocol, the highest mAP@0.5 was achieved by `{best_map_model['model_name']}` ({best_map_model['metrics']['map50']:.4f}).",
        f"The highest F1-score was achieved by `{best_f1_model['model_name']}` ({best_f1_model['metrics']['f1']:.4f}).",
        f"The lowest average inference latency was observed for `{fastest_model['model_name']}` ({fastest_model['latency_ms']['infer_avg']:.2f} ms).",
        "The stability table should be interpreted together with the detection table, because a low empty-detection rate does not necessarily imply better precision when the confidence working point is model-specific.",
        "",
        "### Markdown Tables",
        "",
        "#### Table 1: Detection Performance",
        markdown_table(tables["table1"]["headers"], tables["table1"]["rows"]),
        "",
        "#### Table 2: Runtime Performance",
        markdown_table(tables["table2"]["headers"], tables["table2"]["rows"]),
        "",
        "#### Table 3: Stability Analysis",
        markdown_table(tables["table3"]["headers"], tables["table3"]["rows"]),
        "",
        "#### Per-Class Metrics",
        markdown_table(tables["per_class"]["headers"], tables["per_class"]["rows"]),
        "",
    ]
    (exp_dir / "paper_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\n[Summary][{experiment_name}] Table 1")
    print(markdown_table(tables["table1"]["headers"], tables["table1"]["rows"]))
    print(f"\n[Summary][{experiment_name}] Table 2")
    print(markdown_table(tables["table2"]["headers"], tables["table2"]["rows"]))
    print(f"\n[Summary][{experiment_name}] Table 3")
    print(markdown_table(tables["table3"]["headers"], tables["table3"]["rows"]))

    return compact_summary


def run_experiment(
    records,
    experiment_name,
    experiment_cfg,
    model_specs,
    input_size,
    core,
    out_root,
    dataset_meta,
    input_format="rgb",
    letterbox_color=114,
):
    model_summaries = []
    for model_name, model_path in model_specs:
        conf = experiment_cfg["conf_by_model"][model_name]
        prediction_run = run_predictions_for_model(
            model_path,
            records,
            conf,
            input_size,
            core,
            input_format=input_format,
            letterbox_color=letterbox_color,
        )
        model_summaries.append(build_model_summary(records, prediction_run))
    stage_meta = dict(dataset_meta)
    stage_meta["num_images"] = len(records)
    return write_experiment_outputs(
        out_dir=out_root,
        experiment_name=experiment_name,
        experiment_desc=experiment_cfg["description"],
        dataset_meta=stage_meta,
        model_summaries=model_summaries,
    )


def main():
    parser = argparse.ArgumentParser(description="Paper-oriented RKNN evaluation for port defect detection")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=MODELS_DIR,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=DATA_IMAGES_DIR,
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default=DATA_LABELS_DIR,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(EVAL_DIR)),
    )
    parser.add_argument("--input_size", type=str, default="640x640")
    parser.add_argument("--core", type=str, default="0")
    parser.add_argument(
        "--input_format",
        type=str,
        default="rgb",
        choices=["rgb", "bgr"],
        help="Input color order used before RKNN inference.",
    )
    parser.add_argument(
        "--pad_color",
        "--letterbox_color",
        dest="letterbox_color",
        type=int,
        default=114,
        help="Letterbox padding color. Typical values are 114 and 0.",
    )
    parser.add_argument("--validation_images", type=int, default=10)
    parser.add_argument("--skip_full", action="store_true")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument(
        "--auto_best_conf_scan",
        action="store_true",
        help="Automatically scan confidence thresholds for each model and use the selected best conf in best_working_point.",
    )
    parser.add_argument(
        "--conf_scan_values",
        type=str,
        default=DEFAULT_CONF_SCAN_VALUES,
        help="Comma-separated confidence thresholds used by --auto_best_conf_scan.",
    )
    parser.add_argument(
        "--conf_select_metric",
        type=str,
        default="map50",
        choices=["map50", "f1"],
        help="Primary metric used to select the best confidence threshold.",
    )
    parser.add_argument(
        "--diagnose_alignment_only",
        action="store_true",
        help="Run label/prediction alignment checks and 10-image GT+prediction visualizations only.",
    )
    parser.add_argument(
        "--diagnose_images",
        type=int,
        default=10,
        help="Number of images used for alignment diagnosis visualizations.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    in_w, in_h = map(int, args.input_size.lower().split("x"))
    input_size = (in_w, in_h)

    label_info = detect_label_format(args.label_path, args.image_dir)
    print(f"[Eval] Label format: {label_info['format']}")
    print(f"[Eval] Label detection rationale: {label_info['rationale']}")

    full_records = build_dataset_records(args.image_dir, label_info, limit=args.max_images)
    print(f"[Eval] Dataset images ready: {len(full_records)}")
    gt_per_class = {name: 0 for name in infer_core.CLASSES}
    total_gt_boxes = 0
    for record in full_records:
        total_gt_boxes += int(len(record["gt_classes"]))
        for cid in record["gt_classes"]:
            gt_per_class[infer_core.CLASSES[int(cid)]] += 1

    model_specs = resolve_model_specs(args.model_dir)
    experiments = build_experiments([name for name, _ in model_specs])
    print(f"[Eval] Models selected: {[name for name, _ in model_specs]}")

    dataset_meta = {
        "image_dir": args.image_dir,
        "label_path": args.label_path,
        "label_format": label_info["format"],
        "label_rationale": label_info["rationale"],
        "num_images": len(full_records),
        "num_gt_boxes": total_gt_boxes,
        "gt_per_class": gt_per_class,
        "classes": list(infer_core.CLASSES),
        "input_size": {"width": in_w, "height": in_h},
        "input_format": args.input_format,
        "letterbox_color": int(args.letterbox_color),
    }
    with open(out_root / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, ensure_ascii=False, indent=2)

    if args.auto_best_conf_scan:
        conf_values = parse_conf_scan_values(args.conf_scan_values)
        print(f"[Eval] Auto conf scan enabled: values={conf_values} metric={args.conf_select_metric}")
        best_conf_by_model = scan_confidence_thresholds(
            records=full_records,
            model_specs=model_specs,
            conf_values=conf_values,
            input_size=input_size,
            core=args.core,
            out_root=out_root,
            dataset_meta=dataset_meta,
            select_metric=args.conf_select_metric,
            input_format=args.input_format,
            letterbox_color=args.letterbox_color,
        )
        experiments["best_working_point"]["conf_by_model"] = best_conf_by_model
        experiments["best_working_point"]["description"] = (
            "Best working point selected automatically by confidence scan on the evaluation dataset. "
            f"Primary selection metric: {args.conf_select_metric}."
        )

    if args.diagnose_alignment_only:
        diagnostic_records = full_records[: min(args.diagnose_images, len(full_records))]
        print(f"[Eval] Alignment diagnosis stage on first {len(diagnostic_records)} images")
        report_path = run_alignment_diagnosis(
            all_records=full_records,
            diagnostic_records=diagnostic_records,
            label_info=label_info,
            model_specs=model_specs,
            input_size=input_size,
            core=args.core,
            out_root=out_root,
            image_dir=args.image_dir,
            experiments=experiments,
            input_format=args.input_format,
            letterbox_color=args.letterbox_color,
        )
        print(f"[Eval] Alignment diagnosis finished: {report_path}")
        return

    validation_records = full_records[: min(args.validation_images, len(full_records))]
    print(f"[Eval] Validation stage on first {len(validation_records)} images")
    validation_summaries = []
    for experiment_name, experiment_cfg in experiments.items():
        summary = run_experiment(
            records=validation_records,
            experiment_name=f"validation_{len(validation_records)}_{experiment_name}",
            experiment_cfg=experiment_cfg,
            model_specs=model_specs,
            input_size=input_size,
            core=args.core,
            out_root=out_root,
            dataset_meta=dataset_meta,
            input_format=args.input_format,
            letterbox_color=args.letterbox_color,
        )
        validation_summaries.append(summary)
    validate_stage("validation", [model for exp in validation_summaries for model in exp["models"]])
    print("[Eval] Validation stage passed.")

    if args.skip_full:
        print("[Eval] --skip_full enabled, stopping after validation.")
        return

    print(f"[Eval] Full evaluation stage on {len(full_records)} images")
    all_full_summaries = []
    for experiment_name, experiment_cfg in experiments.items():
        summary = run_experiment(
            records=full_records,
            experiment_name=experiment_name,
            experiment_cfg=experiment_cfg,
            model_specs=model_specs,
            input_size=input_size,
            core=args.core,
            out_root=out_root,
            dataset_meta=dataset_meta,
            input_format=args.input_format,
            letterbox_color=args.letterbox_color,
        )
        all_full_summaries.append(summary)

    with open(out_root / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_meta,
                "experiments": all_full_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    combined_lines = [
        "# RKNN Paper Evaluation Report",
        "",
        "## Dataset",
        f"The evaluation dataset contains {dataset_meta['num_images']} cropped test images.",
        f"Annotation format: {dataset_meta['label_format']}.",
        f"Detection rationale: {dataset_meta['label_rationale']}.",
        "",
        "## Protocols",
        f"1. `unified_conf_0.10`: {experiments['unified_conf_0.10']['description']}",
        f"2. `best_working_point`: {experiments['best_working_point']['description']}",
        "",
    ]
    for exp in all_full_summaries:
        combined_lines.append(f"## {exp['experiment']}")
        combined_lines.append("")
        combined_lines.append(f"Description: {exp['description']}")
        combined_lines.append("")
        exp_dir = out_root / exp["experiment"]
        combined_lines.append((exp_dir / "table1.md").read_text(encoding="utf-8").strip())
        combined_lines.append("")
        combined_lines.append((exp_dir / "table2.md").read_text(encoding="utf-8").strip())
        combined_lines.append("")
        combined_lines.append((exp_dir / "table3.md").read_text(encoding="utf-8").strip())
        combined_lines.append("")

    (out_root / "paper_ready_summary.md").write_text("\n".join(combined_lines) + "\n", encoding="utf-8")
    print(f"[Eval] Finished. Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
