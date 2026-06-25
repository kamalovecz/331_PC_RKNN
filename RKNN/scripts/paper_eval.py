#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-dataset evaluation and visualization script for 0422 RKNN models.

This script is intentionally separate from edge_benchmark_paper.py:
- edge_benchmark_paper.py focuses on FPS / power / latency reporting
- this script focuses on detection-vs-GT validation for the nine 0422 RKNN models

Outputs per model:
1) per_image_metrics.csv
2) model_summary.json
3) vis/*.jpg with GT + prediction difference overlay
4) predictions.json with per-image RKNN detections for downstream AP recomputation

Outputs for the whole run:
1) global_summary.csv
2) global_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from rknnlite.api import RKNNLite


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import MODELS_DIR, DATA_IMAGES_DIR, DATA_LABELS_DIR, PROJECT_ROOT
import infer as base


DEFAULT_MODEL_NAMES = [
    "B3_Lite",
    "NEU_Pretrain_pt_Full_model_Port_defect",
    "Port_RuleLoss_Full_model",
    "PROCESSED_A_GFPN_REPHFE_s42",
    "PROCESSED_A_GFPN_s42",
    "PROCESSED_FULL_MODEL_s42",
    "PROCESSED_REPHFE_s42",
    "PROCESSED_SADH_s42",
    "PROCESSED_YOLOV8N_BASELINE_s42",
]

GT_COLOR = (255, 0, 0)
TP_COLOR = (0, 180, 0)
FP_COLOR = (0, 0, 255)
DUP_COLOR = (0, 165, 255)
FN_COLOR = (0, 255, 255)
TEXT_COLOR = (20, 20, 20)
AP_IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 0422 RKNN models on full test set with GT visualization.")
    parser.add_argument(
        "--model-root",
        default=MODELS_DIR,
        help="Directory containing *_640_fp.rknn models.",
    )
    parser.add_argument(
        "--img-dir",
        default=DATA_IMAGES_DIR,
        help="Image directory.",
    )
    parser.add_argument(
        "--label-dir",
        default=DATA_LABELS_DIR,
        help="YOLO label directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(PROJECT_ROOT) / "0422_Paper" / "0422_Paper_remote"),
        help="Output directory on board.",
    )
    parser.add_argument("--model-names", nargs="+", default=DEFAULT_MODEL_NAMES)
    parser.add_argument("--image-names", nargs="*", default=None, help="Optional image subset for smoke tests.")
    parser.add_argument("--core", default="0", help="NPU core mask: 0/1/2/all.")
    parser.add_argument("--iou-match", type=float, default=0.50, help="IoU threshold for TP/FP/FN matching.")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold used in RKNN post-process.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--pad-color", type=int, default=114)
    parser.add_argument("--min-wh", type=float, default=10.0)
    parser.add_argument("--min-area", type=float, default=120.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=8.0)
    parser.add_argument("--merge-gap", type=int, default=12)
    parser.add_argument("--edge-margin", type=int, default=4)
    parser.add_argument("--min-edge-box", type=int, default=20)
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence override.")
    parser.add_argument("--decode-mode", default="auto", choices=["auto", "dfl", "flat"])
    parser.add_argument("--save-vis", action="store_true", default=True, help="Save visualization images.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"cv2.imread failed: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    encoded.tofile(str(path))


def load_yolo_labels(label_path: Path, image_width: int, image_height: int) -> List[dict]:
    if not label_path.exists():
        return []

    boxes: List[dict] = []
    for line_index, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1]) * image_width
            yc = float(parts[2]) * image_height
            bw = float(parts[3]) * image_width
            bh = float(parts[4]) * image_height
        except ValueError as exc:
            raise ValueError(f"Invalid label line: {label_path}:{line_index}") from exc

        x1 = max(0.0, xc - bw / 2.0)
        y1 = max(0.0, yc - bh / 2.0)
        x2 = min(float(image_width), xc + bw / 2.0)
        y2 = min(float(image_height), yc + bh / 2.0)
        class_name = base.CLASSES[class_id] if 0 <= class_id < len(base.CLASSES) else f"class_{class_id}"
        boxes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return boxes


def convert_predictions(boxes: np.ndarray, class_ids: np.ndarray, scores: np.ndarray) -> List[dict]:
    if boxes is None or len(boxes) == 0:
        return []
    rows: List[dict] = []
    for box, cid, score in zip(boxes, class_ids, scores):
        cid_int = int(cid)
        rows.append(
            {
                "class_id": cid_int,
                "class_name": base.CLASSES[cid_int] if 0 <= cid_int < len(base.CLASSES) else f"class_{cid_int}",
                "conf": float(score),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            }
        )
    rows.sort(key=lambda item: item["conf"], reverse=True)
    return rows


def compute_iou(box_a: dict, box_b: dict) -> float:
    x1 = max(float(box_a["x1"]), float(box_b["x1"]))
    y1 = max(float(box_a["y1"]), float(box_b["y1"]))
    x2 = min(float(box_a["x2"]), float(box_b["x2"]))
    y2 = min(float(box_a["y2"]), float(box_b["y2"]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a["x2"]) - float(box_a["x1"])) * max(0.0, float(box_a["y2"]) - float(box_a["y1"]))
    area_b = max(0.0, float(box_b["x2"]) - float(box_b["x1"])) * max(0.0, float(box_b["y2"]) - float(box_b["y1"]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def match_predictions_to_ground_truth(
    gt_boxes: Sequence[dict],
    pred_boxes: Sequence[dict],
    iou_match: float,
) -> Tuple[List[dict], List[dict]]:
    gt_states: List[dict] = []
    gt_matched = [False] * len(gt_boxes)

    for gt_index, gt in enumerate(gt_boxes):
        gt_states.append({**gt, "gt_index": gt_index, "status": "missed_fn", "matched_pred_index": None, "best_iou": 0.0})

    pred_states: List[dict] = []
    for pred_index, pred in enumerate(sorted(pred_boxes, key=lambda item: float(item.get("conf", 0.0)), reverse=True)):
        best_unmatched_index = None
        best_unmatched_iou = 0.0
        best_any_index = None
        best_any_iou = 0.0

        for gt_index, gt in enumerate(gt_boxes):
            if int(pred["class_id"]) != int(gt["class_id"]):
                continue
            iou = compute_iou(pred, gt)
            if iou > best_any_iou:
                best_any_iou = iou
                best_any_index = gt_index
            if gt_matched[gt_index]:
                continue
            if iou > best_unmatched_iou:
                best_unmatched_iou = iou
                best_unmatched_index = gt_index

        pred_state = {**pred, "pred_index": pred_index, "matched_gt_index": None, "iou_with_gt": 0.0, "match_status": "unmatched_fp"}
        if best_unmatched_index is not None and best_unmatched_iou >= iou_match:
            gt_matched[best_unmatched_index] = True
            gt_states[best_unmatched_index]["status"] = "matched_tp"
            gt_states[best_unmatched_index]["matched_pred_index"] = pred_index
            gt_states[best_unmatched_index]["best_iou"] = best_unmatched_iou
            pred_state["matched_gt_index"] = best_unmatched_index
            pred_state["iou_with_gt"] = best_unmatched_iou
            pred_state["match_status"] = "matched_tp"
        elif best_any_index is not None and best_any_iou >= iou_match:
            pred_state["matched_gt_index"] = best_any_index
            pred_state["iou_with_gt"] = best_any_iou
            pred_state["match_status"] = "duplicate_fp"
        pred_states.append(pred_state)

    return gt_states, pred_states


def compute_image_metrics(gt_states: Sequence[dict], pred_states: Sequence[dict]) -> dict:
    tp = sum(1 for item in pred_states if item["match_status"] == "matched_tp")
    fp = sum(1 for item in pred_states if item["match_status"] != "matched_tp")
    fn = sum(1 for item in gt_states if item["status"] != "matched_tp")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "gt_count": len(gt_states),
        "pred_count": len(pred_states),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0 or precisions.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_class_ap(
    gt_by_image: Dict[str, List[dict]],
    predictions: Sequence[dict],
    iou_threshold: float,
) -> Tuple[float, int, int]:
    gt_used: Dict[str, List[bool]] = {}
    gt_total = 0
    for image_name, gt_boxes in gt_by_image.items():
        gt_used[image_name] = [False] * len(gt_boxes)
        gt_total += len(gt_boxes)

    if gt_total == 0:
        return 0.0, 0, 0

    ranked_preds = sorted(predictions, key=lambda item: float(item.get("conf", 0.0)), reverse=True)
    if not ranked_preds:
        return 0.0, gt_total, 0

    tp = np.zeros(len(ranked_preds), dtype=np.float32)
    fp = np.zeros(len(ranked_preds), dtype=np.float32)

    for pred_index, pred in enumerate(ranked_preds):
        image_name = str(pred["image_name"])
        gt_boxes = gt_by_image.get(image_name, [])
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index >= 0 and best_iou >= float(iou_threshold) and not gt_used[image_name][best_gt_index]:
            gt_used[image_name][best_gt_index] = True
            tp[pred_index] = 1.0
        else:
            fp[pred_index] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(float(gt_total), 1e-12)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    ap = compute_ap_from_pr(recalls, precisions)
    pred_total = len(ranked_preds)
    return ap, gt_total, pred_total


def compute_detection_map(
    class_names: Sequence[str],
    gt_by_class: Dict[int, Dict[str, List[dict]]],
    preds_by_class: Dict[int, List[dict]],
    iou_thresholds: np.ndarray,
) -> Dict[str, object]:
    per_class_rows: List[dict] = []
    ap50_values: List[float] = []
    ap75_values: List[float] = []
    ap5095_values: List[float] = []

    iou_thresholds_list = [float(x) for x in iou_thresholds.tolist()]
    iou_75_index = min(range(len(iou_thresholds_list)), key=lambda idx: abs(iou_thresholds_list[idx] - 0.75))

    for class_id, class_name in enumerate(class_names):
        gt_by_image = gt_by_class.get(class_id, {})
        pred_list = preds_by_class.get(class_id, [])
        ap_per_threshold: List[float] = []
        gt_count = sum(len(v) for v in gt_by_image.values())
        pred_count = len(pred_list)
        for iou_thr in iou_thresholds_list:
            ap, _, _ = compute_class_ap(gt_by_image, pred_list, iou_thr)
            ap_per_threshold.append(ap)

        ap50 = ap_per_threshold[0] if ap_per_threshold else 0.0
        ap75 = ap_per_threshold[iou_75_index] if ap_per_threshold else 0.0
        ap5095 = float(np.mean(ap_per_threshold)) if ap_per_threshold else 0.0

        if gt_count > 0:
            ap50_values.append(ap50)
            ap75_values.append(ap75)
            ap5095_values.append(ap5095)

        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt": gt_count,
                "pred": pred_count,
                "ap50": ap50,
                "ap75": ap75,
                "ap50_95": ap5095,
                "ap_by_iou": {
                    f"{iou_thr:.2f}": ap_value
                    for iou_thr, ap_value in zip(iou_thresholds_list, ap_per_threshold)
                },
            }
        )

    return {
        "map50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "map75": float(np.mean(ap75_values)) if ap75_values else 0.0,
        "map50_95": float(np.mean(ap5095_values)) if ap5095_values else 0.0,
        "iou_thresholds": iou_thresholds_list,
        "per_class": per_class_rows,
    }


def clip_box(box: dict, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(width - 1, round(float(box["x1"])))))
    y1 = int(max(0, min(height - 1, round(float(box["y1"])))))
    x2 = int(max(0, min(width - 1, round(float(box["x2"])))))
    y2 = int(max(0, min(height - 1, round(float(box["y2"])))))
    return x1, y1, x2, y2


def draw_box(image: np.ndarray, box: dict, color: Tuple[int, int, int], label: str, thickness: int) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1, th + 8)
        cv2.rectangle(image, (x1, ty - th - 8), (x1 + tw + 8, ty + baseline - 4), color, -1)
        cv2.putText(image, label, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def render_diff_overlay(image: np.ndarray, gt_states: Sequence[dict], pred_states: Sequence[dict], metrics: dict, image_name: str, model_name: str) -> np.ndarray:
    canvas = image.copy()

    # Draw all GT first so positional bias remains visible under TP/FP overlays.
    for gt in gt_states:
        gt_label = f"GT:{gt['class_name']}"
        draw_box(canvas, gt, GT_COLOR, gt_label, thickness=1)

    for pred in pred_states:
        if pred["match_status"] == "matched_tp":
            color = TP_COLOR
            prefix = "TP"
        elif pred["match_status"] == "duplicate_fp":
            color = DUP_COLOR
            prefix = "Dup"
        else:
            color = FP_COLOR
            prefix = "FP"
        label = f"{prefix}:{pred['class_name']} {pred['conf']:.2f}"
        draw_box(canvas, pred, color, label, thickness=2)

    for gt in gt_states:
        if gt["status"] != "matched_tp":
            draw_box(canvas, gt, FN_COLOR, f"FN:{gt['class_name']}", thickness=2)

    header = (
        f"{model_name} | {image_name} | "
        f"GT={metrics['gt_count']} Pred={metrics['pred_count']} "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}"
    )
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 1400), 34), (255, 255, 255), -1)
    cv2.putText(canvas, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, TEXT_COLOR, 2, cv2.LINE_AA)
    return canvas


def init_model_runtime(model_path: Path, core: str) -> RKNNLite:
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: ret={ret} model={model_path}")
    ret = rknn.init_runtime(core_mask=base.resolve_core_mask(core))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"init_runtime failed: ret={ret} model={model_path}")
    return rknn


def collect_image_names(img_dir: Path, requested_names: Sequence[str] | None) -> List[str]:
    if requested_names:
        normalized_requested: List[str] = []
        for item in requested_names:
            for piece in str(item).split(","):
                name = piece.strip()
                if name:
                    normalized_requested.append(name)
        names = [name for name in normalized_requested if (img_dir / name).is_file()]
    else:
        names = sorted(path.name for path in img_dir.iterdir() if path.is_file() and base.img_check(path.name))
    if not names:
        raise RuntimeError(f"No valid images found under {img_dir}")
    return names


def run_single_model(args: argparse.Namespace, model_name: str, image_names: Sequence[str], global_rows: List[dict]) -> dict:
    model_root = Path(args.model_root)
    model_path = model_root / f"{model_name}_640_fp.rknn"
    if not model_path.exists():
        raise FileNotFoundError(f"Model missing: {model_path}")

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_root = Path(args.out_dir) / model_name
    vis_dir = out_root / "vis"
    ensure_dir(vis_dir)

    profile = base.resolve_model_runtime_profile(str(model_path), args.conf, "auto", args.debug)
    actual_conf = profile["conf"]
    actual_norm_mode = profile["norm_mode"]
    actual_input_format = profile.get("input_format", "rgb")
    actual_input_layout = profile.get("input_layout", "auto")
    bbox_expand_scale = profile.get("bbox_expand_scale", 1.0)
    bbox_expand_pad = profile.get("bbox_expand_pad", 0.0)
    postprocess_variant = profile.get("postprocess_variant", "legacy")
    apply_edge_filter = profile.get("apply_edge_filter", True)

    rknn = init_model_runtime(model_path, args.core)
    output_details = base.try_get_output_details(rknn)

    model_rows: List[dict] = []
    class_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
    gt_by_class: Dict[int, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    preds_by_class: Dict[int, List[dict]] = defaultdict(list)
    prediction_export_rows: List[dict] = []
    started = time.perf_counter()

    try:
        dummy = base.make_dummy_input(
            rknn,
            (640, 640),
            input_format=actual_input_format,
            input_layout=actual_input_layout,
            norm_mode=actual_norm_mode,
            letterbox_color=args.pad_color,
        )
        for _ in range(2):
            _ = rknn.inference(inputs=[dummy])
    except Exception:
        pass

    for index, image_name in enumerate(image_names, start=1):
        img_path = img_dir / image_name
        label_path = label_dir / f"{Path(image_name).stem}.txt"
        image_bgr = read_image(img_path)

        t0 = time.perf_counter()
        inp, ratio, pad = base.prepare_input(
            image_bgr,
            rknn,
            (640, 640),
            input_format=actual_input_format,
            input_layout=actual_input_layout,
            norm_mode=actual_norm_mode,
            letterbox_color=args.pad_color,
            debug=False,
        )
        t1 = time.perf_counter()
        outputs = rknn.inference(inputs=[inp])
        t2 = time.perf_counter()
        boxes, class_ids, scores, mode = base.post_process(
            outputs=outputs,
            output_details=output_details,
            input_size=(640, 640),
            num_classes=len(base.CLASSES),
            conf_thres=actual_conf,
            iou_thres=args.iou,
            max_det=args.max_det,
            decode_mode=args.decode_mode,
            min_wh=args.min_wh,
            min_area=args.min_area,
            max_aspect_ratio=args.max_aspect_ratio,
            merge_gap=args.merge_gap,
            postprocess_variant=postprocess_variant,
            model_name=profile["model_name"],
            flat_head_debug=False,
            conf_scan_values=None,
            flat_force_sigmoid=False,
            mode_diagnostic=False,
        )
        t3 = time.perf_counter()

        if boxes is not None and len(boxes) > 0:
            boxes = base.expand_boxes_xyxy(boxes, (640, 640), scale=bbox_expand_scale, pad=bbox_expand_pad)
            boxes0 = base.scale_boxes_to_original(boxes, ratio, pad, image_bgr.shape)
            if apply_edge_filter:
                boxes0, class_ids, scores = base.filter_edge_boxes(
                    boxes0,
                    class_ids,
                    scores,
                    image_bgr.shape,
                    edge_margin=args.edge_margin,
                    min_edge_box=args.min_edge_box,
                )
        else:
            boxes0 = None

        pred_boxes = convert_predictions(boxes0, class_ids, scores) if boxes0 is not None and len(boxes0) > 0 else []
        gt_boxes = load_yolo_labels(label_path, image_bgr.shape[1], image_bgr.shape[0])

        for gt_box in gt_boxes:
            gt_by_class[int(gt_box["class_id"])][image_name].append(gt_box)
        for pred_box in pred_boxes:
            pred_record = {
                "image_name": image_name,
                **pred_box,
            }
            preds_by_class[int(pred_box["class_id"])].append(pred_record)
            prediction_export_rows.append(pred_record)

        gt_states, pred_states = match_predictions_to_ground_truth(gt_boxes, pred_boxes, args.iou_match)
        metrics = compute_image_metrics(gt_states, pred_states)

        for gt in gt_states:
            class_totals[gt["class_name"]]["gt"] += 1
            if gt["status"] != "matched_tp":
                class_totals[gt["class_name"]]["fn"] += 1
        for pred in pred_states:
            if pred["match_status"] == "matched_tp":
                class_totals[pred["class_name"]]["tp"] += 1
            else:
                class_totals[pred["class_name"]]["fp"] += 1

        vis_path = vis_dir / image_name
        if args.save_vis:
            vis = render_diff_overlay(image_bgr, gt_states, pred_states, metrics, image_name, model_name)
            write_image(vis_path, vis)

        row = {
            "model": model_name,
            "image_name": image_name,
            "gt_count": metrics["gt_count"],
            "pred_count": metrics["pred_count"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "mode": mode,
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
            "vis_path": str(vis_path) if args.save_vis else "",
        }
        model_rows.append(row)
        global_rows.append(row)

        if index == 1 or index == len(image_names) or index % 50 == 0:
            print(
                f"[{model_name}] {index}/{len(image_names)} "
                f"{image_name} tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
                f"infer={(t2 - t1) * 1000.0:.2f}ms"
            )

    elapsed = time.perf_counter() - started
    total_tp = sum(row["tp"] for row in model_rows)
    total_fp = sum(row["fp"] for row in model_rows)
    total_fn = sum(row["fn"] for row in model_rows)
    total_gt = sum(row["gt_count"] for row in model_rows)
    total_pred = sum(row["pred_count"] for row in model_rows)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    ap_metrics = compute_detection_map(
        class_names=base.CLASSES,
        gt_by_class=gt_by_class,
        preds_by_class=preds_by_class,
        iou_thresholds=AP_IOU_THRESHOLDS,
    )

    csv_path = out_root / "per_image_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "model",
            "image_name",
            "gt_count",
            "pred_count",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "mode",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "vis_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_rows:
            writer.writerow(
                {
                    **row,
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "preprocess_ms": f"{row['preprocess_ms']:.4f}",
                    "inference_ms": f"{row['inference_ms']:.4f}",
                    "postprocess_ms": f"{row['postprocess_ms']:.4f}",
                }
            )

    predictions_json_path = out_root / "predictions.json"
    predictions_json_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "model_path": str(model_path),
                "class_names": list(base.CLASSES),
                "image_count": len(model_rows),
                "predictions": prediction_export_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    class_rows = []
    for class_name, info in sorted(class_totals.items()):
        cls_precision = info["tp"] / (info["tp"] + info["fp"]) if (info["tp"] + info["fp"]) > 0 else 0.0
        cls_recall = info["tp"] / (info["tp"] + info["fn"]) if (info["tp"] + info["fn"]) > 0 else 0.0
        ap_row = next((row for row in ap_metrics["per_class"] if row["class_name"] == class_name), None)
        class_rows.append(
            {
                "class_name": class_name,
                "gt": info["gt"],
                "tp": info["tp"],
                "fp": info["fp"],
                "fn": info["fn"],
                "precision": cls_precision,
                "recall": cls_recall,
                "ap50": ap_row["ap50"] if ap_row else 0.0,
                "ap75": ap_row["ap75"] if ap_row else 0.0,
                "ap50_95": ap_row["ap50_95"] if ap_row else 0.0,
            }
        )

    summary = {
        "model": model_name,
        "model_path": str(model_path),
        "image_count": len(model_rows),
        "elapsed_s": elapsed,
        "profile": {
            "conf": actual_conf,
            "input_format": actual_input_format,
            "input_layout": actual_input_layout,
            "norm_mode": actual_norm_mode,
            "postprocess_variant": postprocess_variant,
            "apply_edge_filter": apply_edge_filter,
            "bbox_expand_scale": bbox_expand_scale,
            "bbox_expand_pad": bbox_expand_pad,
            "notes": profile.get("notes", []),
        },
        "overall": {
            "gt_total": total_gt,
            "pred_total": total_pred,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "mean_inference_ms": float(np.mean([row["inference_ms"] for row in model_rows])) if model_rows else None,
            "map50": ap_metrics["map50"],
            "map75": ap_metrics["map75"],
            "map50_95": ap_metrics["map50_95"],
        },
        "per_class": class_rows,
        "per_image_csv": str(csv_path),
        "predictions_json": str(predictions_json_path),
        "ap_eval": {
            "iou_thresholds": ap_metrics["iou_thresholds"],
        },
    }
    (out_root / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    image_names = collect_image_names(img_dir, args.image_names)
    print(f"[Config] image_count={len(image_names)}")
    print(f"[Config] model_count={len(args.model_names)}")
    print(f"[Config] out_dir={out_dir}")

    global_rows: List[dict] = []
    summaries: List[dict] = []

    for model_name in args.model_names:
        print(f"\n[Model] {model_name}")
        summary = run_single_model(args, model_name, image_names, global_rows)
        summaries.append(summary)
        print(
            f"[ModelDone] {model_name} "
            f"P={summary['overall']['precision']:.4f} "
            f"R={summary['overall']['recall']:.4f} "
            f"TP={summary['overall']['tp']} FP={summary['overall']['fp']} FN={summary['overall']['fn']}"
        )

    global_csv = out_dir / "global_summary.csv"
    with global_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "model",
            "image_count",
            "gt_total",
            "pred_total",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "map50",
            "map75",
            "map50_95",
            "mean_inference_ms",
            "elapsed_s",
            "per_image_csv",
            "predictions_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            overall = summary["overall"]
            writer.writerow(
                {
                    "model": summary["model"],
                    "image_count": summary["image_count"],
                    "gt_total": overall["gt_total"],
                    "pred_total": overall["pred_total"],
                    "tp": overall["tp"],
                    "fp": overall["fp"],
                    "fn": overall["fn"],
                    "precision": f"{overall['precision']:.6f}",
                    "recall": f"{overall['recall']:.6f}",
                    "map50": f"{overall['map50']:.6f}",
                    "map75": f"{overall['map75']:.6f}",
                    "map50_95": f"{overall['map50_95']:.6f}",
                    "mean_inference_ms": f"{overall['mean_inference_ms']:.4f}" if overall["mean_inference_ms"] is not None else "",
                    "elapsed_s": f"{summary['elapsed_s']:.4f}",
                    "per_image_csv": summary["per_image_csv"],
                    "predictions_json": summary["predictions_json"],
                }
            )

    payload = {
        "config": {
            "model_root": str(args.model_root),
            "img_dir": str(args.img_dir),
            "label_dir": str(args.label_dir),
            "out_dir": str(args.out_dir),
            "image_count": len(image_names),
            "model_names": list(args.model_names),
            "iou_match": args.iou_match,
            "iou": args.iou,
            "conf_override": args.conf,
        },
        "summaries": summaries,
    }
    (out_dir / "global_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[Done]")
    print(f"  global_csv: {global_csv}")
    print(f"  global_json: {out_dir / 'global_summary.json'}")


if __name__ == "__main__":
    main()
