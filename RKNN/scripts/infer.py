#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
港机表面缺陷 RKNN 推理脚本 V2

当前版本针对如下 RKNN 输出结构做了稳定适配：
1. out[0] = (1, 4+nc, 8400) 或 (1, 8400, 4+nc) 的 flat head
2. out[1], out[2], out[3] = (1, 64+nc, H, W) 的 DFL 多尺度输出

根据你当前日志：
  out[0] shape=(1, 9, 8400)
  out[1] shape=(1, 69, 80, 80)
  out[2] shape=(1, 69, 40, 40)
  out[3] shape=(1, 69, 20, 20)

这版代码在 auto 模式下会优先使用 3 个 DFL 多尺度头进行解码，
flat head 只作为回退路径，不会优先拿来出结果。

额外增强：
1. 支持 per-tensor / per-channel 反量化
2. 增加最小框、最小面积、长宽比过滤
3. 增加边缘小框过滤
4. 增加同类别近邻框合并
5. 保留 FP16 对照功能
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from rknnlite.api import RKNNLite


from paths import (
    MODELS_DIR as FP_MODEL_DIR, DATA_IMAGES_DIR, DATA_LABELS_DIR,
    OUTPUTS_DIR, ALIGN_DUMP_DIR, LAYERWISE_DIAG_DIR, EVAL_DIR,
    DEFAULT_INPUT_SIZE, DEFAULT_CONF as DEFAULT_CONF_THRES, DEFAULT_IOU, DEFAULT_CORE,
    CLASSES, REG_MAX, REG_CH, PYTHON_BIN, PROJECT_ROOT
)

CLASSES = CLASSES  # re-export
DEFAULT_CONF_THRES = 0.35
FP_MODEL_DIR = FP_MODEL_DIR  # from paths.py
FP_MODEL_NAME_MAP = {
    "yolov8n_baseline.rknn": "yolov8n_baseline_fp.rknn",
    "yolov8n_port_RULELOSS.rknn": "yolov8n_port_RULELOSS_fp.rknn",
    "B3-Lite_V2.rknn": "B3-Lite_V2_fp.rknn",
    "B3-Llite-V3.rknn": "B3-Llite-V3_fp.rknn",
}

REG_MAX = 15
REG_CH = 4 * (REG_MAX + 1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def img_check(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp"]


def resolve_core_mask(core_str):
    c = str(core_str).lower()
    table = {
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
        "all": RKNNLite.NPU_CORE_0_1_2,
    }
    return table.get(c, RKNNLite.NPU_CORE_0)


def resolve_model_path(model_path, prefer_fp=False):
    resolved = os.path.abspath(model_path)
    if not prefer_fp:
        return resolved

    base = os.path.basename(resolved)
    if base.endswith("_fp.rknn") and os.path.exists(resolved):
        return resolved

    candidate_names = []
    mapped = FP_MODEL_NAME_MAP.get(base)
    if mapped:
        candidate_names.append(mapped)

    if base.endswith(".rknn"):
        candidate_names.append(base[:-5] + "_fp.rknn")

    candidate_names.append(base)

    seen = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        fp_path = os.path.join(FP_MODEL_DIR, name)
        if os.path.exists(fp_path):
            return fp_path

    raise FileNotFoundError(
        f"prefer_fp=True but no matching FP model found for '{base}' under '{FP_MODEL_DIR}'"
    )


def resolve_model_runtime_profile(model_path, conf, norm_mode, input_layout="auto", debug=False):
    model_name = os.path.basename(model_path)
    user_conf_override = conf is not None
    actual_conf = float(conf) if user_conf_override else DEFAULT_CONF_THRES
    actual_norm_mode = norm_mode
    actual_input_format = "rgb"
    actual_input_layout = input_layout
    bbox_expand_scale = 1.0
    bbox_expand_pad = 0.0
    postprocess_variant = "legacy"
    apply_edge_filter = True
    flat_head_debug = False
    mode_diagnostic = False
    conf_source = "user parameter" if user_conf_override else "global default"
    notes = []

    # Known issue: 0422 FP exports align on board only when RGB + NHWC + div255
    # New flat-head model (PT->ONNX with ultralytics, std=[255,255,255]):
    # RKNN already applies (x-0)/255 internally. Feed uint8 [0,255] directly.
    if model_name == "baseline_test0625.rknn":
        actual_input_format = "rgb"
        actual_input_layout = "nhwc"
        if actual_norm_mode == "auto":
            actual_norm_mode = "none"
        notes.append("feed uint8 [0,255]: RKNN std=[255,255,255] already normalizes")
    # are used together. Keep this override scoped to the new export naming rule
    # so earlier models preserve their existing defaults.
    if model_name.endswith("_640_fp.rknn"):
        actual_input_format = "rgb"
        actual_input_layout = "nhwc"
        if actual_norm_mode == "auto":
            actual_norm_mode = "div255"
        notes.append("0422 fp export uses RGB + NHWC + div255 by default")

    # Known issue: this FP model is sensitive to input preprocessing.
    # Keep uint8-style input by default and avoid div255 unless the user forces it.
    if model_name == "B3-Llite-V3_fp.rknn":
        if actual_norm_mode == "auto":
            actual_norm_mode = "none"
        notes.append("input preprocessing sensitive")

    # Known issue: these FP flat-head exports currently produce very low class scores.
    # Lower the default confidence threshold first so we can observe the logits/probability shape.
    if model_name == "yolov8n_baseline_fp.rknn":
        if not user_conf_override:
            actual_conf = 0.01
            conf_source = "model profile"
        flat_head_debug = bool(debug)
        notes.append("flat head scores are low")
    elif model_name == "yolov8n_port_RULELOSS_fp.rknn":
        if not user_conf_override:
            actual_conf = 0.001
            conf_source = "model profile"
        flat_head_debug = bool(debug)
        notes.append("flat head scores are low")

    # Known issue: this model still needs evidence collection before touching decode rules.
    # Keep behavior unchanged and only emit extra diagnostics when debugging.
    if model_name == "B3-Lite_V2.rknn":
        if not user_conf_override:
            actual_conf = 0.10
            conf_source = "model profile"
        mode_diagnostic = bool(debug)
        notes.append("mode diagnosis enabled")

    # Known issue on the 0417 export: predictions often land inside coarse GT regions as
    # narrow fragments. Apply a minimal model-specific bbox compensation instead of
    # changing DFL / flat decode semantics.
    if model_name == "PROCESSED_A_GFPN_s42_fp.rknn":
        bbox_expand_scale = 3.0
        bbox_expand_pad = 30.0
        notes.append("bbox compensation enabled for coarse-region GT alignment")
    elif model_name == "PROCESSED_FULL_MODEL_s42_fp.rknn":
        bbox_expand_scale = 4.0
        bbox_expand_pad = 10.0
        notes.append("bbox compensation enabled for conservative coarse-region boxes")
    elif model_name == "PROCESSED_YOLOV8N_BASELINE_s42_fp.rknn":
        # Known issue: current board-side extra merge / expand heuristics make flat-head
        # boxes drift away from the PC reference path. Prefer a cleaner finalize chain first.
        postprocess_variant = "pc_aligned"
        apply_edge_filter = False
        notes.append("pc-aligned finalize enabled for flat-head alignment")

    # Known issue on the 0420 NEU export: decoded boxes are consistently too tight.
    # Prefer a cleaner PC-aligned finalize path before relying on bbox compensation.
    elif model_name == "NEU_Pretrain_pt_Full_model_Port_defect_fp.rknn":
        postprocess_variant = "pc_aligned"
        apply_edge_filter = False
        notes.append("pc-aligned finalize enabled for DFL alignment")

    return {
        "model_name": model_name,
        "conf": actual_conf,
        "conf_source": conf_source,
        "input_format": actual_input_format,
        "input_layout": actual_input_layout,
        "norm_mode": actual_norm_mode,
        "bbox_expand_scale": bbox_expand_scale,
        "bbox_expand_pad": bbox_expand_pad,
        "postprocess_variant": postprocess_variant,
        "apply_edge_filter": apply_edge_filter,
        "flat_head_debug": flat_head_debug,
        "mode_diagnostic": mode_diagnostic,
        "notes": notes,
    }


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / float(w), new_h / float(h))
    nw, nh = int(round(w * r)), int(round(h * r))

    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = (new_w - nw) / 2.0
    dh = (new_h - nh) / 2.0

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    out = cv2.copyMakeBorder(
        im_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return out, r, (left, top)


def resolve_letterbox_color(letterbox_color):
    c = int(letterbox_color)
    c = max(0, min(c, 255))
    return (c, c, c)


def scale_boxes_to_original(boxes_xyxy, ratio, pad, orig_shape):
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return boxes_xyxy

    padw, padh = pad
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] -= padw
    boxes[:, [1, 3]] -= padh
    boxes[:, :4] /= max(ratio, 1e-9)

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    return boxes


def expand_boxes_xyxy(boxes_xyxy, canvas_size, scale=1.0, pad=0.0):
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return boxes_xyxy

    scale = float(scale)
    pad = float(pad)
    if abs(scale - 1.0) < 1e-6 and abs(pad) < 1e-6:
        return boxes_xyxy

    boxes = boxes_xyxy.copy().astype(np.float32)
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    bw = (boxes[:, 2] - boxes[:, 0]) * scale + 2.0 * pad
    bh = (boxes[:, 3] - boxes[:, 1]) * scale + 2.0 * pad

    boxes[:, 0] = cx - bw * 0.5
    boxes[:, 1] = cy - bh * 0.5
    boxes[:, 2] = cx + bw * 0.5
    boxes[:, 3] = cy + bh * 0.5

    in_w, in_h = canvas_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, in_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, in_h - 1)
    return boxes


def nms_xyxy(boxes, scores, iou_thresh=0.5):
    if boxes is None or len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def dequant_if_needed(out, detail):
    x = np.array(out)

    if x.dtype not in (np.int8, np.uint8) or detail is None:
        return x.astype(np.float32)

    scale = detail.get("scale", detail.get("qnt_scale", None))
    zp = detail.get("zero_point", detail.get("qnt_zp", None))
    if scale is None or zp is None:
        return x.astype(np.float32)

    scale = np.array(scale, dtype=np.float32)
    zp = np.array(zp, dtype=np.float32)

    if scale.size == 1 and zp.size == 1:
        return (x.astype(np.float32) - float(zp.reshape(-1)[0])) * float(scale.reshape(-1)[0])

    y = x.astype(np.float32)

    if y.ndim == 4:
        if y.shape[1] == scale.size:
            shape = [1, scale.size, 1, 1]
            return (y - zp.reshape(shape)) * scale.reshape(shape)
        if y.shape[-1] == scale.size:
            shape = [1, 1, 1, scale.size]
            return (y - zp.reshape(shape)) * scale.reshape(shape)

    if y.ndim == 3:
        if y.shape[0] == scale.size:
            shape = [scale.size, 1, 1]
            return (y - zp.reshape(shape)) * scale.reshape(shape)
        if y.shape[-1] == scale.size:
            shape = [1, 1, scale.size]
            return (y - zp.reshape(shape)) * scale.reshape(shape)

    return y


def try_get_output_details(rknn_lite):
    try:
        return rknn_lite.get_output_details()
    except Exception:
        return None


def try_get_input_detail(rknn_lite):
    try:
        if hasattr(rknn_lite, "get_inputs_details"):
            details = rknn_lite.get_inputs_details()
            if details:
                return details[0]
    except Exception:
        pass

    try:
        if hasattr(rknn_lite, "get_input_details"):
            details = rknn_lite.get_input_details()
            if details:
                return details[0]
    except Exception:
        pass

    return None


def safe_prob(x):
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    if np.min(x) < 0.0 or np.max(x) > 1.5:
        return sigmoid(np.clip(x, -30, 30))
    return np.clip(x, 0.0, 1.0)


def normalize_head_to_chw(out):
    a = np.array(out)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    return a


def pick_dfl_concat_heads(outputs, num_classes):
    expected_c = REG_CH + num_classes
    heads = []
    for out in outputs:
        a = normalize_head_to_chw(out)
        if a.ndim == 3 and a.shape[0] == expected_c:
            heads.append(a)

    heads.sort(key=lambda x: x.shape[1], reverse=True)
    return heads


def inspect_dfl_concat_heads(outputs, num_classes):
    expected_c = REG_CH + num_classes
    matches = []
    notes = []

    for idx, out in enumerate(outputs):
        raw = np.array(out)
        norm = normalize_head_to_chw(out)
        raw_shape = tuple(raw.shape)
        norm_shape = tuple(norm.shape)

        if norm.ndim != 3:
            notes.append(
                f"out[{idx}] reject dfl: normalized ndim={norm.ndim}, "
                f"raw_shape={raw_shape}, norm_shape={norm_shape}"
            )
            continue

        if norm.shape[0] != expected_c:
            notes.append(
                f"out[{idx}] reject dfl: channel_dim={norm.shape[0]}, "
                f"expected={expected_c}, raw_shape={raw_shape}, norm_shape={norm_shape}"
            )
            continue

        matches.append({"index": idx, "shape": norm_shape, "tensor": norm})
        notes.append(
            f"out[{idx}] accept dfl head: raw_shape={raw_shape}, norm_shape={norm_shape}"
        )

    matches.sort(key=lambda item: item["shape"][1], reverse=True)
    return matches, notes


def pick_flat_head(outputs, num_classes):
    expected_c = 4 + num_classes
    best = None
    best_score = -1.0

    for out in outputs:
        a = np.array(out)

        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2:
            continue

        if a.shape[0] == expected_c and a.shape[1] >= 100:
            score = a.shape[1] / 1000.0 + 10.0
        elif a.shape[1] == expected_c and a.shape[0] >= 100:
            score = a.shape[0] / 1000.0 + 10.0
        else:
            continue

        if score > best_score:
            best_score = score
            best = a

    return best


def inspect_flat_head(outputs, num_classes):
    expected_c = 4 + num_classes
    candidates = []
    notes = []

    for idx, out in enumerate(outputs):
        raw = np.array(out)
        arr = raw
        raw_shape = tuple(raw.shape)
        squeezed = False

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
            squeezed = True

        arr_shape = tuple(arr.shape)
        if arr.ndim != 2:
            notes.append(
                f"out[{idx}] reject flat: ndim={arr.ndim}, "
                f"raw_shape={raw_shape}, norm_shape={arr_shape}"
            )
            continue

        if arr.shape[0] == expected_c and arr.shape[1] >= 100:
            score = arr.shape[1] / 1000.0 + 10.0
            layout = "channels_first"
        elif arr.shape[1] == expected_c and arr.shape[0] >= 100:
            score = arr.shape[0] / 1000.0 + 10.0
            layout = "channels_last"
        else:
            notes.append(
                f"out[{idx}] reject flat: expected one dim={expected_c} and the other >=100, "
                f"raw_shape={raw_shape}, norm_shape={arr_shape}"
            )
            continue

        candidates.append(
            {
                "index": idx,
                "shape": arr_shape,
                "layout": layout,
                "score": score,
                "squeezed": squeezed,
                "tensor": arr,
            }
        )
        notes.append(
            f"out[{idx}] accept flat candidate: raw_shape={raw_shape}, norm_shape={arr_shape}, "
            f"layout={layout}, squeezed={squeezed}"
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0] if candidates else None
    return best, notes


def print_score_stats(tag, values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        print(f"{tag}: empty")
        return
    print(
        f"{tag}: max={float(np.max(arr)):.6f} "
        f"mean={float(np.mean(arr)):.6f} "
        f"p95={float(np.percentile(arr, 95)):.6f} "
        f"p99={float(np.percentile(arr, 99)):.6f}"
    )


def parse_conf_scan_values(spec):
    values = []
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0:
            raise ValueError(f"conf scan threshold must be >= 0, got {value}")
        values.append(value)

    if not values:
        raise ValueError("conf scan thresholds are empty")

    return values


def format_topk_scores(scores, topk=3):
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return "[]"

    top = np.sort(arr)[::-1][:topk]
    return "[" + ", ".join(f"{float(v):.6f}" for v in top) + "]"


def format_box_range(boxes):
    if boxes is None or len(boxes) == 0:
        return "none"

    boxes = np.asarray(boxes, dtype=np.float32)
    return (
        f"x=[{float(np.min(boxes[:, 0])):.2f}, {float(np.max(boxes[:, 2])):.2f}] "
        f"y=[{float(np.min(boxes[:, 1])):.2f}, {float(np.max(boxes[:, 3])):.2f}]"
    )


def print_dfl_decode_alignment_log(
    debug_prefix,
    head_idx,
    feat,
    reg,
    cls_raw,
    dist,
    boxes,
    cls_prob,
    scores,
    input_size,
    conf_thres,
):
    h, w = feat.shape[1], feat.shape[2]
    in_w, in_h = input_size
    stride_x = float(in_w) / float(w)
    stride_y = float(in_h) / float(h)
    score_transform = "sigmoid" if (np.min(cls_raw) < 0.0 or np.max(cls_raw) > 1.5) else "identity_clip"

    print(
        f"{debug_prefix} head{head_idx} pc_align "
        f"shape={tuple(feat.shape)} hw=({h},{w}) "
        f"stride=({stride_x:.2f},{stride_y:.2f}) "
        f"score_transform={score_transform}"
    )
    print(
        f"{debug_prefix} head{head_idx} reg_raw_range "
        f"min={float(np.min(reg)):.6f} max={float(np.max(reg)):.6f} "
        f"mean={float(np.mean(reg)):.6f}"
    )
    print(
        f"{debug_prefix} head{head_idx} dist_range "
        f"l=[{float(np.min(dist[0])):.4f}, {float(np.max(dist[0])):.4f}] "
        f"t=[{float(np.min(dist[1])):.4f}, {float(np.max(dist[1])):.4f}] "
        f"r=[{float(np.min(dist[2])):.4f}, {float(np.max(dist[2])):.4f}] "
        f"b=[{float(np.min(dist[3])):.4f}, {float(np.max(dist[3])):.4f}]"
    )
    print(
        f"{debug_prefix} head{head_idx} box_range_preclip="
        f"{format_box_range(boxes)}"
    )
    print(
        f"{debug_prefix} head{head_idx} cls_raw_range "
        f"min={float(np.min(cls_raw)):.6f} max={float(np.max(cls_raw)):.6f} "
        f"mean={float(np.mean(cls_raw)):.6f}"
    )
    print_score_stats(f"{debug_prefix} head{head_idx} cls_prob_max", np.max(cls_prob, axis=1))
    print_score_stats(f"{debug_prefix} head{head_idx} score", scores)
    print(
        f"{debug_prefix} head{head_idx} scores_ge_conf="
        f"{int(np.sum(scores >= float(conf_thres)))}"
    )

    topk = min(5, len(scores))
    if topk > 0:
        top_idx = np.argsort(scores)[::-1][:topk]
        print(f"{debug_prefix} head{head_idx} top{topk}_decoded_candidates")
        for rank, idx in enumerate(top_idx, 1):
            box = boxes[idx]
            probs = cls_prob[idx]
            print(
                f"  top{rank}: score={float(scores[idx]):.6f} "
                f"cls={int(np.argmax(probs))} "
                f"xyxy=({float(box[0]):.2f},{float(box[1]):.2f},{float(box[2]):.2f},{float(box[3]):.2f}) "
                f"cls_prob={probs.tolist()}"
            )


def summarize_mode_candidates(dfl_matches, flat_candidate):
    if len(dfl_matches) >= 3:
        report = f"more like dfl multi-head ({len(dfl_matches)} matched heads)"
        if flat_candidate is not None:
            report += f"; auxiliary flat-like output at out[{flat_candidate['index']}]"
        return report
    if len(dfl_matches) > 0:
        report = f"more like dfl multi-head, but only {len(dfl_matches)} head(s) matched"
        if flat_candidate is not None:
            report += f"; also has flat-like output at out[{flat_candidate['index']}]"
        return report
    if flat_candidate is not None:
        return f"more like flat head (candidate at out[{flat_candidate['index']}])"
    return "looks like neither dfl multi-head nor flat head"


def run_flat_conf_scan(
    boxes,
    class_ids,
    scores,
    input_size,
    iou_thres,
    max_det,
    min_wh,
    min_area,
    max_aspect_ratio,
    merge_gap,
    postprocess_variant,
    conf_scan_values,
    model_name="",
):
    debug_prefix = f"[FlatScan][{model_name}]"
    for conf_thres in conf_scan_values:
        debug_state = {}
        scan_boxes, _, scan_scores = finalize_boxes(
            boxes,
            class_ids,
            scores,
            input_size,
            conf_thres,
            iou_thres,
            max_det,
            min_wh=min_wh,
            min_area=min_area,
            max_aspect_ratio=max_aspect_ratio,
            merge_gap=merge_gap,
            postprocess_variant=postprocess_variant,
            debug_state=debug_state,
        )
        conf_scores = scores[scores >= float(conf_thres)]
        final_det = 0 if scan_boxes is None else int(len(scan_boxes))
        top1 = "none" if conf_scores.size == 0 else f"{float(np.max(conf_scores)):.6f}"
        print(
            f"{debug_prefix} conf={float(conf_thres):.6f} "
            f"candidates_before_conf={debug_state.get('pre_conf_count', 0)} "
            f"candidates_after_conf={debug_state.get('post_conf_count', 0)} "
            f"candidates_after_nms={debug_state.get('post_nms_count', 0)} "
            f"final_det={final_det} "
            f"top1={top1} "
            f"top3={format_topk_scores(conf_scores, topk=3)} "
            f"box_range={format_box_range(scan_boxes)}"
        )


def decode_dfl_reg(reg):
    _, h, w = reg.shape
    reg = reg.reshape(4, REG_MAX + 1, h, w)
    reg = np.clip(reg, -30, 30)

    reg_max = np.max(reg, axis=1, keepdims=True)
    reg_exp = np.exp(reg - reg_max)
    reg_prob = reg_exp / np.sum(reg_exp, axis=1, keepdims=True)

    bins = np.arange(REG_MAX + 1, dtype=np.float32).reshape(1, REG_MAX + 1, 1, 1)
    dist = np.sum(reg_prob * bins, axis=1)
    return dist


def dist2bbox_xyxy(dist, input_size):
    in_w, in_h = input_size
    _, h, w = dist.shape

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = xx.astype(np.float32) + 0.5
    cy = yy.astype(np.float32) + 0.5

    stride_x = float(in_w) / float(w)
    stride_y = float(in_h) / float(h)

    l, t, r, b = dist[0], dist[1], dist[2], dist[3]

    x1 = (cx - l) * stride_x
    y1 = (cy - t) * stride_y
    x2 = (cx + r) * stride_x
    y2 = (cy + b) * stride_y

    boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
    return boxes


def merge_nearby_same_class_boxes(boxes, scores, class_ids, gap=12):
    if boxes is None or len(boxes) == 0:
        return boxes, scores, class_ids

    used = np.zeros(len(boxes), dtype=bool)
    merged_boxes = []
    merged_scores = []
    merged_cls = []

    for i in range(len(boxes)):
        if used[i]:
            continue

        used[i] = True
        cur_group = [i]

        changed = True
        while changed:
            changed = False

            x1 = np.min(boxes[cur_group, 0])
            y1 = np.min(boxes[cur_group, 1])
            x2 = np.max(boxes[cur_group, 2])
            y2 = np.max(boxes[cur_group, 3])

            ex1 = x1 - gap
            ey1 = y1 - gap
            ex2 = x2 + gap
            ey2 = y2 + gap

            for j in range(len(boxes)):
                if used[j]:
                    continue
                if class_ids[j] != class_ids[i]:
                    continue

                bx1, by1, bx2, by2 = boxes[j]
                overlap = not (bx2 < ex1 or bx1 > ex2 or by2 < ey1 or by1 > ey2)
                if overlap:
                    used[j] = True
                    cur_group.append(j)
                    changed = True

        group_boxes = boxes[cur_group]
        group_scores = scores[cur_group]

        mx1 = np.min(group_boxes[:, 0])
        my1 = np.min(group_boxes[:, 1])
        mx2 = np.max(group_boxes[:, 2])
        my2 = np.max(group_boxes[:, 3])
        mscore = np.max(group_scores)

        merged_boxes.append([mx1, my1, mx2, my2])
        merged_scores.append(mscore)
        merged_cls.append(class_ids[i])

    return (
        np.array(merged_boxes, dtype=np.float32),
        np.array(merged_scores, dtype=np.float32),
        np.array(merged_cls, dtype=np.int32),
    )


def filter_edge_boxes(boxes, class_ids, scores, img_shape, edge_margin=4, min_edge_box=20):
    if boxes is None or len(boxes) == 0:
        return boxes, class_ids, scores

    h, w = img_shape[:2]
    keep = []

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        touch_edge = (
            (x1 <= edge_margin) or
            (y1 <= edge_margin) or
            (x2 >= w - 1 - edge_margin) or
            (y2 >= h - 1 - edge_margin)
        )

        bw = x2 - x1
        bh = y2 - y1

        if touch_edge and (bw < min_edge_box or bh < min_edge_box):
            continue

        keep.append(i)

    if not keep:
        return None, None, None

    keep = np.array(keep, dtype=np.int32)
    return boxes[keep], class_ids[keep], scores[keep]


def finalize_boxes(
    boxes,
    class_ids,
    scores,
    input_size,
    conf_thres,
    iou_thres,
    max_det,
    min_wh=10.0,
    min_area=120.0,
    max_aspect_ratio=8.0,
    merge_gap=12,
    postprocess_variant="legacy",
    debug_state=None,
):
    in_w, in_h = input_size

    if debug_state is not None:
        debug_state["pre_conf_count"] = int(len(boxes))
        debug_state["postprocess_variant"] = postprocess_variant

    keep = scores >= float(conf_thres)
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]
    if debug_state is not None:
        debug_state["post_conf_count"] = int(len(boxes))
    if len(boxes) == 0:
        return None, None, None

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, in_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, in_h - 1)

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    if postprocess_variant == "pc_aligned":
        # Keep the finalize chain close to the PC-side reference:
        # clip -> minimum width/height -> per-class NMS -> topk.
        valid = (w >= 2.0) & (h >= 2.0)
    else:
        area = w * h
        aspect = np.maximum(w / (h + 1e-6), h / (w + 1e-6))
        valid = (
            (w >= float(min_wh)) &
            (h >= float(min_wh)) &
            (area >= float(min_area)) &
            (aspect <= float(max_aspect_ratio))
        )

    boxes = boxes[valid]
    class_ids = class_ids[valid]
    scores = scores[valid]
    if debug_state is not None:
        debug_state["post_geom_count"] = int(len(boxes))
    if len(boxes) == 0:
        return None, None, None

    final_b = []
    final_c = []
    final_s = []

    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        keep_idx = nms_xyxy(boxes[idx], scores[idx], iou_thresh=iou_thres)
        if len(keep_idx) == 0:
            continue

        final_b.append(boxes[idx][keep_idx])
        final_c.append(np.full(len(keep_idx), c, dtype=np.int32))
        final_s.append(scores[idx][keep_idx])

    if not final_b:
        return None, None, None

    boxes = np.concatenate(final_b, axis=0)
    class_ids = np.concatenate(final_c, axis=0)
    scores = np.concatenate(final_s, axis=0)
    if debug_state is not None:
        debug_state["post_nms_count"] = int(len(boxes))

    if postprocess_variant != "pc_aligned":
        boxes, scores, class_ids = merge_nearby_same_class_boxes(
            boxes,
            scores,
            class_ids,
            gap=merge_gap
        )

    order = np.argsort(scores)[::-1]
    if len(order) > max_det:
        order = order[:max_det]

    boxes = boxes[order]
    class_ids = class_ids[order]
    scores = scores[order]
    if debug_state is not None:
        debug_state["final_count"] = int(len(boxes))

    return boxes, class_ids, scores


def decode_dfl_concat_heads(
    heads,
    input_size,
    num_classes,
    conf_thres,
    iou_thres,
    max_det,
    min_wh,
    min_area,
    max_aspect_ratio,
    merge_gap,
    postprocess_variant="legacy",
    model_name="",
    dfl_head_debug=False,
):
    boxes_all = []
    cls_all = []
    scores_all = []

    if dfl_head_debug:
        print(f"[DFLDebug][{model_name}] matched_heads={len(heads)} conf_thres={float(conf_thres):.6f}")

    for head_idx, feat in enumerate(heads):
        if feat.shape[0] != REG_CH + num_classes:
            continue

        reg = feat[:REG_CH]
        cls_raw = feat[REG_CH:REG_CH + num_classes]

        dist = decode_dfl_reg(reg)
        boxes = dist2bbox_xyxy(dist, input_size)

        cls_prob = safe_prob(cls_raw).reshape(num_classes, -1).T
        class_ids = np.argmax(cls_prob, axis=1).astype(np.int32)
        scores = np.max(cls_prob, axis=1).astype(np.float32)

        if dfl_head_debug:
            debug_prefix = f"[DFLDebug][{model_name}]"
            print_dfl_decode_alignment_log(
                debug_prefix=debug_prefix,
                head_idx=head_idx,
                feat=feat,
                reg=reg,
                cls_raw=cls_raw,
                dist=dist,
                boxes=boxes,
                cls_prob=cls_prob,
                scores=scores,
                input_size=input_size,
                conf_thres=conf_thres,
            )

        boxes_all.append(boxes)
        cls_all.append(class_ids)
        scores_all.append(scores)

    if not boxes_all:
        if dfl_head_debug:
            print(f"[DFLDebug][{model_name}] no valid dfl heads entered decode")
        return None, None, None

    boxes = np.concatenate(boxes_all, axis=0)
    class_ids = np.concatenate(cls_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)

    debug_state = {} if dfl_head_debug else None
    result = finalize_boxes(
        boxes,
        class_ids,
        scores,
        input_size,
        conf_thres,
        iou_thres,
        max_det,
        min_wh=min_wh,
        min_area=min_area,
        max_aspect_ratio=max_aspect_ratio,
        merge_gap=merge_gap,
        postprocess_variant=postprocess_variant,
        debug_state=debug_state,
    )

    if dfl_head_debug:
        debug_prefix = f"[DFLDebug][{model_name}]"
        print(f"{debug_prefix} candidates_before_conf={debug_state.get('pre_conf_count', 0)}")
        print(f"{debug_prefix} candidates_after_conf={debug_state.get('post_conf_count', 0)}")
        print(f"{debug_prefix} candidates_after_geom={debug_state.get('post_geom_count', 0)}")
        print(f"{debug_prefix} candidates_after_nms={debug_state.get('post_nms_count', 0)}")
        print(f"{debug_prefix} candidates_final={debug_state.get('final_count', 0)}")
        if result[0] is not None and len(result[0]) > 0:
            print(f"{debug_prefix} final_box_range={format_box_range(result[0])}")
            print(f"{debug_prefix} final_top3={format_topk_scores(result[2], topk=3)}")
        elif debug_state.get("post_conf_count", 0) == 0:
            print(f"{debug_prefix} reject_reason=all candidates filtered by conf_thres")
        elif debug_state.get("post_geom_count", 0) == 0:
            print(f"{debug_prefix} reject_reason=all post-conf candidates filtered by geometry")
        elif debug_state.get("post_nms_count", 0) == 0:
            print(f"{debug_prefix} reject_reason=all post-geometry candidates removed by NMS")
        else:
            print(f"{debug_prefix} reject_reason=no boxes survived final merge/order stage")
    return result


def decode_flat_head(
    flat,
    input_size,
    num_classes,
    conf_thres,
    iou_thres,
    max_det,
    min_wh,
    min_area,
    max_aspect_ratio,
    merge_gap,
    postprocess_variant="legacy",
    model_name="",
    flat_head_debug=False,
    conf_scan_values=None,
    flat_force_sigmoid=False,
):
    if flat is None:
        return None, None, None

    expected_c = 4 + num_classes

    if flat.ndim == 3 and flat.shape[0] == 1:
        flat = flat[0]
    if flat.ndim != 2:
        return None, None, None

    if flat.shape[0] == expected_c:
        arr = flat.T
    elif flat.shape[1] == expected_c:
        arr = flat
    else:
        return None, None, None

    arr = arr[:, :expected_c]
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        return None, None, None

    raw = arr[:, :4].astype(np.float32)
    cls_raw = arr[:, 4:4 + num_classes].astype(np.float32)

    if float(np.max(np.abs(raw))) <= 2.0:
        raw[:, [0, 2]] *= float(input_size[0])
        raw[:, [1, 3]] *= float(input_size[1])

    boxes = np.zeros_like(raw, dtype=np.float32)
    boxes[:, 0] = raw[:, 0] - raw[:, 2] / 2.0
    boxes[:, 1] = raw[:, 1] - raw[:, 3] / 2.0
    boxes[:, 2] = raw[:, 0] + raw[:, 2] / 2.0
    boxes[:, 3] = raw[:, 1] + raw[:, 3] / 2.0

    safe_cls_prob = safe_prob(cls_raw)
    forced_sigmoid_prob = sigmoid(np.clip(cls_raw, -30, 30))
    # Known issue: baseline_fp / ruleloss_fp flat-head scores are very low.
    # Keep the current decoder unchanged by default; only switch score mapping when explicitly requested.
    cls_prob = forced_sigmoid_prob if flat_force_sigmoid else safe_cls_prob
    class_ids = np.argmax(cls_prob, axis=1).astype(np.int32)
    scores = np.max(cls_prob, axis=1).astype(np.float32)

    emit_debug = flat_head_debug or bool(conf_scan_values) or flat_force_sigmoid
    if emit_debug:
        debug_prefix = f"[FlatDebug][{model_name}]"
        raw_scores = np.max(cls_raw, axis=1).astype(np.float32)
        print_score_stats(f"{debug_prefix} cls_raw_global", cls_raw)
        print(
            f"{debug_prefix} cls_raw_range "
            f"min={float(np.min(cls_raw)):.6f} "
            f"max={float(np.max(cls_raw)):.6f} "
            f"has_negative={bool(np.any(cls_raw < 0.0))} "
            f"has_gt1={bool(np.any(cls_raw > 1.0))}"
        )
        print_score_stats(f"{debug_prefix} cls_raw_max_per_candidate", raw_scores)
        print(f"{debug_prefix} score_transform={'forced_sigmoid' if flat_force_sigmoid else 'safe_prob'}")
        print_score_stats(f"{debug_prefix} cls_prob", scores)
        if flat_force_sigmoid:
            print_score_stats(
                f"{debug_prefix} cls_prob_safe_prob",
                np.max(safe_cls_prob, axis=1).astype(np.float32),
            )
            print_score_stats(
                f"{debug_prefix} cls_prob_forced_sigmoid",
                np.max(forced_sigmoid_prob, axis=1).astype(np.float32),
            )

        raw_vs_prob_topk = min(5, len(scores))
        if raw_vs_prob_topk > 0:
            topk_idx = np.argsort(scores)[::-1][:raw_vs_prob_topk]
            print(f"{debug_prefix} safe_prob comparison topk={raw_vs_prob_topk}")
            for rank, idx in enumerate(topk_idx, 1):
                box = boxes[idx]
                print(
                    f"  top{rank}: cls={int(class_ids[idx])} "
                    f"score={float(scores[idx]):.6f} "
                    f"xyxy=({float(box[0]):.2f},{float(box[1]):.2f},{float(box[2]):.2f},{float(box[3]):.2f})"
                )
                print(f"    raw={cls_raw[idx].tolist()}")
                print(f"    safe_prob={safe_cls_prob[idx].tolist()}")
                if flat_force_sigmoid:
                    print(f"    forced_sigmoid={forced_sigmoid_prob[idx].tolist()}")
                else:
                    print(f"    prob={cls_prob[idx].tolist()}")

    debug_state = {} if emit_debug else None

    result = finalize_boxes(
        boxes,
        class_ids,
        scores,
        input_size,
        conf_thres,
        iou_thres,
        max_det,
        min_wh=min_wh,
        min_area=min_area,
        max_aspect_ratio=max_aspect_ratio,
        merge_gap=merge_gap,
        postprocess_variant=postprocess_variant,
        debug_state=debug_state,
    )

    if emit_debug:
        debug_prefix = f"[FlatDebug][{model_name}]"
        print(f"{debug_prefix} candidates_before_conf={debug_state.get('pre_conf_count', 0)}")
        print(f"{debug_prefix} candidates_after_conf={debug_state.get('post_conf_count', 0)}")
        print(f"{debug_prefix} candidates_after_nms={debug_state.get('post_nms_count', 0)}")
        print(f"{debug_prefix} candidates_final={debug_state.get('final_count', 0)}")
        if result[0] is not None and len(result[0]) > 0:
            print(f"{debug_prefix} final_box_range={format_box_range(result[0])}")
            print(f"{debug_prefix} final_top3={format_topk_scores(result[2], topk=3)}")
        else:
            print(f"{debug_prefix} final_box_range=none")
            print(f"{debug_prefix} final_top3=[]")

    if conf_scan_values:
        run_flat_conf_scan(
            boxes=boxes,
            class_ids=class_ids,
            scores=scores,
            input_size=input_size,
            iou_thres=iou_thres,
            max_det=max_det,
            min_wh=min_wh,
            min_area=min_area,
            max_aspect_ratio=max_aspect_ratio,
            merge_gap=merge_gap,
            postprocess_variant=postprocess_variant,
            conf_scan_values=conf_scan_values,
            model_name=model_name,
        )

    return result


def post_process(
    outputs,
    output_details,
    input_size,
    num_classes,
    conf_thres,
    iou_thres,
    max_det,
    decode_mode="auto",
    min_wh=10.0,
    min_area=120.0,
    max_aspect_ratio=8.0,
    merge_gap=12,
    postprocess_variant="legacy",
    model_name="",
    flat_head_debug=False,
    conf_scan_values=None,
    flat_force_sigmoid=False,
    mode_diagnostic=False,
):
    outs = []
    for i, out in enumerate(outputs):
        detail = output_details[i] if (output_details is not None and i < len(output_details)) else None
        outs.append(dequant_if_needed(out, detail))

    if mode_diagnostic:
        diag_prefix = f"[ModeDiag][{model_name}]"
        dfl_matches, dfl_notes = inspect_dfl_concat_heads(outs, num_classes)
        flat_candidate, flat_notes = inspect_flat_head(outs, num_classes)
        print(f"{diag_prefix} output_count={len(outs)}")
        for note in dfl_notes:
            print(f"{diag_prefix} {note}")
        for note in flat_notes:
            print(f"{diag_prefix} {note}")
        print(f"{diag_prefix} candidate_report={summarize_mode_candidates(dfl_matches, flat_candidate)}")
        dfl_heads = [item["tensor"] for item in dfl_matches]
        flat = None if flat_candidate is None else flat_candidate["tensor"]
    else:
        dfl_heads = pick_dfl_concat_heads(outs, num_classes)
        flat = None

    if decode_mode in ("auto", "dfl_concat"):
        if len(dfl_heads) >= 3:
            boxes, class_ids, scores = decode_dfl_concat_heads(
                heads=dfl_heads,
                input_size=input_size,
                num_classes=num_classes,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                min_wh=min_wh,
                min_area=min_area,
                max_aspect_ratio=max_aspect_ratio,
                merge_gap=merge_gap,
                postprocess_variant=postprocess_variant,
                model_name=model_name,
                dfl_head_debug=mode_diagnostic,
            )
            if boxes is not None and len(boxes) > 0:
                if mode_diagnostic:
                    print(f"[ModeDiag][{model_name}] dfl_concat accepted")
                return boxes, class_ids, scores, "dfl_concat"
            if mode_diagnostic:
                print(f"[ModeDiag][{model_name}] dfl_concat rejected after decode")
        elif mode_diagnostic:
            print(
                f"[ModeDiag][{model_name}] dfl_concat rejected before decode: "
                f"matched_heads={len(dfl_heads)} < 3"
            )

        if decode_mode == "dfl_concat":
            return None, None, None, "dfl_concat_fail"

    if decode_mode in ("auto", "flat"):
        if flat is None:
            flat = pick_flat_head(outs, num_classes)
        boxes, class_ids, scores = decode_flat_head(
            flat=flat,
            input_size=input_size,
            num_classes=num_classes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            min_wh=min_wh,
            min_area=min_area,
            max_aspect_ratio=max_aspect_ratio,
            merge_gap=merge_gap,
            postprocess_variant=postprocess_variant,
            model_name=model_name,
            flat_head_debug=(flat_head_debug or mode_diagnostic),
            conf_scan_values=conf_scan_values,
            flat_force_sigmoid=flat_force_sigmoid,
        )
        if boxes is not None and len(boxes) > 0:
            if mode_diagnostic:
                print(f"[ModeDiag][{model_name}] flat accepted")
            return boxes, class_ids, scores, "flat"
        if mode_diagnostic:
            if flat is None:
                print(f"[ModeDiag][{model_name}] flat rejected before decode: no compatible flat candidate")
            else:
                print(f"[ModeDiag][{model_name}] flat rejected after decode")

        if decode_mode == "flat":
            return None, None, None, "flat_fail"

    if mode_diagnostic:
        print(f"[ModeDiag][{model_name}] no mode accepted; returning none")
    return None, None, None, "none"


def draw_dets(image, boxes, class_ids, scores):
    out = image.copy()
    if boxes is None or len(boxes) == 0:
        return out

    colors = [
        (0, 255, 0),
        (255, 180, 0),
        (0, 200, 255),
        (255, 0, 0),
        (255, 0, 255),
    ]

    h, w = out.shape[:2]
    for b, cid, sc in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = [int(v) for v in b]

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        c = int(cid)
        name = CLASSES[c] if 0 <= c < len(CLASSES) else f"cls{c}"
        label = f"{name} {float(sc):.2f}"
        color = colors[c % len(colors)]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - 4, th + base + 2)
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw, ty), color, -1)
        cv2.putText(
            out,
            label,
            (x1, ty - base),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return out


def collect_images(input_path):
    if os.path.isfile(input_path):
        return [input_path] if img_check(input_path) else []

    if not os.path.isdir(input_path):
        return []

    names = sorted(os.listdir(input_path))
    out = []
    for n in names:
        p = os.path.join(input_path, n)
        if os.path.isfile(p) and img_check(p):
            out.append(p)
    return out


def parse_input_layout(input_detail, fallback_hw):
    if input_detail is None:
        return [1, 3, fallback_hw[1], fallback_hw[0]], "uint8"

    model_shape = input_detail.get("shape", [1, 3, fallback_hw[1], fallback_hw[0]])
    model_type = input_detail.get("type", input_detail.get("dtype", "uint8"))
    return model_shape, str(model_type)


def prepare_input(
    im0,
    rknn,
    input_size,
    input_format,
    input_layout,
    norm_mode,
    letterbox_color=114,
    debug=False,
):
    im_lb, ratio, pad = letterbox(
        im0,
        new_shape=input_size,
        color=resolve_letterbox_color(letterbox_color),
    )

    if input_format.lower() == "rgb":
        im_fmt = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)
    else:
        im_fmt = im_lb.copy()

    input_detail = try_get_input_detail(rknn)
    model_shape, model_type = parse_input_layout(input_detail, input_size)

    if input_layout == "auto":
        if len(model_shape) >= 4 and model_shape[-1] == 3:
            layout = "nhwc"
        else:
            layout = "nchw"
    else:
        layout = input_layout.lower()

    expect_float = "float" in str(model_type).lower()

    if layout == "nhwc":
        inp = np.expand_dims(im_fmt, 0)
    else:
        inp = np.expand_dims(im_fmt.transpose(2, 0, 1), 0)

    if norm_mode == "auto":
        apply_div255 = expect_float
    elif norm_mode == "div255":
        apply_div255 = True
    else:
        apply_div255 = False

    force_float32 = (norm_mode == 'none')
    if apply_div255:
        inp = inp.astype(np.float32) / 255.0
    else:
        if expect_float or force_float32:
            inp = inp.astype(np.float32)
        else:
            inp = inp.astype(np.uint8)

    if debug:
        print(f"[Debug] input_shape_expected={model_shape}")
        print(f"[Debug] input_type_expected={model_type}")
        print(f"[Debug] input_layout_used={layout}")
        print(f"[Debug] input_format_used={input_format}")
        print(f"[Debug] norm_mode_used={norm_mode}")
        print(f"[Debug] input_tensor_shape={tuple(inp.shape)} dtype={inp.dtype}")

    return inp, ratio, pad


def make_dummy_input(rknn, input_size, input_format, input_layout, norm_mode, letterbox_color=114):
    dummy_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    inp, _, _ = prepare_input(
        dummy_img,
        rknn,
        input_size,
        input_format=input_format,
        input_layout=input_layout,
        norm_mode=norm_mode,
        letterbox_color=letterbox_color,
        debug=False,
    )
    return inp


def run_model(
    model_path,
    image_list,
    out_dir,
    input_size,
    conf,
    iou,
    max_det,
    core,
    debug,
    decode_mode,
    input_format,
    input_layout,
    norm_mode,
    letterbox_color,
    min_wh,
    min_area,
    max_aspect_ratio,
    merge_gap,
    edge_margin,
    min_edge_box,
    conf_scan_values=None,
    flat_force_sigmoid=False,
):
    os.makedirs(out_dir, exist_ok=True)
    num_classes = len(CLASSES)
    profile = resolve_model_runtime_profile(model_path, conf, norm_mode, input_layout, debug)
    actual_conf = profile["conf"]
    conf_source = profile["conf_source"]
    actual_input_format = profile.get("input_format", input_format)
    actual_input_layout = profile.get("input_layout", input_layout)
    actual_norm_mode = profile["norm_mode"]
    bbox_expand_scale = profile.get("bbox_expand_scale", 1.0)
    bbox_expand_pad = profile.get("bbox_expand_pad", 0.0)
    postprocess_variant = profile.get("postprocess_variant", "legacy")
    apply_edge_filter = profile.get("apply_edge_filter", True)
    model_name = profile["model_name"]
    flat_head_debug = profile["flat_head_debug"]
    mode_diagnostic = profile.get("mode_diagnostic", False)

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: ret={ret}, model={model_path}")

    ret = rknn.init_runtime(core_mask=resolve_core_mask(core))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"init_runtime failed: ret={ret}, model={model_path}")

    output_details = try_get_output_details(rknn)
    if debug and output_details is not None:
        print(f"[Debug] output_details={len(output_details)}")
        for i, d in enumerate(output_details):
            scale = d.get("scale", d.get("qnt_scale", "?"))
            zp = d.get("zero_point", d.get("qnt_zp", "?"))

            if isinstance(scale, (list, tuple, np.ndarray)):
                scale_desc = f"len={len(scale)}"
            else:
                scale_desc = str(scale)

            if isinstance(zp, (list, tuple, np.ndarray)):
                zp_desc = f"len={len(zp)}"
            else:
                zp_desc = str(zp)

            print(
                f"  out[{i}] dtype={d.get('dtype', '?')} "
                f"shape={d.get('shape', '?')} "
                f"scale={scale_desc} zp={zp_desc}"
            )

    try:
        dummy = make_dummy_input(
            rknn,
            input_size,
            input_format=actual_input_format,
            input_layout=actual_input_layout,
            norm_mode=actual_norm_mode,
            letterbox_color=letterbox_color,
        )
        for _ in range(2):
            _ = rknn.inference(inputs=[dummy])
    except Exception as e:
        print(f"[Warn] warmup skipped: {e}")

    stats = []
    printed_runtime_info = False
    print(
        f"[Info] Model profile: name={model_name} "
        f"input_format={actual_input_format} "
        f"input_layout={actual_input_layout} "
        f"norm_mode={actual_norm_mode} "
        f"conf_thres={actual_conf:.6f} (from {conf_source}) "
        f"postprocess_variant={postprocess_variant} "
        f"apply_edge_filter={apply_edge_filter} "
        f"bbox_expand_scale={bbox_expand_scale:.2f} "
        f"bbox_expand_pad={bbox_expand_pad:.2f}"
    )
    for note in profile["notes"]:
        print(f"[Info] Model note: {note}")
    if mode_diagnostic:
        print(
            f"[ModeDiag][{model_name}] model_path={model_path} "
            f"input_format={actual_input_format} input_layout={actual_input_layout} "
            f"decode_mode={decode_mode} norm_mode={actual_norm_mode} "
            f"letterbox_color={int(letterbox_color)}"
        )

    for idx, img_path in enumerate(image_list, 1):
        name = os.path.basename(img_path)
        im0 = cv2.imread(img_path)
        if im0 is None:
            print(f"[{idx}/{len(image_list)}] read failed: {img_path}")
            continue

        t0 = time.perf_counter()

        inp, ratio, pad = prepare_input(
            im0,
            rknn,
            input_size,
            input_format=actual_input_format,
            input_layout=actual_input_layout,
            norm_mode=actual_norm_mode,
            letterbox_color=letterbox_color,
            debug=debug,
        )

        t1 = time.perf_counter()
        outputs = rknn.inference(inputs=[inp])
        t2 = time.perf_counter()

        if not printed_runtime_info:
            print(f"[Info] Model outputs for {model_name}:")
            for i, out in enumerate(outputs):
                arr = np.array(out)
                print(f"  out[{i}] shape={tuple(arr.shape)}")
            printed_runtime_info = True

        if debug:
            print("=" * 70)
            print(f"[{idx}/{len(image_list)}] {name}")
            print(f"[Debug] outputs={len(outputs)}")
            for i, out in enumerate(outputs):
                arr = np.array(out)
                print(
                    f"  out[{i}] shape={tuple(arr.shape)} dtype={arr.dtype} "
                    f"min={float(np.min(arr)):.4f} max={float(np.max(arr)):.4f} "
                    f"mean={float(np.mean(arr)):.4f}"
                )

        boxes, cids, scores, mode = post_process(
            outputs=outputs,
            output_details=output_details,
            input_size=input_size,
            num_classes=num_classes,
            conf_thres=actual_conf,
            iou_thres=iou,
            max_det=max_det,
            decode_mode=decode_mode,
            min_wh=min_wh,
            min_area=min_area,
            max_aspect_ratio=max_aspect_ratio,
            merge_gap=merge_gap,
            postprocess_variant=postprocess_variant,
            model_name=model_name,
            flat_head_debug=flat_head_debug,
            conf_scan_values=conf_scan_values,
            flat_force_sigmoid=flat_force_sigmoid,
            mode_diagnostic=mode_diagnostic,
        )
        t3 = time.perf_counter()

        if boxes is None or len(boxes) == 0:
            out_img = im0
            det = 0
        else:
            boxes = expand_boxes_xyxy(
                boxes,
                input_size,
                scale=bbox_expand_scale,
                pad=bbox_expand_pad,
            )
            boxes0 = scale_boxes_to_original(boxes, ratio, pad, im0.shape)
            if apply_edge_filter:
                boxes0, cids, scores = filter_edge_boxes(
                    boxes0,
                    cids,
                    scores,
                    im0.shape,
                    edge_margin=edge_margin,
                    min_edge_box=min_edge_box,
                )

            if boxes0 is None or len(boxes0) == 0:
                out_img = im0
                det = 0
            else:
                out_img = draw_dets(im0, boxes0, cids, scores)
                det = len(boxes0)

        save_path = os.path.join(out_dir, name)
        cv2.imwrite(save_path, out_img)

        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0

        print(
            f"[{idx}/{len(image_list)}] {name} | mode={mode} | det={det} | "
            f"pre={pre_ms:.2f}ms infer={inf_ms:.2f}ms post={post_ms:.2f}ms"
        )
        if mode_diagnostic:
            print(f"[ModeDiag][{model_name}] final_mode={mode}")

        stats.append((name, mode, det, pre_ms, inf_ms, post_ms))

    rknn.release()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Stable RKNN inference for port defect YOLOv8")

    parser.add_argument("--model_path", type=str, required=True, help="RKNN model path")
    parser.add_argument(
        "--prefer_fp",
        action="store_true",
        help="Resolve model_path to the matching FP model under port_defect_rknn_model if available",
    )
    parser.add_argument("--input", type=str, required=True, help="image path or image folder")
    parser.add_argument("--out_dir", type=str, default="./result_port_defect")
    parser.add_argument("--input_size", type=str, default="640x640", help="e.g. 640x640")

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="optional confidence override; if omitted, model profile or global default is used",
    )
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--core", type=str, default="0", help="0/1/2/all")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--decode_mode",
        type=str,
        default="auto",
        choices=["auto", "dfl_concat", "flat"],
        help="for your current model, auto will prefer dfl_concat when 3 DFL heads exist"
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="rgb",
        choices=["rgb", "bgr"]
    )
    parser.add_argument(
        "--input_layout",
        type=str,
        default="auto",
        choices=["auto", "nhwc", "nchw"]
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "div255"]
    )
    parser.add_argument(
        "--pad_color",
        "--letterbox_color",
        dest="letterbox_color",
        type=int,
        default=114,
        help="single-channel pad color used by letterbox, e.g. 114 or 0",
    )

    parser.add_argument("--min_wh", type=float, default=10.0)
    parser.add_argument("--min_area", type=float, default=120.0)
    parser.add_argument("--max_aspect_ratio", type=float, default=8.0)
    parser.add_argument("--merge_gap", type=int, default=12)

    parser.add_argument("--edge_margin", type=int, default=4)
    parser.add_argument("--min_edge_box", type=int, default=20)
    parser.add_argument(
        "--conf_scan",
        action="store_true",
        help="Run flat-head confidence threshold scan for the current model",
    )
    parser.add_argument(
        "--conf_scan_values",
        type=str,
        default="0.10,0.03,0.01,0.005,0.001",
        help="Comma-separated confidence thresholds used by --conf_scan",
    )
    parser.add_argument(
        "--flat_force_sigmoid",
        action="store_true",
        help="Experimental: apply an extra sigmoid to flat-head cls_raw before thresholding",
    )

    parser.add_argument("--fp16_model_path", type=str, default="", help="optional second model for side-by-side check")

    args = parser.parse_args()
    resolved_model_path = resolve_model_path(args.model_path, prefer_fp=args.prefer_fp)
    conf_scan_values = parse_conf_scan_values(args.conf_scan_values) if args.conf_scan else None

    in_w, in_h = map(int, args.input_size.lower().split("x"))
    input_size = (in_w, in_h)

    images = collect_images(args.input)
    if not images:
        print(f"No valid images found in: {args.input}")
        sys.exit(1)

    print(f"[Info] Preprocess: letterbox({int(args.letterbox_color)})")
    print(f"[Info] Input format: {args.input_format}")
    print(f"[Info] Input layout: {args.input_layout}")
    print(f"[Info] Norm mode: {args.norm_mode}")
    print(f"[Info] Decode mode: {args.decode_mode}")
    print(f"[Info] Class names: {CLASSES}")
    print(f"[Info] Requested model: {args.model_path}")
    print(f"[Info] Running model: {resolved_model_path}")
    if conf_scan_values:
        print(f"[Info] Conf scan values: {conf_scan_values}")
    print(f"[Info] Flat force sigmoid: {args.flat_force_sigmoid}")

    run_model(
        model_path=resolved_model_path,
        image_list=images,
        out_dir=args.out_dir,
        input_size=input_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        core=args.core,
        debug=args.debug,
        decode_mode=args.decode_mode,
        input_format=args.input_format,
        input_layout=args.input_layout,
        norm_mode=args.norm_mode,
        letterbox_color=args.letterbox_color,
        min_wh=args.min_wh,
        min_area=args.min_area,
        max_aspect_ratio=args.max_aspect_ratio,
        merge_gap=args.merge_gap,
        edge_margin=args.edge_margin,
        min_edge_box=args.min_edge_box,
        conf_scan_values=conf_scan_values,
        flat_force_sigmoid=args.flat_force_sigmoid,
    )

    if args.fp16_model_path:
        out_dir_fp16 = args.out_dir.rstrip("/\\") + "_fp16"
        resolved_fp16_model_path = resolve_model_path(
            args.fp16_model_path,
            prefer_fp=args.prefer_fp,
        )
        print(f"\n[Info] FP16 reference requested: {args.fp16_model_path}")
        print(f"[Info] Running FP16 reference model: {resolved_fp16_model_path}")
        run_model(
            model_path=resolved_fp16_model_path,
            image_list=images,
            out_dir=out_dir_fp16,
            input_size=input_size,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            core=args.core,
            debug=args.debug,
            decode_mode=args.decode_mode,
            input_format=args.input_format,
            input_layout=args.input_layout,
            norm_mode=args.norm_mode,
            letterbox_color=args.letterbox_color,
            min_wh=args.min_wh,
            min_area=args.min_area,
            max_aspect_ratio=args.max_aspect_ratio,
            merge_gap=args.merge_gap,
            edge_margin=args.edge_margin,
            min_edge_box=args.min_edge_box,
            conf_scan_values=conf_scan_values,
            flat_force_sigmoid=args.flat_force_sigmoid,
        )
        print(f"[Info] Done. Compare {args.out_dir} vs {out_dir_fp16}")


if __name__ == "__main__":
    main()
