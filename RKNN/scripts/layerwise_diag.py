#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orange Pi / RKNN 端逐层诊断脚本

设计原则：
1. 尽量复用 port_defect_rknn_infer_from_zooV2.py 中现有的 preprocess / inference / postprocess
2. 额外输出 out[1]/out[2]/out[3] 的逐层 topK 统计，便于和 PT 端逐字段对齐
3. 输出 JSON 结构尽量和 PT 端一致

建议放置位置：
- 最好直接放在项目根目录的 scripts/ 下运行
- 如果不在该目录，也可以通过 --repo-root 指向该目录
"""

from __future__ import annotations
from paths import PROJECT_ROOT, DATA_IMAGES_DIR, OUTPUTS_DIR, LAYERWISE_DIAG_DIR

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RKNN layerwise diagnostic export")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=PROJECT_ROOT,
        help="Directory containing port_defect_rknn_infer_from_zooV2.py",
    )
    parser.add_argument("--model-path", type=str, required=True, help="RKNN model path")
    parser.add_argument(
        "--prefer-fp",
        action="store_true",
        help="Reuse existing resolve_model_path(..., prefer_fp=True) logic",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=PROJECT_ROOT,
        help="Image directory, single image path, or comma-separated image names/paths.",
    )
    parser.add_argument(
        "--image-names",
        type=str,
        default="",
        help="Optional comma-separated subset of image names or absolute paths.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--input-size", type=str, default="640x640")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--core", type=str, default="0", help="0/1/2/all")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--decode-mode",
        type=str,
        default="dfl_concat",
        choices=["auto", "dfl_concat", "flat"],
        help="Diagnostic default is dfl_concat to focus on out[1]/out[2]/out[3].",
    )
    parser.add_argument("--input-format", type=str, default="rgb", choices=["rgb", "bgr"])
    parser.add_argument("--input-layout", type=str, default="auto", choices=["auto", "nhwc", "nchw"])
    parser.add_argument("--norm-mode", type=str, default="auto", choices=["auto", "none", "div255"])
    parser.add_argument("--pad-color", type=int, default=114)
    parser.add_argument("--min-wh", type=float, default=10.0)
    parser.add_argument("--min-area", type=float, default=120.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=8.0)
    parser.add_argument("--merge-gap", type=int, default=12)
    parser.add_argument("--edge-margin", type=int, default=4)
    parser.add_argument("--min-edge-box", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./tmp_dfl_head_debug",
    )
    return parser.parse_args()


def ensure_repo_import(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_base_module():
    try:
        import infer as base  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "导入 port_defect_rknn_infer_from_zooV2 失败。"
            "请确认 --repo-root 正确，且脚本运行环境能访问 rknnlite。"
        ) from exc
    return base


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def img_check(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def parse_hw(spec: str) -> Tuple[int, int]:
    spec = spec.lower().strip()
    if "x" not in spec:
        raise ValueError(f"invalid input size: {spec}")
    w, h = spec.split("x", 1)
    return int(w), int(h)


def list_all_images(input_spec: str) -> List[Path]:
    input_spec = str(input_spec).strip()
    if not input_spec:
        return []

    if "," in input_spec:
        out = []
        for item in input_spec.split(","):
            item = item.strip()
            if item:
                out.append(Path(item))
        return out

    p = Path(input_spec)
    if p.is_file():
        return [p] if img_check(p) else []
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.is_file() and img_check(x)])
    return []


def resolve_image_subset(images_arg: str, image_names: str, limit: int) -> List[Path]:
    base_items = list_all_images(images_arg)
    if not base_items:
        return []

    if not image_names:
        chosen = base_items
    else:
        parent_dir = Path(images_arg) if Path(images_arg).is_dir() else None
        resolved: List[Path] = []
        missing: List[str] = []
        for item in [x.strip() for x in image_names.split(",") if x.strip()]:
            candidate = Path(item)
            if candidate.is_absolute() and candidate.exists() and img_check(candidate):
                resolved.append(candidate)
                continue
            if parent_dir is not None:
                in_dir = parent_dir / item
                if in_dir.exists() and img_check(in_dir):
                    resolved.append(in_dir)
                    continue
            matched = [p for p in base_items if p.name == item]
            if matched:
                resolved.append(matched[0])
            else:
                missing.append(item)
        if missing:
            raise FileNotFoundError(f"这些图片未找到: {missing}")
        chosen = resolved

    if limit and limit > 0:
        chosen = chosen[:limit]
    return chosen


def serialize_output_shapes(outputs: Sequence[np.ndarray]) -> List[Dict]:
    items = []
    for idx, out in enumerate(outputs):
        arr = np.asarray(out)
        items.append(
            {
                "output_index": int(idx),
                "shape": [int(x) for x in arr.shape],
                "dtype": str(arr.dtype),
                "min": float(np.min(arr)) if arr.size else None,
                "max": float(np.max(arr)) if arr.size else None,
                "mean": float(np.mean(arr)) if arr.size else None,
            }
        )
    return items


def build_layer_diagnostics(
    base,
    dequanted_outputs: Sequence[np.ndarray],
    input_size: Tuple[int, int],
    topk: int,
) -> Tuple[List[Dict], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    matches, _ = base.inspect_dfl_concat_heads(dequanted_outputs, len(base.CLASSES))
    layers: List[Dict] = []
    boxes_all: List[np.ndarray] = []
    cls_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []

    for layer_idx, match in enumerate(matches):
        feat = np.asarray(match["tensor"], dtype=np.float32)
        out_idx = int(match["index"])
        c, h, w = feat.shape

        reg = feat[:base.REG_CH]
        cls_raw = feat[base.REG_CH:base.REG_CH + len(base.CLASSES)]
        dist = base.decode_dfl_reg(reg)
        boxes = base.dist2bbox_xyxy(dist, input_size)

        cls_raw_flat = cls_raw.reshape(len(base.CLASSES), -1).T.astype(np.float32)
        cls_prob = base.safe_prob(cls_raw_flat)
        class_ids = np.argmax(cls_prob, axis=1).astype(np.int32)
        scores = np.max(cls_prob, axis=1).astype(np.float32)

        boxes_all.append(boxes)
        cls_all.append(class_ids)
        scores_all.append(scores)

        stride_x = float(input_size[0]) / float(w)
        stride_y = float(input_size[1]) / float(h)
        keep_top = min(max(int(topk), 1), len(scores))
        top_idx = np.argsort(scores)[::-1][:keep_top]
        entries: List[Dict] = []
        for rank, flat_idx in enumerate(top_idx, 1):
            gy = int(flat_idx // w)
            gx = int(flat_idx % w)
            cid = int(class_ids[flat_idx])
            entries.append(
                {
                    "rank": int(rank),
                    "flat_index": int(flat_idx),
                    "grid_xy": [gx, gy],
                    "anchor_center_xy": [
                        float((gx + 0.5) * stride_x),
                        float((gy + 0.5) * stride_y),
                    ],
                    "best_class_id": cid,
                    "best_class_name": base.CLASSES[cid] if 0 <= cid < len(base.CLASSES) else str(cid),
                    "best_class_score": float(scores[flat_idx]),
                    "cls_raw_vector": [float(v) for v in cls_raw_flat[flat_idx].tolist()],
                    "cls_prob_vector": [float(v) for v in cls_prob[flat_idx].tolist()],
                    "dfl_ltrb": [float(v) for v in dist[:, gy, gx].tolist()],
                    "decoded_box_xyxy": [float(v) for v in boxes[flat_idx].tolist()],
                }
            )

        layers.append(
            {
                "layer_index": int(layer_idx),
                "output_index": int(out_idx),
                "feat_shape": [int(c), int(h), int(w)],
                "grid_size": [int(w), int(h)],
                "stride_xy": [float(stride_x), float(stride_y)],
                "candidate_count": int(h * w),
                "topk": entries,
            }
        )

    if not boxes_all:
        return layers, None, None, None

    return (
        layers,
        np.concatenate(boxes_all, axis=0),
        np.concatenate(cls_all, axis=0),
        np.concatenate(scores_all, axis=0),
    )


def serialize_final_dets(
    base,
    boxes_in: Optional[np.ndarray],
    class_ids: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    ratio: float,
    pad: Tuple[int, int],
    orig_shape: Tuple[int, int, int],
    input_size: Tuple[int, int],
    bbox_expand_scale: float,
    bbox_expand_pad: float,
    apply_edge_filter: bool,
    edge_margin: int,
    min_edge_box: int,
) -> List[Dict]:
    if boxes_in is None or class_ids is None or scores is None or len(boxes_in) == 0:
        return []

    boxes_work = base.expand_boxes_xyxy(
        boxes_in.copy(),
        input_size,
        scale=bbox_expand_scale,
        pad=bbox_expand_pad,
    )
    boxes0 = base.scale_boxes_to_original(boxes_work, ratio, pad, orig_shape)

    cids = class_ids.copy()
    scrs = scores.copy()
    if apply_edge_filter:
        boxes0, cids, scrs = base.filter_edge_boxes(
            boxes0,
            cids,
            scrs,
            orig_shape,
            edge_margin=edge_margin,
            min_edge_box=min_edge_box,
        )

    if boxes0 is None or len(boxes0) == 0:
        return []

    out: List[Dict] = []
    for idx, (box, cid, score) in enumerate(zip(boxes0, cids, scrs), 1):
        cid_int = int(cid)
        out.append(
            {
                "rank": int(idx),
                "class_id": cid_int,
                "class_name": base.CLASSES[cid_int] if 0 <= cid_int < len(base.CLASSES) else str(cid_int),
                "score": float(score),
                "box_xyxy_orig": [float(v) for v in box.tolist()],
            }
        )
    return out


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    ensure_repo_import(repo_root)
    base = load_base_module()

    input_size = parse_hw(args.input_size)
    image_paths = resolve_image_subset(args.images, args.image_names, args.limit)
    if not image_paths:
        raise FileNotFoundError("未找到可诊断图片，请检查 --images / --image-names")

    resolved_model = base.resolve_model_path(args.model_path, prefer_fp=args.prefer_fp)
    profile = base.resolve_model_runtime_profile(resolved_model, args.conf, args.norm_mode, args.debug)
    actual_conf = profile["conf"]
    actual_input_format = profile.get("input_format", args.input_format)
    actual_input_layout = profile.get("input_layout", args.input_layout)
    actual_norm_mode = profile["norm_mode"]
    bbox_expand_scale = profile.get("bbox_expand_scale", 1.0)
    bbox_expand_pad = profile.get("bbox_expand_pad", 0.0)
    postprocess_variant = profile.get("postprocess_variant", "legacy")
    apply_edge_filter = profile.get("apply_edge_filter", True)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    save_path = out_dir / f"{Path(resolved_model).stem}_rknn_layerwise_diag.json"

    rknn = base.RKNNLite()
    ret = rknn.load_rknn(resolved_model)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: ret={ret}, model={resolved_model}")
    ret = rknn.init_runtime(core_mask=base.resolve_core_mask(args.core))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"init_runtime failed: ret={ret}, model={resolved_model}")

    output_details = base.try_get_output_details(rknn)

    payload = {
        "model_path": str(resolved_model),
        "model_name": profile["model_name"],
        "class_names": list(base.CLASSES),
        "input_size": [int(input_size[0]), int(input_size[1])],
        "mode_requested": args.decode_mode,
        "runtime_profile": {
            "conf": float(actual_conf),
            "input_format": str(actual_input_format),
            "input_layout": str(actual_input_layout),
            "norm_mode": str(actual_norm_mode),
            "postprocess_variant": str(postprocess_variant),
            "apply_edge_filter": bool(apply_edge_filter),
            "bbox_expand_scale": float(bbox_expand_scale),
            "bbox_expand_pad": float(bbox_expand_pad),
            "notes": list(profile.get("notes", [])),
        },
        "images": [],
    }

    try:
        for image_path in image_paths:
            im0 = cv2.imread(str(image_path))
            if im0 is None:
                payload["images"].append(
                    {
                        "image_path": str(image_path),
                        "error": "cv2.imread failed",
                    }
                )
                continue

            inp, ratio, pad = base.prepare_input(
                im0,
                rknn,
                input_size,
                input_format=actual_input_format,
                input_layout=actual_input_layout,
                norm_mode=actual_norm_mode,
                letterbox_color=args.pad_color,
                debug=args.debug,
            )
            outputs = rknn.inference(inputs=[inp])

            dequanted = []
            for idx, out in enumerate(outputs):
                detail = output_details[idx] if (output_details is not None and idx < len(output_details)) else None
                dequanted.append(base.dequant_if_needed(out, detail))

            layers, boxes_all, cls_all, scores_all = build_layer_diagnostics(
                base=base,
                dequanted_outputs=dequanted,
                input_size=input_size,
                topk=int(args.topk),
            )

            if boxes_all is not None and len(boxes_all) > 0:
                debug_state = {}
                _, _, _ = base.finalize_boxes(
                    boxes=boxes_all.copy(),
                    class_ids=cls_all.copy(),
                    scores=scores_all.copy(),
                    input_size=input_size,
                    conf_thres=actual_conf,
                    iou_thres=float(args.iou),
                    max_det=int(args.max_det),
                    min_wh=float(args.min_wh),
                    min_area=float(args.min_area),
                    max_aspect_ratio=float(args.max_aspect_ratio),
                    merge_gap=int(args.merge_gap),
                    postprocess_variant=postprocess_variant,
                    debug_state=debug_state,
                )
            else:
                debug_state = {
                    "pre_conf_count": 0,
                    "post_conf_count": 0,
                    "post_geom_count": 0,
                    "post_nms_count": 0,
                    "final_count": 0,
                }

            final_boxes, final_cids, final_scores, mode = base.post_process(
                outputs=outputs,
                output_details=output_details,
                input_size=input_size,
                num_classes=len(base.CLASSES),
                conf_thres=actual_conf,
                iou_thres=float(args.iou),
                max_det=int(args.max_det),
                decode_mode=args.decode_mode,
                min_wh=float(args.min_wh),
                min_area=float(args.min_area),
                max_aspect_ratio=float(args.max_aspect_ratio),
                merge_gap=int(args.merge_gap),
                postprocess_variant=postprocess_variant,
                model_name=profile["model_name"],
                flat_head_debug=False,
                conf_scan_values=None,
                flat_force_sigmoid=False,
                mode_diagnostic=args.debug or bool(profile.get("mode_diagnostic", False)),
            )

            image_record = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "orig_shape": [int(im0.shape[1]), int(im0.shape[0])],
                "letterbox_ratio": float(ratio),
                "letterbox_pad": [int(pad[0]), int(pad[1])],
                "mode": "none" if mode in ("dfl_concat_fail", "flat_fail", "none") else str(mode),
                "outputs": serialize_output_shapes(outputs),
                "layers": layers,
                "final": {
                    "pre_nms_count": int(debug_state.get("pre_conf_count", 0)),
                    "post_conf_count": int(debug_state.get("post_conf_count", 0)),
                    "post_geom_count": int(debug_state.get("post_geom_count", 0)),
                    "post_nms_count": int(debug_state.get("post_nms_count", 0)),
                    "final_det_count": 0,
                    "detections": [],
                },
            }

            final_dets = serialize_final_dets(
                base=base,
                boxes_in=final_boxes,
                class_ids=final_cids,
                scores=final_scores,
                ratio=ratio,
                pad=pad,
                orig_shape=im0.shape,
                input_size=input_size,
                bbox_expand_scale=bbox_expand_scale,
                bbox_expand_pad=bbox_expand_pad,
                apply_edge_filter=apply_edge_filter,
                edge_margin=int(args.edge_margin),
                min_edge_box=int(args.min_edge_box),
            )
            image_record["final"]["final_det_count"] = int(len(final_dets))
            image_record["final"]["detections"] = final_dets

            payload["images"].append(image_record)
            print(
                f"[RKNN] model={payload['model_name']} image={image_path.name} "
                f"mode={image_record['mode']} final_det={image_record['final']['final_det_count']} "
                f"pre_conf={image_record['final']['post_conf_count']} post_nms={image_record['final']['post_nms_count']}"
            )
    finally:
        rknn.release()

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Error] {exc}")
        traceback.print_exc()
        raise
