#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
板端 RKNN 对齐 dump 脚本

特点：
1. 复用现有 port_defect_rknn_infer_from_zooV2.py 的 preprocess / dequant / decode / postprocess
2. 保存 preprocess 后输入张量、RKNN 原始输出、dequant 后输出、decode 后中间结果
3. 显式打印 preprocess / dequant / decode 对齐检查项
4. 不改现有主推理脚本，只做最小包装
"""

from __future__ import annotations
from paths import MODELS_DIR, DATA_IMAGES_DIR, OUTPUTS_DIR, ALIGN_DUMP_DIR

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import infer as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RKNN alignment dump wrapper")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prefer-fp", action="store_true")
    parser.add_argument("--input", type=str, required=True, help="image path or image folder")
    parser.add_argument("--image-names", type=str, default="", help="comma-separated subset names when --input is a folder")
    parser.add_argument("--out-dir", type=str, default="./tmp_align_dump")
    parser.add_argument("--input-size", type=str, default="640x640")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--core", type=str, default="0")
    parser.add_argument("--decode-mode", type=str, default="dfl_concat", choices=["auto", "dfl_concat", "flat"])
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-npy", action="store_true", help="Save intermediate arrays as .npy")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_hw(text: str) -> Tuple[int, int]:
    w, h = text.lower().split("x", 1)
    return int(w), int(h)


def collect_images(input_path: str, image_names: str, limit: int) -> List[str]:
    all_images = base.collect_images(input_path)
    if image_names:
        wanted = [x.strip() for x in image_names.split(",") if x.strip()]
        indexed = {os.path.basename(p): p for p in all_images}
        resolved = []
        for name in wanted:
            if os.path.isabs(name) and os.path.exists(name):
                resolved.append(name)
            elif name in indexed:
                resolved.append(indexed[name])
            else:
                raise FileNotFoundError(f"image not found in input set: {name}")
        all_images = resolved
    if limit and limit > 0:
        all_images = all_images[:limit]
    return all_images


def output_detail_summary(output_details) -> List[Dict]:
    rows = []
    if output_details is None:
        return rows
    for idx, d in enumerate(output_details):
        scale = d.get("scale", d.get("qnt_scale", None))
        zp = d.get("zero_point", d.get("qnt_zp", None))
        rows.append(
            {
                "output_index": int(idx),
                "shape": d.get("shape", None),
                "dtype": d.get("dtype", None),
                "scale": np.asarray(scale).tolist() if scale is not None else None,
                "zero_point": np.asarray(zp).tolist() if zp is not None else None,
            }
        )
    return rows


def detect_input_layout(rknn, input_size: Tuple[int, int], requested_layout: str, norm_mode: str) -> Dict:
    input_detail = base.try_get_input_detail(rknn)
    model_shape, model_type = base.parse_input_layout(input_detail, input_size)
    if requested_layout == "auto":
        actual_layout = "nhwc" if len(model_shape) >= 4 and model_shape[-1] == 3 else "nchw"
    else:
        actual_layout = requested_layout.lower()

    expect_float = "float" in str(model_type).lower()
    if norm_mode == "auto":
        apply_div255 = expect_float
        actual_norm = "div255" if apply_div255 else "none"
    elif norm_mode == "div255":
        apply_div255 = True
        actual_norm = "div255"
    else:
        apply_div255 = False
        actual_norm = "none"

    return {
        "model_shape": model_shape,
        "model_type": str(model_type),
        "input_layout": actual_layout,
        "expect_float": bool(expect_float),
        "norm_mode_applied": actual_norm,
    }


def build_anchor_centers(h: int, w: int, input_size: Tuple[int, int]) -> np.ndarray:
    in_w, in_h = input_size
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    stride_x = float(in_w) / float(w)
    stride_y = float(in_h) / float(h)
    cx = (xx.astype(np.float32) + 0.5) * stride_x
    cy = (yy.astype(np.float32) + 0.5) * stride_y
    return np.stack([cx, cy], axis=-1)


def save_array_if_needed(path: Path, arr, enabled: bool) -> None:
    if not enabled:
        return
    np.save(str(path), np.asarray(arr))


def decode_heads_for_dump(
    dequanted_outputs: Sequence[np.ndarray],
    input_size: Tuple[int, int],
    topk: int,
) -> List[Dict]:
    matches, notes = base.inspect_dfl_concat_heads(dequanted_outputs, len(base.CLASSES))
    layers = []
    for layer_idx, match in enumerate(matches):
        feat = np.asarray(match["tensor"], dtype=np.float32)
        out_idx = int(match["index"])
        c, h, w = feat.shape
        reg = feat[:base.REG_CH]
        cls_raw = feat[base.REG_CH:base.REG_CH + len(base.CLASSES)]
        dist = base.decode_dfl_reg(reg)
        boxes = base.dist2bbox_xyxy(dist, input_size)
        anchor_centers = build_anchor_centers(h, w, input_size)

        cls_raw_flat = cls_raw.reshape(len(base.CLASSES), -1).T.astype(np.float32)
        cls_prob = base.safe_prob(cls_raw_flat)
        scores = np.max(cls_prob, axis=1).astype(np.float32)
        class_ids = np.argmax(cls_prob, axis=1).astype(np.int32)

        keep_top = min(max(int(topk), 1), len(scores))
        top_idx = np.argsort(scores)[::-1][:keep_top]
        entries = []
        for rank, flat_idx in enumerate(top_idx, 1):
            gy = int(flat_idx // w)
            gx = int(flat_idx % w)
            cid = int(class_ids[flat_idx])
            entries.append(
                {
                    "rank": int(rank),
                    "grid_xy": [gx, gy],
                    "anchor_center_xy": [float(v) for v in anchor_centers[gy, gx].tolist()],
                    "best_class_id": cid,
                    "best_class_name": base.CLASSES[cid] if 0 <= cid < len(base.CLASSES) else str(cid),
                    "best_class_score": float(scores[flat_idx]),
                    "dfl_ltrb": [float(v) for v in dist[:, gy, gx].tolist()],
                    "decoded_box_xyxy": [float(v) for v in boxes[flat_idx].tolist()],
                }
            )

        layers.append(
            {
                "layer_index": int(layer_idx),
                "output_index": int(out_idx),
                "feat_shape": [int(c), int(h), int(w)],
                "stride_xy": [float(input_size[0] / float(w)), float(input_size[1] / float(h))],
                "anchor_center_formula": "((grid_x+0.5)*stride_x, (grid_y+0.5)*stride_y)",
                "dist2bbox_formula": "xyxy from l/t/r/b distances around anchor center",
                "topk": entries,
                "arrays": {
                    "reg": reg,
                    "cls_raw": cls_raw,
                    "dist": dist,
                    "boxes": boxes,
                    "anchor_centers": anchor_centers,
                    "scores": scores,
                    "class_ids": class_ids,
                },
            }
        )
    return layers


def main() -> None:
    args = parse_args()
    input_size = parse_hw(args.input_size)
    images = collect_images(args.input, args.image_names, args.limit)
    if not images:
        raise FileNotFoundError(f"No valid images found in: {args.input}")

    model_path = base.resolve_model_path(args.model_path, prefer_fp=args.prefer_fp)
    profile = base.resolve_model_runtime_profile(model_path, args.conf, args.norm_mode, args.debug)
    actual_conf = profile["conf"]
    actual_input_format = profile.get("input_format", args.input_format)
    actual_input_layout = profile.get("input_layout", args.input_layout)
    actual_norm_mode = profile["norm_mode"]
    postprocess_variant = profile.get("postprocess_variant", "legacy")
    bbox_expand_scale = profile.get("bbox_expand_scale", 1.0)
    bbox_expand_pad = profile.get("bbox_expand_pad", 0.0)
    apply_edge_filter = profile.get("apply_edge_filter", True)

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    rknn = base.RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: ret={ret}, model={model_path}")
    ret = rknn.init_runtime(core_mask=base.resolve_core_mask(args.core))
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"init_runtime failed: ret={ret}, model={model_path}")

    output_details = base.try_get_output_details(rknn)
    input_cfg = detect_input_layout(
        rknn=rknn,
        input_size=input_size,
        requested_layout=actual_input_layout,
        norm_mode=actual_norm_mode,
    )

    print(f"[AlignCheck] model={os.path.basename(model_path)}")
    print(f"[AlignCheck] RGB/BGR={actual_input_format.upper()}")
    print(f"[AlignCheck] letterbox_pad_color={int(args.pad_color)}")
    print(f"[AlignCheck] div255/none={input_cfg['norm_mode_applied']}")
    print(f"[AlignCheck] NHWC/NCHW={input_cfg['input_layout'].upper()}")
    print(f"[AlignCheck] reg_max={base.REG_MAX}")
    print(f"[AlignCheck] DFL_bins={base.REG_MAX + 1}")
    print("[AlignCheck] anchor_center=((grid_x+0.5)*stride_x, (grid_y+0.5)*stride_y)")
    print("[AlignCheck] dist2bbox=xyxy from l/t/r/b distances around anchor center")
    for row in output_detail_summary(output_details):
        print(
            f"[AlignCheck] out[{row['output_index']}] "
            f"scale={row['scale']} zero_point={row['zero_point']} shape={row['shape']}"
        )

    try:
        for img_path in images:
            im0 = cv2.imread(img_path)
            if im0 is None:
                print(f"[Warn] read failed: {img_path}")
                continue

            image_name = os.path.basename(img_path)
            stem = Path(image_name).stem
            image_dir = out_dir / stem
            ensure_dir(image_dir)

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
            save_array_if_needed(image_dir / "preprocess_input.npy", inp, args.save_npy)
            outputs = rknn.inference(inputs=[inp])

            raw_rows = []
            dequanted = []
            for idx, out in enumerate(outputs):
                arr = np.asarray(out)
                raw_rows.append(
                    {
                        "output_index": int(idx),
                        "shape": [int(v) for v in arr.shape],
                        "dtype": str(arr.dtype),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "mean": float(np.mean(arr)),
                    }
                )
                save_array_if_needed(image_dir / f"raw_out_{idx}.npy", arr, args.save_npy)
                detail = output_details[idx] if (output_details is not None and idx < len(output_details)) else None
                deq = base.dequant_if_needed(arr, detail)
                dequanted.append(deq)
                save_array_if_needed(image_dir / f"dequant_out_{idx}.npy", deq, args.save_npy)

            head_layers = decode_heads_for_dump(
                dequanted_outputs=dequanted,
                input_size=input_size,
                topk=args.topk,
            )
            for layer in head_layers:
                layer_idx = layer["layer_index"]
                arrays = layer.pop("arrays")
                save_array_if_needed(image_dir / f"head_{layer_idx}_reg.npy", arrays["reg"], args.save_npy)
                save_array_if_needed(image_dir / f"head_{layer_idx}_cls_raw.npy", arrays["cls_raw"], args.save_npy)
                save_array_if_needed(image_dir / f"head_{layer_idx}_dist.npy", arrays["dist"], args.save_npy)
                save_array_if_needed(image_dir / f"head_{layer_idx}_boxes.npy", arrays["boxes"], args.save_npy)
                save_array_if_needed(image_dir / f"head_{layer_idx}_anchor_centers.npy", arrays["anchor_centers"], args.save_npy)
                save_array_if_needed(image_dir / f"head_{layer_idx}_scores.npy", arrays["scores"], args.save_npy)
                save_array_if_needed(image_dir / f"head_{layer_idx}_class_ids.npy", arrays["class_ids"], args.save_npy)

            boxes, cids, scores, mode = base.post_process(
                outputs=outputs,
                output_details=output_details,
                input_size=input_size,
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
                mode_diagnostic=args.debug or bool(profile.get("mode_diagnostic", False)),
            )

            final_det_count = 0
            final_boxes_orig = []
            if boxes is not None and len(boxes) > 0:
                boxes = base.expand_boxes_xyxy(
                    boxes,
                    input_size,
                    scale=bbox_expand_scale,
                    pad=bbox_expand_pad,
                )
                boxes0 = base.scale_boxes_to_original(boxes, ratio, pad, im0.shape)
                if apply_edge_filter:
                    boxes0, cids, scores = base.filter_edge_boxes(
                        boxes0,
                        cids,
                        scores,
                        im0.shape,
                        edge_margin=args.edge_margin,
                        min_edge_box=args.min_edge_box,
                    )
                if boxes0 is not None and len(boxes0) > 0:
                    final_det_count = int(len(boxes0))
                    final_boxes_orig = [
                        {
                            "class_id": int(cid),
                            "class_name": base.CLASSES[int(cid)],
                            "score": float(sc),
                            "box_xyxy_orig": [float(v) for v in box.tolist()],
                        }
                        for box, cid, sc in zip(boxes0, cids, scores)
                    ]

            payload = {
                "image_path": img_path,
                "image_name": image_name,
                "model_path": model_path,
                "preprocess": {
                "RGB/BGR": args.input_format.upper(),
                "RGB/BGR_effective": actual_input_format.upper(),
                "letterbox_pad_color": int(args.pad_color),
                "div255/none": input_cfg["norm_mode_applied"],
                "NHWC/NCHW": input_cfg["input_layout"].upper(),
                    "model_input_shape": input_cfg["model_shape"],
                    "model_input_type": input_cfg["model_type"],
                    "ratio": float(ratio),
                    "pad": [int(pad[0]), int(pad[1])],
                    "input_tensor_shape": [int(v) for v in np.asarray(inp).shape],
                    "input_tensor_dtype": str(np.asarray(inp).dtype),
                },
                "outputs": {
                    "details": output_detail_summary(output_details),
                    "raw": raw_rows,
                },
                "decode": {
                    "reg_max": base.REG_MAX,
                    "DFL_bins": base.REG_MAX + 1,
                    "anchor_center": "((grid_x+0.5)*stride_x, (grid_y+0.5)*stride_y)",
                    "stride": [layer["stride_xy"] for layer in head_layers],
                    "dist2bbox": "xyxy from l/t/r/b distances around anchor center",
                    "layers": head_layers,
                },
                "final": {
                    "mode": mode,
                    "final_det_count": final_det_count,
                    "detections": final_boxes_orig,
                },
            }
            summary_path = image_dir / "align_dump_summary.json"
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Saved] {summary_path}")
    finally:
        rknn.release()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
