#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_model_output.py

RKNN 模型输出诊断工具（适用于 YOLOv8 单输出 (1, 4+nc, N) / RK3588）
- 支持命令行传参指定 rknn 模型路径
- 自动 query 输入输出属性（dims/fmt/type）
- 支持两种输入模式对比（uint8 vs float32/255）用于定位归一化问题
- 输出类别统计（当输出不含 inf/nan 时才可信）

用法示例：
python3 debug_model_output.py \
  --model_path ../models/PROCESSED_A_GFPN_REPHFE_s42_640_fp.rknn \
  --image ../data/images/test_3.jpg \
  --input_size 736x736 \
  --core 0 \
  --try_float

参数说明：
--try_float: 除了 uint8 输入外，再额外跑一次 float32(/255) 输入，用于判断归一化是否是根因
"""

import os
import sys
import argparse
import numpy as np
import cv2
from rknnlite.api import RKNNLite


NEU_CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def parse_hw(s: str):
    s = s.strip().lower().replace(" ", "")
    if "x" not in s:
        raise ValueError(f"Invalid input_size '{s}', expected like 736x736")
    w, h = s.split("x", 1)
    return int(w), int(h)


def pretty_print_attr(title, attr):
    print(f"{title}:")
    # attr 通常是 dict-like
    if isinstance(attr, dict):
        for k in sorted(attr.keys()):
            print(f"  {k}: {attr[k]}")
    else:
        print(f"  {attr}")


def safe_minmax_mean_std(arr: np.ndarray):
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    a = arr[finite]
    return float(a.min()), float(a.max()), float(a.mean()), float(a.std())


def build_input(img_bgr: np.ndarray, w: int, h: int, as_float: bool, fmt: str):
    """
    根据 fmt 构造输入：
    - fmt: 'NHWC' or 'NCHW'（来自 query 或用户指定）
    - as_float:
      - False: uint8 (0..255)
      - True:  float32 (0..1)
    """
    img = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if as_float:
        img = (img.astype(np.float32) / 255.0)
    else:
        img = img.astype(np.uint8)

    if fmt.upper() == "NCHW":
        img = img.transpose(2, 0, 1)  # HWC -> CHW

    img = np.expand_dims(img, 0)  # add batch
    return img


def infer_once(rknn_lite: RKNNLite, inp: np.ndarray):
    outputs = rknn_lite.inference(inputs=[inp])
    return outputs


def analyze_yolov8_single_output(out: np.ndarray, expected_nc: int = 6, class_names=None):
    """
    针对 YOLOv8 单输出：
      out shape 可能是 (1, C, N) 或 (1, N, C) 或 (C, N) 或 (N, C)
    其中 C = 4 + nc

    返回：解析结果 dict
    """
    res = {"ok": False, "reason": "", "nc": expected_nc}

    arr = out
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # (C,N) or (N,C)

    if arr.ndim != 2:
        res["reason"] = f"Not a 2D tensor after squeeze, got shape {arr.shape}"
        return res

    # 判断哪个维度是 C
    C_candidates = [4 + expected_nc]
    # 如果用户不确定 nc，也可以从 shape 推断，但你这里固定 NEU 为 6
    if arr.shape[0] in C_candidates and arr.shape[1] > arr.shape[0]:
        # (C, N) -> (N, C)
        arr2 = arr.T
        layout = "CN"
    elif arr.shape[1] in C_candidates and arr.shape[0] > arr.shape[1]:
        # (N, C)
        arr2 = arr
        layout = "NC"
    else:
        # 无法匹配，仍然尝试：若其中一个维度==10，则当作 C=10
        if arr.shape[0] == 10 and arr.shape[1] > 10:
            arr2 = arr.T
            layout = "CN(assume C=10)"
            res["nc"] = 6
        elif arr.shape[1] == 10 and arr.shape[0] > 10:
            arr2 = arr
            layout = "NC(assume C=10)"
            res["nc"] = 6
        else:
            res["reason"] = f"Cannot match YOLOv8 single output. shape={arr.shape}, expected C={4+expected_nc}"
            return res

    C = arr2.shape[1]
    nc = C - 4
    if nc <= 0:
        res["reason"] = f"Invalid C={C}, cannot derive classes"
        return res

    boxes = arr2[:, :4]
    cls_raw = arr2[:, 4:]

    # 过滤掉包含 inf/nan 的行（否则类别统计全是假的）
    finite_mask = np.isfinite(boxes).all(axis=1) & np.isfinite(cls_raw).all(axis=1)
    valid_ratio = float(finite_mask.mean())
    boxes_v = boxes[finite_mask]
    cls_v = cls_raw[finite_mask]

    res.update({
        "ok": True,
        "layout": layout,
        "C": C,
        "N": arr2.shape[0],
        "nc_inferred": nc,
        "finite_valid_ratio": valid_ratio,
        "finite_valid_count": int(finite_mask.sum()),
    })

    # 如果几乎都不是 finite，直接返回
    if res["finite_valid_count"] < 10:
        res["warning"] = "Too few finite rows; class debug is not reliable. Fix input fmt/normalization first."
        return res

    # 判断 cls 是否 logits（存在负值或大量 >1）
    cls_min = float(cls_v.min())
    cls_max = float(cls_v.max())
    need_sigmoid = (cls_min < 0.0) or (cls_max > 1.5)

    if need_sigmoid:
        cls_prob = 1.0 / (1.0 + np.exp(-np.clip(cls_v, -30, 30)))
        res["cls_mode"] = "logits->sigmoid"
    else:
        cls_prob = cls_v
        res["cls_mode"] = "prob"

    best_cls = np.argmax(cls_prob, axis=1)
    best_score = np.max(cls_prob, axis=1)

    # 直方图
    hist = np.bincount(best_cls, minlength=nc)

    # 阈值统计
    thr = 0.05
    keep = best_score >= thr
    hist_thr = np.bincount(best_cls[keep], minlength=nc)

    # top-k
    topk = min(10, best_score.shape[0])
    idx = np.argsort(-best_score)[:topk]
    top_items = []
    for i in idx:
        ci = int(best_cls[i])
        name = None
        if class_names and ci < len(class_names):
            name = class_names[ci]
        top_items.append({
            "score": float(best_score[i]),
            "cls": ci,
            "name": name,
        })

    res.update({
        "cls_min": cls_min,
        "cls_max": cls_max,
        "hist_all": hist.tolist(),
        "thr": thr,
        "count_thr": int(keep.sum()),
        "hist_thr": hist_thr.tolist(),
        "top": top_items,
    })
    return res


def main():
    parser = argparse.ArgumentParser(description="RKNN YOLOv8 output debug tool")
    parser.add_argument("--model_path", type=str, required=True, help="Path to RKNN model")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--input_size", type=str, default="736x736", help="WxH, e.g. 736x736")
    parser.add_argument("--core", type=int, default=0, choices=[0, 1, 2], help="NPU core index")
    parser.add_argument("--expected_nc", type=int, default=6, help="Expected number of classes (NEU=6)")
    parser.add_argument("--try_float", action="store_true", help="Also run float32(/255) input test")
    parser.add_argument("--force_fmt", type=str, default="", choices=["", "NHWC", "NCHW"], help="Force input fmt (override query)")
    args = parser.parse_args()

    model_path = args.model_path
    img_path = args.image

    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"ERROR: image not found: {img_path}")
        sys.exit(1)

    w, h = parse_hw(args.input_size)

    print("=" * 80)
    print("RKNN模型输出诊断工具")
    print("=" * 80)

    # 1) load
    print(f"\n[1] 加载模型: {model_path}")
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print(f"错误: 无法加载模型 (返回码: {ret})")
        sys.exit(2)
    print("✓ 模型加载成功")

    # 2) init runtime
    print("\n[2] 初始化运行环境")
    core_mask = {
        0: RKNNLite.NPU_CORE_0,
        1: RKNNLite.NPU_CORE_1,
        2: RKNNLite.NPU_CORE_2,
    }[args.core]
    ret = rknn_lite.init_runtime(core_mask=core_mask)
    if ret != 0:
        print(f"错误: 初始化失败 (返回码: {ret})")
        rknn_lite.release()
        sys.exit(3)
    print("✓ 运行环境初始化成功")

    # 3) query attrs
    print("\n[3] 查询模型输入输出属性 (QUERY_INPUT_ATTR/QUERY_OUTPUT_ATTR)")
    try:
        io_num = rknn_lite.query(RKNNLite.QUERY_IN_OUT_NUM)
        print(f"[IO] n_input={io_num.get('n_input')} n_output={io_num.get('n_output')}")
        in_attr = rknn_lite.query(RKNNLite.QUERY_INPUT_ATTR, 0)
        out_attr = rknn_lite.query(RKNNLite.QUERY_OUTPUT_ATTR, 0)
        pretty_print_attr("[INPUT_ATTR]", in_attr)
        pretty_print_attr("[OUTPUT_ATTR]", out_attr)
        inferred_fmt = str(in_attr.get("fmt", "NHWC")).upper()
    except Exception as e:
        print(f"⚠ query attr 失败（不致命）：{e}")
        inferred_fmt = "NHWC"

    if args.force_fmt:
        fmt = args.force_fmt.upper()
        print(f"[Info] Force input fmt: {fmt} (override query)")
    else:
        fmt = inferred_fmt
        print(f"[Info] Use input fmt from query (or fallback): {fmt}")

    # 4) load image
    print(f"\n[4] 加载测试图片: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("ERROR: cannot read image")
        rknn_lite.release()
        sys.exit(4)
    print(f"✓ 图片加载成功, 原始尺寸: {img_bgr.shape}")

    # 5) run tests
    def run_case(case_name: str, as_float: bool):
        print("\n" + "-" * 80)
        print(f"[CASE] {case_name}")
        inp = build_input(img_bgr, w, h, as_float=as_float, fmt=fmt)
        print(f"[Input] shape={inp.shape}, dtype={inp.dtype}, min={inp.min():.3f}, max={inp.max():.3f}")

        print("[5] 执行推理...")
        try:
            outs = infer_once(rknn_lite, inp)
            print("✓ 推理成功")
        except Exception as e:
            print(f"ERROR: inference failed: {e}")
            return

        print("\n[6] 分析模型输出（原始统计）")
        print(f"输出数量: {len(outs)}")
        for i, out in enumerate(outs):
            stats = safe_minmax_mean_std(out)
            print(f"\n--- 输出 #{i} ---")
            print(f"  shape={out.shape}, dtype={out.dtype}, size={out.size}")
            if stats is None:
                print("  stats: all values are non-finite (inf/nan)")
            else:
                mn, mx, mean, std = stats
                print(f"  finite stats: min={mn:.6f}, max={mx:.6f}, mean={mean:.6f}, std={std:.6f}")
            non_zero = int(np.count_nonzero(out))
            print(f"  non_zero={non_zero} ({non_zero/out.size*100:.2f}%)")
            print(f"  finite_ratio={np.isfinite(out).mean()*100:.2f}%")

        # YOLOv8 单输出类别分析（仅对第一个输出尝试）
        if len(outs) >= 1:
            print("\n[7] YOLOv8 单输出类别/结构解析（仅在 finite 足够时可信）")
            info = analyze_yolov8_single_output(
                outs[0],
                expected_nc=args.expected_nc,
                class_names=NEU_CLASS_NAMES if args.expected_nc == 6 else None,
            )
            if not info.get("ok", False):
                print(f"⚠ 解析失败：{info.get('reason')}")
            else:
                print(f"✓ 解析成功: layout={info['layout']}  N={info['N']}  C={info['C']}  nc_inferred={info['nc_inferred']}")
                print(f"  finite_valid_ratio={info['finite_valid_ratio']*100:.2f}% ({info['finite_valid_count']}/{info['N']})")

                if "warning" in info:
                    print(f"⚠ {info['warning']}")
                else:
                    print(f"  cls_mode={info['cls_mode']}  cls_min={info['cls_min']:.4f}  cls_max={info['cls_max']:.4f}")
                    print(f"  hist_all={info['hist_all']}")
                    print(f"  >=thr({info['thr']}): count={info['count_thr']}, hist_thr={info['hist_thr']}")
                    print("  top scores:")
                    for t in info["top"]:
                        if t["name"] is not None:
                            print(f"    score={t['score']:.4f} cls={t['cls']} ({t['name']})")
                        else:
                            print(f"    score={t['score']:.4f} cls={t['cls']}")

    # 默认先跑 uint8
    run_case("uint8 input (0..255)", as_float=False)

    # 可选再跑 float32(/255)
    if args.try_float:
        run_case("float32 input (/255)", as_float=True)

    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
    rknn_lite.release()


if __name__ == "__main__":
    main()
