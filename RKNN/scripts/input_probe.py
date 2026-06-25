#!/usr/bin/env python3
import argparse
import numpy as np
import cv2
from rknnlite.api import RKNNLite

def stat(name, arr):
    arr = np.array(arr)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}")

def try_infer(rknn, inp, tag):
    try:
        outs = rknn.inference(inputs=[inp])
        # 只看第一个输出的范围（通常就能判断对不对）
        o0 = np.array(outs[0])
        mx = float(o0.max())
        mn = float(o0.min())
        print(f"[{tag}] OK  out0 shape={o0.shape}, min={mn:.6f}, max={mx:.6f}")
        return mx
    except Exception as e:
        print(f"[{tag}] FAIL {e}")
        return -1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--size", type=int, default=736, help="input size, e.g. 736")
    args = ap.parse_args()

    rknn = RKNNLite()
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        print("load_rknn failed:", ret)
        return

    ret = rknn.init_runtime()
    if ret != 0:
        print("init_runtime failed:", ret)
        return

    print("=" * 60)
    print("1) Model IO details (if supported)")
    print("=" * 60)
    # 不同版本RKNNLite接口略有差异：有则打印，没有就跳过
    for fn in ["get_input_details", "get_output_details"]:
        if hasattr(rknn, fn):
            try:
                details = getattr(rknn, fn)()
                print(fn, "=>", details)
            except Exception as e:
                print(fn, "not available:", e)

    print("=" * 60)
    print("2) Read image & basic stats")
    print("=" * 60)
    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        print("cv2.imread failed, check path:", args.img)
        return
    print("orig img_bgr:", img_bgr.shape, img_bgr.dtype, img_bgr.min(), img_bgr.max())

    img_bgr = cv2.resize(img_bgr, (args.size, args.size))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4种组合：BGR/RGB × NHWC/NCHW（uint8）
    inp_bgr_nhwc = img_bgr.astype(np.uint8)
    inp_rgb_nhwc = img_rgb.astype(np.uint8)
    inp_bgr_nchw = inp_bgr_nhwc.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
    inp_rgb_nchw = inp_rgb_nhwc.transpose(2, 0, 1)[None, ...]

    # 有些模型是 NHWC 但要求带 batch，有些不要求；这里也试一下带 batch 的 NHWC
    inp_bgr_nhwc_b = inp_bgr_nhwc[None, ...]  # (1,H,W,3)
    inp_rgb_nhwc_b = inp_rgb_nhwc[None, ...]

    print("=" * 60)
    print("3) Input candidate stats")
    print("=" * 60)
    stat("bgr_nhwc", inp_bgr_nhwc)
    stat("rgb_nhwc", inp_rgb_nhwc)
    stat("bgr_nchw", inp_bgr_nchw)
    stat("rgb_nchw", inp_rgb_nchw)
    stat("bgr_nhwc_b", inp_bgr_nhwc_b)
    stat("rgb_nhwc_b", inp_rgb_nhwc_b)

    print("=" * 60)
    print("4) Inference probe (which input looks correct?)")
    print("=" * 60)
    scores = {}
    scores["bgr_nhwc"]   = try_infer(rknn, inp_bgr_nhwc,   "bgr_nhwc")
    scores["rgb_nhwc"]   = try_infer(rknn, inp_rgb_nhwc,   "rgb_nhwc")
    scores["bgr_nchw"]   = try_infer(rknn, inp_bgr_nchw,   "bgr_nchw")
    scores["rgb_nchw"]   = try_infer(rknn, inp_rgb_nchw,   "rgb_nchw")
    scores["bgr_nhwc_b"] = try_infer(rknn, inp_bgr_nhwc_b, "bgr_nhwc_b")
    scores["rgb_nhwc_b"] = try_infer(rknn, inp_rgb_nhwc_b, "rgb_nhwc_b")

    best = max(scores, key=lambda k: scores[k])
    print("=" * 60)
    print("Best candidate by out0.max():", best, "=>", scores[best])
    print("NOTE: 这只是快速判断；后续用该输入格式去做完整后处理/NMS画框。")
    print("=" * 60)

    rknn.release()

if __name__ == "__main__":
    main()
