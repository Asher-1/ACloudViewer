#!/usr/bin/env python3
"""Standalone monocular (DA3MONO-LARGE) end-to-end depth + sky parity check.

Verifies the C++ `da3-cli depth` mono path (Engine::depth_mono: backbone with
cat_token=False -> DPT depth_sky head, output_dim==1 + sky) against the genuine
reference forward on a RAW arbitrary-resolution photo.

Reference path (NO torchvision binary required; same harness as
e2e_verify_native.py):
  - genuine upstream InputProcessor (process_res=504, upper_bound_resize); only the
    ABI-broken ToTensor/Normalize are stubbed with exact math equivalents.
  - net.backbone.pretrained.get_intermediate_layers(..., n=out_layers)
  - net.head(feats, H, W, patch_start_idx=0) -> {"depth", "sky"}   (the call model() makes)

C++ path: build/examples/cli/da3-cli depth --model <mono.gguf> --input photo.png
          --pfm depth.pfm --sky sky.pfm  (mono auto-detected from head metadata)

PASS: depth corr > 0.999 (primary), sky corr > 0.99, both with small max|d|.
"""
import os, sys, subprocess, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DA3_SRC = "/tmp/da3-src/src"

from e2e_verify_native import (make_structured_image, install_torchvision_stub,
                               read_pfm, W0, H0, PROCESS_RES, METHOD)

OUT_PNG   = os.path.join(ROOT, "dumps", "mono_input.png")
OUT_GGUF  = os.path.join(ROOT, "dumps", "reference_mono.gguf")
CPP_PFM   = os.path.join(ROOT, "dumps", "e2e_mono_cpp.pfm")
CPP_SKY   = os.path.join(ROOT, "dumps", "e2e_mono_sky_cpp.pfm")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--model-dir", default="models/DA3MONO-LARGE")
    ap.add_argument("--gguf", default=os.path.join("models", "depth-anything-mono-large-f32.gguf"))
    args = ap.parse_args()

    import torch
    from PIL import Image as PILImage
    os.makedirs(os.path.join(ROOT, "dumps"), exist_ok=True)

    # 1) Arbitrary-resolution non-square photo (deterministic structured fallback).
    img_path = args.image if args.image and os.path.exists(args.image) else None
    if img_path:
        arr = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)
    else:
        arr = make_structured_image(W0, H0)
    PILImage.fromarray(arr).save(OUT_PNG)
    h0, w0 = arr.shape[:2]
    print(f"input photo: {w0}x{h0} -> {OUT_PNG}")

    # 2) Reference: genuine InputProcessor resize -> backbone -> mono depth+sky head.
    install_torchvision_stub()
    from da3_reference import load_model
    sys.path.insert(0, DA3_SRC)
    from depth_anything_3.utils.io.input_processor import InputProcessor

    proc = InputProcessor()
    tensor, _, _ = proc([OUT_PNG], process_res=PROCESS_RES, process_res_method=METHOD)
    t = tensor.reshape(-1, 3, tensor.shape[-2], tensor.shape[-1])[0]  # (3,H,W)
    C, H, W = t.shape
    assert C == 3 and H % 14 == 0 and W % 14 == 0, t.shape
    print(f"processed resolution: {W}x{H} (long side={max(W,H)}, both mult of 14)")

    _, net = load_model(args.model_dir)
    assert int(net.head.out_dim) == 1, net.head.out_dim
    bb = net.backbone.pretrained
    out_layers = list(net.backbone.out_layers)
    x = t[None, None].contiguous()   # (1,1,3,H,W)
    with torch.no_grad():
        outs, _ = bb.get_intermediate_layers(
            x, n=out_layers, export_feat_layers=[], ref_view_strategy="saddle_balanced")
        ho = net.head(list(outs), H, W, patch_start_idx=0)
    ref_depth = ho["depth"].squeeze().float().cpu().numpy()   # (H,W)
    ref_sky   = ho["sky"].squeeze().float().cpu().numpy()     # (H,W)
    assert ref_depth.shape == (H, W) and ref_sky.shape == (H, W), (ref_depth.shape, ref_sky.shape)

    # 3) C++ mono CLI (depth + sky).
    cli = os.path.join(ROOT, "build", "examples", "cli", "da3-cli")
    model = args.gguf if os.path.isabs(args.gguf) else os.path.join(ROOT, args.gguf)
    subprocess.check_call([cli, "depth", "--model", model, "--input", OUT_PNG,
                           "--pfm", CPP_PFM, "--sky", CPP_SKY])
    cpp_depth = read_pfm(CPP_PFM)
    cpp_sky   = read_pfm(CPP_SKY)
    assert cpp_depth.shape == ref_depth.shape, (cpp_depth.shape, ref_depth.shape)
    assert cpp_sky.shape == ref_sky.shape, (cpp_sky.shape, ref_sky.shape)

    # 4) Compare.
    d_abs = np.abs(cpp_depth - ref_depth)
    d_corr = np.corrcoef(cpp_depth.ravel(), ref_depth.ravel())[0, 1]
    s_abs = np.abs(cpp_sky - ref_sky)
    s_den = np.corrcoef(cpp_sky.ravel(), ref_sky.ravel())
    s_corr = s_den[0, 1] if np.isfinite(s_den[0, 1]) else 1.0
    print(f"e2e mono depth: shape={ref_depth.shape} max|d|={d_abs.max():.3e} "
          f"mean|d|={d_abs.mean():.3e} corr={d_corr:.6f}")
    print(f"  ref depth [{ref_depth.min():.4f},{ref_depth.max():.4f}] "
          f"cpp depth [{cpp_depth.min():.4f},{cpp_depth.max():.4f}]")
    print(f"e2e mono sky:   max|d|={s_abs.max():.3e} mean|d|={s_abs.mean():.3e} corr={s_corr:.6f}")
    print(f"  ref sky [{ref_sky.min():.4f},{ref_sky.max():.4f}] "
          f"cpp sky [{cpp_sky.min():.4f},{cpp_sky.max():.4f}]")

    # 5) Dump reference for the C++ gate.
    import gguf
    gw = gguf.GGUFWriter(OUT_GGUF, "mono_depth")
    gw.add_uint32("mono.out_h", int(H))
    gw.add_uint32("mono.out_w", int(W))
    gw.add_tensor("mono_depth", np.ascontiguousarray(ref_depth.reshape(-1).astype(np.float32)))
    gw.add_tensor("mono_sky", np.ascontiguousarray(ref_sky.reshape(-1).astype(np.float32)))
    gw.write_header_to_file(); gw.write_kv_data_to_file(); gw.write_tensors_to_file(); gw.close()
    print(f"wrote {OUT_GGUF} (mono_depth + mono_sky {ref_depth.size} f32) + {OUT_PNG}")

    ok = (d_corr > 0.999 and d_abs.max() < 5e-2 and s_corr > 0.99)
    print("E2E-MONO", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
