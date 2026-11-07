#!/usr/bin/env python3
"""Compare raw QuerySplat f32 dumps against the upstream cpp_ggml bins.

The dumps come from test_querysplat_capi --dump-gaussians/--dump-renders
(little-endian f32, upstream layouts: gaussians N x 14 in
pos3|opacity|scale3|rot4|color order, renders S x 3 x H x W planes
concatenated). The upstream reference bins are the outputs of the pinned
querysplat-ggml working copy's eval run (cpp_ggml cli, f32 tier).

Usage:
  compare_upstream_bins.py --ours-gaussians P --ref-gaussians P \
      [--ours-renders P --ref-renders P] --json OUT

Verdict thresholds (the registered integration-parity gates):
  - gaussians Pearson correlation >= 0.9995 (f32 tier; float-association
    noise between CPU/CUDA GEMM orders absorbs well below this),
  - per-render-view PSNR >= 20 dB (the renders re-project the predicted
    gaussians; small camera/gaussian deltas compound in screen space).
"""

import argparse
import json
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-gaussians", required=True)
    parser.add_argument("--ref-gaussians", required=True)
    parser.add_argument("--ours-renders")
    parser.add_argument("--ref-renders")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    report: dict = {"schema": "querysplat.parity/1"}
    ok = True

    ours = np.fromfile(args.ours_gaussians, dtype="<f4")
    ref = np.fromfile(args.ref_gaussians, dtype="<f4")
    if ours.size != ref.size:
        print(f"gaussians size mismatch: ours {ours.size} vs ref {ref.size}",
              file=sys.stderr)
        return 1
    if ours.size % 14 != 0:
        print(f"gaussians buffer not a multiple of 14: {ours.size}",
              file=sys.stderr)
        return 1
    n = ours.size // 14
    ours = ours.reshape(n, 14)
    ref = ref.reshape(n, 14)

    fields = ["pos_x", "pos_y", "pos_z", "opacity", "scale_x", "scale_y",
              "scale_z", "rot_0", "rot_1", "rot_2", "rot_3", "col_r",
              "col_g", "col_b"]
    per_field = {}
    for j, name in enumerate(fields):
        a = ours[:, j].astype(np.float64)
        b = ref[:, j].astype(np.float64)
        va, vb = a.var(), b.var()
        corr = float(np.corrcoef(a, b)[0, 1]) if va > 0 and vb > 0 else 1.0
        per_field[name] = {
            "pearson": corr,
            "max_abs_diff": float(np.abs(a - b).max()),
            "mean_abs_diff": float(np.abs(a - b).mean()),
        }
    flat_corr = float(np.corrcoef(ours.ravel().astype(np.float64),
                                  ref.ravel().astype(np.float64))[0, 1])
    report["gaussians"] = {
        "count": int(n),
        "pearson": flat_corr,
        "per_field": per_field,
        "threshold_pearson": 0.9995,
    }
    if not np.isfinite(flat_corr) or flat_corr < 0.9995:
        ok = False

    if args.ours_renders and args.ref_renders:
        o = np.fromfile(args.ours_renders, dtype="<f4")
        r = np.fromfile(args.ref_renders, dtype="<f4")
        if o.size != r.size or o.size % 3 != 0:
            print(f"renders size mismatch: ours {o.size} vs ref {r.size}",
                  file=sys.stderr)
            return 1
        total = o.size // 3  # S x H x W (3 channels folded in)
        # infer the view count from the reference layout (S x 3 x H x W):
        # the H x W plane factor is shared
        views = round(total / (512 * 512))
        if total != views * 512 * 512:
            print(f"renders not a multiple of 3x512x512: {total}",
                  file=sys.stderr)
            return 1
        o = o.reshape(views, 3, 512, 512).astype(np.float64)
        r = r.reshape(views, 3, 512, 512).astype(np.float64)
        per_view = []
        for v in range(views):
            mse = float(((o[v] - r[v]) ** 2).mean())
            psnr = (10.0 * np.log10(1.0 / mse) if mse > 0 else 99.0)
            per_view.append({"view": v, "psnr_db": round(psnr, 3)})
            if psnr < 20.0:
                ok = False
        render_corr = float(np.corrcoef(o.ravel(), r.ravel())[0, 1])
        report["renders"] = {
            "views": views,
            "pearson": render_corr,
            "per_view_psnr_db": per_view,
            "threshold_psnr_db": 20.0,
        }
        if not np.isfinite(render_corr):
            ok = False

    report["verdict"] = "PASS" if ok else "FAIL"
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("gaussians", "verdict")
                      if k in report}, indent=1)[:600])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
