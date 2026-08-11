#!/usr/bin/env python3
"""Build the upstream Depth Anything V2 model + a deterministic input fixture.
Shared by convert_da2_to_gguf.py and dump_da2.py so weights and the test input
are identical across conversion and parity dumps.

Relative models: /tmp/da2-src/depth_anything_v2/dpt.py (DepthAnythingV2).
Metric models:   /tmp/da2-src/metric_depth/depth_anything_v2/dpt.py (adds max_depth).
ImageNet normalization; the fixture is already a multiple of 14 so the resize
policy is not exercised here (documented for the converter/dump scripts)."""
import os, sys, numpy as np, torch

DA2_SRC        = os.environ.get("DA2_SRC", "/tmp/da2-src")
DA2_METRIC_SRC = os.path.join(DA2_SRC, "metric_depth")

DA2_CONFIGS = {
    "vits": dict(encoder="vits", features=64,  out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
    "vitg": dict(encoder="vitg", features=384, out_channels=[1536, 1536, 1536, 1536]),
}
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_da2_model(encoder: str, ckpt_path: str, max_depth: float = 0.0):
    """Return the upstream DepthAnythingV2 (eval/f32/CPU). Relative if max_depth==0,
    else the metric_depth variant constructed with max_depth."""
    cfg = dict(DA2_CONFIGS[encoder])
    if max_depth and max_depth > 0:
        if DA2_METRIC_SRC not in sys.path:
            sys.path.insert(0, DA2_METRIC_SRC)
        from depth_anything_v2.dpt import DepthAnythingV2  # metric variant
        net = DepthAnythingV2(**cfg, max_depth=float(max_depth))
    else:
        if DA2_SRC not in sys.path:
            sys.path.insert(0, DA2_SRC)
        from depth_anything_v2.dpt import DepthAnythingV2  # relative variant
        net = DepthAnythingV2(**cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(sd)
    net.eval()
    return net


def fixed_input(k: int = 16) -> np.ndarray:
    """Deterministic structured RGB image (3, 14k, 14k), ImageNet-normalized, CHW f32."""
    n = 14 * k
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / float(n)
    rgb = np.stack([0.5 + 0.5 * np.sin(6.0 * xx),
                    0.5 + 0.5 * np.cos(5.0 * yy),
                    0.5 + 0.5 * np.sin(4.0 * (xx + yy))], axis=0)  # (3,n,n) in [0,1]
    chw = (rgb - NORM_MEAN[:, None, None]) / NORM_STD[:, None, None]
    return np.ascontiguousarray(chw, dtype=np.float32)
