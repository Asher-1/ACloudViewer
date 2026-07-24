#!/usr/bin/env python3
"""End-to-end nested metric-depth verification (FINAL milestone): the C++ da3 CLI
nested pipeline vs the original NestedDepthAnything3Net `net(x)` on a 224x224 image.

Both sides receive identical 224x224 uint8 input. The reference runs the full nested
forward (anyview GIANT + metric ViT-L branches + metric-scaling + least-squares
alignment + sky handling) -> metric-scale `depth`. The C++ side runs
  da3-cli depth --model <anyview.gguf> --metric-model <metric.gguf> --input img --pfm out.pfm
which executes Engine::depth_metric (both ggml backbones + heads + NestedAligner).

SLOW: the reference forward runs the giant + large backbones on CPU (minutes).
"""
import os, sys, subprocess, argparse, numpy as np, torch
# Pre-import e3nn.o3 before the torchvision stub is installed (see dump_nested.py).
import e3nn.o3  # noqa: F401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")
from da3_reference import load_model, FIX_H, FIX_W
from PIL import Image as PILImage

NORM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
NORM_STD  = np.array([0.229, 0.224, 0.225], np.float32)
MODEL_DIR = "models/DA3NESTED-GIANT-LARGE"


def get_image(path_or_none):
    if path_or_none and os.path.exists(path_or_none):
        img = PILImage.open(path_or_none).convert("RGB").resize((FIX_W, FIX_H), PILImage.BICUBIC)
        return np.array(img, dtype=np.uint8)
    # Deterministic structured fallback (gradients + blocks -> real depth structure).
    yy, xx = np.mgrid[0:FIX_H, 0:FIX_W].astype(np.float32)
    r = (np.sin(xx / 30.0) * 0.5 + 0.5)
    g = (np.cos(yy / 40.0) * 0.5 + 0.5)
    b = ((xx + yy) / (2 * FIX_W))
    arr = np.stack([r, g, b], -1)
    arr[60:120, 60:120, :] = 0.9
    arr[140:200, 30:90, :] = 0.2
    return (arr * 255).astype(np.uint8)


def ref_nested(uint8_img):
    _, net = load_model(MODEL_DIR)
    assert hasattr(net, "da3") and hasattr(net, "da3_metric"), type(net).__name__
    x = (uint8_img.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    t = torch.from_numpy(x).permute(2, 0, 1)[None, None].contiguous()  # (1,1,3,H,W)
    with torch.no_grad():
        full = net(t)
    depth = full["depth"].squeeze().float().cpu().numpy()      # (H,W) metric-scale
    scale = float(full["scale_factor"])
    ext = full["extrinsics"].reshape(3, 4).float().cpu().numpy()
    return depth, scale, ext


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(w * h * 4), dtype="<f4" if scale < 0 else ">f4").reshape(h, w)
        return np.flipud(data).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--anyview", default="models/depth-anything-nested-anyview.gguf")
    ap.add_argument("--metric", default="models/depth-anything-nested-metric.gguf")
    ap.add_argument("--cli", default="build/examples/cli/da3-cli")
    a = ap.parse_args()
    os.makedirs("dumps", exist_ok=True)

    uint8 = get_image(a.image)
    PILImage.fromarray(uint8).save("dumps/e2e_nested_input.png")  # lossless

    print("[ref] running nested net(x) (SLOW: giant + large)...", flush=True)
    ref_depth, ref_scale, ref_ext = ref_nested(uint8)

    print("[cpp] running da3-cli nested metric pipeline...", flush=True)
    out = subprocess.check_output(
        [a.cli, "depth", "--model", a.anyview, "--metric-model", a.metric,
         "--input", "dumps/e2e_nested_input.png", "--pfm", "dumps/e2e_nested_cpp.pfm"]).decode()
    print(out.strip())
    cpp = read_pfm("dumps/e2e_nested_cpp.pfm")
    assert cpp.shape == ref_depth.shape, (cpp.shape, ref_depth.shape)

    absd = np.abs(cpp - ref_depth)
    rel = absd / (np.abs(ref_depth) + 1e-6)
    corr = np.corrcoef(cpp.ravel(), ref_depth.ravel())[0, 1]
    print(f"e2e nested depth: shape={ref_depth.shape} max|d|={absd.max():.3e} "
          f"mean|d|={absd.mean():.3e} median_rel={np.median(rel):.3e} corr={corr:.6f}")
    print(f"ref range [{ref_depth.min():.4f},{ref_depth.max():.4f}] "
          f"cpp range [{cpp.min():.4f},{cpp.max():.4f}]  ref scale={ref_scale:.6f}")

    ok = absd.max() < 5e-3 and corr > 0.999
    print("E2E_NESTED", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
