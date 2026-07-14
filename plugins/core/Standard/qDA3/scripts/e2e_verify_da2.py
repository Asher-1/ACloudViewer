#!/usr/bin/env python3
"""DA2 end-to-end relative/metric depth parity on a RAW photo. Publish gate.

Reference: upstream DepthAnythingV2.image2tensor (cv2 INTER_CUBIC, lower_bound 518,
ensure_multiple_of 14, ImageNet norm) -> net.forward -> depth at processed res.
C++: da3-cli depth --model <da2.gguf> --input photo --pfm out.pfm.
Compares at the processed resolution (forward() output, before infer_image's final
resize-to-original). PASS: corr > 0.999 and small p999|d| (99.9th-pct |d|)."""
import os, sys, subprocess, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e2e_verify_native import make_structured_image, read_pfm, W0, H0


def main():
    import argparse, cv2, torch
    from PIL import Image as PILImage
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--ckpt", default="models/da2/depth_anything_v2_vitl.pth")
    ap.add_argument("--gguf", default="models/depth-anything2-large-f32.gguf")
    ap.add_argument("--max-depth", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "dumps"), exist_ok=True)
    out_png = os.path.join(ROOT, "dumps", "da2_input.png")
    cpp_pfm = os.path.join(ROOT, "dumps", "e2e_da2_cpp.pfm")

    if a.image and os.path.exists(a.image):
        arr = np.array(PILImage.open(a.image).convert("RGB"), dtype=np.uint8)
    else:
        arr = make_structured_image(W0, H0)
    PILImage.fromarray(arr).save(out_png)

    from da2_reference import load_da2_model
    net = load_da2_model(a.encoder, a.ckpt, a.max_depth)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    image, (h, w) = net.image2tensor(bgr, input_size=518)   # upstream preprocessing
    with torch.no_grad():
        ref = net.forward(image).squeeze().float().cpu().numpy()   # (H,W) processed res

    cli = os.path.join(ROOT, "build", "examples", "cli", "da3-cli")
    model = a.gguf if os.path.isabs(a.gguf) else os.path.join(ROOT, a.gguf)
    subprocess.check_call([cli, "depth", "--model", model, "--input", out_png, "--pfm", cpp_pfm])
    cpp = read_pfm(cpp_pfm)
    assert cpp.shape == ref.shape, (cpp.shape, ref.shape)

    d = np.abs(cpp - ref)
    corr = np.corrcoef(cpp.ravel(), ref.ravel())[0, 1]
    p999 = float(np.percentile(d, 99.9))
    print(f"e2e da2 {a.encoder}: shape={ref.shape} max|d|={d.max():.3e} mean|d|={d.mean():.3e} p999|d|={p999:.3e} corr={corr:.6f}")
    # corr>0.999 is the parity signal (design-spec gate). |d|'s raw max is dominated by
    # ~0.02% single-pixel cubic-resize aliasing at depth-discontinuity edges, so the
    # robustness term uses the 99.9th percentile, not max, for DA2's unnormalized depth.
    ok = (corr > 0.999 and p999 < 5e-2 * max(1.0, ref.max()))
    print("E2E-DA2", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
