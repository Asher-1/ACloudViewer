#!/usr/bin/env python3
"""End-to-end 3D-Gaussian verification: the C++ `da3 reconstruct` CLI vs the
original DA3-GIANT PyTorch model `net(x, infer_gs=True)` on an identical 224x224
uint8 image. Compares the world-space gaussian attributes parsed from the C++
.ply against the reference adapter outputs.

The C++ side runs the full GIANT pipeline (backbone -> depth + cam pose -> GSDPT
raw_gs -> GaussianAdapter), so this exercises the WHOLE chain, not just the
adapter (which tests/test_gs_adapter.cpp already gates at 2e-3 from dumped
inputs). A single giant CPU forward is slow (minutes) on each side -- expected.

Skips (exit 0 with a notice) if the giant gguf / DA3-GIANT checkpoint is absent.
"""
import os, sys, subprocess, struct, argparse
import numpy as np
# Pre-import e3nn (used by the reference gs_adapter's rotate_sh) BEFORE
# da3_reference.load_model installs its torchvision stub. e3nn drags in
# torch._dynamo, whose registration walks sys.modules via `inspect`; the stub
# answers any attribute with a dummy class and crashes that walk. Importing it
# first lets the dynamo chain complete against the real module table. (Same
# caveat as scripts/dump_giant.py.)
try:
    import e3nn.o3  # noqa: F401
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

GGUF = os.path.join(ROOT, "models", "depth-anything-giant-f32.gguf")
MODEL_DIR = os.path.join(ROOT, "models", "DA3-GIANT")
CLI = os.path.join(ROOT, "build", "examples", "cli", "da3-cli")


def structured_image():
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    r = np.sin(xx / 30.0) * 0.5 + 0.5
    g = np.cos(yy / 40.0) * 0.5 + 0.5
    b = (xx + yy) / (2 * 224)
    arr = np.stack([r, g, b], -1)
    arr[60:120, 60:120, :] = 0.9
    arr[140:200, 30:90, :] = 0.2
    return (arr * 255).astype(np.uint8)


def parse_ply(path):
    """Parse the standard 3DGS binary-LE ply written by ply_export.cpp."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        assert f.readline().strip() == b"format binary_little_endian 1.0"
        props, n = [], 0
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                props.append(line.split()[-1].decode())
            elif line == b"end_header":
                break
        rec = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))
    cols = {name: rec[:, i] for i, name in enumerate(props)}
    means = np.stack([cols["x"], cols["y"], cols["z"]], -1)
    f_dc = np.stack([cols["f_dc_0"], cols["f_dc_1"], cols["f_dc_2"]], -1)
    opacity = 1.0 / (1.0 + np.exp(-cols["opacity"]))            # undo logit
    scales = np.exp(np.stack([cols["scale_0"], cols["scale_1"], cols["scale_2"]], -1))
    rot = np.stack([cols["rot_0"], cols["rot_1"], cols["rot_2"], cols["rot_3"]], -1)
    return means, f_dc, opacity, scales, rot


def ref_gaussians(uint8_img):
    from da3_reference import load_model
    import torch
    _, net = load_model(MODEL_DIR)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (uint8_img.astype(np.float32) / 255.0 - mean) / std
    t = torch.from_numpy(x).permute(2, 0, 1)[None, None].contiguous()
    with torch.no_grad():
        full = net(t, infer_gs=True)
    g = full["gaussians"]
    return {
        "means": g.means.squeeze(0).float().cpu().numpy(),
        "f_dc": g.harmonics.squeeze(0)[..., 0].float().cpu().numpy(),
        "opacity": g.opacities.squeeze(0).reshape(-1).float().cpu().numpy(),
        "scales": g.scales.squeeze(0).float().cpu().numpy(),
        "rot": g.rotations.squeeze(0).float().cpu().numpy(),
    }


def report(name, got, ref, rtol=2e-3, atol=2e-3):
    d = np.abs(got - ref)
    tol = atol + rtol * np.abs(ref)
    nviol = int((d > tol).sum())
    print(f"  {name:9s} max|d|={d.max():.3e} mean|d|={d.mean():.3e} viol={nviol}/{d.size}")
    return d.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if not (os.path.exists(GGUF) and os.path.isdir(MODEL_DIR) and os.path.exists(CLI)):
        print("e2e_verify_gs: giant gguf / DA3-GIANT / da3-cli missing -> SKIP")
        return 0

    from PIL import Image as PILImage
    if args.image and os.path.exists(args.image):
        uint8 = np.array(PILImage.open(args.image).convert("RGB").resize((224, 224), PILImage.BICUBIC), dtype=np.uint8)
    else:
        uint8 = structured_image()

    os.makedirs(os.path.join(ROOT, "dumps"), exist_ok=True)
    png = os.path.join(ROOT, "dumps", "e2e_gs_input.png")
    ply = os.path.join(ROOT, "dumps", "e2e_gs_cpp.ply")
    PILImage.fromarray(uint8).save(png)

    print("running C++ da3 reconstruct (giant; slow) ...")
    subprocess.run([CLI, "reconstruct", "--model", GGUF, "--input", png, "--ply", ply], check=True)
    c_means, c_fdc, c_op, c_scales, c_rot = parse_ply(ply)

    print("running reference net(infer_gs=True) (giant; slow) ...")
    ref = ref_gaussians(uint8)

    print("e2e gaussian parity (C++ full pipeline vs reference):")
    mx = []
    mx.append(report("means", c_means, ref["means"]))
    mx.append(report("scales", c_scales, ref["scales"]))
    mx.append(report("f_dc", c_fdc, ref["f_dc"]))
    mx.append(report("opacity", c_op, ref["opacity"]))
    # quaternion: account for the +/- double cover sign ambiguity.
    drot = np.minimum(np.abs(c_rot - ref["rot"]).max(-1), np.abs(c_rot + ref["rot"]).max(-1))
    print(f"  {'rot':9s} max|d|(sign-aware)={drot.max():.3e} mean={drot.mean():.3e}")

    if not args.keep:
        for p in (png, ply):
            try: os.remove(p)
            except OSError: pass
    print("e2e_verify_gs: DONE (full-pipeline gaussian comparison printed above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
