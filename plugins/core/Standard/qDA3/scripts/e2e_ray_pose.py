#!/usr/bin/env python3
"""End-to-end verification of the RAY->POSE solver (use_ray_pose part B).

Compares the genuine reference `net(x, use_ray_pose=True)` extrinsics/intrinsics against
the C++ `da3-cli depth --ray-pose` on the SAME real, native-resolution (504) image.

TOLERANCE IS LOOSE BY NATURE (and documented):
  The reference RANSAC (ransac_find_homography_weighted_fast_batch) samples candidate
  point groups via torch.randperm AND randomly subsamples the consensus inlier set
  (>8000 -> 8000) -> the reference pose itself is nondeterministic across runs. The C++
  production path uses its OWN seeded deterministic sampling + subsample (a different but
  equally valid consensus). So the two solvers consume DIFFERENT point subsets and only
  agree up to the RANSAC consensus variation (plus f32-torch vs f64-host SVD/QR). The
  RIGOROUS bit-tight parity (fed identical indices) is proven separately by the T3 gate
  (tests/test_ray_pose.cpp, rotation max|d|~2e-7). Here we assert the e2e production
  pose is CLOSE:
      rotation geodesic angle < 1.0 deg
      focal (fx,fy) relative error < 2%
      principal point relative error < 2%
"""
import os, sys, types, json, subprocess, argparse, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DA3_SRC = "/tmp/da3-src/src"

OUT_PNG = os.path.join(ROOT, "dumps", "e2e_ray_pose_input.png")
PROCESS_RES = 504
METHOD = "upper_bound_resize"
SEED = 1234


def install_torchvision_stub():
    import torch
    tv = types.ModuleType("torchvision")
    tvt = types.ModuleType("torchvision.transforms")

    class ToTensor:
        def __call__(self, img):
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
            return t.permute(2, 0, 1).contiguous()

    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
        def __call__(self, t):
            return (t - self.mean) / self.std

    class CenterCrop:
        def __init__(self, size): self.size = size
        def __call__(self, t):
            H, W = t.shape[-2:]; th, tw = self.size
            top = max(0, (H - th) // 2); left = max(0, (W - tw) // 2)
            return t[..., top:top + th, left:left + tw]

    tvt.ToTensor = ToTensor; tvt.Normalize = Normalize; tvt.CenterCrop = CenterCrop
    tv.transforms = tvt
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = tvt


def make_structured_image(w, h, seed=20240615):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (xx / (w - 1) * 200.0 + 30.0)
    g = (yy / (h - 1) * 180.0 + 40.0)
    b = (((np.sin(xx * 0.02) + np.cos(yy * 0.018)) * 0.5 + 0.5) * 180.0 + 30.0)
    img = np.stack([r, g, b], axis=-1)
    check = (((xx.astype(int) // 9) + (yy.astype(int) // 9)) % 2) * 35.0
    img += check[..., None]
    for (cx, cy, rad, col) in [(150, 130, 80, (230, 70, 60)),
                               (480, 300, 95, (50, 200, 110)),
                               (330, 210, 55, (60, 80, 235)),
                               (560, 90, 45, (240, 220, 60))]:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
        for c in range(3):
            img[..., c][m] = col[c]
    img[300:380, 60:200, :] = np.array([120, 90, 200], np.float32)
    img[40:110, 250:430, :] = np.array([200, 200, 90], np.float32)
    img += rng.integers(0, 10, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def geodesic_deg(Ra, Rb):
    M = Ra.T @ Rb
    c = (np.trace(M) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    import torch
    from PIL import Image as PILImage
    os.makedirs(os.path.join(ROOT, "dumps"), exist_ok=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--model-dir", default="models/DA3-BASE")
    ap.add_argument("--gguf", default=os.path.join("models", "depth-anything-base-aux-f32.gguf"))
    args = ap.parse_args()

    if args.image and os.path.exists(args.image):
        arr = np.array(PILImage.open(args.image).convert("RGB"), dtype=np.uint8)
    else:
        arr = make_structured_image(640, 427)
    PILImage.fromarray(arr, "RGB").save(OUT_PNG)
    print(f"input photo: {arr.shape[1]}x{arr.shape[0]} -> {OUT_PNG}")

    install_torchvision_stub()
    from da3_reference import load_model
    sys.path.insert(0, DA3_SRC)
    from depth_anything_3.utils.io.input_processor import InputProcessor

    proc = InputProcessor()
    tensor, _, _ = proc([OUT_PNG], process_res=PROCESS_RES, process_res_method=METHOD)
    t = tensor.reshape(-1, 3, tensor.shape[-2], tensor.shape[-1])[0]
    C, H, W = t.shape
    print(f"processed resolution: {W}x{H}")

    _, net = load_model(args.model_dir)
    x = t[None, None].contiguous()
    with torch.no_grad():
        torch.manual_seed(SEED)
        out = net(x, use_ray_pose=True)
    ext_ref = out.extrinsics.reshape(3, 4).float().cpu().numpy()   # c2w 3x4
    K_ref = out.intrinsics.reshape(3, 3).float().cpu().numpy()
    R_ref = ext_ref[:3, :3]
    print("reference ext(c2w)=\n", np.round(ext_ref, 6))
    print("reference K=\n", np.round(K_ref, 4))

    # C++ ray-pose
    cli = os.path.join(ROOT, "build", "examples", "cli", "da3-cli")
    model = args.gguf if os.path.isabs(args.gguf) else os.path.join(ROOT, args.gguf)
    pose_json = os.path.join(ROOT, "dumps", "e2e_ray_pose_cpp.json")
    subprocess.check_call([cli, "depth", "--model", model, "--input", OUT_PNG,
                           "--pose", pose_json, "--ray-pose"])
    with open(pose_json) as f:
        pj = json.load(f)
    ext_cpp = np.array(pj["extrinsics"], dtype=np.float64)  # 3x4
    K_cpp = np.array(pj["intrinsics"], dtype=np.float64)
    R_cpp = ext_cpp[:3, :3]
    print("C++ ext(c2w)=\n", np.round(ext_cpp, 6))
    print("C++ K=\n", np.round(K_cpp, 4))

    geo = geodesic_deg(R_ref, R_cpp)
    fx_rel = abs(K_cpp[0, 0] - K_ref[0, 0]) / abs(K_ref[0, 0])
    fy_rel = abs(K_cpp[1, 1] - K_ref[1, 1]) / abs(K_ref[1, 1])
    cx_rel = abs(K_cpp[0, 2] - K_ref[0, 2]) / max(abs(K_ref[0, 2]), 1.0)
    cy_rel = abs(K_cpp[1, 2] - K_ref[1, 2]) / max(abs(K_ref[1, 2]), 1.0)
    t_abs = float(np.max(np.abs(ext_cpp[:3, 3] - ext_ref[:3, 3])))
    print(f"\n[e2e ray-pose] rotation geodesic={geo:.4f} deg")
    print(f"               focal rel: fx={fx_rel*100:.3f}% fy={fy_rel*100:.3f}%")
    print(f"               pp rel:    cx={cx_rel*100:.3f}% cy={cy_rel*100:.3f}%")
    print(f"               translation max|d|={t_abs:.4e}")

    ok = (geo < 1.0) and (fx_rel < 0.02) and (fy_rel < 0.02) and (cx_rel < 0.02) and (cy_rel < 0.02)
    print("E2E-RAY-POSE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
