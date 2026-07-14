#!/usr/bin/env python3
"""Native-resolution end-to-end depth verification (M9-T2).

Verifies the C++ `da3-cli depth` (which now runs the REAL DA3 upper_bound_resize
policy via Engine::depth_native) against the genuine reference forward on a RAW
arbitrary-resolution real photo.

Reference path (NO torchvision binary required):
  - preprocess with the GENUINE upstream `InputProcessor`
    (process_res=504, process_res_method="upper_bound_resize"); only the
    ToTensor/Normalize transforms are stubbed with their exact math equivalents
    because the installed torchvision wheel is ABI-broken (0.27 vs torch 2.12).
    The actual resize policy (cv2 INTER_CUBIC/INTER_AREA + round-to-14) is the
    real upstream code, identical to what model.inference() would feed the net.
  - net.backbone.pretrained.get_intermediate_layers(..., n=[5,7,9,11])
  - net.head(feats, H, W, patch_start_idx=0) -> depth   (same call model() makes)

C++ path: build/examples/cli/da3-cli depth --input photo.png --pfm out.pfm
  (native resolution by default; --legacy-resize would force the old floor path)

PASS: corr > 0.999 and max|d| small (resize is bit-exact + forward is f32-parity).

Also dumps dumps/reference_native.gguf (native_depth + out_h/out_w) and
dumps/native_input.png for the C++ gate tests/test_engine_depth_native.cpp.
"""
import os, sys, types, subprocess, argparse, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DA3_SRC = "/tmp/da3-src/src"

OUT_PNG   = os.path.join(ROOT, "dumps", "native_input.png")
OUT_GGUF  = os.path.join(ROOT, "dumps", "reference_native.gguf")
CPP_PFM   = os.path.join(ROOT, "dumps", "e2e_native_cpp.pfm")
PROCESS_RES = 504
METHOD = "upper_bound_resize"
W0, H0 = 640, 427          # non-square, neither dim a multiple of 14


def make_structured_image(w, h, seed=20240615):
    """Deterministic structured non-square RGB photo-like content: smooth depth-ish
    gradients + geometric shapes + high-frequency texture to stress the resampler."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # large smooth gradients (sky/ground feel)
    r = (xx / (w - 1) * 200.0 + 30.0)
    g = (yy / (h - 1) * 180.0 + 40.0)
    b = (((np.sin(xx * 0.02) + np.cos(yy * 0.018)) * 0.5 + 0.5) * 180.0 + 30.0)
    img = np.stack([r, g, b], axis=-1)
    # high-frequency checkerboard texture
    check = (((xx.astype(int) // 9) + (yy.astype(int) // 9)) % 2) * 35.0
    img += check[..., None]
    # geometric shapes (foreground objects)
    for (cx, cy, rad, col) in [(150, 130, 80, (230, 70, 60)),
                               (480, 300, 95, (50, 200, 110)),
                               (330, 210, 55, (60, 80, 235)),
                               (560, 90, 45, (240, 220, 60))]:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
        for c in range(3):
            img[..., c][m] = col[c]
    # a couple of rectangles
    img[300:380, 60:200, :] = np.array([120, 90, 200], np.float32)
    img[40:110, 250:430, :] = np.array([200, 200, 90], np.float32)
    img += rng.integers(0, 10, size=img.shape).astype(np.float32)  # mild deterministic grain
    return np.clip(img, 0, 255).astype(np.uint8)


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


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(w * h * 4), dtype="<f4" if scale < 0 else ">f4").reshape(h, w)
        return np.flipud(data).copy()   # writer emits rows bottom-to-top


def main():
    import torch
    from PIL import Image as PILImage
    os.makedirs(os.path.join(ROOT, "dumps"), exist_ok=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", default=None,
                    help="optional input image (else a deterministic structured photo)")
    ap.add_argument("--model-dir", default="models/DA3-BASE",
                    help="reference checkpoint dir passed to load_model()")
    ap.add_argument("--gguf", default=os.path.join("models", "depth-anything-base-f32.gguf"),
                    help="C++ gguf model for da3-cli depth")
    args = ap.parse_args()

    # 1) Real arbitrary-resolution non-square photo (deterministic structured fallback).
    img_path = args.image if args.image and os.path.exists(args.image) else None
    if img_path:
        arr = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)
    else:
        arr = make_structured_image(W0, H0)
    PILImage.fromarray(arr, "RGB").save(OUT_PNG)   # lossless PNG, both sides read identical pixels
    h0, w0 = arr.shape[:2]
    print(f"input photo: {w0}x{h0} -> {OUT_PNG}")

    # 2) Reference: genuine InputProcessor resize -> backbone -> head.
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
    bb = net.backbone.pretrained
    # Use the checkpoint's genuine out_layers (config-driven), not a hardcoded list:
    # base/small use [5,7,9,11], large uses [11,15,19,23], giant [19,27,33,39].
    out_layers = list(net.backbone.out_layers)
    x = t[None, None].contiguous()   # (1,1,3,H,W)
    with torch.no_grad():
        outs, _ = bb.get_intermediate_layers(
            x, n=out_layers, export_feat_layers=[], ref_view_strategy="saddle_balanced")
        ho = net.head(list(outs), H, W, patch_start_idx=0)
    ref = ho["depth"].squeeze().float().cpu().numpy()   # (H,W)
    assert ref.shape == (H, W), ref.shape

    # 3) C++ native CLI.
    cli = os.path.join(ROOT, "build", "examples", "cli", "da3-cli")
    model = args.gguf if os.path.isabs(args.gguf) else os.path.join(ROOT, args.gguf)
    subprocess.check_call([cli, "depth", "--model", model, "--input", OUT_PNG, "--pfm", CPP_PFM])
    cpp = read_pfm(CPP_PFM)
    assert cpp.shape == ref.shape, (cpp.shape, ref.shape)

    # 4) Compare.
    absd = np.abs(cpp - ref)
    corr = np.corrcoef(cpp.ravel(), ref.ravel())[0, 1]
    print(f"e2e native depth: shape={ref.shape} max|d|={absd.max():.3e} "
          f"mean|d|={absd.mean():.3e} corr={corr:.6f}")
    print(f"ref range [{ref.min():.4f},{ref.max():.4f}] cpp range [{cpp.min():.4f},{cpp.max():.4f}]")

    # 5) Dump reference native depth for the C++ gate.
    import gguf
    gw = gguf.GGUFWriter(OUT_GGUF, "native_depth")
    gw.add_uint32("native.out_h", int(H))
    gw.add_uint32("native.out_w", int(W))
    gw.add_tensor("native_depth", np.ascontiguousarray(ref.reshape(-1).astype(np.float32)))
    gw.write_header_to_file(); gw.write_kv_data_to_file(); gw.write_tensors_to_file(); gw.close()
    print(f"wrote {OUT_GGUF} (native_depth {ref.size} f32) + {OUT_PNG}")

    ok = corr > 0.999 and absd.max() < 5e-2
    print("E2E-NATIVE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
