#!/usr/bin/env python3
"""Dump a REAL-image preprocessing reference for the DA3 upper_bound_resize policy.

Generates a deterministic structured non-square image (640x427), saves it as a
lossless PNG (so the C++ stb decode and the Python PIL decode see identical
pixels), runs the GENUINE upstream `InputProcessor` (the cv2 INTER_AREA /
INTER_CUBIC resize code is exercised unmodified), and writes the processed
tensor + output size to a gguf the C++ gate asserts against.

torchvision is stubbed with exact ToTensor/Normalize equivalents (these ops are
mathematically unambiguous) only because the installed torchvision binary is
incompatible with this torch build; the resize path - the actual parity target -
is the real upstream cv2 code.
"""
import os, sys, types
import numpy as np

DA3_SRC = "/tmp/da3-src/src"
OUT_GGUF = "dumps/reference_preproc_real.gguf"
OUT_PNG  = "dumps/preproc_real_input.png"
W0, H0   = 640, 427          # non-square, neither dim a multiple of 14
PROCESS_RES = 504
METHOD = "upper_bound_resize"


def make_structured_image(w, h, seed=1234):
    """Deterministic high-frequency RGB content to stress the resampler."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (xx / (w - 1) * 255.0)
    g = (yy / (h - 1) * 255.0)
    b = (((np.sin(xx * 0.12) + np.cos(yy * 0.15)) * 0.5 + 0.5) * 255.0)
    img = np.stack([r, g, b], axis=-1)
    # checkerboard high-frequency overlay
    check = (((xx.astype(int) // 7) + (yy.astype(int) // 7)) % 2) * 60.0
    img += check[..., None]
    # a couple of bright disks
    for (cx, cy, rad, col) in [(160, 120, 70, (255, 40, 40)),
                               (470, 300, 90, (30, 220, 90)),
                               (320, 213, 50, (40, 60, 240))]:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
        for c in range(3):
            img[..., c][m] = col[c]
    img += rng.integers(0, 12, size=img.shape).astype(np.float32)  # mild deterministic noise
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


def main():
    os.makedirs("dumps", exist_ok=True)
    import cv2
    from PIL import Image
    install_torchvision_stub()
    sys.path.insert(0, DA3_SRC)
    from depth_anything_3.utils.io.input_processor import InputProcessor

    arr = make_structured_image(W0, H0)
    Image.fromarray(arr, "RGB").save(OUT_PNG)            # lossless PNG
    print(f"wrote {OUT_PNG}  ({W0}x{H0})")

    # Sanity: report the two resize steps for transparency.
    longest = max(W0, H0); scale = PROCESS_RES / longest
    nw, nh = round(W0 * scale), round(H0 * scale)
    print(f"step1 longest-side: {W0}x{H0} -> {nw}x{nh}  scale={scale:.4f}  "
          f"interp={'CUBIC' if scale > 1 else 'AREA'}")
    def nm(x, p=14):
        d = (x // p) * p; u = d + p
        return u if abs(u - x) <= abs(x - d) else d
    fw, fh = max(1, nm(nw)), max(1, nm(nh))
    if (fw, fh) != (nw, nh):
        print(f"step2 div14:        {nw}x{nh} -> {fw}x{fh}  "
              f"interp={'CUBIC' if (fw > nw or fh > nh) else 'AREA'}")
    else:
        print(f"step2 div14:        {nw}x{nh} already divisible by 14 (no-op)")

    proc = InputProcessor()
    tensor, _, _ = proc([OUT_PNG], process_res=PROCESS_RES, process_res_method=METHOD)
    # tensor: (N,3,H,W) for N=1 image -> take the single (3,H,W).
    t = tensor.reshape(-1, 3, tensor.shape[-2], tensor.shape[-1])[0]
    C, H, W = t.shape
    assert (C, H, W) == (3, fh, fw), (t.shape, (3, fh, fw))
    print(f"reference processed tensor: (3,{H},{W})")

    import gguf
    w = gguf.GGUFWriter(OUT_GGUF, "preproc_real")
    w.add_uint32("preproc.out_h", int(H))
    w.add_uint32("preproc.out_w", int(W))
    w.add_tensor("proc_image", np.ascontiguousarray(t.cpu().numpy().reshape(-1).astype(np.float32)))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {OUT_GGUF}  (proc_image flat {t.numel()} f32)")


if __name__ == "__main__":
    main()
