#!/usr/bin/env python3
"""End-to-end MULTI-VIEW verification: C++ da3 CLI (multi --input) vs the original DA3
PyTorch model on 2 structured 224x224 views. Both sides receive identical uint8 inputs;
the reference runs net(x_mv) with x_mv stacked as [1,2,3,224,224]."""
import os, sys, subprocess, json, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")
from da3_reference import load_model, fixed_input_multiview
from PIL import Image as PILImage

def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(w*h*4), dtype="<f4" if scale < 0 else ">f4").reshape(h, w)
        return np.flipud(data).copy()  # writer emits rows bottom-to-top

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/depth-anything-base-f32.gguf")
    ap.add_argument("--cli", default="build/examples/cli/da3-cli")
    a = ap.parse_args()
    os.makedirs("dumps", exist_ok=True)

    # Build 2 structured views (same fixture used by the dump/backbone gates).
    x_mv, raws = fixed_input_multiview(S=2, seed=0)   # x_mv (1,2,3,224,224); raws: 2x uint8 (224,224,3)
    paths = []
    for i, raw in enumerate(raws):
        p = f"dumps/e2e_mv_input{i}.png"
        PILImage.fromarray(raw).save(p)  # lossless
        paths.append(p)

    # Reference: original net over the stacked multi-view input.
    _, net = load_model()
    with torch.no_grad():
        full = net(x_mv)
    ref_depth = full["depth"].reshape(2, 224, 224).float().cpu().numpy()
    ref_ext = full["extrinsics"].reshape(2, 3, 4).float().cpu().numpy()
    ref_intr = full["intrinsics"].reshape(2, 3, 3).float().cpu().numpy()

    # C++ CLI multi-view.
    cmd = [a.cli, "depth", "--model", a.model,
           "--input", paths[0], "--input", paths[1], "--out-prefix", "dumps/e2e_mv"]
    subprocess.check_call(cmd)

    ok = True
    lines = []
    for v in range(2):
        cpp_depth = read_pfm(f"dumps/e2e_mv_view{v}.pfm")
        with open(f"dumps/e2e_mv_view{v}.json") as f:
            pj = json.load(f)
        cpp_ext = np.array(pj["extrinsics"], dtype=np.float32)   # (3,4)
        cpp_intr = np.array(pj["intrinsics"], dtype=np.float32)  # (3,3)
        assert cpp_depth.shape == ref_depth[v].shape, (cpp_depth.shape, ref_depth[v].shape)
        d_abs = np.abs(cpp_depth - ref_depth[v])
        corr = np.corrcoef(cpp_depth.ravel(), ref_depth[v].ravel())[0, 1]
        ext_d = np.abs(cpp_ext - ref_ext[v])
        intr_d = np.abs(cpp_intr - ref_intr[v])
        lines.append(f"e2e mv: view{v} depth max|d|={d_abs.max():.3e} mean|d|={d_abs.mean():.3e} "
                     f"corr={corr:.6f} / pose ext max|d|={ext_d.max():.3e} intr max|d|={intr_d.max():.3e}")
        ok &= (d_abs.max() < 5e-3) and (corr > 0.999) and (ext_d.max() < 1e-2)
    for l in lines:
        print(l)
    print("E2E_MV", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
