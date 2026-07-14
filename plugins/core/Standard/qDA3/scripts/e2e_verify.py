#!/usr/bin/env python3
"""End-to-end depth verification: C++ da3 CLI vs the original DA3 PyTorch model on a
real image, on the verified 224x224 path. Both sides receive identical 224x224 uint8 input."""
import os, sys, subprocess, struct, json, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")
from da3_reference import load_model
from PIL import Image as PILImage

def get_image(path_or_none):
    if path_or_none and os.path.exists(path_or_none):
        img = PILImage.open(path_or_none).convert("RGB").resize((224,224), PILImage.BICUBIC)
        return np.array(img, dtype=np.uint8)
    # Deterministic STRUCTURED fallback (not noise): smooth gradients + shapes so depth has real structure.
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    r = (np.sin(xx/30.0)*0.5+0.5)
    g = (np.cos(yy/40.0)*0.5+0.5)
    b = ((xx+yy)/(2*224))
    arr = np.stack([r,g,b], -1)
    # add a couple of blocks
    arr[60:120, 60:120, :] = 0.9
    arr[140:200, 30:90, :] = 0.2
    return (arr*255).astype(np.uint8)

def ref_depth(uint8_img):
    _, net = load_model()
    bb = net.backbone.pretrained
    mean = np.array([0.485,0.456,0.406], np.float32); std = np.array([0.229,0.224,0.225], np.float32)
    x = (uint8_img.astype(np.float32)/255.0 - mean)/std
    t = torch.from_numpy(x).permute(2,0,1)[None,None].contiguous()
    with torch.no_grad():
        outs,_ = bb.get_intermediate_layers(t, n=[5,7,9,11], export_feat_layers=[], ref_view_strategy="saddle_balanced")
        ho = net.head(list(outs), 224, 224, patch_start_idx=0)
    return ho["depth"].squeeze().float().cpu().numpy()  # (224,224)

def ref_pose(uint8_img):
    # Reference camera pose via the FULL default forward (cam_dec path), same
    # 224x224 uint8 image as ref_depth. Returns (extrinsics 3x4, intrinsics 3x3).
    _, net = load_model()
    mean = np.array([0.485,0.456,0.406], np.float32); std = np.array([0.229,0.224,0.225], np.float32)
    x = (uint8_img.astype(np.float32)/255.0 - mean)/std
    t = torch.from_numpy(x).permute(2,0,1)[None,None].contiguous()
    with torch.no_grad():
        full = net(t)
    ext = full["extrinsics"].reshape(3,4).float().cpu().numpy()
    intr = full["intrinsics"].reshape(3,3).float().cpu().numpy()
    return ext, intr

def read_pfm(path):
    with open(path,"rb") as f:
        assert f.readline().strip()==b"Pf"
        w,h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(w*h*4), dtype="<f4" if scale<0 else ">f4").reshape(h,w)
        # writer emitted rows bottom-to-top (row H-1 first); flip back to top-to-bottom
        return np.flipud(data).copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", default="models/depth-anything-base-f32.gguf")
    ap.add_argument("--cli", default="build/examples/cli/da3-cli")
    a = ap.parse_args()
    os.makedirs("dumps", exist_ok=True)
    uint8 = get_image(a.image)
    PILImage.fromarray(uint8).save("dumps/e2e_input.png")  # lossless
    ref = ref_depth(uint8)
    subprocess.check_call([a.cli, "depth", "--model", a.model, "--input", "dumps/e2e_input.png",
                           "--pfm", "dumps/e2e_cpp.pfm"])
    cpp = read_pfm("dumps/e2e_cpp.pfm")
    assert cpp.shape == ref.shape, (cpp.shape, ref.shape)
    absd = np.abs(cpp-ref)
    rel = absd/ (np.abs(ref)+1e-6)
    corr = np.corrcoef(cpp.ravel(), ref.ravel())[0,1]
    print(f"e2e depth: shape={ref.shape} max|d|={absd.max():.3e} mean|d|={absd.mean():.3e} "
          f"median_rel={np.median(rel):.3e} corr={corr:.6f}")
    print(f"ref range [{ref.min():.4f},{ref.max():.4f}] cpp range [{cpp.min():.4f},{cpp.max():.4f}]")

    # --- pose e2e: C++ CLI --pose vs original net(img) extrinsics/intrinsics ---
    ref_ext, ref_intr = ref_pose(uint8)
    subprocess.check_call([a.cli, "depth", "--model", a.model, "--input", "dumps/e2e_input.png",
                           "--pose", "dumps/e2e_pose.json"])
    with open("dumps/e2e_pose.json") as f:
        pj = json.load(f)
    cpp_ext = np.array(pj["extrinsics"], dtype=np.float32)   # (3,4)
    cpp_intr = np.array(pj["intrinsics"], dtype=np.float32)  # (3,3)
    ext_d = np.abs(cpp_ext - ref_ext)
    intr_d = np.abs(cpp_intr - ref_intr)
    print(f"e2e pose: ext max|d|={ext_d.max():.3e} intr max|d|={intr_d.max():.3e}")

    # PASS criterion: near-exact on the verified path (depth AND pose)
    pose_ok = ext_d.max() < 1e-2
    ok = absd.max() < 5e-3 and corr > 0.999 and pose_ok
    print("E2E", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
