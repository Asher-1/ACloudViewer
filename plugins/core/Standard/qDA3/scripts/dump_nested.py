#!/usr/bin/env python3
"""Dump gold reference tensors for DA3NESTED-GIANT-LARGE (N=1), the FINAL milestone.

Runs three CPU forwards on the fixed 224x224 input (VERY SLOW -- giant+large):
  1. net.da3(x)         -> raw anyview (giant) depth/conf/extrinsics/intrinsics
                           (pre metric-scaling/alignment).
  2. net.da3_metric(x)  -> metric branch depth (post internal sky-fill) + sky,
                           plus the pure DPT-head depth/sky (via hook) for the
                           head-isolation parity gate, plus the metric backbone
                           out-layer feats.
  3. net(x)             -> the FINAL nested output: aligned metric-scale depth,
                           scale_factor, metric extrinsics/intrinsics, is_metric.

Produces dumps/reference_nested.gguf (flattened f32, row-major) + dumps/
manifest_nested.json. Gold reference for the C++ M6 parity gates (M6-T2/T3).
"""
import os, json, sys, numpy as np, torch
# See dump_giant.py: pre-import e3nn.o3 before load_model installs the torchvision stub.
import e3nn.o3  # noqa: F401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
from scripts.da3_reference import load_model, fixed_input, FIX_H, FIX_W, PATCH

MODEL = "models/DA3NESTED-GIANT-LARGE"
OUT = "dumps/reference_nested.gguf"
MANIFEST = "dumps/manifest_nested.json"
METRIC_OUT_LAYERS = [4, 11, 17, 23]


def flat(t):
    return t.detach().reshape(-1).float().contiguous()


def main():
    os.makedirs("dumps", exist_ok=True)
    _, net = load_model(MODEL)
    assert hasattr(net, "da3") and hasattr(net, "da3_metric"), type(net).__name__
    bb_m = net.da3_metric.backbone.pretrained
    assert len(bb_m.blocks) == 24, len(bb_m.blocks)
    assert bb_m.embed_dim == 1024, bb_m.embed_dim
    assert not hasattr(bb_m.blocks[0].mlp, "w12"), "metric ViT-L expected classic MLP fc1/fc2"

    x, raw = fixed_input()
    cap, shapes = {}, {}

    # ---- 1) raw anyview (giant) branch --------------------------------------
    print("[1/3] anyview (giant) forward (SLOW)...", flush=True)
    with torch.no_grad():
        any_out = net.da3(x)
    depth_any = any_out["depth"].squeeze()
    assert tuple(depth_any.shape) == (FIX_H, FIX_W), depth_any.shape
    cap["depth_any"] = flat(depth_any);                 shapes["depth_any"] = [FIX_H, FIX_W]
    cap["depth_conf_any"] = flat(any_out["depth_conf"]); shapes["depth_conf_any"] = [FIX_H, FIX_W]
    cap["extrinsics_any"] = flat(any_out["extrinsics"]); shapes["extrinsics_any"] = list(any_out["extrinsics"].squeeze().shape)
    cap["intrinsics_any"] = flat(any_out["intrinsics"]); shapes["intrinsics_any"] = list(any_out["intrinsics"].squeeze().shape)

    # ---- 2) metric branch: hook backbone feats + pure DPT head --------------
    print("[2/3] metric (large) forward (SLOW)...", flush=True)
    feat_cap, head_cap = {}, {}

    def bb_hook(_m, _inp, out):
        feats = out[0] if isinstance(out, (tuple, list)) else out
        for L, entry in zip(METRIC_OUT_LAYERS, feats):
            t = entry[0] if isinstance(entry, (tuple, list)) else entry
            feat_cap[L] = t.detach().clone()

    def head_hook(_m, _inp, out):
        head_cap["depth"] = out["depth"].detach().clone()
        if "sky" in out:
            head_cap["sky"] = out["sky"].detach().clone()

    h1 = net.da3_metric.backbone.register_forward_hook(bb_hook)
    h2 = net.da3_metric.head.register_forward_hook(head_hook)
    try:
        with torch.no_grad():
            metric_out = net.da3_metric(x)
    finally:
        h1.remove(); h2.remove()

    depth_metric_raw = metric_out["depth"].squeeze()  # post internal sky-fill
    sky = metric_out["sky"].squeeze()
    assert tuple(depth_metric_raw.shape) == (FIX_H, FIX_W), depth_metric_raw.shape
    cap["depth_metric_raw"] = flat(depth_metric_raw); shapes["depth_metric_raw"] = [FIX_H, FIX_W]
    cap["sky"] = flat(sky);                           shapes["sky"] = [FIX_H, FIX_W]
    if "depth_conf" in metric_out:
        cap["depth_conf_metric"] = flat(metric_out["depth_conf"]); shapes["depth_conf_metric"] = [FIX_H, FIX_W]
    # pure DPT-head depth (pre internal sky-fill) for the head-isolation gate
    dh = head_cap["depth"].squeeze()
    cap["depth_metric_head"] = flat(dh); shapes["depth_metric_head"] = [FIX_H, FIX_W]
    if "sky" in head_cap:
        cap["sky_head"] = flat(head_cap["sky"].squeeze()); shapes["sky_head"] = [FIX_H, FIX_W]
    # metric backbone out-layer feats
    for L in METRIC_OUT_LAYERS:
        t = feat_cap[L]
        cap[f"feat_m_{L}"] = flat(t)
        shapes[f"feat_m_{L}"] = list(t.shape)

    # ---- 3) FINAL nested forward --------------------------------------------
    print("[3/3] FINAL nested forward (SLOWEST: giant + large)...", flush=True)
    with torch.no_grad():
        full = net(x)
    depth_final = full["depth"].squeeze()
    assert tuple(depth_final.shape) == (FIX_H, FIX_W), depth_final.shape
    cap["depth_final"] = flat(depth_final);                shapes["depth_final"] = [FIX_H, FIX_W]
    cap["depth_conf_final"] = flat(full["depth_conf"]);    shapes["depth_conf_final"] = [FIX_H, FIX_W]
    cap["extrinsics_final"] = flat(full["extrinsics"]);    shapes["extrinsics_final"] = list(full["extrinsics"].squeeze().shape)
    cap["intrinsics_final"] = flat(full["intrinsics"]);    shapes["intrinsics_final"] = list(full["intrinsics"].squeeze().shape)
    scale_factor = float(full["scale_factor"])
    is_metric = int(full.get("is_metric", 0))
    cap["scale_factor"] = torch.tensor([scale_factor], dtype=torch.float32); shapes["scale_factor"] = [1]

    cap["input_image"] = flat(x);                                          shapes["input_image"] = list(x.shape)
    cap["raw_image"] = torch.from_numpy(raw.astype(np.float32)).reshape(-1); shapes["raw_image"] = [FIX_H, FIX_W, 3]

    # ---- verification --------------------------------------------------------
    assert torch.isfinite(depth_final).all() and bool((depth_final > 0).all()), "depth_final must be positive/finite"
    assert np.isfinite(scale_factor) and scale_factor > 0, f"bad scale_factor {scale_factor}"
    assert torch.isfinite(depth_metric_raw).all(), "depth_metric_raw non-finite"
    assert is_metric == 1, f"expected is_metric=1, got {is_metric}"

    w = gguf.GGUFWriter(OUT, "reference_nested")
    for k, v in cap.items():
        w.add_tensor(k, np.ascontiguousarray(v.cpu().numpy().reshape(-1).astype(np.float32)))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    meta = {
        "H": FIX_H, "W": FIX_W, "patch": PATCH,
        "metric_out_layers": METRIC_OUT_LAYERS,
        "metric_embed_dim": 1024, "metric_depth": 24, "metric_ffn_type": "mlp",
        "scale_factor": scale_factor, "is_metric": is_metric,
        "shapes": shapes, "atol": 2e-3, "rtol": 2e-3,
    }
    with open(MANIFEST, "w") as f:
        json.dump(meta, f, indent=2)

    print("wrote", OUT)
    for k in cap:
        print(f"  {k}: {shapes.get(k, [cap[k].numel()])}")
    print(f"VERIFY: depth_final range [{depth_final.min():.4f}, {depth_final.max():.4f}] "
          f"mean {depth_final.mean():.4f}  scale_factor {scale_factor:.6f}  is_metric {is_metric}")
    print(f"        depth_any range [{depth_any.min():.4f}, {depth_any.max():.4f}]  "
          f"depth_metric_raw range [{depth_metric_raw.min():.4f}, {depth_metric_raw.max():.4f}]  "
          f"sky range [{sky.min():.4f}, {sky.max():.4f}]")


if __name__ == "__main__":
    main()
