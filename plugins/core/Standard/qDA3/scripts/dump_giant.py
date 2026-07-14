#!/usr/bin/env python3
"""Dump gold reference tensors for DA3-GIANT (N=1): backbone out-layers, giant
depth+pose, the GSDPT raw_gs/conf, and the final world-space 3D Gaussians.

Produces dumps/reference_giant.gguf (flattened f32, row-major) + dumps/
manifest_giant.json. These are the GOLD reference the C++ GIANT parity gates
(M5-T2/T3/T4) assert against. The giant is 40 layers / embed 1536; a single CPU
forward is SLOW (minutes) -- that is expected.
"""
import os, json, sys, numpy as np, torch
# Pre-import e3nn (used by gs_adapter's rotate_sh) HERE, before load_model()
# installs its torchvision stub. e3nn's import drags in torch._dynamo, whose
# registration walks sys.modules via `inspect`; the torchvision stub answers any
# attribute (incl. __file__) with a dummy class, which crashes that walk. Doing
# the import first lets the dynamo chain complete against the real module table.
import e3nn.o3  # noqa: F401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
from scripts.da3_reference import load_model, fixed_input, FIX_H, FIX_W, PATCH

OUT = "dumps/reference_giant.gguf"
MANIFEST = "dumps/manifest_giant.json"
OUT_LAYERS = [19, 27, 33, 39]


def main():
    os.makedirs("dumps", exist_ok=True)
    _, net = load_model("models/DA3-GIANT")
    x, raw = fixed_input()
    bb = net.backbone.pretrained
    assert len(bb.blocks) == 40, len(bb.blocks)
    assert bb.embed_dim == 1536, bb.embed_dim
    assert hasattr(bb.blocks[0].mlp, "w12"), "expected SwiGLU mlp.w12 on the giant"
    assert net.gs_head is not None and net.gs_adapter is not None, "giant must have gs_head/gs_adapter"
    cap = {}

    # --- backbone out-layer features + camera tokens --------------------------
    with torch.no_grad():
        outs, _aux = bb.get_intermediate_layers(
            x, n=OUT_LAYERS, export_feat_layers=[],
            ref_view_strategy="saddle_balanced")
    for L, o in zip(OUT_LAYERS, outs):
        cap[f"feat_g_{L}"] = o[0].detach().reshape(-1).float().contiguous()  # [1,1,256,3072]
        cap[f"cam_g_{L}"] = o[1].detach().reshape(-1).float().contiguous()   # [1,1,3072]
    feat0 = outs[0][0]
    assert tuple(feat0.shape) == (1, 1, 256, 3072), feat0.shape

    # --- full giant forward with the 3DGS branch ------------------------------
    # Hook gs_head to grab its raw_gs / raw_gs_conf (camera-space gaussian params
    # BEFORE the gs_adapter geometry). The forward returns a Dict reshaped to
    # [B,S,...]; B=S=1 here.
    gs_cap = {}

    def gs_hook(_m, _inp, out):
        gs_cap["raw_gs"] = out["raw_gs"].detach().clone()        # [1,1,37,224,224]
        gs_cap["raw_gs_conf"] = out["raw_gs_conf"].detach().clone()  # [1,1,224,224]

    h = net.gs_head.register_forward_hook(gs_hook)
    try:
        with torch.no_grad():
            full = net(x, infer_gs=True)
    finally:
        h.remove()

    depth_g = full["depth"].squeeze()        # [224,224]
    assert tuple(depth_g.shape) == (FIX_H, FIX_W), depth_g.shape
    cap["depth_g"] = depth_g.detach().reshape(-1).float().contiguous()
    cap["depth_conf_g"] = full["depth_conf"].detach().reshape(-1).float().contiguous()
    cap["extrinsics_g"] = full["extrinsics"].detach().reshape(-1).float().contiguous()  # 3x4
    cap["intrinsics_g"] = full["intrinsics"].detach().reshape(-1).float().contiguous()  # 3x3

    # raw_gs / conf from the gs_head hook. activate_head_gs emits channels-last,
    # so raw_gs is [H,W,37] (not [37,H,W]) and the conf is [H,W].
    raw_gs = gs_cap["raw_gs"].squeeze()      # [224,224,37]
    raw_gs_conf = gs_cap["raw_gs_conf"].squeeze()  # [224,224]
    assert tuple(raw_gs.shape) == (FIX_H, FIX_W, 37), raw_gs.shape
    assert tuple(raw_gs_conf.shape) == (FIX_H, FIX_W), raw_gs_conf.shape
    cap["raw_gs"] = raw_gs.detach().reshape(-1).float().contiguous()
    cap["gs_conf"] = raw_gs_conf.detach().reshape(-1).float().contiguous()

    # --- world-space gaussians (gs_adapter output) ----------------------------
    gaussians = full.get("gaussians")
    assert gaussians is not None, "net(infer_gs=True) produced no gaussians"
    gs_means = gaussians.means.squeeze(0)        # [N,3]
    gs_scales = gaussians.scales.squeeze(0)      # [N,3]
    gs_rotations = gaussians.rotations.squeeze(0)  # [N,4]
    gs_harmonics = gaussians.harmonics.squeeze(0)  # [N,3,d_sh]
    gs_opacities = gaussians.opacities.squeeze(0)  # [N] or [N,1]
    N = FIX_H * FIX_W
    assert tuple(gs_means.shape) == (N, 3), gs_means.shape
    cap["gs_means"] = gs_means.detach().reshape(-1).float().contiguous()
    cap["gs_scales"] = gs_scales.detach().reshape(-1).float().contiguous()
    cap["gs_rotations"] = gs_rotations.detach().reshape(-1).float().contiguous()
    cap["gs_harmonics"] = gs_harmonics.detach().reshape(-1).float().contiguous()
    cap["gs_opacities"] = gs_opacities.detach().reshape(-1).float().contiguous()

    cap["input_image"] = x.detach().reshape(-1).float().contiguous()
    cap["raw_image"] = torch.from_numpy(raw.astype(np.float32)).reshape(-1)  # (224,224,3) HWC 0..255

    # pre-flatten shapes for the manifest
    shapes = {
        "feat_g_19": [256, 3072], "feat_g_27": [256, 3072],
        "feat_g_33": [256, 3072], "feat_g_39": [256, 3072],
        "cam_g_19": [3072], "cam_g_27": [3072], "cam_g_33": [3072], "cam_g_39": [3072],
        "depth_g": [FIX_H, FIX_W], "depth_conf_g": [FIX_H, FIX_W],
        "extrinsics_g": list(full["extrinsics"].squeeze().shape),
        "intrinsics_g": list(full["intrinsics"].squeeze().shape),
        "raw_gs": [FIX_H, FIX_W, 37], "gs_conf": [FIX_H, FIX_W],
        "gs_means": [N, 3], "gs_scales": [N, 3], "gs_rotations": [N, 4],
        "gs_harmonics": list(gs_harmonics.shape), "gs_opacities": list(gs_opacities.shape),
        "input_image": list(x.shape), "raw_image": [FIX_H, FIX_W, 3],
    }

    # --- verification ---------------------------------------------------------
    assert torch.isfinite(depth_g).all() and bool((depth_g > 0).all()), "depth_g must be positive/finite"
    for k in ("gs_means", "gs_scales", "gs_rotations", "gs_harmonics", "gs_opacities"):
        v = cap[k]
        assert torch.isfinite(v).all(), f"{k} contains non-finite values"

    w = gguf.GGUFWriter(OUT, "reference_giant")
    for k, v in cap.items():
        w.add_tensor(k, np.ascontiguousarray(v.cpu().numpy().reshape(-1).astype(np.float32)))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    with open(MANIFEST, "w") as f:
        json.dump({"H": FIX_H, "W": FIX_W, "patch": PATCH, "out_layers": OUT_LAYERS,
                   "embed_dim": 1536, "depth": 40, "ffn_type": "swiglu",
                   "shapes": shapes, "atol": 2e-3, "rtol": 2e-3}, f, indent=2)

    print("wrote", OUT)
    for k in cap:
        print(f"  {k}: {shapes.get(k, [cap[k].numel()])}")
    print("VERIFY: depth_g positive+finite, gaussians finite, gs_means", list(gs_means.shape))


if __name__ == "__main__":
    main()
