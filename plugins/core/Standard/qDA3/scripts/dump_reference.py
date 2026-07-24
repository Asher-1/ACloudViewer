#!/usr/bin/env python3
"""Dump gold per-component reference tensors for DA3-BASE backbone (N=1).

Produces dumps/reference.gguf (flattened f32 tensors, row-major) and
dumps/manifest.json (pre-flatten shapes + tolerances). These are the GOLD
reference the C++ backbone parity gates (Tasks 12 and 15) assert against.
"""
import os, json, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
from scripts.da3_reference import load_model, fixed_input, FIX_H, FIX_W, PATCH

OUT = "dumps/reference.gguf"
MANIFEST = "dumps/manifest.json"
OUT_LAYERS = [5, 7, 9, 11]


def main():
    os.makedirs("dumps", exist_ok=True)
    _, net = load_model()
    x, raw = fixed_input()
    bb = net.backbone.pretrained
    cap = {}

    # Capture the tokens right after cls-prepend + interpolated-pos-embed add,
    # before block 0, by wrapping prepare_tokens_with_masks.
    orig = bb.prepare_tokens_with_masks

    def wrapped(xx, *a, **k):
        out = orig(xx, *a, **k)
        cap["pos_embed_added"] = out.detach().clone()
        return out

    bb.prepare_tokens_with_masks = wrapped
    try:
        with torch.no_grad():
            outs, _aux = bb.get_intermediate_layers(
                x, n=OUT_LAYERS, export_feat_layers=[],
                ref_view_strategy="saddle_balanced")
    finally:
        bb.prepare_tokens_with_masks = orig

    feats = [o[0] for o in outs]
    cams = [o[1] for o in outs]
    for L, f, c in zip(OUT_LAYERS, feats, cams):
        cap[f"feat_{L}"] = f.detach().contiguous().float()
        cap[f"cam_token_{L}"] = c.detach().contiguous().float()
    cap["input_image"] = x.detach().contiguous().float()
    cap["raw_image"] = torch.from_numpy(raw.astype(np.float32))  # (224,224,3) HWC, values 0..255
    cap["pos_embed_added"] = cap["pos_embed_added"].detach().contiguous().float()

    # --- 2D RoPE isolated parity fixture (Task 11) -----------------------------
    # Uses the REAL reference module so the C++ rope is gated against ground truth.
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D
    rope = RotaryPositionEmbedding2D(frequency=100.0)
    hd, T = 64, 4
    g = torch.Generator().manual_seed(1)
    rin = torch.randn(1, 1, T, hd, generator=g)                          # (B,heads,N,hd)
    rpos = torch.tensor([[[1, 1], [1, 2], [2, 1], [2, 2]]], dtype=torch.long)  # (1,N,2) y,x
    with torch.no_grad():
        rout = rope(rin, rpos)
    cap["rope_in"] = rin.detach().contiguous().float()
    cap["rope_out"] = rout.detach().contiguous().float()
    cap["rope_pos"] = rpos.detach().contiguous().float()

    # --- DualDPT depth head reference (Task M2) --------------------------------
    # The head consumes the raw `outs` structure (a list of (feature, cam) tuples,
    # exactly what get_intermediate_layers returns). Hook the post-resize stages
    # and the post-output_conv1 fused tensor for layer-isolation debugging.
    head = net.head
    handles = []

    def _mk_hook(key):
        def _h(_m, _inp, out):
            cap[key] = out.detach().contiguous().float()
        return _h

    for s in range(4):
        handles.append(head.resize_layers[s].register_forward_hook(_mk_hook(f"head_stage{s}")))
    handles.append(head.scratch.output_conv1.register_forward_hook(_mk_hook("head_fused")))
    try:
        with torch.no_grad():
            head_out = net.head(list(outs), FIX_H, FIX_W, patch_start_idx=0)
    finally:
        for hd in handles:
            hd.remove()

    # --- Isolated conv parity fixtures (Task M2-T3) ----------------------------
    # Real head submodules on deterministic random input, to gate the riskiest
    # ggml graph ops (conv-transpose, strided conv, 1x1 conv) against ground truth.
    with torch.no_grad():
        g2 = torch.Generator().manual_seed(7)
        ct_in = torch.randn(1, 96, 16, 16, generator=g2)
        ct_out = net.head.resize_layers[0](ct_in)           # ConvTranspose k4s4 -> (1,96,64,64)
        cap["convt0_in"] = ct_in.float().contiguous(); cap["convt0_out"] = ct_out.detach().float().contiguous()
        cv_in = torch.randn(1, 768, 16, 16, generator=g2)
        cv_out = net.head.resize_layers[3](cv_in)           # Conv k3s2p1 -> (1,768,8,8)
        cap["convs3_in"] = cv_in.float().contiguous(); cap["convs3_out"] = cv_out.detach().float().contiguous()
        proj_in = torch.randn(1, 256, 1536, generator=g2)   # a token block [B, N, C]
        # projects[0] expects [B,C,ph,pw]; emulate the head's permute/reshape on a 16x16 grid:
        pin = proj_in.permute(0, 2, 1).reshape(1, 1536, 16, 16)
        pj_out = net.head.projects[0](pin)                  # Conv 1x1 1536->96 -> (1,96,16,16)
        cap["proj0_in"] = pin.float().contiguous(); cap["proj0_out"] = pj_out.detach().float().contiguous()

    head_depth = head_out["depth"].squeeze()        # (224,224)
    head_depth_conf = head_out["depth_conf"].squeeze()
    assert tuple(head_depth.shape) == (FIX_H, FIX_W), head_depth.shape
    assert torch.isfinite(head_depth).all() and bool((head_depth > 0).all()), "depth must be positive/finite (exp)"
    assert bool((head_depth_conf >= 1.0).all()), "depth_conf must be >= 1 (expp1)"
    cap["head_depth"] = head_depth.detach().contiguous().float()
    cap["head_depth_conf"] = head_depth_conf.detach().contiguous().float()

    # UV positional embedding (224x224x64), BEFORE the *0.1 ratio scaling.
    from depth_anything_3.model.utils.head_utils import create_uv_grid, position_grid_to_embed
    uv = create_uv_grid(FIX_W, FIX_H, aspect_ratio=1.0)
    uv_emb = position_grid_to_embed(uv, 64)   # (224,224,64)
    cap["uv_embed_64"] = uv_emb.detach().float().contiguous()

    # --- M3: camera pose (default cam_dec path) -------------------------------
    # Run the FULL default forward (use_ray_pose=False -> cam_dec path) and
    # capture: the cam_dec input (feats[-1][1] = the layer-11 camera token),
    # the raw pose encoding (9,), and the resulting extrinsics (3x4 = w2c) and
    # intrinsics (3x3). cam_token_in must equal the already-dumped cam_token_11.
    pose_cap = {}
    def cam_hook(_m, inp, out):
        pose_cap["cam_token_in"] = inp[0].detach().clone()   # feats[-1][1] passed to cam_dec
        pose_cap["pose_enc"] = out.detach().clone()
    h = net.cam_dec.register_forward_hook(cam_hook)
    with torch.no_grad():
        full = net(x)                      # default forward: use_ray_pose=False -> cam_dec path
    h.remove()
    cap["pose_enc"] = pose_cap["pose_enc"].reshape(-1).float().contiguous()           # (9,)
    cap["cam_token_in"] = pose_cap["cam_token_in"].reshape(-1).float().contiguous()   # (1536,)
    cap["extrinsics"] = full["extrinsics"].reshape(-1).float().contiguous()           # (12,) = 3x4
    cap["intrinsics"] = full["intrinsics"].reshape(-1).float().contiguous()           # (9,) = 3x3

    w = gguf.GGUFWriter(OUT, "reference")
    for k, v in cap.items():
        arr = np.ascontiguousarray(v.cpu().numpy().reshape(-1).astype(np.float32))
        w.add_tensor(k, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    shapes = {k: list(v.shape) for k, v in cap.items()}
    with open(MANIFEST, "w") as f:
        json.dump({"H": FIX_H, "W": FIX_W, "patch": PATCH, "out_layers": OUT_LAYERS,
                   "shapes": shapes, "atol": 2e-3, "rtol": 2e-3}, f, indent=2)

    print("wrote", OUT)
    for k, v in cap.items():
        print(f"  {k}: {list(v.shape)}")

    # ---- M4: multi-view (S=2) reference ----
    from scripts.da3_reference import fixed_input_multiview
    x_mv, raws = fixed_input_multiview(S=2, seed=0)
    mvcap = {}
    with torch.no_grad():
        outs_mv, _ = bb.get_intermediate_layers(
            x_mv, n=[5, 7, 9, 11], export_feat_layers=[],
            ref_view_strategy="saddle_balanced")
        full_mv = net(x_mv)
    MV_OUT_LAYERS = [5, 7, 9, 11]
    for L, o in zip(MV_OUT_LAYERS, outs_mv):
        mvcap[f"feat_mv_{L}"] = o[0].detach().reshape(-1).float().contiguous()   # [1,2,256,1536]->flat
        mvcap[f"cam_mv_{L}"] = o[1].detach().reshape(-1).float().contiguous()    # [1,2,1536]->flat
    mvcap["depth_mv"] = full_mv["depth"].detach().reshape(-1).float().contiguous()        # [1,2,224,224]
    mvcap["extrinsics_mv"] = full_mv["extrinsics"].detach().reshape(-1).float().contiguous()  # [1,2,3,4]
    mvcap["intrinsics_mv"] = full_mv["intrinsics"].detach().reshape(-1).float().contiguous()  # [1,2,3,3]
    mvcap["raw_mv_0"] = torch.from_numpy(raws[0].astype(np.float32))
    mvcap["raw_mv_1"] = torch.from_numpy(raws[1].astype(np.float32))
    wmv = gguf.GGUFWriter("dumps/reference_mv.gguf", "reference_mv")
    for k, v in mvcap.items():
        wmv.add_tensor(k, np.ascontiguousarray(v.cpu().numpy().reshape(-1).astype(np.float32)))
    wmv.write_header_to_file(); wmv.write_kv_data_to_file(); wmv.write_tensors_to_file(); wmv.close()
    with open("dumps/manifest_mv.json", "w") as f:
        json.dump({"S": 2, "H": FIX_H, "W": FIX_W, "out_layers": MV_OUT_LAYERS,
                   "shapes": {k: list(v.shape) for k, v in mvcap.items()}}, f, indent=2)
    print("wrote dumps/reference_mv.gguf:", {k: list(v.shape) for k, v in mvcap.items()})

    # ---- M4b: multi-view (S=4) reference WITH reference-view selection ----
    # S=4 >= THRESH_FOR_REF_SELECTION(=3) so the backbone selects a reference view
    # at layer alt_start-1 (=3), reorders, processes, and restores original order at
    # the out-layers (dumped feats are in ORIGINAL view order).
    import depth_anything_3.model.dinov2.vision_transformer as vt_mod
    from scripts.da3_reference import fixed_input_multiview_distinct
    x_mv4, raws4 = fixed_input_multiview_distinct(S=4, seed=0)
    mv4cap = {}
    # Monkeypatch select_reference_view to capture the chosen b_idx reliably.
    _real_select = vt_mod.select_reference_view
    _bidx_holder = {}
    def _capturing_select(xx, *a, **kw):
        bi = _real_select(xx, *a, **kw)
        _bidx_holder["b_idx"] = int(bi.reshape(-1)[0].item())
        return bi
    vt_mod.select_reference_view = _capturing_select
    try:
        with torch.no_grad():
            outs_mv4, _ = bb.get_intermediate_layers(
                x_mv4, n=[5, 7, 9, 11], export_feat_layers=[],
                ref_view_strategy="saddle_balanced")
            full_mv4 = net(x_mv4)
    finally:
        vt_mod.select_reference_view = _real_select
    assert "b_idx" in _bidx_holder, "reference-view selection did not run for S=4"
    MV_OUT_LAYERS = [5, 7, 9, 11]
    for L, o in zip(MV_OUT_LAYERS, outs_mv4):
        mv4cap[f"feat_mv4_{L}"] = o[0].detach().reshape(-1).float().contiguous()   # [1,4,256,1536]->flat
        mv4cap[f"cam_mv4_{L}"] = o[1].detach().reshape(-1).float().contiguous()    # [1,4,1536]->flat
    mv4cap["depth_mv4"] = full_mv4["depth"].detach().reshape(-1).float().contiguous()        # [1,4,224,224]
    mv4cap["extrinsics_mv4"] = full_mv4["extrinsics"].detach().reshape(-1).float().contiguous()  # [1,4,3,4]
    mv4cap["intrinsics_mv4"] = full_mv4["intrinsics"].detach().reshape(-1).float().contiguous()  # [1,4,3,3]
    for v in range(4):
        mv4cap[f"raw_mv4_{v}"] = torch.from_numpy(raws4[v].astype(np.float32))
    mv4cap["refsel_b_idx"] = torch.tensor([float(_bidx_holder["b_idx"])])
    wmv4 = gguf.GGUFWriter("dumps/reference_mv4.gguf", "reference_mv4")
    for k, v in mv4cap.items():
        wmv4.add_tensor(k, np.ascontiguousarray(v.cpu().numpy().reshape(-1).astype(np.float32)))
    wmv4.write_header_to_file(); wmv4.write_kv_data_to_file(); wmv4.write_tensors_to_file(); wmv4.close()
    with open("dumps/manifest_mv4.json", "w") as f:
        json.dump({"S": 4, "H": FIX_H, "W": FIX_W, "out_layers": MV_OUT_LAYERS,
                   "b_idx": _bidx_holder["b_idx"],
                   "shapes": {k: list(v.shape) for k, v in mv4cap.items()}}, f, indent=2)
    print("wrote dumps/reference_mv4.gguf b_idx=%d:" % _bidx_holder["b_idx"],
          {k: list(v.shape) for k, v in mv4cap.items()})


if __name__ == "__main__":
    main()
