#!/usr/bin/env python3
"""Convert DA3-BASE to a single self-contained GGUF: config as KV, backbone weights as f32."""
import argparse, sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
import scripts.gguf_keys as K
from scripts.da3_reference import load_model


def write_da3_gguf(net, output, checkpoint_name, with_aux=False):
    """Write a full DepthAnything3Net (backbone + DualDPT head + cam_dec + optional
    gs_head) to a self-contained GGUF. Reused by the nested converter for the
    anyview (giant) branch (net.da3)."""
    bb = net.backbone.pretrained if hasattr(net.backbone, "pretrained") else net.backbone
    embed_dim = bb.embed_dim
    depth = bb.n_blocks
    num_heads = bb.num_heads
    head_dim = embed_dim // num_heads
    # FFN type: classic MLP (fc1/fc2) vs SwiGLU (w12/w3, giant). For SwiGLU the
    # "hidden" width the C++ loader cares about is w3.in_features (the post-gate
    # dim, 4096 for the giant); for plain MLP it is fc1.out_features.
    mlp0 = bb.blocks[0].mlp
    if hasattr(mlp0, "w12"):
        ffn_type = "swiglu"
        mlp_hidden = mlp0.w3.in_features
    else:
        ffn_type = "mlp"
        mlp_hidden = mlp0.fc1.out_features
    pos_rows = bb.pos_embed.shape[1] - 1
    M = int(round(pos_rows ** 0.5))
    assert M * M == pos_rows, f"pos_embed rows {pos_rows} not a perfect square"
    # LayerScale gamma is learned per-channel and exported faithfully as the
    # ls1/ls2 tensors; this scalar KV is informational only and not used at
    # inference. Do NOT derive it from gamma.mean() — that averages a learned
    # per-channel vector and bears no relation to the original init constant.
    init_values = 0.0
    qkv_bias = bb.blocks[0].attn.qkv.bias is not None

    w = gguf.GGUFWriter(output, K.ARCH)
    w.add_string(K.KV["arch"], K.ARCH)
    w.add_string(K.KV["checkpoint_name"], checkpoint_name)
    w.add_uint32(K.KV["patch_size"], 14)
    w.add_uint32(K.KV["vit.embed_dim"], int(embed_dim))
    w.add_uint32(K.KV["vit.depth"], int(depth))
    w.add_uint32(K.KV["vit.num_heads"], int(num_heads))
    w.add_uint32(K.KV["vit.head_dim"], int(head_dim))
    w.add_uint32(K.KV["vit.mlp_hidden"], int(mlp_hidden))
    w.add_uint32(K.KV["vit.num_register"], int(bb.num_register_tokens))
    w.add_float32(K.KV["vit.init_values"], init_values)
    w.add_int32(K.KV["vit.alt_start"], int(bb.alt_start))
    w.add_int32(K.KV["vit.rope_start"], int(bb.rope_start))
    w.add_int32(K.KV["vit.qknorm_start"], int(bb.qknorm_start))
    w.add_float32(K.KV["vit.rope_freq"], 100.0)
    w.add_bool(K.KV["vit.cat_token"], bool(bb.cat_token))
    w.add_bool(K.KV["vit.qkv_bias"], bool(qkv_bias))
    w.add_float32(K.KV["vit.ln_eps"], 1e-6)
    w.add_float32(K.KV["vit.interp_offset"], float(bb.interpolate_offset))
    w.add_bool(K.KV["vit.interp_antialias"], bool(bb.interpolate_antialias))
    w.add_uint32(K.KV["vit.pos_embed_grid"], int(M))
    w.add_string(K.KV["vit.ffn_type"], ffn_type)
    out_layers = list(getattr(net.backbone, "out_layers", [5, 7, 9, 11]))
    w.add_array(K.KV["vit.out_layers"], [int(v) for v in out_layers])
    w.add_array(K.KV["img.mean"], [0.485, 0.456, 0.406])
    w.add_array(K.KV["img.std"], [0.229, 0.224, 0.225])
    w.add_string(K.KV["img.resize_mode"], "upper_bound")
    w.add_uint32(K.KV["img.resize_target"], 504)

    # --- DualDPT depth head config -------------------------------------------
    head = net.head
    # features = the common fusion width (out-channels of any layer{i}_rn); 128 for BASE.
    head_features = int(head.scratch.layer1_rn.out_channels)
    head_out_channels = [int(head.projects[i].out_channels) for i in range(4)]
    w.add_uint32(K.KV["head.features"], head_features)
    w.add_array(K.KV["head.out_channels"], head_out_channels)
    w.add_uint32(K.KV["head.output_dim"], 2)
    w.add_bool(K.KV["head.pos_embed"], bool(head.pos_embed))
    w.add_uint32(K.KV["head.down_ratio"], int(head.down_ratio))
    w.add_string(K.KV["head.activation"], str(head.activation))
    w.add_string(K.KV["head.conf_activation"], str(head.conf_activation))

    # --- DualDPT auxiliary ray head config (opt-in) --------------------------
    # OFF by default so the published depth GGUFs are byte-identical. When ON, set
    # head.has_aux + aux dims; the aux tensors are emitted in the head loop below.
    aux_levels = int(getattr(head, "aux_levels", 4))
    if with_aux:
        w.add_bool(K.KV["head.has_aux"], True)
        w.add_uint32(K.KV["head.aux_ray_dim"], 6)
        w.add_uint32(K.KV["head.aux_levels"], aux_levels)

    # --- camera pose decoder (cam_dec MLP) config ----------------------------
    w.add_uint32(K.KV["cam.dim_in"], 1536)

    written, skipped = 0, []
    for name, t in net.backbone.named_parameters():
        canon = name.split("pretrained.")[-1] if "pretrained." in name else name
        g = K.rename_backbone(canon)
        if g is None:
            skipped.append(name)
            continue
        arr = np.ascontiguousarray(
            t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32
        )
        w.add_tensor(g, arr)
        written += 1
    if written == 0:
        raise SystemExit("error: no backbone tensors mapped; check rename_backbone prefix")
    # The backbone loop iterates only backbone params, so every one should map.
    # A nonzero skip means rename_backbone is missing a rule (e.g. a new variant's
    # mask_token / register_tokens) and would silently drop weights — fail loudly.
    if skipped:
        raise SystemExit(
            f"error: {len(skipped)} backbone param(s) unmapped (rename_backbone gap): {skipped[:5]}"
        )

    # --- DualDPT head main-path tensors --------------------------------------
    head_written, skipped_aux, head_unmapped, aux_written = 0, [], [], 0
    for name, t in net.head.named_parameters():
        g = K.rename_head(name)
        if g is None:
            if K.is_head_aux(name):
                # Opt-in aux ray head: emit the finest-level aux tensors (--with-aux);
                # otherwise skip (default OFF -> byte-identical depth GGUF).
                ga = K.rename_head_aux(name, aux_levels - 1) if with_aux else None
                if ga is None:
                    skipped_aux.append(name)
                    continue
                arr = np.ascontiguousarray(
                    t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32
                )
                w.add_tensor(ga, arr)
                aux_written += 1
                continue
            else:
                head_unmapped.append(name)
            continue
        arr = np.ascontiguousarray(
            t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32
        )
        w.add_tensor(g, arr)
        head_written += 1
    # A non-aux unmapped head param is a real gap in rename_head and would silently
    # drop a weight needed by the depth path -> fail loudly.
    if head_unmapped:
        raise SystemExit(
            f"error: {len(head_unmapped)} head param(s) unmapped and not aux "
            f"(rename_head gap): {head_unmapped[:8]}"
        )
    if head_written == 0:
        raise SystemExit("error: no head tensors mapped; check rename_head")

    # --- camera pose decoder (cam_dec) tensors -------------------------------
    # cam_dec is the default pose path's MLP; it has no optional/aux params, so
    # every one of its 10 params must map (fail loudly otherwise).
    cam_written, cam_unmapped = 0, []
    for name, t in net.cam_dec.named_parameters():
        g = K.rename_cam(name)
        if g is None:
            cam_unmapped.append(name)
            continue
        arr = np.ascontiguousarray(
            t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32
        )
        w.add_tensor(g, arr)
        cam_written += 1
    if cam_unmapped:
        raise SystemExit(
            f"error: {len(cam_unmapped)} cam_dec param(s) unmapped "
            f"(rename_cam gap): {cam_unmapped[:8]}"
        )
    if cam_written == 0:
        raise SystemExit("error: no cam_dec tensors mapped; check rename_cam")

    # --- GSDPT 3D-Gaussian head (gs_head) ------------------------------------
    # Present only on the giant (DA3-BASE has net.gs_head is None -> skip). The
    # gs_adapter carries no learned weights, so only KV is needed for it.
    gs_written = 0
    if getattr(net, "gs_head", None) is not None:
        gs = net.gs_head
        gs_features = int(gs.scratch.layer1_rn.out_channels)
        gs_out_channels = [int(gs.projects[i].out_channels) for i in range(4)]
        w.add_uint32(K.KV["gs.output_dim"], int(gs.out_dim))
        w.add_uint32(K.KV["gs.features"], gs_features)
        w.add_array(K.KV["gs.out_channels"], gs_out_channels)
        ga = net.gs_adapter
        w.add_uint32(K.KV["gs.sh_degree"], int(ga.sh_degree))
        w.add_float32(K.KV["gs.scale_min"], float(ga.gaussian_scale_min))
        w.add_float32(K.KV["gs.scale_max"], float(ga.gaussian_scale_max))
        w.add_bool(K.KV["gs.pred_offset_depth"], bool(ga.pred_offset_depth))
        w.add_bool(K.KV["gs.pred_offset_xy"], bool(ga.pred_offset_xy))
        w.add_bool(K.KV["gs.pred_color"], bool(ga.pred_color))

        gs_unmapped = []
        for name, t in gs.named_parameters():
            g = K.rename_gs(name)
            if g is None:
                gs_unmapped.append(name)
                continue
            arr = np.ascontiguousarray(
                t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32
            )
            w.add_tensor(g, arr)
            gs_written += 1
        if gs_unmapped:
            raise SystemExit(
                f"error: {len(gs_unmapped)} gs_head param(s) unmapped "
                f"(rename_gs gap): {gs_unmapped[:8]}"
            )
        if gs_written == 0:
            raise SystemExit("error: no gs_head tensors mapped; check rename_gs")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {output}: backbone_tensors={written} skipped={len(skipped)}")
    print(f"head_tensors={head_written} aux_tensors={aux_written} skipped_aux={len(skipped_aux)}")
    print(f"cam_tensors={cam_written}")
    print(f"ffn_type={ffn_type} mlp_hidden={mlp_hidden} gs_tensors={gs_written}")
    return {"backbone": written, "head": head_written, "cam": cam_written,
            "gs": gs_written, "ffn_type": ffn_type, "mlp_hidden": mlp_hidden}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DA3-BASE")
    ap.add_argument("--output", default="models/depth-anything-base-f32.gguf")
    ap.add_argument("--with-aux", action="store_true",
                    help="emit the opt-in DualDPT auxiliary ray head tensors "
                         "(default OFF -> byte-identical to the published depth GGUF)")
    a = ap.parse_args()

    _, net = load_model(a.model)
    write_da3_gguf(net, a.output, os.path.basename(a.model.rstrip("/")), with_aux=a.with_aux)


if __name__ == "__main__":
    main()
