#!/usr/bin/env python3
"""Convert the STANDALONE monocular checkpoint DA3MONO-LARGE into a single
self-contained GGUF.

DA3MONO-LARGE is a DepthAnything3Net whose backbone is a plain ViT-L (24 layers,
embed 1024, classic MLP FFN, alt/rope/qknorm_start=-1, cat_token=False,
out_layers [4,11,17,23]) and whose head is a single-head DPT (output_dim=1,
activation=exp, norm=Identity/"idt", pos_embed=False) WITH a parallel sky head
(sky_activation=relu). It is architecturally identical to the nested metric
branch (convert_nested_to_gguf.write_metric_gguf) but STANDALONE -> written under
the vit.*/head.* prefixes the C++ engine reads directly (not m_vit.*/m_head.*).

The net also carries cam_enc/cam_dec/gs_head, but MONOCULAR DEPTH uses neither;
only the backbone + depth/sky head are converted.
"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
import scripts.gguf_keys as K
from scripts.da3_reference import load_model


def write_mono_gguf(net, output, checkpoint_name):
    bb = net.backbone.pretrained if hasattr(net.backbone, "pretrained") else net.backbone
    embed_dim = bb.embed_dim
    depth = bb.n_blocks
    num_heads = bb.num_heads
    head_dim = embed_dim // num_heads
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
    qkv_bias = bb.blocks[0].attn.qkv.bias is not None

    w = gguf.GGUFWriter(output, K.ARCH)
    w.add_string(K.KV["arch"], K.ARCH)
    w.add_string(K.KV["checkpoint_name"], checkpoint_name)
    w.add_uint32(K.KV["patch_size"], 14)
    # --- ViT-L backbone config (vit.*) ---------------------------------------
    w.add_uint32(K.KV["vit.embed_dim"], int(embed_dim))
    w.add_uint32(K.KV["vit.depth"], int(depth))
    w.add_uint32(K.KV["vit.num_heads"], int(num_heads))
    w.add_uint32(K.KV["vit.head_dim"], int(head_dim))
    w.add_uint32(K.KV["vit.mlp_hidden"], int(mlp_hidden))
    w.add_string(K.KV["vit.ffn_type"], ffn_type)
    w.add_uint32(K.KV["vit.num_register"], int(bb.num_register_tokens))
    w.add_float32(K.KV["vit.init_values"], 0.0)
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
    out_layers = list(getattr(net.backbone, "out_layers", [4, 11, 17, 23]))
    w.add_array(K.KV["vit.out_layers"], [int(v) for v in out_layers])
    w.add_array(K.KV["img.mean"], [0.485, 0.456, 0.406])
    w.add_array(K.KV["img.std"], [0.229, 0.224, 0.225])
    w.add_string(K.KV["img.resize_mode"], "upper_bound")
    w.add_uint32(K.KV["img.resize_target"], 504)

    # --- single-head DPT (+ sky head) config (head.*) ------------------------
    head = net.head
    head_features = int(head.scratch.layer1_rn.out_channels)
    head_out_channels = [int(head.projects[i].out_channels) for i in range(4)]
    norm_type = "idt" if isinstance(head.norm, torch.nn.Identity) else "layer"
    w.add_uint32(K.KV["head.features"], head_features)
    w.add_array(K.KV["head.out_channels"], head_out_channels)
    w.add_uint32(K.KV["head.output_dim"], int(head.out_dim))     # =1 (NOT hardcoded 2)
    w.add_bool(K.KV["head.pos_embed"], bool(head.pos_embed))     # False for mono
    w.add_uint32(K.KV["head.down_ratio"], int(head.down_ratio))
    w.add_string(K.KV["head.activation"], str(head.activation))
    w.add_string(K.KV["head.conf_activation"], str(head.conf_activation))
    w.add_string(K.KV["head.sky_activation"], str(head.sky_activation))
    w.add_string(K.KV["head.norm_type"], norm_type)

    # --- backbone tensors (vit.*) --------------------------------------------
    written, skipped = 0, []
    for name, t in net.backbone.named_parameters():
        canon = name.split("pretrained.")[-1] if "pretrained." in name else name
        g = K.rename_backbone(canon)
        if g is None:
            skipped.append(name)
            continue
        arr = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
        w.add_tensor(g, arr)
        written += 1
    if skipped:
        raise SystemExit(
            f"error: {len(skipped)} backbone param(s) unmapped "
            f"(rename_backbone gap): {skipped[:5]}")
    if written == 0:
        raise SystemExit("error: no backbone tensors mapped")

    # --- depth + sky head tensors (head.*) -----------------------------------
    head_written, head_unmapped = 0, []
    for name, t in net.head.named_parameters():
        g = K.rename_mono_head(name)
        if g is None:
            head_unmapped.append(name)
            continue
        arr = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
        w.add_tensor(g, arr)
        head_written += 1
    if head_unmapped:
        raise SystemExit(
            f"error: {len(head_unmapped)} mono head param(s) unmapped "
            f"(rename_mono_head gap): {head_unmapped[:8]}")
    if head_written == 0:
        raise SystemExit("error: no mono head tensors mapped")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {output}: backbone_tensors={written} head_tensors={head_written} unmapped=0")
    print(f"  vit: embed={embed_dim} depth={depth} ffn={ffn_type} heads={num_heads} "
          f"mlp_hidden={mlp_hidden} cat_token={bool(bb.cat_token)} out_layers={out_layers}")
    print(f"  head: features={head_features} out_channels={head_out_channels} "
          f"output_dim={head.out_dim} act={head.activation} conf={head.conf_activation} "
          f"sky_act={head.sky_activation} norm_type={norm_type} pos_embed={head.pos_embed}")
    return {"backbone": written, "head": head_written}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DA3MONO-LARGE")
    ap.add_argument("--output", default="models/depth-anything-mono-large-f32.gguf")
    a = ap.parse_args()

    _, net = load_model(a.model)
    assert hasattr(net, "head") and not hasattr(net, "da3"), \
        f"expected standalone DepthAnything3Net (got {type(net).__name__})"
    assert int(net.head.out_dim) == 1, f"expected mono head out_dim==1, got {net.head.out_dim}"
    write_mono_gguf(net, a.output, os.path.basename(a.model.rstrip("/")))


if __name__ == "__main__":
    main()
