#!/usr/bin/env python3
"""Convert an upstream Depth Anything V2 checkpoint (.pth) into a single
self-contained GGUF that the existing metadata-driven engine runs via the new
DA2 relative-depth route.

DA2 is a plain DINOv2 backbone (no RoPE/QK-norm/registers/cross-view) + a standard
DPT head (output_dim=1, no UV pos-embed, no head.norm). The existing
gguf_keys.rename_backbone / rename_head map DA2's exact param names unchanged; DA2
simply omits camera_token / q_norm / head.norm. Relative models return inverse
depth (ReLU); metric models (Hypersim=20 indoor, VKITTI=80 outdoor) scale by
max_depth, recorded as head.max_depth."""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
import scripts.gguf_keys as K
from scripts.da2_reference import load_da2_model, DA2_CONFIGS

INTERMEDIATE_IDX = {
    "vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23], "vitg": [9, 19, 29, 39],
}

# DA2's DINOv2 carries `mask_token` (nn.Parameter, zeros) used ONLY during masked
# pretraining (`prepare_tokens_with_masks` applies it iff masks is not None, which
# never happens at depth inference). DA3's backbone does not expose it, so the
# shared K.rename_backbone has no rule for it on purpose. Skip it here (DA2-local)
# rather than adding a rule to gguf_keys.py, which would affect DA3.
_DA2_SKIP_BACKBONE = {"mask_token"}


def write_da2_gguf(net, encoder, output, checkpoint_name, max_depth=0.0):
    bb = net.pretrained
    head = net.depth_head
    embed_dim = bb.embed_dim
    depth = bb.n_blocks
    num_heads = bb.num_heads
    head_dim = embed_dim // num_heads
    mlp0 = bb.blocks[0].mlp
    if hasattr(mlp0, "w12"):
        ffn_type, mlp_hidden = "swiglu", mlp0.w3.in_features
    else:
        ffn_type, mlp_hidden = "mlp", mlp0.fc1.out_features
    pos_rows = bb.pos_embed.shape[1] - 1
    M = int(round(pos_rows ** 0.5))
    assert M * M == pos_rows, f"pos_embed rows {pos_rows} not a perfect square"
    qkv_bias = bb.blocks[0].attn.qkv.bias is not None
    out_layers = INTERMEDIATE_IDX[encoder]
    head_features = int(head.scratch.layer1_rn.out_channels)
    head_out_channels = [int(head.projects[i].out_channels) for i in range(4)]

    w = gguf.GGUFWriter(output, K.ARCH)
    w.add_string(K.KV["arch"], "depthanything2")                 # <-- discriminator value
    w.add_string(K.KV["checkpoint_name"], checkpoint_name)
    w.add_uint32(K.KV["patch_size"], 14)
    # --- DINOv2 backbone (vit.*) — plain: rope/alt/qknorm all -1, no registers ---
    w.add_uint32(K.KV["vit.embed_dim"], int(embed_dim))
    w.add_uint32(K.KV["vit.depth"], int(depth))
    w.add_uint32(K.KV["vit.num_heads"], int(num_heads))
    w.add_uint32(K.KV["vit.head_dim"], int(head_dim))
    w.add_uint32(K.KV["vit.mlp_hidden"], int(mlp_hidden))
    w.add_string(K.KV["vit.ffn_type"], ffn_type)
    w.add_uint32(K.KV["vit.num_register"], 0)
    w.add_float32(K.KV["vit.init_values"], 1.0)
    w.add_int32(K.KV["vit.alt_start"], -1)
    w.add_int32(K.KV["vit.rope_start"], -1)
    w.add_int32(K.KV["vit.qknorm_start"], -1)
    w.add_float32(K.KV["vit.rope_freq"], 100.0)
    w.add_bool(K.KV["vit.cat_token"], False)
    w.add_bool(K.KV["vit.qkv_bias"], bool(qkv_bias))
    w.add_float32(K.KV["vit.ln_eps"], 1e-6)
    w.add_float32(K.KV["vit.interp_offset"], float(bb.interpolate_offset))
    w.add_bool(K.KV["vit.interp_antialias"], bool(bb.interpolate_antialias))
    w.add_uint32(K.KV["vit.pos_embed_grid"], int(M))
    w.add_array(K.KV["vit.out_layers"], [int(v) for v in out_layers])
    # --- preprocessing (DA2: lower_bound 518, ImageNet) ---
    w.add_array(K.KV["img.mean"], [0.485, 0.456, 0.406])
    w.add_array(K.KV["img.std"], [0.229, 0.224, 0.225])
    w.add_string(K.KV["img.resize_mode"], "lower_bound")
    w.add_uint32(K.KV["img.resize_target"], 518)
    # --- DPT head (head.*) — single channel, no UV pos-embed, no head.norm ---
    w.add_uint32(K.KV["head.features"], head_features)
    w.add_array(K.KV["head.out_channels"], head_out_channels)
    w.add_uint32(K.KV["head.output_dim"], 1)
    w.add_bool(K.KV["head.pos_embed"], False)
    # Final activation: relative head ends in ReLU; metric head ends in Sigmoid
    # (then x max_depth). The C++ route keys on max_depth>0, but the metadata is
    # written authoritatively so the GGUF is self-describing.
    w.add_string(K.KV["head.activation"], "sigmoid" if (max_depth and max_depth > 0) else "relu")
    w.add_string(K.KV["head.norm_type"], "idt")
    if max_depth and max_depth > 0:
        w.add_float32(K.KV["head.max_depth"], float(max_depth))

    # --- backbone tensors (vit.*): net.pretrained param names are already prefix-free, then rename_backbone ---
    written, skipped = 0, []
    for name, t in net.pretrained.named_parameters():
        if name in _DA2_SKIP_BACKBONE:
            continue
        g = K.rename_backbone(name)
        if g is None:
            skipped.append(name); continue
        w.add_tensor(g, np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32))
        written += 1
    if skipped:
        raise SystemExit(f"error: {len(skipped)} backbone param(s) unmapped: {skipped[:8]}")

    # --- head tensors (head.*): net.depth_head params are already prefix-free ---
    head_written, head_unmapped = 0, []
    for name, t in net.depth_head.named_parameters():
        g = K.rename_head(name)
        if g is None:
            head_unmapped.append(name); continue
        w.add_tensor(g, np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32))
        head_written += 1
    if head_unmapped:
        raise SystemExit(f"error: {len(head_unmapped)} head param(s) unmapped: {head_unmapped[:8]}")

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {output}: backbone={written} head={head_written} unmapped=0 "
          f"arch=depthanything2 encoder={encoder} max_depth={max_depth}")
    return {"backbone": written, "head": head_written, "unmapped": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(DA2_CONFIGS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-depth", type=float, default=0.0)
    a = ap.parse_args()
    net = load_da2_model(a.encoder, a.ckpt, a.max_depth)
    write_da2_gguf(net, a.encoder, a.output,
                   a.name or os.path.basename(a.ckpt), a.max_depth)


if __name__ == "__main__":
    main()
