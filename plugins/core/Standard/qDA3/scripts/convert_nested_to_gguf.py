#!/usr/bin/env python3
"""Convert DA3NESTED-GIANT-LARGE into TWO self-contained GGUFs:

  - models/depth-anything-nested-anyview.gguf : net.da3 (the GIANT anyview branch,
    DepthAnything3Net with SwiGLU ViT-g backbone + DualDPT + cam_dec + GSDPT).
    Written with the EXACT same logic as the standalone giant (write_da3_gguf).
  - models/depth-anything-nested-metric.gguf : net.da3_metric (the metric branch,
    ViT-L MLP backbone -> m_vit.* + single-head DPT with a sky head -> m_head.*).

The nested net is NestedDepthAnything3Net (model.da3 + model.da3_metric). Loading
the 6.76GB checkpoint is SLOW (minutes) but no forward is run here.
"""
import argparse, os, sys, numpy as np, torch
# Pre-import e3nn.o3 BEFORE load_model() installs its torchvision stub. The nested
# net's gs_adapter pulls in e3nn (sh_helpers), whose import drags in torch._dynamo;
# its registration walks sys.modules via inspect, and the torchvision stub answers
# any attribute (incl. __file__) with a dummy class that crashes that walk. Doing
# the real import first lets the dynamo chain complete against the real module table.
import e3nn.o3  # noqa: F401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
import scripts.gguf_keys as K
from scripts.da3_reference import load_model
from scripts.convert_da3_to_gguf import write_da3_gguf


def write_metric_gguf(net, output, checkpoint_name):
    """Write the metric DepthAnything3Net branch (net.da3_metric) to a GGUF:
    ViT-L backbone under m_vit.*, single-head DPT (+ sky head) under m_head.*.
    No cam_dec / gs_head on the metric branch."""
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
    # --- metric ViT-L backbone config (m_vit.*) ------------------------------
    w.add_uint32(K.KV["m_vit.embed_dim"], int(embed_dim))
    w.add_uint32(K.KV["m_vit.depth"], int(depth))
    w.add_uint32(K.KV["m_vit.num_heads"], int(num_heads))
    w.add_uint32(K.KV["m_vit.head_dim"], int(head_dim))
    w.add_uint32(K.KV["m_vit.mlp_hidden"], int(mlp_hidden))
    w.add_string(K.KV["m_vit.ffn_type"], ffn_type)
    w.add_uint32(K.KV["m_vit.num_register"], int(bb.num_register_tokens))
    w.add_float32(K.KV["m_vit.init_values"], 0.0)
    w.add_int32(K.KV["m_vit.alt_start"], int(bb.alt_start))
    w.add_int32(K.KV["m_vit.rope_start"], int(bb.rope_start))
    w.add_int32(K.KV["m_vit.qknorm_start"], int(bb.qknorm_start))
    w.add_float32(K.KV["m_vit.rope_freq"], 100.0)
    w.add_bool(K.KV["m_vit.cat_token"], bool(bb.cat_token))
    w.add_bool(K.KV["m_vit.qkv_bias"], bool(qkv_bias))
    w.add_float32(K.KV["m_vit.ln_eps"], 1e-6)
    w.add_float32(K.KV["m_vit.interp_offset"], float(bb.interpolate_offset))
    w.add_bool(K.KV["m_vit.interp_antialias"], bool(bb.interpolate_antialias))
    w.add_uint32(K.KV["m_vit.pos_embed_grid"], int(M))
    out_layers = list(getattr(net.backbone, "out_layers", [4, 11, 17, 23]))
    w.add_array(K.KV["m_vit.out_layers"], [int(v) for v in out_layers])

    # --- metric DPT head config (m_head.*) -----------------------------------
    head = net.head
    head_features = int(head.scratch.layer1_rn.out_channels)
    head_out_channels = [int(head.projects[i].out_channels) for i in range(4)]
    norm_type = "idt" if isinstance(head.norm, torch.nn.Identity) else "layer"
    w.add_uint32(K.KV["m_head.features"], head_features)
    w.add_array(K.KV["m_head.out_channels"], head_out_channels)
    w.add_uint32(K.KV["m_head.output_dim"], int(head.out_dim))
    w.add_uint32(K.KV["m_head.down_ratio"], int(head.down_ratio))
    w.add_string(K.KV["m_head.activation"], str(head.activation))
    w.add_string(K.KV["m_head.conf_activation"], str(head.conf_activation))
    w.add_string(K.KV["m_head.sky_activation"], str(head.sky_activation))
    w.add_string(K.KV["m_head.norm_type"], norm_type)

    # --- metric backbone tensors ---------------------------------------------
    written, skipped = 0, []
    for name, t in net.backbone.named_parameters():
        canon = name.split("pretrained.")[-1] if "pretrained." in name else name
        g = K.rename_metric_backbone(canon)
        if g is None:
            skipped.append(name)
            continue
        arr = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
        w.add_tensor(g, arr)
        written += 1
    if skipped:
        raise SystemExit(
            f"error: {len(skipped)} metric backbone param(s) unmapped "
            f"(rename_metric_backbone gap): {skipped[:5]}")
    if written == 0:
        raise SystemExit("error: no metric backbone tensors mapped")

    # --- metric DPT head tensors (main path + sky head) ----------------------
    head_written, head_unmapped = 0, []
    for name, t in net.head.named_parameters():
        g = K.rename_metric_head(name)
        if g is None:
            head_unmapped.append(name)
            continue
        arr = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)
        w.add_tensor(g, arr)
        head_written += 1
    if head_unmapped:
        raise SystemExit(
            f"error: {len(head_unmapped)} metric head param(s) unmapped "
            f"(rename_metric_head gap): {head_unmapped[:8]}")
    if head_written == 0:
        raise SystemExit("error: no metric head tensors mapped")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {output}: m_vit_tensors={written} m_head_tensors={head_written}")
    print(f"  m_vit: embed={embed_dim} depth={depth} ffn={ffn_type} heads={num_heads} "
          f"mlp_hidden={mlp_hidden} out_layers={out_layers}")
    print(f"  m_head: features={head_features} out_channels={head_out_channels} "
          f"output_dim={head.out_dim} act={head.activation} conf={head.conf_activation} "
          f"sky_act={head.sky_activation} norm_type={norm_type}")
    return {"m_vit": written, "m_head": head_written}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DA3NESTED-GIANT-LARGE")
    ap.add_argument("--anyview-output", default="models/depth-anything-nested-anyview.gguf")
    ap.add_argument("--metric-output", default="models/depth-anything-nested-metric.gguf")
    a = ap.parse_args()

    _, net = load_model(a.model)
    assert hasattr(net, "da3") and hasattr(net, "da3_metric"), \
        f"expected NestedDepthAnything3Net (got {type(net).__name__})"
    cp = os.path.basename(a.model.rstrip("/"))

    print("=== anyview (giant) branch ===")
    a_counts = write_da3_gguf(net.da3, a.anyview_output, cp + "-anyview")
    print("=== metric (large+sky) branch ===")
    m_counts = write_metric_gguf(net.da3_metric, a.metric_output, cp + "-metric")
    print("=== summary ===")
    print("anyview:", a_counts)
    print("metric:", m_counts)


if __name__ == "__main__":
    main()
