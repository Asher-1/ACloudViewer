#!/usr/bin/env python3
"""Dump gold reference tensors for the DualDPT AUXILIARY ray head (DA3-BASE, N=1).

Produces dumps/reference_rays.gguf (flattened f32, row-major) + dumps/manifest_rays.json.
This is the GOLD reference the C++ aux-ray parity gate (test_head_rays) asserts against.

The aux branch is a fully independent pyramid alongside the (already parity-verified)
main depth head, sharing only the resized backbone feats (l1_rn..l4_rn). Only the FINEST
aux pyramid level is returned. The aux output is at the refinenet1_aux resolution
(8*pw x 8*ph = 128x128 for the 224 fixture), NOT upsampled to the input resolution.

Captured isolation checkpoints (NCHW from torch hooks):
  aux_refine1 : output of refinenet1_aux           (B,128,128,128) -> pre output_conv1_aux
  aux_out1    : output of output_conv1_aux[last]   (B, 64,128,128) -> post 5-conv reduce
Final outputs (reference order, row-major):
  ray         : [H_aux, W_aux, 6]  (linear/identity activation)
  ray_conf    : [H_aux, W_aux]     (conf_activation = expp1 = exp(x)+1)
"""
import os, json, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
from scripts.da3_reference import load_model, fixed_input, FIX_H, FIX_W, PATCH

OUT = "dumps/reference_rays.gguf"
MANIFEST = "dumps/manifest_rays.json"
OUT_LAYERS = [5, 7, 9, 11]


def main():
    os.makedirs("dumps", exist_ok=True)
    _, net = load_model()
    x, _raw = fixed_input()
    bb = net.backbone.pretrained
    head = net.head
    aux_levels = int(head.aux_levels)
    last = aux_levels - 1

    with torch.no_grad():
        outs, _aux = bb.get_intermediate_layers(
            x, n=OUT_LAYERS, export_feat_layers=[],
            ref_view_strategy="saddle_balanced")

    cap = {}
    handles = []

    def _mk_hook(key):
        def _h(_m, _inp, out):
            cap[key] = out.detach().contiguous().float()
        return _h

    handles.append(head.scratch.refinenet1_aux.register_forward_hook(_mk_hook("aux_refine1")))
    handles.append(head.scratch.output_conv1_aux[last].register_forward_hook(_mk_hook("aux_out1")))
    try:
        with torch.no_grad():
            head_out = net.head(list(outs), FIX_H, FIX_W, patch_start_idx=0)
    finally:
        for h in handles:
            h.remove()

    name_aux = head.head_aux  # "ray"
    ray = head_out[name_aux].squeeze(0).squeeze(0)             # (H_aux, W_aux, 6)
    ray_conf = head_out[f"{name_aux}_conf"].squeeze(0).squeeze(0)  # (H_aux, W_aux)

    assert ray.dim() == 3 and ray.shape[-1] == 6, ray.shape
    H_aux, W_aux = int(ray.shape[0]), int(ray.shape[1])
    assert tuple(ray_conf.shape) == (H_aux, W_aux), ray_conf.shape
    assert torch.isfinite(ray).all(), "ray must be finite"
    assert torch.isfinite(ray_conf).all() and bool((ray_conf > 0).all()), "ray_conf must be >0 (expp1)"

    cap["ray"] = ray.detach().contiguous().float()
    cap["ray_conf"] = ray_conf.detach().contiguous().float()

    w = gguf.GGUFWriter(OUT, "reference_rays")
    for k, v in cap.items():
        arr = np.ascontiguousarray(v.cpu().numpy().reshape(-1).astype(np.float32))
        w.add_tensor(k, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    shapes = {k: list(v.shape) for k, v in cap.items()}
    with open(MANIFEST, "w") as f:
        json.dump({"H_in": FIX_H, "W_in": FIX_W, "patch": PATCH,
                   "H_aux": H_aux, "W_aux": W_aux, "ray_dim": 6,
                   "aux_levels": aux_levels, "out_layers": OUT_LAYERS,
                   "shapes": shapes, "atol": 2e-3, "rtol": 2e-3}, f, indent=2)

    print("wrote", OUT)
    for k, v in cap.items():
        print(f"  {k}: {list(v.shape)}  finite={bool(torch.isfinite(v).all())}")
    print(f"aux resolution: H_aux={H_aux} W_aux={W_aux}")
    print(f"ray range: [{float(ray.min()):.4f}, {float(ray.max()):.4f}]")
    print(f"ray_conf range: [{float(ray_conf.min()):.4f}, {float(ray_conf.max()):.4f}]")


if __name__ == "__main__":
    main()
