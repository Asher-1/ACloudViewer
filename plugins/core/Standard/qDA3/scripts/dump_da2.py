#!/usr/bin/env python3
"""Dump DA2 gold tensors for the C++ parity gates: the 4 tapped intermediate
features (post final-norm, cls-stripped — exactly get_intermediate_layers(norm=True))
and the final forward() depth, on the shared fixed input.

Feature layout matches DinoBackbone::forward cat_token=false: feat[o] = vit.norm(x)
for patch tokens 1..N, flat index = token*embed + channel."""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf
from scripts.da2_reference import load_da2_model, fixed_input
from scripts.convert_da2_to_gguf import INTERMEDIATE_IDX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--ckpt", default="models/da2/depth_anything_v2_vitl.pth")
    ap.add_argument("--max-depth", type=float, default=0.0)
    ap.add_argument("--output", default="dumps/reference_da2.gguf")
    ap.add_argument("-k", type=int, default=16)   # fixture is 14k x 14k
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.output), exist_ok=True)

    net = load_da2_model(a.encoder, a.ckpt, a.max_depth)
    chw = fixed_input(a.k)                                   # (3,n,n)
    n = chw.shape[-1]
    x = torch.from_numpy(chw)[None]                          # (1,3,n,n)
    idx = INTERMEDIATE_IDX[a.encoder]
    with torch.no_grad():
        feats = net.pretrained.get_intermediate_layers(
            x, idx, return_class_token=True)                 # tuple of (patch[1,N,C], cls)
        depth = net(x)                                       # (1,n,n)

    gw = gguf.GGUFWriter(a.output, "da2_ref")
    gw.add_uint32("da2.n", int(n))
    gw.add_tensor("input_image", np.ascontiguousarray(chw.reshape(-1), dtype=np.float32))
    for li, (patch, _cls) in zip(idx, feats):
        arr = patch[0].contiguous().float().cpu().numpy().reshape(-1)  # (N*C,) token-major
        gw.add_tensor(f"feat_da2_{li}", np.ascontiguousarray(arr, dtype=np.float32))
    gw.add_tensor("depth_da2", np.ascontiguousarray(depth[0].float().cpu().numpy().reshape(-1),
                                                     dtype=np.float32))
    gw.write_header_to_file(); gw.write_kv_data_to_file(); gw.write_tensors_to_file(); gw.close()
    print(f"wrote {a.output}: n={n} feats={list(idx)} depth={int(n)}x{int(n)}")


if __name__ == "__main__":
    main()
