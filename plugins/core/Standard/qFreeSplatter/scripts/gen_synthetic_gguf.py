#!/usr/bin/env python3
"""Emit a structurally valid (zero-weight) free-splatter GGUF.

This is a DEV/CI artifact, not a real model: it lets the full load path
(model_file -> map_tensors -> realize_weights) and the CLI be exercised on CPU
without the upstream checkpoint. Real weights come from convert.py.

  python scripts/gen_synthetic_gguf.py out.gguf [--scene|--object|--object-2dgs]
"""
from __future__ import annotations

import argparse

import numpy as np
import gguf

ARCH = "free-splatter"


def build(out: str, variant: str) -> None:
    n_layer, n_embd, n_ff, n_head, head_dim = 24, 1024, 4096, 16, 64
    patch, image, in_ch, sh_degree = 8, 512, 3, 1
    gauss = 22 if variant == "object-2dgs" else 23
    sh_residual = variant == "scene"
    use_2dgs = variant == "object-2dgs"
    ph = pw = patch
    tpv = (image // patch) ** 2  # tokens per view

    w = gguf.GGUFWriter(out, ARCH)
    w.add_uint32(f"{ARCH}.block_count", n_layer)
    w.add_uint32(f"{ARCH}.embedding_length", n_embd)
    w.add_uint32(f"{ARCH}.feed_forward_length", n_ff)
    w.add_uint32(f"{ARCH}.attention.head_count", n_head)
    w.add_uint32(f"{ARCH}.attention.key_length", head_dim)
    w.add_uint32(f"{ARCH}.vision.patch_size", patch)
    w.add_uint32(f"{ARCH}.vision.image_size", image)
    w.add_uint32(f"{ARCH}.vision.in_channels", in_ch)
    w.add_uint32(f"{ARCH}.vision.gaussian_channels", gauss)
    w.add_uint32(f"{ARCH}.sh_degree", sh_degree)
    w.add_bool(f"{ARCH}.sh_residual", sh_residual)
    w.add_bool(f"{ARCH}.use_2dgs", use_2dgs)
    w.add_float32(f"{ARCH}.attention.layer_norm_epsilon", 1e-5)
    w.add_float32(f"{ARCH}.scale_min_act", 1e-4)
    w.add_float32(f"{ARCH}.scale_max_act", 0.02)

    def z(name, *shape):  # zero tensor, numpy (out, in, ...) -> ggml ne reversed
        w.add_tensor(name, np.zeros(shape, dtype=np.float32))

    z("patch_embed.weight", n_embd, in_ch, ph, pw)  # (1024, 3, 8, 8) -> ne [8,8,3,1024]
    z("pos_embed", tpv, n_embd)                            # (4096, 1024)
    z("view_embed.ref", n_embd)
    z("view_embed.src", n_embd)
    for i in range(n_layer):
        p = f"blk.{i}."
        z(p + "attn_norm.weight", n_embd)
        z(p + "attn_q.weight", n_embd, n_embd)
        z(p + "attn_k.weight", n_embd, n_embd)
        z(p + "attn_v.weight", n_embd, n_embd)
        z(p + "attn_out.weight", n_embd, n_embd)
        z(p + "ffn_norm.weight", n_embd)
        z(p + "ffn_up.weight", n_ff, n_embd)
        z(p + "ffn_down.weight", n_embd, n_ff)
    z("output_norm.weight", n_embd)
    z("unpatch.weight", ph * pw * gauss, n_embd)  # (1472, 1024)
    z("unpatch.bias", ph * pw * gauss)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out} (variant={variant}, gaussian_channels={gauss})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--variant", choices=["scene", "object", "object-2dgs"], default="scene")
    args = ap.parse_args()
    build(args.out, args.variant)


if __name__ == "__main__":
    main()
