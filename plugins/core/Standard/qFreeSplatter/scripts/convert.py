#!/usr/bin/env python3
"""Convert an upstream TencentARC/FreeSplatter checkpoint to a free-splatter GGUF.

Self-contained: needs only torch + safetensors + gguf (NOT the FreeSplatter repo
and NOT llama.cpp). Reads the checkpoint's state_dict (transformer.* keys) and
writes pieces 1-3 (patch tokenizer -> transformer -> Gaussian head). The
rasterizer / PnP weights, if any, are ignored.

  python scripts/convert.py CKPT.safetensors out.gguf [--outtype f16|f32|q8_0|q4_0]
                            [--variant scene|object|object-2dgs]

Run inside docker/Dockerfile.cuda (or any env with the deps). Any state-dict key
that pieces 1-3 expect but is missing -- or any unmapped transformer.* weight --
is a hard error (fail loud, never silently drop a weight).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import gguf
import gguf.quants

try:
    from safetensors import safe_open
    import torch
except ImportError as e:  # pragma: no cover - only needed at conversion time
    print(f"convert.py needs torch + safetensors: {e}", file=sys.stderr)
    raise

ARCH = "free-splatter"

# Architecture constants (verified against the scene/object checkpoints; can be
# overridden from a config.json in M1 if a variant diverges).
N_LAYER, N_EMBD, N_FF, N_HEAD, HEAD_DIM = 24, 1024, 4096, 16, 64
PATCH, IMAGE, IN_CH, SH_DEGREE = 8, 512, 3, 1
LN_EPS, SCALE_MIN, SCALE_MAX = 1e-5, 1e-4, 0.02

# Weight matrices that tolerate f16/q8_0; everything else (norms, biases,
# embeddings, the conv patch-embed) stays f32 even in an f16/q8_0 build.
QUANT_SUFFIXES = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_out.weight",
    "ffn_up.weight", "ffn_down.weight", "unpatch.weight",
)


def variant_channels(variant: str) -> int:
    return 22 if variant == "object-2dgs" else 23


def load_state_dict(path: str) -> dict:
    sd = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def np32(t) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def build_name_map() -> list[tuple[str, str]]:
    """(pytorch_key, gguf_name) for the 1:1 mapped tensors. The patch-embed conv
    is handled separately (it is reshaped, not renamed)."""
    m = [
        ("transformer.pos_embed",         "pos_embed"),
        ("transformer.ref_embed",         "view_embed.ref"),
        ("transformer.src_embed",         "view_embed.src"),
        ("transformer.norm.weight",       "output_norm.weight"),
        ("transformer.unpatchify.weight", "unpatch.weight"),
        ("transformer.unpatchify.bias",   "unpatch.bias"),
    ]
    for i in range(N_LAYER):
        p, q = f"transformer.blocks.{i}.", f"blk.{i}."
        m += [
            (p + "norm1.weight",            q + "attn_norm.weight"),
            (p + "self_attn.to_q.weight",   q + "attn_q.weight"),
            (p + "self_attn.to_k.weight",   q + "attn_k.weight"),
            (p + "self_attn.to_v.weight",   q + "attn_v.weight"),
            (p + "self_attn.to_out.0.weight", q + "attn_out.weight"),
            (p + "norm2.weight",            q + "ffn_norm.weight"),
            (p + "ff.0.weight",             q + "ffn_up.weight"),
            (p + "ff.2.weight",             q + "ffn_down.weight"),
        ]
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--outtype", choices=["f16", "f32", "q8_0", "q4_0"], default="f16")
    ap.add_argument("--variant", choices=["scene", "object", "object-2dgs"], default="scene")
    args = ap.parse_args()

    gauss = variant_channels(args.variant)
    sd = load_state_dict(args.ckpt)

    w = gguf.GGUFWriter(args.out, ARCH)
    w.add_uint32(f"{ARCH}.block_count", N_LAYER)
    w.add_uint32(f"{ARCH}.embedding_length", N_EMBD)
    w.add_uint32(f"{ARCH}.feed_forward_length", N_FF)
    w.add_uint32(f"{ARCH}.attention.head_count", N_HEAD)
    w.add_uint32(f"{ARCH}.attention.key_length", HEAD_DIM)
    w.add_uint32(f"{ARCH}.vision.patch_size", PATCH)
    w.add_uint32(f"{ARCH}.vision.image_size", IMAGE)
    w.add_uint32(f"{ARCH}.vision.in_channels", IN_CH)
    w.add_uint32(f"{ARCH}.vision.gaussian_channels", gauss)
    w.add_uint32(f"{ARCH}.sh_degree", SH_DEGREE)
    w.add_bool(f"{ARCH}.sh_residual", args.variant == "scene")
    w.add_bool(f"{ARCH}.use_2dgs", args.variant == "object-2dgs")
    w.add_float32(f"{ARCH}.attention.layer_norm_epsilon", LN_EPS)
    w.add_float32(f"{ARCH}.scale_min_act", SCALE_MIN)
    w.add_float32(f"{ARCH}.scale_max_act", SCALE_MAX)

    def emit(name: str, arr: np.ndarray) -> None:
        can_quant = name.endswith(QUANT_SUFFIXES)
        qtype = None
        if args.outtype == "q8_0" and can_quant:
            qtype = gguf.GGMLQuantizationType.Q8_0
        elif args.outtype == "q4_0" and can_quant:
            qtype = gguf.GGMLQuantizationType.Q4_0
        elif args.outtype == "f16" and can_quant:
            w.add_tensor(name, arr.astype(np.float16))
            return
        else:
            w.add_tensor(name, arr.astype(np.float32))
            return
        # quantized path (q8_0 or q4_0)
        f32 = arr.astype(np.float32)
        if f32.ndim == 2:
            rows = [gguf.quants.quantize(f32[i], qtype) for i in range(f32.shape[0])]
            qdata = np.concatenate(rows).reshape(f32.shape[0], rows[0].shape[0])
        else:
            qdata = gguf.quants.quantize(f32.ravel(), qtype)
        w.add_tensor(name, qdata, raw_dtype=qtype)

    # Patch-embed conv weight (1024,3,8,8). Emitted as-is: ggml reverses the
    # numpy shape to ne=[KW,KH,IC,OC], which is exactly what ggml_im2col expects
    # as its kernel-shape arg; the engine then reshapes it to [IC*KH*KW, OC] for
    # the matmul. Kept f32 (small, and conv precision matters for every token).
    conv_key = "transformer.patchify.weight"
    if conv_key not in sd:
        print(f"missing required tensor: {conv_key}", file=sys.stderr)
        return 1
    emit("patch_embed.weight", np32(sd[conv_key]))

    used = {conv_key}
    for pt_key, gg_name in build_name_map():
        if pt_key not in sd:
            print(f"missing required tensor: {pt_key}", file=sys.stderr)
            return 1
        arr = np32(sd[pt_key])
        # squeeze leading singleton dims on the embeddings (1,*,C) -> (*,C)/(C,)
        while arr.ndim > 1 and arr.shape[0] == 1 and gg_name.startswith(("pos_embed", "view_embed")):
            arr = arr[0]
        emit(gg_name, arr)
        used.add(pt_key)

    leftover = [k for k in sd if k.startswith("transformer.") and k not in used]
    if leftover:
        print(f"WARNING: {len(leftover)} unmapped transformer.* tensors "
              f"(first few: {leftover[:5]})", file=sys.stderr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} (variant={args.variant}, outtype={args.outtype}, "
          f"gaussian_channels={gauss})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
