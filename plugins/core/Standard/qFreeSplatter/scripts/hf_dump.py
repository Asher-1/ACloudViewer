#!/usr/bin/env python3
"""Dump PyTorch reference activation taps for the free-splatter front half.

Runs the upstream FreeSplatter Transformer on a fixed synthetic multi-view input
and writes, per tap, the exact float64 reference the C++ engine targets, in the
format scripts/compare_taps.py reads. Attention is computed explicitly in float64
(head-looped, so memory stays bounded and xformers/CUDA are not needed); every
other op runs in float64 too, so the reference is the EXACT value, not a second
noisy float path.

Taps are STREAMED to disk as they are produced (a full-depth run holds ~14 GB of
activations otherwise). With --head, also dumps result_norm, the unpatchify
output (head_logits) and the raw gaussians (forward_gaussians = transformer +
SH residual, pre-activation -- the authoritative "raw logits").

  python scripts/hf_dump.py --ckpt CKPT.safetensors --out DIR \
                            --views 2 --blocks 24 --head --seed 20260624

The input images.f32 is written alongside so the engine consumes identical bytes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

OUTPUT_DIM = 23   # scene / object-3dgs head width
C0 = 0.28209479177387814


def install_xformers_stub() -> None:
    def mea(q, k, v, attn_bias=None, scale=None, **kw):
        d = q.shape[-1]
        s = scale if scale is not None else 1.0 / math.sqrt(d)
        out = torch.empty_like(q)
        for i in range(q.shape[0]):
            out[i] = torch.softmax((q[i] @ k[i].transpose(-2, -1)) * s, dim=-1) @ v[i]
        return out
    xf = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")
    ops.memory_efficient_attention = mea
    xf.ops = ops
    sys.modules["xformers"], sys.modules["xformers.ops"] = xf, ops


def load_model(ckpt: str, upstream_dir: str):
    install_xformers_stub()
    sys.path.insert(0, upstream_dir)
    from transformer import Transformer  # type: ignore
    from safetensors import safe_open

    model = Transformer(image_size=512, patch_size=8, input_dim=3, inner_dim=1024,
                        output_dim=OUTPUT_DIM, n_heads=16, depth=24)
    sd = {}
    with safe_open(ckpt, framework="pt") as f:
        for k in f.keys():
            sd[k[len("transformer."):] if k.startswith("transformer.") else k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:3], unexpected[:3])
    return model.double().eval()


class Streamer:
    """Writes each tap to <dir>/<name>.f32 immediately; accumulates only meta."""
    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"taps": {}}

    def emit(self, name: str, arr) -> None:
        a = np.ascontiguousarray(arr)
        if a.ndim == 1:
            a = a[None, :]
        a = a.astype("<f4")
        a.tofile(self.dir / f"{name}.f32")
        self.meta["taps"][name] = {"shape": list(a.shape), "dtype": "f32"}

    def finish(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        print(f"wrote {len(self.meta['taps'])} taps to {self.dir}")


def sq(x):   # (1,S,D) -> (S,D) f64 numpy
    return x.detach().squeeze(0).cpu().numpy()


def flat(tok):  # (1,N,hw,C) -> (N*hw, C)
    return rearrange(tok, "b n hw c -> (b n hw) c").detach().cpu().numpy()


def attn_f64(sa, x, st, pre):
    h = sa.heads
    q, k, v = sa.to_q(x), sa.to_k(x), sa.to_v(x)
    st.emit(f"{pre}.q", sq(q)); st.emit(f"{pre}.k", sq(k)); st.emit(f"{pre}.v", sq(v))
    qh, kh, vh = (rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v))
    s = 1.0 / math.sqrt(qh.shape[-1])
    o = torch.empty_like(qh)
    for i in range(qh.shape[0]):
        o[i] = torch.softmax((qh[i] @ kh[i].transpose(-2, -1)) * s, dim=-1) @ vh[i]
    o = rearrange(o, "(b h) n d -> b n (h d)", h=h)
    out = sa.to_out(o)
    st.emit(f"{pre}.attn_out", sq(out))
    return out


@torch.inference_mode()
def run(model, images, n_blocks, do_head, st):
    B, N, C, H, W = images.shape
    st.emit("images", images.reshape(N, C * H * W).cpu().numpy())

    imgs = rearrange(images, "b n c h w -> (b n) c h w")
    tok = model.patchify(imgs)
    tok = rearrange(tok, "bn c h w -> bn (h w) c")
    tok = rearrange(tok, "(b n) hw c -> b n hw c", b=B)
    st.emit("patch_embed", flat(tok))
    tok = tok + model.interpolate_pos_encoding(tok, W, H).unsqueeze(1)
    st.emit("tokens_pos", flat(tok))
    ve = torch.cat([model.ref_embed, model.src_embed.repeat(1, N - 1, 1)], dim=1)
    tok = tok + ve.unsqueeze(2)
    st.emit("tokens_in", flat(tok))

    x = rearrange(tok, "b n hw c -> b (n hw) c")
    for i in range(n_blocks):
        blk = model.blocks[i]; pre = f"l{i}"
        bsa = blk.norm1(x);     st.emit(f"{pre}.attn_norm", sq(bsa))
        x = x + attn_f64(blk.self_attn, bsa, st, pre); st.emit(f"{pre}.attn_resid", sq(x))
        n2 = blk.norm2(x);      st.emit(f"{pre}.ffn_norm", sq(n2))
        up = blk.ff[0](n2);     st.emit(f"{pre}.ffn_up", sq(up))
        ge = blk.ff[1](up);     st.emit(f"{pre}.gelu", sq(ge))
        dn = blk.ff[2](ge);     st.emit(f"{pre}.ffn_down", sq(dn))
        x = x + dn;             st.emit(f"{pre}.l_out", sq(x))

    if do_head and n_blocks == len(model.blocks):
        rn = model.norm(x);     st.emit("result_norm", sq(rn))
        logits = model.unpatchify(rn)            # (B,S,1472)
        st.emit("head_logits", sq(logits))
        # forward_gaussians: unshuffle + SH residual (pre-activation raw logits).
        g = rearrange(logits, "b (n h w) c -> b n h w c", n=N, h=H // 8, w=W // 8)
        g = rearrange(g, "b n h w (p q c) -> b n (h p) (w q) c", p=8, q=8, c=OUTPUT_DIM)
        resid = torch.zeros_like(g)
        resid[..., 3:6] = (rearrange(images, "b n c h w -> b n h w c") - 0.5) / C0
        g = g + resid
        st.emit("gaussians_raw", rearrange(g, "b n h w c -> (b n h w) c").cpu().numpy())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--upstream", default=".cache/upstream")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--seed", type=int, default=20260624)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    imgs_np = rng.random((args.views, 3, 512, 512), dtype=np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    imgs_np.tofile(args.out / "images.f32")

    model = load_model(args.ckpt, args.upstream)
    images = torch.from_numpy(imgs_np).double().unsqueeze(0)

    st = Streamer(args.out)
    run(model, images, args.blocks, args.head, st)
    st.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
