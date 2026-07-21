#!/usr/bin/env python3
"""Time the upstream FreeSplatter Transformer front-half in PyTorch.

This is the *performance* reference (cf. hf_dump.py, which is the float64
*numerical* reference and is deliberately slow). It runs the real transformer at
native precision with an efficient attention kernel (torch SDPA, the same
flash/mem-efficient path our engine's ggml_flash_attn_ext targets), with proper
warmup and CUDA synchronization, and reports per-forward latency so it can be
compared head-to-head with free_splatter-bench.

What is timed matches the engine's forward: patchify -> +pos/view embed ->
24 transformer blocks -> final norm -> unpatchify (the [N,512,512,23] raw
logits). The engine additionally applies the SH residual + activations, a
sub-millisecond elementwise tail on top of this; it is not included here (the
upstream forward stops at the raw logits).

  nix develop -c bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; \
    .venv-torch-cu128/bin/python scripts/bench_torch.py \
      --ckpt .cache/freesplatter-scene.safetensors --device cuda --dtype fp16 --views 2'

The last line is machine-parseable (scripts/bench.sh reads it):
  RESULT torch device=cuda dtype=fp16 views=N ... median_ms=.. views_per_s=..
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import types

import torch

OUTPUT_DIM = 23


def install_sdpa_stub() -> None:
    """Provide xformers.ops.memory_efficient_attention backed by torch SDPA.

    The upstream CrossAttention calls xops.memory_efficient_attention(q,k,v) with
    q,k,v shaped ((b h), n, d); SDPA on that 3D layout computes exactly
    softmax(q káµ€ / sqrt(d)) v per (batch*head), the fast flash/mem-efficient path.
    Original FreeSplatter uses the real xformers flash kernel; SDPA's flash
    backend is the equivalent. attention is ~57% of the FLOPs at S=8192, so the
    kernel choice matters -- bench_torch's --sdpa pins it for a fair comparison.
    """
    def mea(q, k, v, attn_bias=None, scale=None, **kw):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    xf = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")
    ops.memory_efficient_attention = mea
    xf.ops = ops
    sys.modules["xformers"], sys.modules["xformers.ops"] = xf, ops


def sdpa_context(which: str):
    """Pin the SDPA backend (flash/mem/math) so attention kernel choice is
    explicit rather than the heuristic default. Returns a context manager."""
    import contextlib
    if which == "auto":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = {
        "flash": [SDPBackend.FLASH_ATTENTION],
        "mem":   [SDPBackend.EFFICIENT_ATTENTION],
        "math":  [SDPBackend.MATH],
    }[which]
    return sdpa_kernel(backends)


def load_model(ckpt: str, upstream_dir: str, dtype, device):
    install_sdpa_stub()
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
    return model.eval().to(device=device, dtype=dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--upstream", default=".cache/upstream")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default=None, choices=["fp32", "fp16", "bf16"],
                    help="default: fp32 on cpu, fp16 on cuda")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0, help="cpu intra-op threads (0=torch default)")
    ap.add_argument("--compile", action="store_true", help="wrap the model in torch.compile (first warmup absorbs compilation)")
    ap.add_argument("--sdpa", default="auto", choices=["auto", "flash", "mem", "math"],
                    help="pin the SDPA attention backend (default: auto/heuristic)")
    ap.add_argument("--seed", type=int, default=20260624)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("error: CUDA requested but torch.cuda.is_available() is False "
              "(need a cu1xx torch build + /run/opengl-driver/lib on LD_LIBRARY_PATH)", file=sys.stderr)
        return 1
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    dt = args.dtype or ("fp16" if args.device == "cuda" else "fp32")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dt]
    device = torch.device(args.device)

    t_load0 = time.perf_counter()
    model = load_model(args.ckpt, args.upstream, dtype, device)
    if args.compile:
        model = torch.compile(model)
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t_load0) * 1e3

    g = torch.Generator().manual_seed(args.seed)
    images = torch.rand(1, args.views, 3, 512, 512, generator=g).to(device=device, dtype=dtype)

    S = args.views * (512 // 8) * (512 // 8)
    name = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    print(f"torch {torch.__version__}  device={args.device} ({name})  dtype={dt}  "
          f"views={args.views}  S={S}  threads={torch.get_num_threads()}  load={load_ms:.1f} ms")

    def once() -> float:
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), sdpa_context(args.sdpa):  # fresh ctx each call (one-shot)
            _ = model(images)
        if args.device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(args.warmup):
        once()
    ts = []
    for i in range(args.iters):
        dt_ms = once()
        ts.append(dt_ms)
        print(f"  iter {i:2d}: {dt_ms:8.1f} ms")

    ts_sorted = sorted(ts)
    mn, mx = ts_sorted[0], ts_sorted[-1]
    med = ts_sorted[len(ts_sorted) // 2]
    mean = statistics.fmean(ts)
    vps = args.views / (med / 1e3)
    print(f"\nsummary: min={mn:.1f}  median={med:.1f}  mean={mean:.1f}  max={mx:.1f} ms   "
          f"({vps:.2f} views/s, {1e3/med:.2f} scenes/s)")
    print(f"RESULT torch device={args.device} dtype={dt} views={args.views} "
          f"load_ms={load_ms:.1f} min_ms={mn:.1f} median_ms={med:.1f} mean_ms={mean:.1f} "
          f"max_ms={mx:.1f} views_per_s={vps:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
