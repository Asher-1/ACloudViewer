#!/usr/bin/env python3
"""TRELLIS.2 backend A/B timing harness (same methodology as
scripts/ggml_upgrade_verify.py: identical inputs, one variable changed,
machine-readable comparison output).

Runs the AICore TRELLIS.2 pipeline (libAICore.so) on one or more compute
backends for a fixed image + quality + seed and reports per-stage wall times
plus geometry hashes, so backend precision/speed regressions are visible in
one table:

    python3 core/AICore/src/tasks/trellis/tools/trellis_backend_ab.py \
        --models ~/models --image test.png --device cuda,vulkan --pipeline 512

Requires: AICore built (build_app/bin/libAICore.so), the TRELLIS GGUFs in
--models (dino/ss_flow/ss_dec for coarse; + slat_flow/shape_dec/shape_enc/
tex_dec/tex_slat_flow_512 for 512), optionally rmbg.
"""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time

# aicore_trellis_stage / pipeline_type / background_mode (trellis_capi.h).
STAGE_NAMES = {
    0: "preprocess", 1: "dino", 2: "ss_flow", 3: "ss_dec", 4: "slat_flow",
    5: "shape_dec", 6: "mesh", 7: "upsample", 8: "slat_flow_hr",
    9: "shape_dec_hr", 10: "texture",
}
PIPE_COARSE, PIPE_512, PIPE_1024 = 1, 2, 3
PIPE_ALIASES = {"coarse": PIPE_COARSE, "512": PIPE_512, "1024": PIPE_1024}


def find_lib(root_hint=None):
    candidates = []
    if root_hint:
        candidates.append(os.path.join(root_hint, "libAICore.so"))
    here = os.path.dirname(os.path.abspath(__file__))
    # core/AICore/src/tasks/trellis/tools -> repo root is six levels up.
    candidates.append(os.path.join(here, "..", "..", "..", "..", "..", "..",
                                   "build_app", "bin", "libAICore.so"))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise SystemExit("libAICore.so not found (hint: --lib)")


# void (*)(void* user, int stage, int step, int total)
ProgressFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int)


class Timer:
    """Accumulates per-stage wall times from the stage-entry callbacks."""

    def __init__(self):
        self.stage_ms = {}
        self._last = None
        self._t0 = time.perf_counter()

    def __call__(self, _user, stage, step, total):
        now = time.perf_counter()
        if self._last is not None and self._last != stage:
            self.stage_ms[self._last] = (self.stage_ms.get(self._last, 0.0)
                                         + (now - self._t0) * 1000.0)
            self._t0 = now
        self._last = stage

    def finish(self):
        if self._last is not None:
            now = time.perf_counter()
            self.stage_ms[self._last] = (self.stage_ms.get(self._last, 0.0)
                                         + (now - self._t0) * 1000.0)


def build_paths(models, pipeline):
    """aicore_trellis_model_paths field order; "" omits a model."""
    need = ["dino_q8.gguf", "ss_flow_q8.gguf", "ss_dec_q8.gguf"]
    if pipeline in (PIPE_512, PIPE_1024):
        need += ["slat_flow_q8.gguf", "", "shape_dec_f16.gguf",
                 "shape_enc_f16.gguf", "tex_dec_f16.gguf",
                 "tex_slat_flow_512_q8.gguf"]
    if pipeline == PIPE_1024:
        need[4] = "slat_flow_1024_q8.gguf"
        need[9] = "tex_slat_flow_1024_q8.gguf"
    paths = [os.path.join(models, n) if n else "" for n in need]
    for p in paths:
        if p and not os.path.exists(p):
            raise SystemExit(f"missing model: {p}")
    return paths


def run_once(lib, models, image_bytes, device, pipeline, steps, seed):
    opts = lib.aicore_trellis_options_new()
    lib.aicore_trellis_options_set_device(opts, device.encode())
    lib.aicore_trellis_options_set_threads(opts, 0)

    class ModelPaths(ctypes.Structure):
        _fields_ = [(n, ctypes.c_char_p) for n in
                    ("dino_gguf", "ss_flow_gguf", "ss_dec_gguf",
                     "slat_flow_gguf", "slat_hr_flow_gguf", "shape_dec_gguf",
                     "shape_enc_gguf", "tex_dec_gguf", "tex_flow_gguf",
                     "tex_flow_hr_gguf")]

    mp = ModelPaths(*[p.encode() for p in build_paths(models, pipeline)])
    ctx = lib.aicore_trellis_load_opts(ctypes.byref(mp), opts)
    lib.aicore_trellis_options_free(opts)
    if not ctx:
        raise SystemExit("model load failed")

    class Params(ctypes.Structure):
        _fields_ = [("pipeline_type", ctypes.c_int),
                    ("background_mode", ctypes.c_int),
                    ("seed", ctypes.c_uint64), ("steps", ctypes.c_int),
                    ("guidance", ctypes.c_float),
                    ("texture_steps", ctypes.c_int),
                    ("preview_stride", ctypes.c_int),
                    ("keyframes", ctypes.c_int)]

    params = Params(pipeline_type=pipeline, background_mode=0, seed=seed,
                    steps=steps, guidance=-1.0, texture_steps=steps,
                    preview_stride=-1, keyframes=0)

    timer = Timer()
    cb = ProgressFn(timer)
    err = ctypes.create_string_buffer(512)
    mesh = lib.aicore_trellis_generate_ex(
        ctx, image_bytes, len(image_bytes), ctypes.byref(params), cb, None,
        ctypes.cast(None, ctypes.c_void_p), None, err, 512)
    timer.finish()
    if not mesh:
        raise SystemExit(f"generate failed: {err.value.decode()}")
    nv = lib.aicore_trellis_mesh_n_verts(mesh)
    nt = lib.aicore_trellis_mesh_n_tris(mesh)
    verts = ctypes.string_at(lib.aicore_trellis_mesh_verts(mesh),
                             nv * 3 * 4) if nv else b""
    geo = hashlib.sha256(verts).hexdigest()[:12]
    result = {
        "device": device,
        "verts": nv,
        "tris": nt,
        "geometry_sha12": geo,
        "stage_ms": {STAGE_NAMES.get(k, str(k)): round(v, 1)
                     for k, v in sorted(timer.stage_ms.items())},
        "total_ms": round(sum(timer.stage_ms.values()), 1),
    }
    lib.aicore_trellis_mesh_free(mesh)
    lib.aicore_trellis_free(ctx)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda,vulkan",
                    help="comma list: cuda, vulkan, cpu")
    ap.add_argument("--pipeline", default="coarse",
                    help="coarse | 512 | 1024")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lib", default=None, help="path to libAICore.so")
    ap.add_argument("--torch-lib", default=None,
                    help="dir holding libtorch.so (required when the build "
                         "enabled AICore_USE_CUMESH; preloaded so dlopen can "
                         "resolve the libaicore_cumesh dependency chain)")
    args = ap.parse_args()

    lib_path = find_lib(args.lib)
    if args.torch_lib:
        # dlopen resolves DT_NEEDED breadth-first; preloading the torch core
        # in dependency order lets libaicore_cumesh.so resolve afterwards.
        for name in ("libc10.so", "libc10_cuda.so", "libtorch_cpu.so",
                     "libtorch_cuda.so", "libtorch.so"):
            ctypes.CDLL(os.path.join(args.torch_lib, name),
                        mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(lib_path)
    lib.aicore_trellis_abi_version.restype = ctypes.c_int
    # Explicit prototypes: without argtypes ctypes narrows pointer returns to
    # 32-bit ints and every handle deref segfaults.
    _P = ctypes.c_void_p
    lib.aicore_trellis_options_new.restype = _P
    lib.aicore_trellis_options_free.argtypes = [_P]
    lib.aicore_trellis_options_set_device.argtypes = [_P, ctypes.c_char_p]
    lib.aicore_trellis_options_set_threads.argtypes = [_P, ctypes.c_int]
    lib.aicore_trellis_load_opts.argtypes = [_P, _P]
    lib.aicore_trellis_load_opts.restype = _P
    lib.aicore_trellis_free.argtypes = [_P]
    lib.aicore_trellis_generate_ex.argtypes = [
        _P, ctypes.c_void_p, ctypes.c_int, _P, ProgressFn, _P, _P, _P,
        ctypes.c_char_p, ctypes.c_int]
    lib.aicore_trellis_generate_ex.restype = _P
    lib.aicore_trellis_mesh_free.argtypes = [_P]
    lib.aicore_trellis_mesh_n_verts.argtypes = [_P]
    lib.aicore_trellis_mesh_n_verts.restype = ctypes.c_int
    lib.aicore_trellis_mesh_n_tris.argtypes = [_P]
    lib.aicore_trellis_mesh_n_tris.restype = ctypes.c_int
    lib.aicore_trellis_mesh_verts.argtypes = [_P]
    lib.aicore_trellis_mesh_verts.restype = ctypes.POINTER(ctypes.c_char)
    print(f"lib={lib_path} abi={lib.aicore_trellis_abi_version()}")

    pipeline = PIPE_ALIASES[args.pipeline]
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    results = []
    for device in [d.strip() for d in args.device.split(",") if d.strip()]:
        print(f"=== running device={device} pipeline={args.pipeline} "
              f"steps={args.steps} ...", flush=True)
        t0 = time.perf_counter()
        r = run_once(lib, args.models, image_bytes, device, pipeline,
                     args.steps, args.seed)
        r["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    print("\n## A/B summary\n")
    keys = sorted({k for r in results for k in r["stage_ms"]})
    header = "| stage (ms) | " + " | ".join(r["device"] for r in results) + " |"
    print(header)
    print("|---" * (len(results) + 1) + "|")
    for k in keys:
        row = " | ".join(str(r["stage_ms"].get(k, "-")) for r in results)
        print(f"| {k} | {row} |")
    row = " | ".join(str(r["total_ms"]) for r in results)
    print(f"| **total** | {row} |")
    row = " | ".join(f"{r['verts']} v / {r['tris']} t ({r['geometry_sha12']})"
                     for r in results)
    print(f"\ngeometry: {row}")


if __name__ == "__main__":
    main()
