#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
#
# InstantMesh backend A/B probe (ctypes over libAICore). Mirrors the trellis
# probe contract: one JSON object per run on stdout (machine-readable row for
# the all-model validation runner), then a human A/B table.
#
#   usage: instantmesh_backend_ab.py --models DIR --image IMG
#          [--device auto,cuda,vulkan,cpu] [--preset full|geometry]
#          [--quant f16|q8|f32] [--steps 75] [--seed 42] [--grid-res 64]
#          [--no-texture] [--lib PATH]

import argparse
import ctypes
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path


class ModelPaths(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32)] + [
        (name, ctypes.c_char_p) for name in
        ("cond_gguf", "unet_gguf", "vae_gguf", "dino_gguf", "transformer_gguf", "synthesizer_gguf")]


class Params(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("steps", ctypes.c_int), ("seed", ctypes.c_uint64),
                ("guidance", ctypes.c_float), ("grid_res", ctypes.c_int),
                ("grid_scale", ctypes.c_float), ("texture_res", ctypes.c_int),
                ("bake_texture", ctypes.c_int)]


class PipelineTimings(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("valid_fields", ctypes.c_uint32)] + [
        (name + "_ms", ctypes.c_double) for name in
        ("preprocess", "inference", "postprocess", "serialization", "e2e")]


def public_timings(value):
    """Publish only stages measured by the public contract; never derive E2E from progress or stage sums."""
    if value.abi_version != 1:
        raise ValueError("pipeline timings ABI mismatch")
    if value.valid_fields & 18 != 18:
        raise ValueError("pipeline timings missing inference or e2e")
    result = {}
    for bit, name in enumerate(("preprocess", "inference", "postprocess", "serialization", "e2e")):
        if value.valid_fields & (1 << bit):
            number = getattr(value, name + "_ms")
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"invalid {name}_ms")
            result[name + "_ms"] = number
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize_timings(rows):
    if not rows:
        raise ValueError("no timed forwards")
    result = {}
    # Do not fabricate zero samples for absent stages; every forward has E2E/inference.
    for key in sorted(set.intersection(*(set(row) for row in rows))):
        name = key.removesuffix("_ms")
        values = [row[key] for row in rows]
        result[name + "_samples_ms"] = values
        result[name + "_p50_ms"] = percentile(values, 0.5)
        result[name + "_p95_ms"] = percentile(values, 0.95)
    result["total_ms"] = result["e2e_p50_ms"]
    result["stage_ms"] = {key.removesuffix("_p50_ms"): value
                          for key, value in result.items()
                          if key.endswith("_p50_ms") and not key.startswith("e2e_")}
    return result


def validate_backend(requested, actual):
    wanted, _, index = requested.lower().partition(":")
    actual = actual.lower()
    if not actual:
        raise ValueError("missing actual backend")
    if wanted == "auto":
        return
    matches = (actual != "cpu" if wanted == "gpu" else
               actual == "cpu" if wanted == "cpu" else
               wanted in ("cuda", "vulkan", "metal") and actual.startswith(wanted))
    if not matches or (index and actual != wanted + index):
        raise ValueError(f"backend mismatch: requested={requested}, actual={actual}")


def resolve_library(path):
    """Explicit file or build/install directory; never fall back to old candidates in other builds or the system."""
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    names = (("AICore.dll", "libAICore.dll") if os.name == "nt" else
             ("libAICore.dylib",) if sys.platform == "darwin" else ("libAICore.so",))
    for directory in (path / "bin", path / "lib", path):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"AICore library unavailable: {path}")


# ctypes progress callback (user, stage, ...) -> None; module-level so
# the prototype declarations in main() can reference it.
ProgressFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int)


def build_paths(models, preset, quant):
    """aicore_instantmesh_model_paths field order; "" omits a model."""
    def name(stem, q):
        return f"{stem}_{q}.gguf"

    need = ["", "", "", "", "", ""]
    if preset == "full":
        need[0] = "zero123pp_cond_f32.gguf"
        need[1] = name("zero123pp_unet", quant if quant != "q8" else "f16")
        need[2] = name("zero123pp_vae", quant if quant != "q8" else "f16")
    need[3] = name("dino", quant)
    need[4] = name("lrm_transformer", quant)
    need[5] = name("synthesizer", quant)
    paths = [os.path.join(models, n) if n else "" for n in need]
    for p in paths:
        if p and not os.path.exists(p):
            raise SystemExit(f"missing model: {p}")
    return paths


def inspect_mesh(lib, mesh, texture):
    nv = lib.aicore_instantmesh_mesh_n_verts(mesh)
    nt = lib.aicore_instantmesh_mesh_n_tris(mesh)
    if nv <= 0 or nt <= 0:
        raise ValueError(f"degenerate mesh: verts={nv} tris={nt}")

    def read_floats(pointer, count, label):
        if not pointer:
            raise ValueError(f"missing {label}")
        data = ctypes.string_at(pointer, count * 4)
        values = struct.unpack(f"={count}f", data)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite {label}")
        return data, values

    verts, floats = read_floats(lib.aicore_instantmesh_mesh_verts(mesh), nv * 3, "vertices")
    colors, values = read_floats(lib.aicore_instantmesh_mesh_vcolors(mesh), nv * 3, "colors")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("invalid vertex color range")
    pointer = lib.aicore_instantmesh_mesh_tris(mesh)
    if not pointer:
        raise ValueError("missing triangles")
    tris = ctypes.string_at(pointer, nt * 3 * 4)
    if any(index < 0 or index >= nv for index in struct.unpack(f"={nt * 3}i", tris)):
        raise ValueError("out-of-range triangle index")
    tw, th = lib.aicore_instantmesh_mesh_texture_w(mesh), lib.aicore_instantmesh_mesh_texture_h(mesh)
    texels = uvs = b""
    if texture:
        if not (0 < tw <= 4096 and 0 < th <= 4096):
            raise ValueError("missing or invalid texture atlas")
        uvs, values = read_floats(lib.aicore_instantmesh_mesh_uvs(mesh), nt * 6, "UV")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("invalid UV range")
        pointer = lib.aicore_instantmesh_mesh_texture(mesh)
        if not pointer:
            raise ValueError("missing RGBA texture")
        texels = ctypes.string_at(pointer, tw * th * 4)
    elif tw or th:
        raise ValueError("unexpected texture in vertex-color mode")
    axes = [floats[axis::3] for axis in range(3)]
    return {"verts": nv, "tris": nt, "finite": True,
            "has_texture": bool(tw), "texture_w": tw, "texture_h": th,
            "geometry_sha12": hashlib.sha256(verts + tris).hexdigest()[:12],
            "color_sha256": hashlib.sha256(colors).hexdigest(),
            "texture_sha256": hashlib.sha256(uvs + texels).hexdigest() if texture else "",
            "bbox_min": [min(axis) for axis in axes], "bbox_max": [max(axis) for axis in axes]}


def run_once(lib, models, image_bytes, device, preset, quant, steps, seed,
             grid_res, texture, warmups=1, iterations=2, texture_res=2048, threads=0):
    """Load once per device; warmup/timed forwards share a context, released on every exception path."""
    paths = build_paths(models, preset, quant)
    mp = ModelPaths(ctypes.sizeof(ModelPaths), lib.aicore_instantmesh_abi_version(),
                    *[path.encode() for path in paths])
    opts = lib.aicore_instantmesh_options_new()
    if not opts:
        raise MemoryError("options allocation failed")
    err = ctypes.create_string_buffer(1024)
    try:
        lib.aicore_instantmesh_options_set_device(opts, device.encode())
        lib.aicore_instantmesh_options_set_threads(opts, threads)
        t0 = time.perf_counter()
        ctx = lib.aicore_instantmesh_load_opts_ex(ctypes.byref(mp), opts, err, len(err))
        load_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        lib.aicore_instantmesh_options_free(opts)
    if not ctx:
        raise ValueError(f"model load failed: {err.value.decode(errors='replace')}")
    try:
        actual = lib.aicore_instantmesh_backend(ctx).decode()
        validate_backend(device, actual)
        multiview = bool(lib.aicore_instantmesh_has_multiview(ctx))
        if not lib.aicore_instantmesh_is_ready(ctx) or multiview != (preset == "full"):
            raise ValueError("pipeline readiness/model role mismatch")
        params = Params(ctypes.sizeof(Params), lib.aicore_instantmesh_abi_version(),
                        steps, seed, -1.0, grid_res, -1.0, texture_res, int(texture))
        timings, wall_samples, fingerprints = [], [], []
        result = {}
        for index in range(warmups + iterations):
            start = time.perf_counter()
            mesh = lib.aicore_instantmesh_generate(
                ctx, image_bytes, len(image_bytes), ctypes.byref(params), ProgressFn(), None,
                err, len(err))
            forward_wall_ms = (time.perf_counter() - start) * 1000.0
            if not mesh:
                raise ValueError(f"generate failed: {err.value.decode(errors='replace')}")
            try:
                value = PipelineTimings()
                if lib.aicore_instantmesh_last_pipeline_timings(ctx, ctypes.byref(value)):
                    raise ValueError("public pipeline timings unavailable")
                measured = public_timings(value)
                result = inspect_mesh(lib, mesh, texture)
                if index >= warmups:
                    timings.append(measured)
                    wall_samples.append(forward_wall_ms)
                    fingerprints.append(result["geometry_sha12"])
            finally:
                lib.aicore_instantmesh_mesh_free(mesh)
        result.update(summarize_timings(timings))
        result.update({"device": device, "actual_backend": actual, "preset": preset,
                       "quant": quant, "multiview": multiview, "steps": steps, "seed": seed,
                       "grid_res": grid_res, "texture_res": texture_res, "threads": threads,
                       "load_ms": load_ms, "warmups": warmups, "iterations": iterations,
                       "context_loads": 1, "forward_wall_samples_ms": wall_samples,
                       "forward_geometry_hashes": fingerprints, "timings": timings[-1],
                       # Hashes/bboxes are diagnostic only; no complete accuracy conclusion without independent references.
                       "accuracy_status": "INCOMPLETE", "independent_reference_complete": 0})
        return result
    finally:
        lib.aicore_instantmesh_free(ctx)


def diffusion_output_metrics(decoder_input, reference_latents, decoded, reference_rgb):
    """Diagnostic-only: F32 latent unscale and NCHW-to-RGB semantics, with separate errors for all six views."""
    import numpy as np

    if (decoder_input.shape != reference_latents.shape or decoder_input.ndim != 1 or
            decoded.shape != (3, 960, 640) or reference_rgb.shape != (960, 640, 3)):
        raise ValueError("diffusion capture shape mismatch")
    if not all(np.isfinite(value).all() for value in
               (decoder_input, reference_latents, decoded, reference_rgb)):
        raise ValueError("non-finite diffusion capture/reference")
    # Match the production operation order; float64 is used only to accumulate errors.
    unscaled = reference_latents / np.float32(0.75) + np.float32(0.22)
    raw_latents = (decoder_input - np.float32(0.22)) * np.float32(0.75)
    pixels = decoded / np.float32(0.5) * np.float32(0.8)
    pixels = np.clip(pixels / np.float32(2) + np.float32(0.5), 0, 1)
    rgb = (pixels * np.float32(255) + np.float32(0.5)).astype(np.uint8).transpose(1, 2, 0)
    metrics = {"accuracy_status": "INCOMPLETE", "independent_reference_complete": 0}

    def errors(prefix, actual, expected):
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        metrics[prefix + "_max_abs"] = float(np.abs(difference).max())
        metrics[prefix + "_rmse"] = float(np.sqrt(np.mean(difference * difference)))
        metrics[prefix + "_sign_changes"] = int(np.count_nonzero((actual < 0) != (expected < 0)))

    errors("decoder_input", decoder_input, unscaled)
    errors("final_latents", raw_latents, reference_latents)
    for view in range(6):
        row, col = divmod(view, 2)
        area = (slice(row * 320, (row + 1) * 320), slice(col * 320, (col + 1) * 320))
        errors(f"view{view}_rgb", rgb[area], reference_rgb[area])
        for name in ("max_abs", "rmse"):
            metrics[f"view{view}_rgb_{name}_bytes"] = metrics.pop(f"view{view}_rgb_{name}")
    return metrics


def compare_diffusion_capture(reference, capture):
    """Authenticate retained real captures before comparison; this is not a frozen full-accuracy gate."""
    import numpy as np
    from PIL import Image

    reference, capture = Path(reference), Path(capture)
    provenance = json.loads((reference / "provenance.json").read_text())
    expected = {
        "kind": "independent-75-step-diffusion-reference-not-complete-geometry-texture-or-performance",
        "upstream_revision": "32b14574edc9caba95f03672bfa1eeddd13c3af7",
        "generator": "cpp_ggml/convert/dump_e2e.py",
        "generator_sha256": "10a714c70b0c9e87defe2295e47e0bc1b0fbc59b5b9158e21bb6fb7b38d26cb9",
        "checkpoint_revision": "2da07e89919e1a130c9b5add1584c70c7aa065fd",
        "override_revision": "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
        "input_sha256": "db468bad8a04f1474a8d68140c07501b013b3ec6124b911fb7852675d64c05ee",
        "parameters": {"steps": 75, "seed": 42, "dtype": "fp32", "device": "cpu", "threads": 2}}
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError("diffusion reference identity mismatch: " + key)
    digests, sizes = provenance["files_sha256"], provenance["files_bytes"]
    members = {"img_vae.bin", "img_clip.bin", "cond_pos.bin", "cond_neg.bin",
               "latents0.bin", "ref_latents.bin", "ref_grid.png"}
    members.update(f"ref_view_{view}.png" for view in range(6))
    members.update(f"{prefix}_{step:03d}.bin" for step in range(75)
                   for prefix in ("cond_noise", "step_noise", "latents", "r_input", "w_input", "context", "eps"))
    if set(digests) != members or set(sizes) != members or sizes["ref_latents.bin"] != 4 * 120 * 80 * 4:
        raise ValueError("incomplete independent diffusion reference inventory")
    # Stream each payload; peak arrays are below 40 MiB and no model/backend is loaded.
    for name, digest in digests.items():
        if Path(name).name != name:
            raise ValueError("invalid diffusion reference member name")
        file = reference / name
        hasher = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        if file.stat().st_size != sizes[name] or hasher.hexdigest() != digest:
            raise ValueError("diffusion reference size/digest mismatch: " + name)
    launch = json.loads((capture / "launch.json").read_text())
    if launch.get("input_sha256") != expected["input_sha256"]:
        raise ValueError("diffusion capture input identity mismatch")
    records = [json.loads(line[len("IM_CAPTURE "):])
               for line in (capture / "gui.log").read_text().splitlines() if line.startswith("IM_CAPTURE ")]
    payloads = {}
    for name, count in (("decode-input.bin", 4 * 120 * 80), ("decode-output.bin", 3 * 960 * 640)):
        rows = [record for record in records if record.get("member") == name]
        data = (capture / name).read_bytes()
        if len(rows) != 1 or len(data) != count * 4 or hashlib.sha256(data).hexdigest() != rows[0].get("sha256"):
            raise ValueError("diffusion capture size/digest mismatch: " + name)
        if name == "decode-input.bin" and rows[0].get("shape") != [1, 120, 80]:
            raise ValueError("diffusion capture decoder dimensions mismatch")
        payloads[name] = np.frombuffer(data, dtype="<f4")
    decoded = payloads["decode-output.bin"].reshape(3, 960, 640)
    with Image.open(reference / "ref_grid.png") as image:
        expected_rgb = np.asarray(image.convert("RGB"))
    # Authenticate each independent crop too; a grid/crop disagreement must not be averaged away.
    for view in range(6):
        row, col = divmod(view, 2)
        with Image.open(reference / f"ref_view_{view}.png") as image:
            crop = np.asarray(image.convert("RGB"))
        if not np.array_equal(crop, expected_rgb[row * 320:(row + 1) * 320, col * 320:(col + 1) * 320]):
            raise ValueError("independent diffusion crop/grid mismatch")
    result = diffusion_output_metrics(payloads["decode-input.bin"],
                                     np.fromfile(reference / "ref_latents.bin", dtype="<f4"), decoded, expected_rgb)
    result.update({"kind": "captured-diffusion-output-diagnostic-not-full-accuracy-or-performance",
                   "reference_authenticated": 1, "reference_members": len(digests),
                   "reference_provenance_sha256": hashlib.sha256((reference / "provenance.json").read_bytes()).hexdigest(),
                   "capture_package_sha256": launch["package_sha256"],
                   "capture_plugin_sha256": launch["plugin_sha256"], "capture_git_head": launch["git_head"]})
    return result


def load_library(path):
    directory = resolve_library(path).parent
    # Windows dependency DLL-directory handles must outlive library use; leave system DLL search paths unchanged.
    dll_directory = os.add_dll_directory(str(directory)) if os.name == "nt" else None
    lib = ctypes.CDLL(str(resolve_library(path)))
    lib._dll_directory = dll_directory
    lib.aicore_instantmesh_abi_version.restype = ctypes.c_int
    # Explicit prototypes: without argtypes ctypes narrows pointer arguments
    # to 32-bit ints and every handle deref segfaults (see trellis probe).
    _P = ctypes.c_void_p
    lib.aicore_instantmesh_options_new.restype = _P
    lib.aicore_instantmesh_options_free.argtypes = [_P]
    lib.aicore_instantmesh_options_set_device.argtypes = [_P, ctypes.c_char_p]
    lib.aicore_instantmesh_options_set_threads.argtypes = [_P, ctypes.c_int]
    lib.aicore_instantmesh_options_set_steps.argtypes = [_P, ctypes.c_int]
    lib.aicore_instantmesh_options_set_seed.argtypes = [_P, ctypes.c_uint64]
    lib.aicore_instantmesh_options_set_grid_res.argtypes = [_P, ctypes.c_int]
    lib.aicore_instantmesh_options_set_bake_texture.argtypes = [_P, ctypes.c_int]
    lib.aicore_instantmesh_load_opts_ex.argtypes = [_P, _P, ctypes.c_char_p, ctypes.c_int]
    lib.aicore_instantmesh_load_opts_ex.restype = _P
    lib.aicore_instantmesh_backend.argtypes = [_P]
    lib.aicore_instantmesh_backend.restype = ctypes.c_char_p
    lib.aicore_instantmesh_is_ready.argtypes = [_P]
    lib.aicore_instantmesh_is_ready.restype = ctypes.c_int
    lib.aicore_instantmesh_last_pipeline_timings.argtypes = [_P, ctypes.POINTER(PipelineTimings)]
    lib.aicore_instantmesh_last_pipeline_timings.restype = ctypes.c_int
    lib.aicore_instantmesh_free.argtypes = [_P]
    lib.aicore_instantmesh_has_multiview.argtypes = [_P]
    lib.aicore_instantmesh_has_multiview.restype = ctypes.c_int
    lib.aicore_instantmesh_generate.argtypes = [
        _P, ctypes.c_void_p, ctypes.c_int, _P, ProgressFn, _P,
        ctypes.c_char_p, ctypes.c_int]
    lib.aicore_instantmesh_generate.restype = _P
    lib.aicore_instantmesh_mesh_free.argtypes = [_P]
    lib.aicore_instantmesh_mesh_n_verts.argtypes = [_P]
    lib.aicore_instantmesh_mesh_n_verts.restype = ctypes.c_int
    lib.aicore_instantmesh_mesh_n_tris.argtypes = [_P]
    lib.aicore_instantmesh_mesh_n_tris.restype = ctypes.c_int
    lib.aicore_instantmesh_mesh_verts.argtypes = [_P]
    lib.aicore_instantmesh_mesh_verts.restype = ctypes.POINTER(ctypes.c_char)
    for name in ("tris", "vcolors", "uvs", "texture"):
        accessor = getattr(lib, "aicore_instantmesh_mesh_" + name)
        accessor.argtypes = [_P]
        accessor.restype = _P
    for name in ("w", "h"):
        accessor = getattr(lib, "aicore_instantmesh_mesh_texture_" + name)
        accessor.argtypes = [_P]
        accessor.restype = ctypes.c_int
    if lib.aicore_instantmesh_abi_version() != 2:
        raise ValueError("InstantMesh ABI mismatch (expected 2)")
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models")
    ap.add_argument("--image")
    ap.add_argument("--compare-diffusion-capture", help="Retained GDB capture directory; diagnosis only, loads no model")
    ap.add_argument("--reference", help="Authenticated independent diffusion reference directory")
    ap.add_argument("--diagnostic-output", help="New JSON receipt; refuses to overwrite historical evidence")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--preset", default="full", choices=["full", "geometry"])
    ap.add_argument("--quant", default="f16", choices=["f16", "q8", "f32"])
    ap.add_argument("--steps", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-res", type=int, default=64)
    ap.add_argument("--texture-res", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmups", "--warmup-runs", type=int, default=1)
    ap.add_argument("--iterations", "--inference-runs", type=int, default=2)
    ap.add_argument("--no-texture", action="store_true")
    ap.add_argument("--lib", help="AICore library file or build/install directory")
    args = ap.parse_args()
    if args.compare_diffusion_capture:
        if not args.reference:
            ap.error("--compare-diffusion-capture requires --reference")
        try:
            result = compare_diffusion_capture(args.reference, args.compare_diffusion_capture)
            serialized = json.dumps(result, sort_keys=True, allow_nan=False)
            if args.diagnostic_output:
                with Path(args.diagnostic_output).open("x") as stream:
                    stream.write(serialized + "\n")
            print(serialized)
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(str(error)) from None
        return
    if not args.models or not args.image or not args.lib:
        ap.error("inference requires --models, --image and --lib")
    for valid, label in ((args.warmups >= 0, "warmups"), (args.iterations >= 1, "iterations"),
                         (0 <= args.seed < 2**64, "seed"), (2 <= args.steps <= 1000, "steps"),
                         (2 <= args.grid_res <= 256, "grid-res"),
                         (1 <= args.texture_res <= 4096, "texture-res"),
                         (args.threads >= 0, "threads")):
        if not valid:
            ap.error(f"invalid --{label}")
    devices = [value.strip() for value in args.device.split(",") if value.strip()]
    if not devices:
        ap.error("empty --device")
    try:
        lib = load_library(args.lib)
        image_bytes = Path(args.image).read_bytes()
        results = []
        for device in devices:
            t0 = time.perf_counter()
            row = run_once(lib, args.models, image_bytes, device, args.preset,
                           args.quant, args.steps, args.seed, args.grid_res,
                           not args.no_texture, args.warmups, args.iterations,
                           args.texture_res, args.threads)
            row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
            results.append(row)
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    except (OSError, ValueError, MemoryError) as error:
        raise SystemExit(str(error)) from None

    print("\n## Backend measurements (not controlled baseline/candidate A/B)\n")
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
