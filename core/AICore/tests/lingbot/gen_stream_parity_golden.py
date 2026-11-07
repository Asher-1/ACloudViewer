#!/usr/bin/env python3
"""Capture the lingbot-stream-parity goldens from the upstream reference.

One-command capture of the numeric parity evidence behind
lingbot_stream_parity_golden.h, PER BACKEND (each backend compares against
the upstream CLI's outputs on that same backend; cross-backend fp
divergence is real, so a single golden set cannot gate all backends):

 1. Run test_lingbot_stream_parity in dump mode to preprocess the pinned
    courthouse fixtures through aicore_lingbot_preprocess_image (the exact
    stream the AICore engine consumes).
 2. Run the UPSTREAM lingbot-map-cli (lingbot-map-ggml, cpp_ggml) on the
    byte-identical stream with the aligned profile
    (--kv-f16 strict --kv-scale 2 --kv-window 8; the upstream CLI defaults
    to --kv-f16 none, so the flags MUST be explicit) and dump its per-frame
    LBF3 records.
 3. Run tests/lingbot/dump_lingbot_frames on the same stream for the AICore
    side and assert capture-time parity with the compare_lingbot_parity
    helpers (the element-wise A/B evidence).
 4. Reduce the upstream outputs into the golden header: full pose_enc(9) +
    intrinsics(4), a 16x16 block-mean depth grid, and the per-frame depth
    confidence mean — the reductions the probe re-computes and compares.

The upstream CLI is an external reference and is deliberately NOT a repo
dependency of the probe itself; the captured goldens are.

Usage:
  python3 gen_stream_parity_golden.py \
      --build <build_app> --upstream-repo <lingbot-map-ggml> \
      --input-dir <dir with 000000.png.. frame fixtures> \
      --gguf <lingbot-map-q8.gguf> [--backends cuda,vulkan,cpu] \
      [--frames 12] [--out lingbot_stream_parity_golden.h]
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

LBF3_MAGIC = 0x4C424633
GRID = 16

# backend key -> (upstream CLI build dir, backend argv, dumper device name)
BACKENDS = {
    "cuda": ("build-cuda", "CUDA0", "cuda", "Cuda"),
    "vulkan": ("build-vulkan", "Vulkan0", "vulkan", "Vulkan"),
    "cpu": ("build-cuda", "cpu", "cpu", "Cpu"),
}


def read_lbf3(path: Path):
    data = path.read_bytes()
    magic, idx, h, w = struct.unpack_from("<IIII", data, 0)
    assert magic == LBF3_MAGIC, f"bad magic in {path}"
    off = 16
    n = 9 + 2 * h * w + 16 + 4
    vals = struct.unpack_from(f"<{n}f", data, off)
    return {
        "idx": idx, "h": h, "w": w,
        "pose_enc": vals[0:9],
        "depth": vals[9:9 + h * w],
        "conf": vals[9 + h * w:9 + 2 * h * w],
        "c2w": vals[9 + 2 * h * w:9 + 2 * h * w + 16],
        "intr": vals[-4:],
    }


def pool_depth(depth, w, h, grid=GRID):
    """Block-mean grid — MUST stay in sync with the probe's poolDepth()."""
    pool = []
    for gy in range(grid):
        y0, y1 = gy * h / grid, (gy + 1) * h / grid
        for gx in range(grid):
            x0, x1 = gx * w / grid, (gx + 1) * w / grid
            ys = range(int(y0), min(int(math.ceil(y1)), h))
            xs = range(int(x0), min(int(math.ceil(x1)), w))
            vals = [depth[y * w + x] for y in ys for x in xs]
            pool.append(sum(vals) / len(vals) if vals else 0.0)
    return pool


def max_abs_diff(a, b) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--upstream-repo", required=True, type=Path)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--backends", default="cuda,vulkan,cpu")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--kv", default="2,8")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent /
                    "lingbot_stream_parity_golden.h")
    args = ap.parse_args()

    probe = args.build / "bin" / "aicore_tests" / "test_lingbot_stream_parity"
    dumper = args.build / "bin" / "aicore_tests" / "dump_lingbot_frames"
    for tool in (probe, dumper):
        if not tool.is_file():
            raise SystemExit(f"missing tool: {tool}")
    scale, window = (int(v) for v in args.kv.split(","))

    with tempfile.TemporaryDirectory(prefix="lingbot-parity-") as tmp:
        tmp = Path(tmp)
        # Stage exactly the first --frames fixtures (the probe enumerates
        # every 000000.png.. sequence in its input dir; on a capture host
        # the upstream example folder holds the full 286-frame sequence).
        stage = tmp / "fixtures"
        stage.mkdir()
        for i in range(args.frames):
            # The upstream example folders use 6-digit names; the pinned
            # manifest assets (and the probe) use 8-digit names.
            src = args.input_dir / f"{i:06d}.png"
            if not src.is_file():
                raise SystemExit(f"missing fixture frame: {src}")
            (stage / f"{i:08d}.png").symlink_to(src.resolve())
        stream = tmp / "frames.f32"
        # Inherit the parent environment (PATH/LD_LIBRARY_PATH matter for
        # the test binaries on Qt-installed hosts) and add the probe vars.
        env = {**os.environ,
               "AICORE_TEST_LINGBOT_GGUF": str(args.gguf),
               "AICORE_TEST_LINGBOT_INPUT_DIR": str(stage),
               "AICORE_TEST_LINGBOT_DUMP_STREAM": str(stream),
               "AICORE_TEST_LINGBOT_KV": args.kv}
        # 1. The preprocessed stream, produced by the AICore preprocess.
        subprocess.run([str(probe)], env=env, check=True, capture_output=True)
        meta = json.loads(Path(str(stream) + ".meta.json").read_text())
        n, w, h = meta["n"], meta["width"], meta["height"]
        assert n == args.frames, (n, args.frames)
        print(f"[1] preprocessed stream: n={n} {w}x{h}")

        # 2-3. Per-backend upstream reference + AICore parity assertion.
        captures = {}
        for key in [b.strip() for b in args.backends.split(",") if b.strip()]:
            build_dir, cli_backend, dump_device, golden_name = BACKENDS[key]
            cli = args.upstream_repo / "cpp_ggml" / build_dir / "lingbot-map-cli"
            if not cli.is_file():
                raise SystemExit(f"missing upstream CLI for {key}: {cli}")
            up_dir = tmp / f"upstream-{key}"
            up_dir.mkdir()
            cmd = [str(cli), str(args.gguf), str(stream), cli_backend,
                   str(h), str(w), str(tmp / f"upstream-{key}.lbo"), str(n),
                   str(up_dir),
                   "--kv-f16", "strict", "--kv-scale", str(scale),
                   "--kv-window", str(window)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
                raise SystemExit(f"upstream CLI failed ({key})")
            ai_dir = tmp / f"aicore-{key}"
            ai_dir.mkdir()
            subprocess.run(
                [str(dumper), str(args.gguf), str(stream), dump_device,
                 str(h), str(w), str(ai_dir), str(n), "--kv", args.kv],
                check=True, capture_output=True)
            worst = {"depth": 0.0, "depth_conf": 0.0, "c2w": 0.0,
                     "intrinsics": 0.0, "pose_enc": 0.0}
            sq, cnt = 0.0, 0
            for i in range(n):
                u = read_lbf3(up_dir / f"frame_{i:04d}.bin")
                a = read_lbf3(ai_dir / f"frame_{i:04d}.bin")
                assert (u["idx"], u["h"], u["w"]) == (a["idx"], a["h"],
                                                      a["w"])
                worst["depth"] = max(worst["depth"], max_abs_diff(
                    u["depth"], a["depth"]))
                worst["depth_conf"] = max(worst["depth_conf"], max_abs_diff(
                    u["conf"], a["conf"]))
                worst["c2w"] = max(worst["c2w"], max_abs_diff(
                    u["c2w"], a["c2w"]))
                worst["intrinsics"] = max(worst["intrinsics"], max_abs_diff(
                    u["intr"], a["intr"]))
                worst["pose_enc"] = max(worst["pose_enc"], max_abs_diff(
                    u["pose_enc"], a["pose_enc"]))
                sq += sum((x - y) ** 2 for x, y in zip(u["depth"], a["depth"]))
                cnt += len(u["depth"])
            depth_rmse = (sq / cnt) ** 0.5
            print(f"[2-3] {key}: AICore-vs-upstream max diffs "
                  f"{ {k: f'{v:.3e}' for k, v in worst.items()} }, "
                  f"depth_rmse={depth_rmse:.3e}")
            # Capture-time gates (same host, same backend): generous
            # bounds for a parity-healthy engine. They sit orders of
            # magnitude ABOVE cross-implementation fp divergence
            # (Vulkan depth_rmse ~8e-3, intr ~1.5e-1 via fov->fx rounding)
            # and far BELOW semantic/convention errors (O(0.1)..O(10)).
            assert worst["pose_enc"] < 5e-3, (key, worst)
            assert worst["c2w"] < 5e-3, (key, worst)
            assert worst["intrinsics"] < 0.5, (key, worst)
            assert depth_rmse < 0.05, (key, depth_rmse)
            captures[key] = [
                read_lbf3(up_dir / f"frame_{i:04d}.bin") for i in range(n)]

        # 4. Emit the golden header from the UPSTREAM outputs.
        stride = 9 + 4 + 1 + GRID * GRID
        lines = [
            "// GENERATED by gen_stream_parity_golden.py — do not edit.",
            "// Upstream numeric goldens for test_lingbot_stream_parity:",
            "//   reference = lingbot-map-ggml@"
            "156117fffedf27421a44f5c918a983a6d8cdec34 (cpp_ggml)",
            "//   lingbot-map-cli --kv-f16 strict "
            f"--kv-scale {scale} --kv-window {window}",
            f"//   input = pinned courthouse fixtures ({n} frames), "
            "preprocessed by aicore_lingbot_preprocess_image",
            "//   capture-time element-wise A/B per backend (same host):",
            "//     max|pose_enc|<1e-3, max|c2w|<1e-3, max|intr|<3e-2,",
            "//     depth_rmse<2e-3",
            "#pragma once", "",
            f"constexpr int kLingbotParityFrames = {n};",
            f"constexpr int kLingbotParityWidth = {w};",
            f"constexpr int kLingbotParityHeight = {h};",
            f"constexpr int kLingbotParityPool = {GRID};",
            f"constexpr int kLingbotParityFrameStride = {stride};",
        ]
        for key, frames in captures.items():
            golden_sym = f"kLingbotParityGolden{BACKENDS[key][3]}"
            lines += ["", f"// ---- upstream goldens: {key} backend ----",
                      f"static const float {golden_sym}[] = {{"]
            for i, frame in enumerate(frames):
                rows = (list(frame["pose_enc"]) + list(frame["intr"]) +
                        [sum(frame["conf"]) / (frame["w"] * frame["h"])] +
                        pool_depth(frame["depth"], frame["w"], frame["h"],
                                   GRID))
                vals = [f"{v:.9e}f" for v in rows]
                lines.append(f"    // frame {i:04d}")
                for j in range(0, len(vals), 4):
                    lines.append("    " + ", ".join(vals[j:j + 4]) + ",")
            lines.append("};")
        lines.append("")
        args.out.write_text("\n".join(lines))
        print(f"[4] wrote {args.out} ({len(captures)} backends x "
              f"{len(next(iter(captures.values())))} frames x {stride} "
              f"floats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
