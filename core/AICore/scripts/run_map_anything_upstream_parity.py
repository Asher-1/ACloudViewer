#!/usr/bin/env python3
"""One validate_all row: AICore MapAnything vs upstream cpp_ggml, same bytes.

Truth is the upstream map-anything-ggml ``vggt-cli`` executed on the exact
f32 (S,3,H,W) tensor AICore consumed.  The row is:

  1. test_map_anything_upstream_frames writes frames.bin, runs
     aicore_map_anything_run_frames on --backend and exports every typed
     output under the vggt-cli file names (candidate/).
  2. ``<reference>/cpp_ggml/build-<backend>/bin/vggt-cli --bin frames.bin``
     runs the same tensor on the same backend kind (ref/).
  3. compare_map_anything_parity.py --profile frames requires atol 0 on every
     common raw tensor.

Fail-closed contract:

* no reference configured, or no reference build for this backend -> exit 77
  (validate_all records "skipped", which fails the gate unless the caller
  passes --allow-incomplete);
* reference source identity differs from the pinned revision / ggml revision
  / ggml working-tree diff -> exit 1 (a different program is not the truth
  this row was calibrated against);
* reference resolved a different backend than requested -> exit 1;
* candidate exports a tensor the reference does not, or the reference exports
  one outside UPSTREAM_ONLY_OUTPUTS -> exit 1 (an uncompared output is an
  unproven output).

The last stdout line is the probe's validation JSON extended with
``upstream_parity`` (max_diff, raw_files_compared, reference identity), so
validate_all's structured accuracy checks and fingerprint stability apply.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMPARATOR = SCRIPT_DIR / "compare_map_anything_parity.py"
SKIP = 77

# vggt-cli backend names (cpp_ggml src/backend.cpp init_backend).
UPSTREAM_BACKEND_NAMES = {"cpu": "CPU", "cuda": "CUDA0", "vulkan": "Vulkan0",
                          "metal": "Metal"}
# Upstream debug exports with no public AICore result accessor:
#   pose_raw.bin    -- pi3/pi3x/mapanything camera head before c2w assembly
#                      (AICore exposes the assembled pose.bin, compared);
#   points_conf.bin -- original VGGT point-head confidence (AICore exposes
#                      the depth-head confidence, compared as depth_conf.bin).
UPSTREAM_ONLY_OUTPUTS = {"pose_raw.bin", "points_conf.bin"}
# map_anything_merged vendors the reference ggml's flash-attention sources
# for mapany_* nodes (src/ggml-cuda/mapany/mapany-fattn.cuh): CUDA stream-k
# partitions by the compiled kernel's occupancy, so only the reference's own
# kernel source reproduces its reduction order.
MAPANY_PATCH = (SCRIPT_DIR.parents[2] / "3rdparty" / "llama" / "ggml-patches" /
                "map_anything_merged" / "0001-map-anything-ggml-parity.patch")
MAPANY_TWIN_DIR = "src/ggml-cuda/mapany/ggml-v0.24.0/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True,
                        help="AICore build tree; its library dirs are removed "
                             "from the reference process environment.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image2", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--reference", default="",
                        help="map-anything-ggml checkout root; empty = not configured.")
    parser.add_argument("--expect-revision", required=True)
    parser.add_argument("--expect-ggml-revision", required=True)
    parser.add_argument("--expect-ggml-diff-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def skip(message: str) -> int:
    print(f"SKIP: {message}", file=sys.stderr)
    return SKIP


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout


def reference_identity(root: Path) -> dict:
    """Source identity of the reference program (not of its binaries).

    The top-level diff ignores submodules and untracked files; the ggml
    submodule is pinned by commit plus the sha256 of its full working-tree
    diff, because the calibrated reference carries an uncommitted kernel.
    """
    ggml = root / "cpp_ggml" / "third_party" / "ggml"
    return {
        "revision": git(root, "rev-parse", "HEAD").decode().strip(),
        "tree_diff_sha256": hashlib.sha256(git(
            root, "diff", "HEAD", "--binary", "--ignore-submodules=all")).hexdigest(),
        "ggml_revision": git(ggml, "rev-parse", "HEAD").decode().strip(),
        "ggml_diff_sha256": hashlib.sha256(
            git(ggml, "diff", "HEAD", "--binary")).hexdigest(),
    }


def vendored_twin(patch: Path) -> dict[str, bytes]:
    """Files the patch creates under MAPANY_TWIN_DIR, keyed by base name.

    New-file hunks only ("@@ -0,0 +1,N @@"): the content is every '+' line
    of the hunk. O(patch lines).
    """
    files: dict[str, bytes] = {}
    current = None
    for line in patch.read_bytes().splitlines(keepends=True):
        if line.startswith(b"diff --git "):
            current = None
        elif line.startswith(b"+++ b/"):
            path = line[6:].rstrip(b"\n").decode()
            current = path[len(MAPANY_TWIN_DIR):] \
                if path.startswith(MAPANY_TWIN_DIR) else None
            if current is not None:
                files[current] = b""
        elif current is not None and line.startswith(b"+"):
            files[current] += line[1:]
        elif current is not None and line.startswith(b"\\ No newline"):
            files[current] = files[current].removesuffix(b"\n")
    return files


def twin_drift(root: Path, patch: Path = MAPANY_PATCH) -> list[str]:
    """Vendored twin files that are not the reference ggml's own bytes."""
    twin = vendored_twin(patch)
    if not twin:
        return [f"{patch} vendors no {MAPANY_TWIN_DIR} sources"]
    ggml = root / "cpp_ggml" / "third_party" / "ggml"
    drift = []
    for name, data in sorted(twin.items()):
        try:
            upstream = git(ggml, "show", f"HEAD:src/ggml-cuda/{name}")
        except subprocess.CalledProcessError:
            upstream = None
        if upstream != data:
            drift.append(name)
    return drift


def newer_source(root: Path, binary: Path) -> Path | None:
    """First reference source file modified after ``binary`` was linked.

    Covers what vggt-cli is compiled from: cpp_ggml sources/headers/CMake,
    the ggml submodule and the ggml-patches CMake applies to it. O(files)
    stat calls (a few thousand).
    """
    built = binary.stat().st_mtime
    cpp = root / "cpp_ggml"
    ggml = cpp / "third_party" / "ggml"
    files = [cpp / "CMakeLists.txt", ggml / "CMakeLists.txt"]
    for tree in (cpp / "src", cpp / "include", ggml / "src", ggml / "include",
                 cpp / "third_party" / "ggml-patches"):
        files.extend(path for path in tree.rglob("*") if path.is_file())
    for path in files:
        if path.is_file() and path.stat().st_mtime > built:
            return path
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_env(build: Path, root: Path) -> dict:
    """Environment for vggt-cli without any foreign libggml.

    vggt-cli resolves its own ggml through RUNPATH, but LD_LIBRARY_PATH takes
    precedence; an entry that provides another libggml (the AICore build
    tree, or any copy of its libraries) loads that patched ggml into the
    reference process (measured: GGML_ASSERT abort / a different program).
    Entries inside the candidate build and entries holding a libggml* outside
    the reference checkout are dropped, so toolkit paths such as the CUDA
    runtime remain visible.
    """
    env = dict(os.environ)
    build = build.resolve()
    root = root.resolve()

    def foreign(entry: str) -> bool:
        path = Path(entry).resolve()
        if path.is_relative_to(build):
            return True
        return not path.is_relative_to(root) and path.is_dir() and \
            any(path.glob("libggml*"))

    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        if key not in env:
            continue
        kept = [entry for entry in env[key].split(os.pathsep)
                if entry and not foreign(entry)]
        if kept:
            env[key] = os.pathsep.join(kept)
        else:
            del env[key]
    return env


def foreign_ggml(cli: Path, env: dict, root: Path) -> list[str]:
    """libggml* the dynamic loader would bind for ``cli`` outside ``root``.

    Linux only (ldd); elsewhere the environment filter above is the guard.
    """
    ldd = shutil.which("ldd")
    if ldd is None:
        return []
    out = subprocess.run([ldd, str(cli)], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True).stdout
    root = root.resolve()
    bad = []
    for line in out.splitlines():
        name, _, target = line.strip().partition(" => ")
        if not name.startswith("libggml"):
            continue
        resolved = target.split(" (")[0].strip()
        if not resolved or resolved == "not found" or \
                not Path(resolved).resolve().is_relative_to(root):
            bad.append(f"{name} => {resolved or '?'}")
    return bad


def run(command: list, env: dict | None, log: Path, timeout: int) -> tuple[int, str]:
    completed = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               timeout=timeout, check=False)
    log.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, completed.stdout


def last_json(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def main() -> int:
    args = parse_args()
    backend = args.backend.lower()
    if backend not in UPSTREAM_BACKEND_NAMES:
        return skip(f"no upstream vggt-cli backend name for '{args.backend}'")
    if not args.reference:
        return skip("upstream map-anything-ggml reference not configured "
                    "(validate_all --reference map_anything_ggml=<checkout>)")
    root = Path(args.reference).expanduser().resolve()
    cli = root / "cpp_ggml" / f"build-{backend}" / "bin" / "vggt-cli"
    if not cli.is_file():
        return skip(f"no upstream reference build for {backend}: {cli}")
    try:
        identity = reference_identity(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        return fail(f"cannot read reference source identity under {root}: {exc}")
    expected = {"revision": args.expect_revision,
                "tree_diff_sha256": hashlib.sha256(b"").hexdigest(),
                "ggml_revision": args.expect_ggml_revision,
                "ggml_diff_sha256": args.expect_ggml_diff_sha256}
    drift = [f"{key} {identity[key]} != pinned {value}"
             for key, value in expected.items() if identity[key] != value]
    if drift:
        return fail("reference source differs from the pinned truth: " +
                    "; ".join(drift))
    if backend == "cuda":
        stale_twin = twin_drift(root)
        if stale_twin:
            # The pinned reference moved its ggml (or the twin was edited):
            # re-vendor the reference's files into map_anything_merged.
            return fail("map_anything_merged flash-attention twin is not the "
                        "reference ggml source: " + ", ".join(stale_twin))
    stale = newer_source(root, cli)
    if stale is not None:
        # The identity above pins the SOURCE; a binary built before its last
        # edit is a different program than the pinned one.
        return fail(f"reference build {cli} is older than its source {stale}; "
                    f"rebuild build-{backend}")
    ref_env = reference_env(args.build, root)
    foreign = foreign_ggml(cli, ref_env, root)
    if foreign:
        return fail("reference vggt-cli would bind a libggml outside the "
                    f"reference checkout: {', '.join(foreign)}")

    # Thread count is resolved here so both programs use the same value
    # (both treat 0 as hardware concurrency, but only an explicit number is
    # auditable in the report).
    threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "ref").mkdir(parents=True)

    code, text = run([str(args.probe), str(args.model), args.image, args.image2,
                      backend, str(out), str(args.views), "1", "0", str(threads)],
                     None, out / "candidate.log", args.timeout)
    if code != 0:
        print(text[-4000:], file=sys.stderr)
        return code if code == SKIP else fail(f"candidate probe exit {code}")
    probe_json = last_json(text)
    geometry = json.loads((out / "geometry.json").read_text(encoding="utf-8"))
    if probe_json is None:
        return fail("candidate probe printed no validation JSON")

    code, text = run([str(cli), "--model", str(args.model),
                      "--bin", str(out / "frames.bin"),
                      "--H", str(geometry["H"]), "--W", str(geometry["W"]),
                      "--S", str(geometry["S"]), "--threads", str(threads),
                      "--out-prefix", str(out / "ref" / "up")],
                     ref_env, out / "reference.log", args.timeout)
    if code != 0:
        print(text[-4000:], file=sys.stderr)
        return fail(f"reference vggt-cli exit {code}")
    meta = json.loads((out / "ref" / "up.meta.json").read_text(encoding="utf-8"))
    if meta.get("backend") != UPSTREAM_BACKEND_NAMES[backend]:
        return fail(f"reference resolved backend {meta.get('backend')!r}, "
                    f"requested {UPSTREAM_BACKEND_NAMES[backend]!r}")
    if [meta.get(key) for key in ("S", "H", "W")] != \
            [geometry[key] for key in ("S", "H", "W")]:
        return fail(f"reference geometry {meta} != candidate {geometry}")

    candidate = {path.name for path in (out / "candidate").glob("*.bin")}
    reference = {path.name.removeprefix("up.")
                 for path in (out / "ref").glob("up.*.bin")}
    unmatched = sorted(candidate - reference)
    unexpected = sorted(reference - candidate - UPSTREAM_ONLY_OUTPUTS)
    if unmatched or unexpected:
        return fail(f"output sets differ: candidate-only {unmatched}, "
                    f"reference-only {unexpected}")

    report_path = out / "parity.json"
    code, text = run([sys.executable, str(COMPARATOR), "--profile", "frames",
                      "--candidate-raw", str(out / "candidate"),
                      "--reference-raw", str(out / "ref"),
                      "--report", str(report_path)],
                     None, out / "compare.log", args.timeout)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    max_diff = max((stats["max_abs"] for stats in report["raw"].values()
                    if "max_abs" in stats), default=float("inf"))
    probe_json["upstream_parity"] = {
        "profile": "frames",
        "architecture": geometry["architecture"],
        "views": geometry["S"], "height": geometry["H"], "width": geometry["W"],
        "threads": threads,
        "max_diff": max_diff,
        "raw_files_compared": len(report.get("raw_files", [])),
        "per_file_max_abs": {name: stats["max_abs"]
                             for name, stats in report["raw"].items()
                             if "max_abs" in stats},
        "failures": report["failures"],
        "reference": dict(identity, backend=meta["backend"],
                          cli_sha256=sha256_file(cli)),
    }
    print(json.dumps(probe_json, sort_keys=True))
    if code != 0 or not report["pass"]:
        return fail("upstream frames parity: " + "; ".join(report["failures"][:6]))
    # A passing row is fully summarized by parity.json (every max_abs is 0);
    # its ~20 MB of raw tensors per model are only kept for failing rows,
    # where they are the diagnosis input.
    (out / "frames.bin").unlink()
    shutil.rmtree(out / "candidate")
    for path in (out / "ref").glob("*.bin"):
        path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
