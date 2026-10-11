#!/usr/bin/env python3
"""Materialize the RAM demo reference in a transient, pinned upstream cache.

The repository deliberately contains no generated RAM logits.  This wrapper
resolves the official PyTorch graph from the pinned recognize-anything-ggml
checkout for the exact published GGUF and, for CPU Q8, also records the
upstream CPU activation-quantized graph.  It then delegates the actual probe
to ``test_tagging_probe`` with the resulting cache as ``--reference-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REVISION = "136d09dff71802bc7b6abedb326785bca8b2fcb4"
SCRIPT = "cpp_ggml/scripts/parity_reference.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family(model: Path) -> str:
    if model.name.startswith("ram_plus_"):
        return "ram_plus"
    if model.name.startswith("tag2text_"):
        return "tag2text"
    if model.name.startswith("ram_"):
        return "ram"
    raise ValueError(f"model is not in the RAM catalog: {model.name}")


def cache_for(root: Path, upstream: Path, model: Path, image: Path) -> Path:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(
            f"AICORE_TEST_RAM_UPSTREAM must be revision {REVISION}; got {revision}"
        )
    digest = hashlib.sha256()
    digest.update(b"ram-demo-reference-v2\0")
    digest.update(revision.encode())
    for path in (model, image):
        digest.update(path.name.encode())
        digest.update(sha256(path).encode())
    destination = root / digest.hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "provenance.txt").write_text(
        f"upstream_revision={revision}\n"
        f"model={model}\nmodel_sha256={sha256(model)}\n"
        f"image={image}\nimage_sha256={sha256(image)}\n",
        encoding="utf-8",
    )
    return destination


def materialize(args: argparse.Namespace, unknown: list[str]) -> int:
    upstream_text = os.environ.get("AICORE_TEST_RAM_UPSTREAM")
    if not upstream_text:
        raise RuntimeError(
            "Set AICORE_TEST_RAM_UPSTREAM to the pinned recognize-anything-ggml checkout"
        )
    upstream = Path(upstream_text).resolve()
    model = Path(args.model).resolve()
    image = Path(args.image).resolve()
    if not (model.is_file() and image.is_file()):
        raise RuntimeError(f"missing model or image: {model}, {image}")
    cache = cache_for(Path(args.reference_cache).resolve(), upstream, model, image)
    base = model.name[:-5] + "-demo1-pytorch-gguf-logits.f32"
    pytorch = cache / base
    if not pytorch.is_file():
        output = cache / "pytorch"
        command = [
            sys.executable,
            str(upstream / SCRIPT),
            "--model",
            family(model),
            "--gguf",
            str(model),
            "--image",
            str(image),
            "--output",
            str(output),
        ]
        subprocess.run(command, cwd=upstream, check=True)
        generated = output / "logits.bin"
        if not generated.is_file():
            raise RuntimeError(f"upstream did not write {generated}")
        shutil.copyfile(generated, pytorch)

    if args.backend.lower() == "cpu" and model.name.endswith("-q8_0.gguf"):
        cpu = cache / (model.name[:-5] + ".cpu.f32")
        if not cpu.is_file():
            binary = upstream / "cpp_ggml/build-cpu/ram-ggml"
            if not binary.is_file():
                raise RuntimeError(
                    f"CPU Q8 truth requires the upstream binary: {binary}"
                )
            effective_threads = args.threads if args.threads > 0 else 8
            # validate_all prepends the candidate build to the loader path.
            # Both builds export the same ggml SONAME, so inheriting that path
            # can make the upstream executable load AICore's CPU module and
            # crash before it writes its reference.  The upstream executable
            # has an embedded RUNPATH to its own ggml build; clear the
            # candidate-only overrides and let the dynamic loader use it.
            runtime_env = os.environ.copy()
            if os.name != "nt":
                runtime_env.pop(
                    "DYLD_LIBRARY_PATH" if sys.platform == "darwin"
                    else "LD_LIBRARY_PATH", None)
            runtime_env.pop("GGML_BACKEND_PATH", None)
            runtime_env["OMP_NUM_THREADS"] = str(effective_threads)
            runtime_env["OMP_DYNAMIC"] = "FALSE"
            subprocess.run(
                [
                    str(binary),
                    "--model",
                    str(model),
                    "--image",
                    str(image),
                    "--backend",
                    "cpu",
                    "--threads",
                    str(effective_threads),
                    "--warmup",
                    "1",
                    "--repeat",
                    "1",
                    "--logits-out",
                    str(cpu),
                ],
                cwd=upstream,
                env=runtime_env,
                check=True,
            )
    automatic_ram_q8 = (args.backend.lower() == "cpu" and family(model) == "ram"
                        and model.name.endswith("-q8_0.gguf")
                        and args.cpu_precision == 0)
    reference_script = Path(__file__).with_name("cpu_precision_reference.py")
    arithmetic_key = sha256(reference_script)
    automatic = cache / (model.name[:-5] + ".cpu-auto-pytorch.f32")
    arithmetic_provenance = cache / "cpu-auto-pytorch.sha256"
    if automatic.is_file() and (not arithmetic_provenance.is_file() or
                               arithmetic_provenance.read_text().strip() != arithmetic_key):
        automatic.unlink()
    if automatic_ram_q8 and not automatic.is_file():
        output = cache / "cpu-auto-pytorch"
        subprocess.run([
            sys.executable, str(reference_script),
            "--upstream", str(upstream), "--gguf", str(model), "--demo",
            "--threads", str(args.threads if args.threads > 0 else 8),
            "--model", "ram", "--image", str(image), "--output", str(output),
        ], cwd=upstream, check=True)
        shutil.copyfile(output / "logits.bin", automatic)
        arithmetic_provenance.write_text(arithmetic_key + "\n")
    # The wrapper consumes the model/backend/image options while preparing the
    # cache, so forward them explicitly to the real probe.  Dropping these
    # arguments would make the probe return its "missing model" skip code.
    command = [
        str(Path(args.probe).resolve()),
        "--model", str(model),
        "--backend", args.backend,
        "--image", str(image),
        "--threads", str(args.threads),
        "--cpu-precision", str(args.cpu_precision),
        *unknown,
    ]
    command.extend(("--reference-dir", str(cache)))
    result = subprocess.run(command, text=True, capture_output=True)
    sys.stderr.write(result.stderr)
    if result.returncode == 0 and automatic_ram_q8:
        report = json.loads(result.stdout.strip().splitlines()[-1])
        # Exact upstream arithmetic remains a separate, measured compatibility
        # contract. Its raw-logit bound is not transferred to AUTO arithmetic.
        compatibility = [str(Path(args.probe).resolve()), "--model", str(model),
                         "--backend", args.backend, "--image", str(image),
                         "--threads", str(args.threads), "--cpu-precision", "1",
                         "--warmup", "0", "--repeat", "1",
                         "--reference-dir", str(cache)]
        old = subprocess.run(compatibility, text=True, capture_output=True, check=True)
        compatibility_report = json.loads(old.stdout.strip().splitlines()[-1])
        report["compatibility_accuracy_max_diff"] = compatibility_report["accuracy_max_diff"]
        report["compatibility_raw_logit_reference"] = "upstream_cpu_q8"
        sys.stdout.write(json.dumps(report) + "\n")
    elif result.returncode == 0:
        report = json.loads(result.stdout.strip().splitlines()[-1])
        report["compatibility_accuracy_max_diff"] = report["accuracy_max_diff"]
        sys.stdout.write(json.dumps(report) + "\n")
    else:
        sys.stdout.write(result.stdout)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--reference-cache", default=str(
        Path(tempfile.gettempdir()) / "aicore-tagging-reference-cache"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-precision", type=int, choices=(0, 1), default=0)
    args, unknown = parser.parse_known_args()
    return materialize(args, unknown)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"RAM demo reference preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
