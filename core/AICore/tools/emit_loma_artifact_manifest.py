#!/usr/bin/env python3
"""Emit the publish manifest for generated LoMa GGUF artifacts.

The manifest is the hand-off boundary between the controlled conversion host
and the shared model release. It intentionally includes only GGUF runtime
weights. Reference ``.bin`` files in the same build directory are ONNX-derived
test fixtures and must never be published to the model cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def precision(filename: str) -> str:
    for value in ("f32", "f16", "q8_0"):
        if filename.endswith(f".{value}.gguf"):
            return value
    return "unknown"


def cache_policy(model_precision: str) -> tuple[str, str]:
    if model_precision == "f32":
        return ("runtime_catalog",
                "F32 is the only precision admitted to the automatic LoMa cache.")
    return ("experimental_only",
            "Excluded from automatic cache until a complete real-image "
            "accuracy and reconstruction gate passes.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.artifacts.is_dir():
        raise SystemExit(f"artifact directory does not exist: {args.artifacts}")
    models = []
    for path in sorted(args.artifacts.glob("*.gguf")):
        model_precision = precision(path.name)
        policy, policy_reason = cache_policy(model_precision)
        models.append({
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "precision": model_precision,
            "cache_policy": policy,
            "cache_policy_reason": policy_reason,
        })
    if not models:
        raise SystemExit(f"no GGUF models under {args.artifacts}")

    fixtures = sorted(path.name for path in args.artifacts.glob("*.bin"))
    payload = {
        "schema": 1,
        "runtime_format": "gguf",
        "artifacts": models,
        "excluded_test_fixtures": fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
