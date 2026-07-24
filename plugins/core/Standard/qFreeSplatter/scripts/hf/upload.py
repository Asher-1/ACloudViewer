#!/usr/bin/env python3
"""Publish free-splatter.cpp GGUF weights + model card to a Hugging Face repo.

The GGUFs are produced by scripts/convert.py and live outside git (in .cache/).
This pushes them, plus MODEL_CARD.md (uploaded as the repo's README.md), to a HF
model repo so the project README's "Releases" link has somewhere to point.

Setup:
    pip install huggingface_hub
    huggingface-cli login            # or: export HF_TOKEN=hf_...

Usage:
    python scripts/hf/upload.py --repo <org-or-user>/<name>
    python scripts/hf/upload.py --repo me/freesplatter.cpp-gguf --private
    python scripts/hf/upload.py --repo me/freesplatter.cpp-gguf \
        --files .cache/freesplatter-scene-f16.gguf .cache/freesplatter-object-f16.gguf
    python scripts/hf/upload.py --repo me/freesplatter.cpp-gguf --dry-run

By default it uploads every *.gguf under .cache/ (override with --files) and the
model card next to this script. Nothing is uploaded with --dry-run.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CARD = HERE / "MODEL_CARD.md"


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload GGUF weights + model card to Hugging Face.")
    ap.add_argument("--repo", required=True, help="target repo id, e.g. localai-org/freesplatter.cpp-gguf")
    ap.add_argument("--files", nargs="*", default=None,
                    help="GGUF files to upload (default: .cache/*.gguf)")
    ap.add_argument("--card", default=str(CARD), help="model card markdown (uploaded as README.md)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token (default: $HF_TOKEN or a prior `huggingface-cli login`)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be uploaded and exit")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(str(ROOT / ".cache" / "*.gguf")))
    files = [Path(f) for f in files]
    missing = [f for f in files if not f.is_file()]
    if missing:
        print("error: missing files: " + ", ".join(map(str, missing)), file=sys.stderr)
        return 1
    if not files:
        print("error: no GGUF files found (build them with scripts/convert.py, or pass --files)", file=sys.stderr)
        return 1
    card = Path(args.card)
    if not card.is_file():
        print(f"error: model card not found: {card}", file=sys.stderr)
        return 1

    total_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"repo:  {args.repo}  ({'private' if args.private else 'public'})")
    print(f"card:  {card}  ->  README.md")
    print("files:")
    for f in files:
        print(f"  {f}  ({f.stat().st_size/1e9:.2f} GB)")
    print(f"total: {total_gb:.2f} GB")

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"\nuploading README.md (model card) …")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    for f in files:
        print(f"uploading {f.name} …")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=args.repo, repo_type="model")
    print(f"\ndone -> https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
