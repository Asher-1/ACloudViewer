"""Shared small helpers for the AICore validation scripts and tools.

These used to be re-implemented per script (sha256_file had a dozen
divergent copies); the maintenance rule lives in
`.agents/skills/acloudviewer-aicore-plugin/SKILL.md` (Scripts & tools).
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streamed SHA-256 hex digest of a file (1 MiB chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()
