#!/usr/bin/env python3
"""Resolve the exact Paddle weights an AICore OCR GGUF was converted from.

A Paddle reference is only a parity reference when it runs the same weights
as the GGUF under test. Every OCR GGUF records its source in
``ocr.source.name`` / ``ocr.source.sha256`` (the ``inference.pdiparams``
digest). The PaddleX 3.0.1 official model of the same family is NOT always
that file: ``PP-OCRv4_mobile_det`` shares 202 of 234 parameters with the
PaddleOCR ``ch_PP-OCRv4_det_infer`` release the det GGUF is converted from,
but its 16 RSEFPN SE convolutions carry different weights and all-zero
biases. On real pages that changes the det map (sum 9120 vs 10161 on
medal_table.png) and the box set (93 vs 96 boxes), so a comparison against
the PaddleX default measures a weight difference, not the AICore runtime.

``source_model_dir`` provisions the release archive, verifies the digest
against the GGUF and returns a PaddleX-loadable model directory (the
release's legacy inference.pdmodel/pdiparams plus the PaddleX
``inference.yml`` of the matching official model, which only carries the
pre/post-process config). Models whose PaddleX official parameters are the
GGUF source (SLANet_plus, en_PP-OCRv4_mobile_rec) need no override, but
resolving them through here still proves it.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_ocr_fixtures import Gguf  # noqa: E402

# GGUF ocr.source.name -> (release archive URL, PaddleX official model whose
# inference.yml describes the same pre/post-process).
SOURCES = {
    "ch_PP-OCRv4_det_infer": (
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
        "PP-OCRv4_mobile_det"),
    "ch_PP-OCRv4_rec_infer": (
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
        "PP-OCRv4_mobile_rec"),
    "en_PP-OCRv4_rec_infer": (
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        "en_PP-OCRv4_mobile_rec"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gguf_source(gguf_path: Path) -> tuple[str, str]:
    kv = Gguf(str(gguf_path)).kv
    name, sha = kv.get("ocr.source.name"), kv.get("ocr.source.sha256")
    if not name or not sha:
        raise SystemExit(f"{gguf_path}: no ocr.source.name/sha256 metadata")
    return name, sha


def provision(name: str, sha256: str, cache_root: Path, paddlex_yml: Path,
              url: str | None = None) -> Path:
    """Download/extract `name`, verify its pdiparams digest, add the yml."""
    if url is None:
        if name not in SOURCES:
            raise SystemExit(f"no release archive known for GGUF source {name}")
        url = SOURCES[name][0]
    target = cache_root / name
    params = target / "inference.pdiparams"
    if not params.is_file() or sha256_file(params) != sha256:
        cache_root.mkdir(parents=True, exist_ok=True)
        archive = cache_root / f"{name}.tar"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and Path(m.name).name in
                       ("inference.pdmodel", "inference.pdiparams")]
            target.mkdir(parents=True, exist_ok=True)
            for member in members:
                source = tar.extractfile(member)
                with (target / Path(member.name).name).open("wb") as out:
                    shutil.copyfileobj(source, out)
        archive.unlink()
    got = sha256_file(params) if params.is_file() else "missing"
    if got != sha256:
        raise SystemExit(f"{name}: inference.pdiparams sha256 {got} != GGUF "
                         f"ocr.source.sha256 {sha256}")
    shutil.copyfile(paddlex_yml, target / "inference.yml")
    return target


def source_model_dir(gguf_path: Path, cache_root: Path) -> Path:
    """PaddleX model_dir holding exactly the GGUF's source weights."""
    name, sha = gguf_source(gguf_path)
    if name not in SOURCES:
        raise SystemExit(f"{gguf_path}: no release archive known for {name}")
    from paddlex.inference.utils.official_models import official_models
    official = Path(official_models[SOURCES[name][1]])
    if sha256_file(official / "inference.pdiparams") == sha:
        return official
    return provision(name, sha, cache_root, official / "inference.yml")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path, source_model_dir(Path(path), Path.home() / ".cache" /
                                     "aicore_paddle_sources"))
