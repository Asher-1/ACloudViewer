"""Family-level HF migration: upload every GitHub-hosted catalog row of the
task families that contain at least one >1GB file (DA3, FreeSplatter, LoMa,
AnyCalib). Byte-identical upload (SHA-256 verified against the catalog before
upload), idempotent (already-uploaded rows are skipped), remote size checked
afterwards. Catalog URL switching happens in the C++ sources separately.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO_BY_PREFIX = [
    ("da3_models/", "Asher-1/DA3_GGUF"),
    ("freesplatter_models/", "Asher-1/FreeSplatter_GGUF"),
    ("loma_models/", "Asher-1/LoMa_GGUF"),
    ("anycalib_models/", "Asher-1/AnyCalib_GGUF"),
]

CACHE_ROOT = Path.home() / "cloudViewer_data" / "extract"
PLAN = Path("/tmp/hf-family-plan.json")


def repo_for(relative_path: str) -> str:
    for prefix, repo in REPO_BY_PREFIX:
        if relative_path.startswith(prefix):
            return repo
    raise SystemExit(f"no repo mapping for {relative_path}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    plan = json.loads(PLAN.read_text())
    api = HfApi()
    receipts, failures = [], []
    for row in plan:
        rel = row["relative_path"]
        local = CACHE_ROOT / rel
        repo = repo_for(rel)
        filename = Path(rel).name

        digest = sha256_of(local)
        if digest != row["sha256"]:
            failures.append({"relative_path": rel, "error": "local sha mismatch"})
            print(json.dumps(failures[-1]), flush=True)
            continue

        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True,
                        private=False)

        existing = api.get_paths_info(repo, [filename], repo_type="model")
        entry = existing[0] if existing else None
        lfs = getattr(entry, "lfs", None) if entry else None
        remote_size = (lfs.size if lfs else 0) or (entry.size if entry else 0)
        local_size = local.stat().st_size
        if remote_size == local_size:
            receipts.append({"relative_path": rel, "repo": repo,
                             "bytes": local_size, "skipped": True,
                             "size_ok": True})
            print(json.dumps(receipts[-1]), flush=True)
            continue

        started = time.time()
        api.upload_file(path_or_fileobj=str(local), path_in_repo=filename,
                        repo_id=repo, repo_type="model")
        elapsed = time.time() - started

        entries = api.get_paths_info(repo, [filename], repo_type="model")
        entry = entries[0] if entries else None
        lfs = getattr(entry, "lfs", None) if entry else None
        remote_size = (lfs.size if lfs else 0) or (entry.size if entry else 0)
        ok = remote_size == local_size
        receipt = {"relative_path": rel, "repo": repo, "bytes": local_size,
                   "remote_size": remote_size, "upload_seconds": round(elapsed, 1),
                   "size_ok": ok}
        receipts.append(receipt)
        if not ok:
            failures.append(receipt)
        print(json.dumps(receipt), flush=True)

    Path("/tmp/hf-family-receipts.json").write_text(
        json.dumps(receipts, indent=1) + "\n")
    print(f"processed {len(receipts)} rows, failures: {len(failures)}",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
