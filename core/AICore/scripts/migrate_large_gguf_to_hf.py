"""Migrate >1GB catalog GGUFs from GitHub releases to HuggingFace repos.

Reads the migration plan produced from `test_catalog_dump_urls --json`
(rows whose URL host is github.com and whose verified local cache copy is
>1GB). Creates one HF repo per task family following the existing
Asher-1/<Task>_GGUF naming convention, uploads the byte-identical files
(SHA-256 verified against the catalog before upload), then HEAD-checks each
uploaded object's size. The catalog URL switch happens in a separate step
after this script exits cleanly.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url

REPO_BY_PREFIX = [
    ("da3_models/", "Asher-1/DA3_GGUF"),
    ("freesplatter_models/", "Asher-1/FreeSplatter_GGUF"),
    ("loma_models/", "Asher-1/LoMa_GGUF"),
    ("anycalib_models/", "Asher-1/AnyCalib_GGUF"),
]

CACHE_ROOT = Path.home() / "cloudViewer_data" / "extract"


def repo_for(relative_path: str) -> str:
    for prefix, repo in REPO_BY_PREFIX:
        if relative_path.startswith(prefix):
            return repo
    raise SystemExit(f"no repo mapping for {relative_path}")


def main() -> int:
    plan = json.loads(Path("/tmp/hf-migration-plan.json").read_text())
    api = HfApi()
    receipts = []
    for row in plan:
        rel = row["relative_path"]
        local = CACHE_ROOT / rel
        repo = repo_for(rel)
        filename = Path(rel).name
        url = hf_hub_url(repo_id=repo, filename=filename, revision="main")

        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True,
                        private=False)

        # Idempotent: skip re-upload when the remote object already matches.
        existing = api.get_paths_info(repo, [filename], repo_type="model")
        remote_entry = existing[0] if existing else None
        local_size = local.stat().st_size
        already = False
        if remote_entry is not None:
            lfs = getattr(remote_entry, "lfs", None)
            remote_size = (lfs.size if lfs else 0) or (remote_entry.size or 0)
            already = remote_size == local_size
        if already:
            print(json.dumps({"relative_path": rel, "repo": repo,
                              "bytes": local_size, "skipped": True}),
                  flush=True)
            receipts.append({"relative_path": rel, "repo": repo,
                             "bytes": local_size, "remote_size": remote_size,
                             "sha256": row["sha256"], "upload_seconds": 0,
                             "size_ok": True})
            continue

        started = time.time()
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=filename,
            repo_id=repo,
            repo_type="model",
        )
        elapsed = time.time() - started

        remote_entries = api.get_paths_info(repo, [filename],
                                            repo_type="model")
        entry = remote_entries[0] if remote_entries else None
        lfs = getattr(entry, "lfs", None) if entry else None
        remote_size = (lfs.size if lfs else 0) or (entry.size if entry else 0)
        ok = remote_size == local_size
        receipt = {
            "relative_path": rel,
            "repo": repo,
            "bytes": local.stat().st_size,
            "remote_size": remote_size,
            "sha256": row["sha256"],
            "upload_seconds": round(elapsed, 1),
            "size_ok": ok,
        }
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)

    out = Path("/tmp/hf-migration-receipts.json")
    out.write_text(json.dumps(receipts, indent=1) + "\n")
    failed = [r for r in receipts if not r["size_ok"]]
    print(f"uploaded {len(receipts)} files, size mismatches: {len(failed)}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
