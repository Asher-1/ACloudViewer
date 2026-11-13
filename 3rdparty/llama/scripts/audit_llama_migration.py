#!/usr/bin/env python3
"""Audit a ggml-to-llama.cpp migration without modifying either source tree.

This is a source/coverage inventory, not an inference validation runner. Model
acceptance stays in core/AICore/scripts/validate_all.py. An optional merge
directory contains review candidates, including explicit conflict markers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess


QUANT_SUFFIX = re.compile(r"(?:^|[-_.])(f32|f16|bf16|q\d[^.]*)\.gguf$", re.I)


def files(root: Path) -> dict[Path, Path]:
    return {p.relative_to(root): p for p in root.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(root).parts}


def identity(path: Path) -> dict:
    return {"path": str(path.absolute()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tree_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for relative, path in sorted(files(root).items()):
        digest.update(relative.as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def validate_report_path(output: Path, roots: tuple[Path, ...]) -> None:
    if any(output.resolve().is_relative_to(root.resolve()) for root in roots):
        raise ValueError("report output must be outside the source inputs")


def merge_inventory(baseline: Path, patched: Path, upstream: Path,
                    destination: Path | None = None) -> list[dict]:
    """Three-way merge every downstream-modified file; never prefer a side.

    Cost is linear in the source bytes plus git's per-file diff/merge cost.
    Each merge is isolated because one failed hunk must not hide later files.
    """
    before, after, target = files(baseline), files(patched), files(upstream)
    if destination is not None:
        if destination.exists():
            raise ValueError(f"merge destination already exists: {destination}")
        # Reject output within any input, which would recursively copy itself.
        for root in (baseline, patched, upstream):
            if destination.resolve().is_relative_to(root.resolve()):
                raise ValueError("merge destination must be outside the inputs")
        shutil.copytree(upstream, destination)
    rows = []
    for relative in sorted(before.keys() | after.keys()):
        b, p, n = before.get(relative), after.get(relative), target.get(relative)
        if b and p and b.read_bytes() == p.read_bytes():
            continue
        merged = None
        if b is None:
            status = "add_add" if n else "added"
            if not n:
                merged = p.read_bytes()
        elif p is None:
            status = "local_deleted"
        elif n is None:
            status = "upstream_deleted"
        else:
            result = subprocess.run(
                ["git", "merge-file", "-p", "--diff3", str(n.absolute()),
                 str(b.absolute()), str(p.absolute())], capture_output=True)
            if result.returncode < 0 or result.returncode > 127:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            status = "merged" if result.returncode == 0 else "conflict"
            merged = result.stdout
        row = {"file": relative.as_posix(), "status": status,
               "conflicts": merged.count(b"<<<<<<< ") if merged else 0}
        rows.append(row)
        if destination is not None and merged is not None:
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(merged)
    return rows


def catalog_inventory(catalog: Path, manifest: dict, assets: Path) -> dict:
    rows, tasks = [], manifest["tasks"]
    for line in catalog.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        relative = PurePosixPath(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe catalog path: {relative}")
        if entry["task"] not in tasks:
            raise ValueError(f"catalog task absent from manifest: {entry['task']}")
        match = QUANT_SUFFIX.search(relative.name)
        quant = match[1].lower() if match else "unspecified"
        consumers = []
        for scenario in manifest["scenarios"]:
            # A single physical model can have several catalog owners (YOLO /
            # ReID). Its inference consumer may belong to the other task.
            # owned_globs describes ownership, not an inference consumer.
            globs = ([scenario["model_glob"]]
                     if scenario.get("for_each_model") and scenario.get("model_glob")
                     else scenario.get("covered_globs", []))
            if any(fnmatch.fnmatchcase(str(relative), g) for g in globs):
                consumers.append(f"{scenario['task']}/{scenario['id']}")
        rows.append({**entry, "filename_quant_hint": quant,
                     "selected": quant in {"f16", "q8_0", "q8", "unspecified"},
                     "cache_file_present": (assets / relative).is_file(),
                     "consumers": consumers})
    summaries = []
    for task in tasks:
        owned = [r for r in rows if r["task"] == task]
        selected = [r for r in owned if r["selected"]]
        summaries.append({"task": task, "catalog_rows": len(owned),
                          "filename_quant_hints": dict(Counter(
                              r["filename_quant_hint"] for r in owned)),
                          "selected_rows": len(selected),
                          "cache_files_present": sum(r["cache_file_present"]
                                                     for r in selected),
                          "all_without_consumer": [r["relative_path"] for r in owned
                                                   if not r["consumers"]],
                          "without_consumer": [r["relative_path"] for r in selected
                                               if not r["consumers"]]})
    return {"catalog_identity": identity(catalog), "tasks": summaries,
            "rows": rows, "cache_note": "Presence only; SHA and GGUF tensor "
            "types must be checked by the common validation runner/probes.",
            "selected_unique_files": len({r["relative_path"] for r in rows
                                          if r["selected"]}),
            "all_catalog_unique_files": len({r["relative_path"] for r in rows}),
            "all_without_consumer": [r["relative_path"] for r in rows
                                     if not r["consumers"]],
            "inference_executed": False}


def header_inventory(patched: Path, upstream: Path, aicore: Path) -> dict:
    def symbols(paths):
        result = set()
        for p in paths:
            result.update(re.findall(r"\b(ggml_\w+)\s*\(",
                                     p.read_text(errors="replace")))
        return result
    old = symbols((patched / "include").rglob("*.h"))
    new = symbols((upstream / "include").rglob("*.h"))
    used = symbols(p for p in aicore.rglob("*")
                   if p.is_file() and p.suffix in {".h", ".hpp", ".cpp", ".c"})
    return {"used_names_absent_from_upstream_headers": sorted((old - new) & used),
            "note": "Lexical inventory includes downstream extensions; this is "
            "not a signature, ABI, link, or numerical compatibility test."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "patched", "llama", "repo", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--merge-dir", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--assets", type=Path,
                        default=Path.home() / "cloudViewer_data/extract")
    args = parser.parse_args()
    upstream = args.llama / "ggml"
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit")
    for path in (args.baseline, args.patched, upstream, args.repo / "core/AICore"):
        if not path.is_dir():
            parser.error(f"source directory missing: {path}")
    try:
        validate_report_path(args.output, (args.baseline, args.patched, args.llama))
        rows = merge_inventory(args.baseline, args.patched, upstream,
                               args.merge_dir)
        report = {"llama_revision": args.revision, "modified_files": rows,
                  "source_tree_sha256": {
                      "baseline": tree_identity(args.baseline),
                      "patched": tree_identity(args.patched),
                      "upstream_ggml": tree_identity(upstream)},
                  "file_status_counts": dict(Counter(r["status"] for r in rows)),
                  "conflicts": sum(r["conflicts"] for r in rows),
                  "headers": header_inventory(args.patched, upstream,
                                              args.repo / "core/AICore/src"),
                  "inference_executed": False}
        if args.catalog:
            manifest_path = args.repo / "core/AICore/scripts/validation_manifest.json"
            report["manifest_identity"] = identity(manifest_path)
            report["catalog"] = catalog_inventory(
                args.catalog, json.loads(manifest_path.read_text()), args.assets)
        unresolved = [r for r in rows if r["status"] not in {"added", "merged"}]
        report["verdict"] = "SOURCE_REVIEW_REQUIRED" if unresolved else "SOURCE_MERGE_ONLY"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in
                          ("verdict", "file_status_counts", "conflicts")}))
        return 1 if unresolved else 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
