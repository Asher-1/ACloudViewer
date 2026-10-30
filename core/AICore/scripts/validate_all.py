#!/usr/bin/env python3
"""Run the complete AICore model/task regression matrix.

The runner is intentionally orchestration-only. Task-specific C++ probes own
the meaning of accuracy and inference timing; this script owns discovery,
repeatability, coverage, baseline comparison, and reporting.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MANIFEST = SCRIPT_DIR / "validation_manifest.json"
TIMING_SUFFIXES = ("_ms", "_mib")
HASH_KEYS = ("hash", "sha", "digest")
DOWNLOAD_USER_AGENT = "ACloudViewer-AICore-validation/1"
DEFAULT_VRAM_OVERHEAD_MIB = 1024.0
VRAM_SKIP_STATUS = "vram_skipped"
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def tiered_tasks(manifest: dict[str, Any]) -> set[str]:
    """Tasks that declare a lightweight subset of their scenarios.

    A task opts in by marking at least one scenario with "light": true or
    "light_globs". In the default (light) tier such tasks run only that
    subset; every other task is unaffected. "--full" removes the
    restriction and is the only complete-matrix evidence.
    """
    return {scenario["task"] for scenario in manifest["scenarios"]
            if scenario.get("light") or scenario.get("light_globs")}


def scenario_runs_in_tier(scenario: dict[str, Any], task: str,
                          tiered: set[str], full: bool) -> bool:
    if full or task not in tiered:
        return True
    return bool(scenario.get("light") or scenario.get("light_globs"))


@dataclass(frozen=True)
class RunSpec:
    scenario_id: str
    task: str
    model_id: str
    model_paths: tuple[Path, ...]
    command: tuple[str, ...]
    env: dict[str, str]
    accuracy_gate: str
    metric_parser: str
    fingerprint_policy: str
    require_fingerprint: bool
    report_path: Path
    accuracy: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    # VRAM budget: explicit working-set estimate in MiB, or model bytes plus
    # an activation/working-set overhead when the scenario does not declare
    # one. Only consulted for GPU backends; see vram_gate_decision().
    vram_estimate_mib: float | None = None
    vram_overhead_mib: float | None = None

    @property
    def key(self) -> str:
        return f"{self.task}/{self.scenario_id}/{self.model_id}"


def query_free_vram_mib() -> float | None:
    """Free VRAM (MiB) of the most spacious NVIDIA GPU, or None.

    Probes run on a single device, so the best single GPU is the honest
    upper bound of what a probe can allocate. None means the query is
    unavailable (no nvidia-smi, non-NVIDIA platform) and the caller must
    fail open instead of guessing.
    """
    output = command_output([
        "nvidia-smi", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"])
    if not output:
        return None
    values: list[float] = []
    for line in output.splitlines():
        try:
            values.append(float(line.strip().rstrip("MiB")))
        except ValueError:
            return None
    return max(values) if values else None


def estimate_spec_vram_mib(spec: RunSpec, default_overhead: float) -> float:
    """Estimated VRAM need in MiB: declared value, or model bytes + overhead."""
    if spec.vram_estimate_mib is not None:
        return float(spec.vram_estimate_mib)
    model_mib = sum(
        path.stat().st_size for path in spec.model_paths if path.is_file())
    return model_mib / (1024.0 * 1024.0) + max(
        0.0, spec.vram_overhead_mib if spec.vram_overhead_mib is not None
        else default_overhead)


def vram_gate_decision(spec: RunSpec, args: argparse.Namespace
                       ) -> dict[str, float] | None:
    """Return {'needed_mib', 'free_mib'} when VRAM is insufficient, else None.

    CPU-only runs and hosts without a queryable GPU never gate (fail-open);
    gating only exists to replace a guaranteed OOM crash with an explicit,
    reported skip.
    """
    if args.backend == "cpu":
        return None
    free = query_free_vram_mib()
    if free is None:
        return None
    needed = estimate_spec_vram_mib(spec, args.vram_overhead_mib)
    if needed <= free:
        return None
    return {"needed_mib": needed, "free_mib": free}


@dataclass
class Attempt:
    returncode: int
    duration_ms: float
    metrics: dict[str, float] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    output_tail: str = ""


@dataclass(frozen=True)
class ModelAsset:
    task: str
    relative_path: PurePosixPath
    url: str
    sha256: str
    size_bytes: int = 0

    def destination(self, root: Path) -> Path:
        return root.joinpath(*self.relative_path.parts)


@dataclass(frozen=True)
class InputAsset:
    """Pinned functional fixture, optionally delivered as a ZIP archive."""

    asset_id: str
    tasks: frozenset[str]
    relative_path: PurePosixPath
    url: str
    sha256: str
    size_bytes: int = 0
    archive: bool = False
    extracted_files: tuple[tuple[PurePosixPath, str, int], ...] = ()

    def destination(self, root: Path) -> Path:
        return root.joinpath(*self.relative_path.parts)


def selected_tasks(manifest: dict[str, Any], task_text: str) -> set[str]:
    selected = ({task.strip() for task in task_text.split(",") if task.strip()}
                if task_text else set(manifest["tasks"]))
    unknown = selected - set(manifest["tasks"])
    if unknown:
        raise ValueError("unknown tasks: " + ", ".join(sorted(unknown)))
    return selected


def select_scenarios(manifest: dict[str, Any], scenario_text: str) -> dict[str, Any]:
    """Return a manifest view restricted to exact scenario ids."""
    selected = {item.strip() for item in scenario_text.split(",") if item.strip()}
    if not selected:
        return manifest
    known = {row["id"] for row in manifest["scenarios"]}
    unknown = selected - known
    if unknown:
        raise ValueError("unknown scenarios: " + ", ".join(sorted(unknown)))
    narrowed = dict(manifest)
    narrowed["scenarios"] = [row for row in manifest["scenarios"]
                             if row["id"] in selected]
    return narrowed


def scenario_glob_excluded(path: Path, scenario: dict[str, Any]) -> bool:
    """True when the scenario's model_glob_exclude rules reject this model.

    Excludes exist for multi-modal models (BEVFusion, CaDDN) that share a
    family glob with point-only models but need the dedicated scene scenario.
    Patterns are catalog-relative (``openpcdet_models/kitti_caddn.*.gguf``)
    while callers may hold absolute paths, so the bare file name matches too.
    """
    name = path.name
    for pattern in scenario.get("model_glob_exclude") or []:
        if fnmatch.fnmatchcase(str(path), pattern):
            return True
        base = pattern.rsplit("/", 1)[-1]
        if base != pattern and fnmatch.fnmatchcase(name, base):
            return True
    return False


def manifest_model_patterns(manifest: dict[str, Any],
                            selected: set[str], *, full: bool = True,
                            tiered: set[str] | None = None) -> set[str]:
    """Catalog glob patterns that the selected task set consumes.

    In the light tier, tiered tasks contribute only their declared
    lightweight subset, so the model-cache preflight verifies and downloads
    exactly what the run will execute instead of every catalog model of the
    task. Non-tiered tasks keep their full pattern set.
    """
    tiered = tiered if tiered is not None else tiered_tasks(manifest)
    patterns: set[str] = set()
    for scenario in manifest["scenarios"]:
        task = scenario["task"]
        if task not in selected:
            continue
        if not scenario_runs_in_tier(scenario, task, tiered, full):
            continue
        light_globs = scenario.get("light_globs") or []
        light_active = not full and task in tiered and light_globs
        for key in ("model_glob", "owned_globs", "covered_globs",
                    "required_globs"):
            values = light_globs if light_active and key != "required_globs" \
                else scenario.get(key, [])
            if isinstance(values, str):
                values = [values]
            patterns.update(value for value in values if value)
    return patterns


def parse_catalog_output(output: str, selected: set[str],
                         dependency_patterns: set[str] | None = None,
                         task_light_patterns: dict[str, set[str]] | None = None,
                         task_selection: dict[str, set[str]] | None = None
                         ) -> list[ModelAsset]:
    assets: dict[PurePosixPath, ModelAsset] = {}
    tasks_with_assets: set[str] = set()
    patterns = dependency_patterns or set()
    task_light_patterns = task_light_patterns or {}
    selection_filter = task_selection
    task_selection = task_selection or {}
    for line_number, line in enumerate(output.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"model catalog line {line_number} is not JSON: {error}") from error
        task = raw.get("task")
        relative = PurePosixPath(str(raw.get("relative_path", "")))
        url = str(raw.get("url", ""))
        digest = str(raw.get("sha256", "")).lower()
        size = raw.get("size_bytes", 0)
        if (relative.is_absolute() or ".." in relative.parts or
                len(relative.parts) != 2 or
                not relative.parts[0].endswith("_models") or
                relative.suffix.lower() != ".gguf"):
            raise ValueError(
                f"unsafe model catalog destination on line {line_number}: {relative}")
        if urllib.parse.urlparse(url).scheme not in {"https", "http", "file"}:
            raise ValueError(
                f"unsupported model URL on line {line_number}: {url!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"missing pinned SHA-256 on line {line_number}: {relative}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(
                f"invalid model size on line {line_number}: {size!r}")
        task_required = task_light_patterns.get(task)
        if task not in selected and not any(
                fnmatch.fnmatchcase(str(relative), pattern)
                for pattern in patterns):
            continue
        if task in selected and task_required is not None and not any(
                fnmatch.fnmatchcase(str(relative), pattern)
                for pattern in task_required):
            # Light tier: this task declared a lightweight subset, so catalog
            # rows outside it are neither prefetched nor executed here. The
            # complete matrix is restored with --full.
            continue
        selection = task_selection.get(task)
        if task in selected and selection is not None and not any(
                fnmatch.fnmatchcase(str(relative), pattern)
                for pattern in selection):
            # Explicit --models selection: only the named models (and the
            # bundle scenarios/required dependencies they pull in) are
            # prefetched and executed.
            continue
        asset = ModelAsset(task, relative, url, digest, size)
        previous = assets.get(relative)
        if previous is not None:
            # One physical asset may be consumed by several tasks (reid
            # embeds the shared yolo classify GGUFs and owns no model files
            # of its own). The first task row keeps cache/regression
            # ownership, but a conflicting url/digest/size declaration for
            # the same file is still catalog drift and must fail here.
            if (previous.url, previous.sha256, previous.size_bytes) != (
                    asset.url, asset.sha256, asset.size_bytes):
                raise ValueError(
                    f"conflicting model catalog rows for {relative}")
        assets.setdefault(relative, asset)
        if task in selected:
            tasks_with_assets.add(task)
    missing_tasks = selected - tasks_with_assets
    if selection_filter is not None:
        if not tasks_with_assets:
            raise ValueError(
                "--models matched no catalog rows for tasks: " +
                ", ".join(sorted(selected)))
    elif missing_tasks:
        raise ValueError(
            "model catalog has no assets for tasks: " +
            ", ".join(sorted(missing_tasks)))
    return [assets[path] for path in sorted(assets, key=str)]


def manifest_catalog_drift(manifest: dict[str, Any], catalog_output: str,
                           selected: set[str]) -> list[str]:
    """Model patterns of the selected tasks that match no catalog row.

    The preflight only downloads catalog rows a pattern selects, so a pattern
    naming an unpublished file (renamed release asset, wrong quant suffix) is
    dropped silently: it surfaces later as "missing asset", or never, when a
    sibling glob excludes the real file and no scenario runs it. local_only
    scenarios consume user-installed assets and are exempt.
    """
    paths = []
    for line in catalog_output.splitlines():
        if line.strip():
            paths.append(str(json.loads(line).get("relative_path", "")))
    drift: list[str] = []
    # O(patterns x rows) fnmatch; both are a few hundred at most.
    for scenario in manifest["scenarios"]:
        if scenario["task"] not in selected or scenario.get("local_only"):
            continue
        for key in ("model_glob", "owned_globs", "covered_globs",
                    "required_globs", "light_globs"):
            values = scenario.get(key) or []
            if isinstance(values, str):
                values = [values]
            drift.extend(
                f"{scenario['id']}.{key}: {pattern}" for pattern in values
                if pattern and not any(fnmatch.fnmatchcase(path, pattern)
                                       for path in paths))
    return drift


def load_model_catalog(args: argparse.Namespace, manifest: dict[str, Any],
                       selected: set[str]) -> list[ModelAsset]:
    binary = (args.catalog_binary.resolve() if args.catalog_binary else
              find_binary(args.build, "test_catalog_dump_urls"))
    if not binary.is_file():
        raise RuntimeError(
            f"model catalog binary not found: {binary}; build target "
            "test_catalog_dump_urls first")
    env = os.environ.copy()
    add_runtime_library_path(env, args.build)
    try:
        completed = subprocess.run(
            [str(binary), "--json"], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"could not execute model catalog: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"model catalog failed with code {completed.returncode}: {detail}")
    drift = manifest_catalog_drift(manifest, completed.stdout, selected)
    if drift:
        raise ValueError("manifest model patterns match no catalog asset: " +
                         "; ".join(drift))
    full = bool(getattr(args, "full", False))
    tiered = tiered_tasks(manifest)
    model_filter = [
        item.strip() for item in (getattr(args, "models", "") or "").split(",")
        if item.strip()]
    if model_filter:
        # Explicit selection: bypass the tier and prefetch exactly the named
        # models plus the bundle scenarios and required dependencies they
        # pull in. User patterns address basenames, so they are expanded to
        # relative-path globs (fnmatch's * also spans the model directory).
        selection = model_selection_globs(manifest, selected, model_filter)
        for pattern in model_filter:
            expanded = {pattern, f"*/{pattern}"}
            if not pattern.lower().endswith(".gguf"):
                expanded.add(f"*/{pattern}.gguf")
            selection.update(expanded)
        task_selection = {task: set(selection) for task in selected}
        return parse_catalog_output(
            completed.stdout, selected, selection,
            task_selection=task_selection)
    light_patterns = {
        task: manifest_model_patterns(
            manifest, {task}, full=False, tiered=tiered)
        for task in tiered}
    # Only tasks with exclusively explicit local_only scenarios need no downloads; empty or mixed published tasks still require a catalog.
    # Keep task publication responsibility in one pass; one local scenario must not exempt its published models.
    tasks_with_scenarios: set[str] = set()
    published_tasks: set[str] = set()
    for scenario in manifest['scenarios']:
        task = scenario['task']
        if task in selected:
            tasks_with_scenarios.add(task)
            if not scenario.get('local_only'):
                published_tasks.add(task)
    catalog_selected = published_tasks | (selected - tasks_with_scenarios)
    return parse_catalog_output(
        completed.stdout, catalog_selected,
        manifest_model_patterns(manifest, selected, full=full, tiered=tiered),
        task_light_patterns=None if full else light_patterns)


def parse_input_assets(manifest: dict[str, Any], selected: set[str],
                       scenario_ids: set[str] | None = None
                       ) -> list[InputAsset]:
    selected_scenarios = [
        scenario for scenario in manifest["scenarios"]
        if scenario["task"] in selected and
        (scenario_ids is None or scenario["id"] in scenario_ids)]
    scoped_input_ids: set[str] | None = None
    if selected_scenarios and all("input_ids" in scenario
                                  for scenario in selected_scenarios):
        scoped_input_ids = {
            str(asset_id) for scenario in selected_scenarios
            for asset_id in scenario.get("input_ids", [])}
    known_input_ids = {
        str(raw.get("id", "")) for raw in manifest.get("input_assets", [])}
    unknown_input_ids = (scoped_input_ids or set()) - known_input_ids
    if unknown_input_ids:
        raise ValueError("scenario input_ids name unknown assets: " +
                         ", ".join(sorted(unknown_input_ids)))

    assets: list[InputAsset] = []
    for index, raw in enumerate(manifest.get("input_assets", []), 1):
        asset_id = str(raw.get("id", ""))
        tasks = frozenset(str(task) for task in raw.get("tasks", []))
        if not tasks & selected or (scoped_input_ids is not None and
                                    asset_id not in scoped_input_ids):
            continue
        relative = PurePosixPath(str(raw.get("relative_path", "")))
        url = str(raw.get("url", ""))
        digest = str(raw.get("sha256", "")).lower()
        size = raw.get("size_bytes", 0)
        archive = bool(raw.get("archive", False))
        if (not asset_id or relative.is_absolute() or ".." in relative.parts or
                not relative.parts):
            raise ValueError(f"unsafe input asset destination on row {index}: "
                             f"{relative}")
        if urllib.parse.urlparse(url).scheme not in {"https", "http", "file"}:
            raise ValueError(f"unsupported input asset URL on row {index}: {url!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"missing pinned SHA-256 for input asset {asset_id}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid input asset size for {asset_id}: {size!r}")
        extracted: list[tuple[PurePosixPath, str, int]] = []
        for member in raw.get("extracted_files", []):
            member_path = PurePosixPath(str(member.get("relative_path", "")))
            member_digest = str(member.get("sha256", "")).lower()
            member_size = member.get("size_bytes", 0)
            if (not archive or member_path.is_absolute() or
                    ".." in member_path.parts or not member_path.parts or
                    not re.fullmatch(r"[0-9a-f]{64}", member_digest) or
                    not isinstance(member_size, int) or member_size < 0):
                raise ValueError(f"invalid extracted input asset for {asset_id}")
            extracted.append((member_path, member_digest, member_size))
        if archive != bool(extracted):
            raise ValueError(f"archive input asset {asset_id} must declare "
                             "pinned extracted files")
        assets.append(InputAsset(asset_id, tasks, relative, url, digest, size,
                                 archive, tuple(extracted)))
    if not assets:
        input_tokens = ("{image}", "{image2}", "{face_image}", "{sam_image}",
                        "{yolo_image}")
        needs_input = any(
            any(token in value for token in input_tokens)
            for scenario in selected_scenarios
            for value in list(scenario.get("args", [])) +
            list(scenario.get("env", {}).values()))
        if needs_input:
            raise ValueError("manifest has no input assets for selected tasks")
    return assets


def sha256_file(path: Path) -> str:
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(cache_key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = value
    return value


def verify_asset(asset: ModelAsset | InputAsset, root: Path) -> tuple[bool, str]:
    destination = asset.destination(root)
    if not destination.is_file():
        return False, "missing"
    try:
        if asset.size_bytes and destination.stat().st_size != asset.size_bytes:
            return False, "size mismatch"
        if sha256_file(destination) != asset.sha256:
            return False, "SHA-256 mismatch"
    except OSError as error:
        return False, str(error)
    return True, "verified"


def download_asset(asset: ModelAsset | InputAsset, root: Path, timeout: int,
                   retries: int) -> tuple[Path, int]:
    destination = asset.destination(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        partial: Path | None = None
        try:
            request = urllib.request.Request(
                asset.url, headers={"User-Agent": DOWNLOAD_USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected_header = response.headers.get("Content-Length")
                expected = int(expected_header) if expected_header else 0
                digest = hashlib.sha256()
                received = 0
                with tempfile.NamedTemporaryFile(
                        mode="wb", prefix=f".{destination.name}.", suffix=".part",
                        dir=destination.parent, delete=False) as stream:
                    partial = Path(stream.name)
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if expected and received != expected:
                raise OSError(
                    f"Content-Length mismatch: expected {expected}, got {received}")
            if asset.size_bytes and received != asset.size_bytes:
                raise OSError(
                    f"catalog size mismatch: expected {asset.size_bytes}, got {received}")
            actual_digest = digest.hexdigest()
            if actual_digest != asset.sha256:
                raise OSError(
                    f"SHA-256 mismatch: expected {asset.sha256}, got {actual_digest}")
            os.replace(partial, destination)
            partial = None
            stat = destination.stat()
            _FILE_HASH_CACHE[(str(destination.resolve()), stat.st_size,
                              stat.st_mtime_ns)] = actual_digest
            return destination, received
        except Exception as error:  # noqa: BLE001 - preserve network detail
            last_error = error
            if partial:
                try:
                    partial.unlink()
                except FileNotFoundError:
                    pass
            if attempt < retries:
                time.sleep(min(attempt, 3))
    raise RuntimeError(f"download failed after {retries} attempt(s): {last_error}")


def ensure_assets(assets: list[ModelAsset] | list[InputAsset], root: Path, *,
                  label: str, offline: bool, timeout: int, retries: int,
                  jobs: int) -> tuple[dict[str, Any], list[str]]:
    root.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[ModelAsset | InputAsset, str]] = []
    cached = 0
    for index, asset in enumerate(assets, 1):
        valid, reason = verify_asset(asset, root)
        if valid:
            cached += 1
        else:
            pending.append((asset, reason))
        print(f"[{label}-cache {index}/{len(assets)}] "
              f"{asset.relative_path}: {reason}", flush=True)

    summary: dict[str, Any] = {
        "root": str(root), "expected": len(assets), "verified_cached": cached,
        "downloaded": 0, "downloaded_bytes": 0, "unavailable": [],
    }
    if not pending:
        return summary, []
    if offline:
        summary["unavailable"] = [
            str(asset.relative_path) for asset, _ in pending
        ]
        return summary, [
            f"{asset.relative_path}: {reason} (offline mode)"
            for asset, reason in pending
        ]

    failures: list[str] = []
    workers = max(1, min(jobs, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_asset, asset, root, timeout, retries):
            (asset, reason)
            for asset, reason in pending
        }
        for future in as_completed(futures):
            asset, previous_reason = futures[future]
            try:
                destination, received = future.result()
                summary["downloaded"] += 1
                summary["downloaded_bytes"] += received
                print(f"[{label}-download] OK {destination} ({received} bytes)",
                      flush=True)
            except Exception as error:  # noqa: BLE001 - aggregate all failures
                summary["unavailable"].append(str(asset.relative_path))
                failures.append(
                    f"{asset.relative_path}: {previous_reason}; {error}")
                print(f"[{label}-download] FAIL {asset.relative_path}: {error}",
                      file=sys.stderr, flush=True)
    return summary, sorted(failures)


def ensure_model_assets(assets: list[ModelAsset], root: Path, *, offline: bool,
                        timeout: int, retries: int,
                        jobs: int) -> tuple[dict[str, Any], list[str]]:
    return ensure_assets(assets, root, label="model", offline=offline,
                         timeout=timeout, retries=retries, jobs=jobs)


def verify_extracted_input(asset: InputAsset, root: Path) -> tuple[bool, str]:
    for relative, digest, size in asset.extracted_files:
        destination = root.joinpath(*relative.parts)
        if not destination.is_file():
            return False, f"missing extracted file {relative}"
        try:
            if size and destination.stat().st_size != size:
                return False, f"size mismatch for extracted file {relative}"
            if sha256_file(destination) != digest:
                return False, f"SHA-256 mismatch for extracted file {relative}"
        except OSError as error:
            return False, str(error)
    return True, "verified"


def extract_input_archive(asset: InputAsset, archive: Path, root: Path) -> None:
    try:
        is_zip = zipfile.is_zipfile(archive)
        with (zipfile.ZipFile(archive) if is_zip else tarfile.open(archive, "r:*")) as stream:
            wanted = {str(path) for path, _, _ in asset.extracted_files}
            members = {}
            for entry in stream.infolist() if is_zip else stream.getmembers():
                name = entry.filename if is_zip else entry.name
                normalized = PurePosixPath(name)
                # Official Paddle archives may prefix regular members with
                # './'. Normalize that spelling without accepting traversal,
                # absolute paths or ambiguous duplicate members.
                if normalized.is_absolute() or ".." in normalized.parts:
                    continue
                key = str(normalized)
                if key not in wanted:
                    continue
                if key in members:
                    raise RuntimeError(f"ambiguous archive member: {key}")
                members[key] = entry
            for relative, digest, size in asset.extracted_files:
                member = str(relative)
                if member not in members:
                    raise RuntimeError(f"archive is missing {member}")
                entry = members[member]
                destination = root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if is_zip:
                    if (entry.is_dir() or
                            (entry.external_attr >> 16) & 0o170000 == 0o120000 or
                            (size and entry.file_size != size)):
                        raise RuntimeError(f"invalid archive member: {member}")
                    source = stream.open(entry)
                else:
                    # Read only pinned regular members. Never extract paths,
                    # device entries, symbolic links or hard links from a tar.
                    if not entry.isfile() or (size and entry.size != size):
                        raise RuntimeError(f"invalid archive member: {member}")
                    source = stream.extractfile(entry)
                with source:
                    payload = source.read()
                if size and len(payload) != size:
                    raise RuntimeError(f"archive member size mismatch: {member}")
                if hashlib.sha256(payload).hexdigest() != digest:
                    raise RuntimeError(f"archive member SHA-256 mismatch: {member}")
                with tempfile.NamedTemporaryFile(
                        mode="wb", prefix=f".{destination.name}.", suffix=".part",
                        dir=destination.parent, delete=False) as temporary:
                    partial = Path(temporary.name)
                    temporary.write(payload)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(partial, destination)
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as error:
        raise RuntimeError(f"could not extract {asset.relative_path}: {error}") from error


def ensure_input_assets(assets: list[InputAsset], root: Path, *, offline: bool,
                        timeout: int, retries: int,
                        jobs: int) -> tuple[dict[str, Any], list[str]]:
    files = [asset for asset in assets if not asset.archive]
    archives = [asset for asset in assets if asset.archive]
    file_summary, failures = ensure_assets(
        files, root, label="input", offline=offline, timeout=timeout,
        retries=retries, jobs=jobs)
    archive_root = root.parent / "download"
    archive_summary, archive_failures = ensure_assets(
        archives, archive_root, label="input", offline=offline, timeout=timeout,
        retries=retries, jobs=jobs)
    failures.extend(archive_failures)
    for asset in archives:
        valid, reason = verify_asset(asset, archive_root)
        if not valid:
            continue
        extracted, extract_reason = verify_extracted_input(asset, root)
        if extracted:
            continue
        if offline:
            failures.append(f"{asset.relative_path}: {extract_reason} (offline mode)")
            archive_summary["unavailable"].append(str(asset.relative_path))
            continue
        try:
            extract_input_archive(asset, asset.destination(archive_root), root)
            print(f"[input-extract] OK {asset.relative_path}", flush=True)
        except RuntimeError as error:
            failures.append(f"{asset.relative_path}: {error}")
            archive_summary["unavailable"].append(str(asset.relative_path))
    summary = {
        "root": str(root), "expected": len(assets),
        "verified_cached": (file_summary["verified_cached"] +
                            archive_summary["verified_cached"]),
        "downloaded": file_summary["downloaded"] + archive_summary["downloaded"],
        "downloaded_bytes": (file_summary["downloaded_bytes"] +
                             archive_summary["downloaded_bytes"]),
        "unavailable": sorted(set(file_summary["unavailable"] +
                                  archive_summary["unavailable"])),
    }
    return summary, sorted(set(failures))


def filter_unavailable_specs(
        specs: list[RunSpec], root: Path,
        unavailable: list[str]) -> tuple[list[RunSpec], list[str]]:
    blocked_paths = {
        root.joinpath(*PurePosixPath(relative).parts).resolve()
        for relative in unavailable
    }
    runnable: list[RunSpec] = []
    skipped: list[str] = []
    for spec in specs:
        blocked = sorted(
            path for path in spec.model_paths if path.resolve() in blocked_paths)
        if blocked:
            relative = ", ".join(
                str(path.resolve().relative_to(root)) for path in blocked)
            skipped.append(f"{spec.key}: unavailable model(s): {relative}")
        else:
            runnable.append(spec)
    return runnable, skipped


def load_manifest(path: Path, *, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    manifest = json.loads(data)
    if identity is not None:
        identity.update(path=str(path), bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest())
    if manifest.get("schema") not in (1, 2):
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")
    if (not manifest.get("tasks") or not manifest.get("scenarios") or
            not manifest.get("input_assets")):
        raise ValueError("manifest must declare tasks, scenarios, and input assets")
    for scenario in manifest["scenarios"]:
        normalize_accuracy_gate(scenario)
    return manifest


def normalize_accuracy_gate(scenario: dict[str, Any]) -> dict[str, Any]:
    """Return the executable accuracy contract for one manifest scenario.

    ``accuracy_gate`` is retained as a human-readable compatibility field.
    New scenarios should use the structured ``accuracy`` object. Existing
    rows are explicitly marked ``probe_legacy`` until their probe exports
    named numeric metrics; this prevents a prose claim from being mistaken
    for an enforced threshold.
    """
    legacy = scenario.get("accuracy_gate", "")
    gate = scenario.get("accuracy")
    if gate is None:
        if not legacy:
            raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                             "must declare accuracy or accuracy_gate")
        return {
            "reference": "probe_legacy",
            "description": legacy,
            "checks": [],
            "require_metrics": [],
            "structured": False,
        }
    if not isinstance(gate, dict):
        raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                         "accuracy must be an object")
    reference = gate.get("reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                         "accuracy.reference must be a non-empty string")
    reference_id = gate.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                         "accuracy.reference_id must be a non-empty string")
    checks = gate.get("checks", [])
    required = gate.get("require_metrics", [])
    if not isinstance(checks, list) or not isinstance(required, list):
        raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                         "accuracy checks/require_metrics must be lists")
    allowed_ops = {"<", "<=", ">", ">=", "==", "!=", "finite"}
    normalized_checks: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                             "accuracy check must be an object")
        metric = check.get("metric")
        operator = check.get("op")
        if (not isinstance(metric, str) or not metric or
                operator not in allowed_ops):
            raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                             "accuracy check needs metric and valid op")
        if operator != "finite" and not isinstance(
                check.get("value"), (int, float)):
            raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                             f"accuracy check {metric} needs numeric value")
        normalized = {"metric": metric, "op": operator}
        if operator != "finite":
            normalized["value"] = float(check["value"])
        if check.get("label"):
            normalized["label"] = str(check["label"])
        candidate_only = check.get("candidate_only", False)
        if not isinstance(candidate_only, bool):
            raise ValueError(
                f"scenario {scenario.get('id', '<unknown>')} accuracy "
                f"check {metric} candidate_only must be boolean")
        if candidate_only:
            normalized["candidate_only"] = True
        backends = check.get("backends")
        if backends is not None:
            if (not isinstance(backends, list) or not backends or
                    any(not isinstance(item, str) or not item
                        for item in backends)):
                raise ValueError(
                    f"scenario {scenario.get('id', '<unknown>')} accuracy "
                    f"check {metric} backends must be non-empty names")
            normalized["backends"] = list(backends)
        normalized_checks.append(normalized)
    if any(not isinstance(item, str) or not item for item in required):
        raise ValueError(f"scenario {scenario.get('id', '<unknown>')} "
                         "accuracy.require_metrics must contain names")
    if not normalized_checks and not required and reference != "probe_invariant":
        raise ValueError(
            f"scenario {scenario.get('id', '<unknown>')} accuracy must "
            "declare checks or require_metrics; only probe_invariant may "
            "delegate non-numeric invariants to a failing probe exit code")
    normalized_gate = dict(gate)
    normalized_gate["checks"] = normalized_checks
    normalized_gate["require_metrics"] = list(required)
    normalized_gate["structured"] = True
    return normalized_gate


def resolve_metric(metrics: dict[str, float], name: str) -> float | None:
    """Resolve an exact or unique leaf metric path from a probe report."""
    if name in metrics:
        return metrics[name]
    suffix = "/" + name
    matches = [value for key, value in metrics.items()
               if key.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def evaluate_accuracy_gate(gate: dict[str, Any],
                           metrics: dict[str, float],
                           backend: str | None = None,
                           baseline: bool = False) -> list[str]:
    """Evaluate structured checks applicable to the selected backend.

    ``candidate_only`` checks describe a newly introduced candidate contract.
    A pre-change baseline retains the metric for diagnosis but is not required
    to satisfy a capability it predates.
    """
    failures: list[str] = []
    for name in gate.get("require_metrics", []):
        value = resolve_metric(metrics, name)
        if value is None or not isinstance(value, (int, float)) or not \
                math.isfinite(float(value)):
            failures.append(f"required accuracy metric missing/non-finite: {name}")
    for check in gate.get("checks", []):
        if baseline and check.get("candidate_only", False):
            continue
        check_backends = check.get("backends")
        if (backend is not None and check_backends is not None and
                backend not in check_backends):
            continue
        name = check["metric"]
        value = resolve_metric(metrics, name)
        if value is None or not math.isfinite(float(value)):
            failures.append(f"accuracy metric missing/non-finite: {name}")
            continue
        operator = check["op"]
        expected = check.get("value")
        passed = {
            "<": value < expected,
            "<=": value <= expected,
            ">": value > expected,
            ">=": value >= expected,
            "==": value == expected,
            "!=": value != expected,
            "finite": True,
        }[operator]
        if not passed:
            label = check.get("label", name)
            failures.append(f"accuracy gate failed: {label} {value:.8g} "
                            f"{operator} {expected}")
    return failures


def release_accuracy_contract_failures(specs: list[RunSpec]) -> list[str]:
    """Reject prose-only accuracy claims from release evidence.

    Legacy probes may still enforce useful assertions internally, so they
    remain available to the developer profile.  A release row, however, must
    expose a structured reference identity and either a machine-evaluated
    metric contract or an explicit probe-owned invariant; otherwise the
    runner cannot prove what was gated. Probe invariants establish structural
    correctness only, not upstream semantic accuracy.
    """
    return [
        f"{spec.key}: release profile requires structured accuracy; "
        "migrate accuracy_gate prose to accuracy.reference/reference_id and "
        "checks/require_metrics or an explicit probe_invariant"
        for spec in specs if not spec.accuracy.get("structured", False)
    ]


def release_manifest_accuracy_contract_failures(
        manifest: dict[str, Any], args: argparse.Namespace,
        selected: set[str], catalog: list[ModelAsset]) -> list[str]:
    """Fail release accuracy metadata before downloading large model files.

    The catalog has already been narrowed by tier and ``--models``. Match its
    relative paths back to the owning scenarios, excluding required dependency
    globs so a shared dependency does not select unrelated bundle rows.
    ``release_accuracy_contract_failures`` remains as a post-expansion defense.
    """
    model_filter = [
        item.strip() for item in (getattr(args, "models", "") or "").split(",")
        if item.strip()]
    full = bool(getattr(args, "full", False)) or bool(model_filter)
    tiered = tiered_tasks(manifest)
    catalog_paths = [str(asset.relative_path) for asset in catalog]
    failures: list[str] = []
    for scenario in manifest["scenarios"]:
        task = scenario["task"]
        if task not in selected or not scenario_runs_in_tier(
                scenario, task, tiered, full):
            continue
        if model_filter and not model_id_matches(scenario["id"], model_filter):
            keys = (("model_glob",) if scenario.get("for_each_model") else
                    ("owned_globs", "covered_globs", "model_glob"))
            patterns: list[str] = []
            for key in keys:
                values = scenario.get(key, [])
                if isinstance(values, str):
                    values = [values]
                patterns.extend(values)
            if not any(fnmatch.fnmatchcase(path, pattern)
                       for path in catalog_paths for pattern in patterns):
                continue
        gate = normalize_accuracy_gate(scenario)
        if not gate.get("structured", False):
            failures.append(
                f"{task}/{scenario['id']}: release profile requires "
                "structured accuracy; migrate accuracy_gate prose to "
                "accuracy.reference/reference_id and checks/require_metrics "
                "or an explicit probe_invariant")
    return failures


def find_binary(build: Path, name: str) -> Path:
    suffixes = (".exe", "") if os.name == "nt" else ("",)
    # bin/plugin_test holds plugin-level unit tests (for example the
    # qRelateAnything temporal tracker tests) that manifest rows may consume.
    for directory in (build / "bin" / "aicore_tests", build / "bin",
                      build / "bin" / "plugin_test"):
        for suffix in suffixes:
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    return (build / "bin" / "aicore_tests" / name).resolve()


def model_quant(path: Path) -> str:
    match = re.search(r"-(f32|f16|q8_0|q8|q4_0|q4_k)\.gguf$", path.name,
                      re.IGNORECASE)
    return match.group(1).lower() if match else "f16"


def resolve_token(text: str, values: dict[str, str], build: Path,
                  references: dict[str, str] | None = None) -> str:
    """Expand ``{binary:X}``, ``{reference:NAME}`` and plain ``{key}`` tokens.

    An unconfigured reference expands to the empty string; the consuming
    probe must exit 77 on it so the row is reported as skipped (a gate
    failure unless --allow-incomplete), never silently passed.
    """
    def binary_replace(match: re.Match[str]) -> str:
        return str(find_binary(build, match.group(1)))

    def reference_replace(match: re.Match[str]) -> str:
        return (references or {}).get(match.group(1), "")

    text = re.sub(r"\{binary:([^}]+)\}", binary_replace, text)
    text = re.sub(r"\{reference:([^}]+)\}", reference_replace, text)
    return text.format_map(values)


def parse_references(items: list[str]) -> dict[str, str]:
    """Parse repeated ``--reference NAME=PATH`` into an absolute-path map."""
    references: dict[str, str] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--reference expects NAME=PATH, got {item!r}")
        # A venv's python is often a symlink to the base interpreter. Resolving
        # it changes sys.prefix and silently discards the selected packages.
        references[name] = str(Path(path).expanduser().absolute())
    return references


def existing_files(assets: Path, pattern: str) -> list[Path]:
    return sorted(path.resolve() for path in assets.glob(pattern) if path.is_file())


def model_id_matches(model_id: str, patterns: list[str]) -> bool:
    """True when a spec's model id (model file name, or the scenario id of a
    bundle scenario) matches one of the --models patterns. The .gguf suffix
    is optional in the pattern.
    """
    stem = (model_id[:-len(".gguf")] if model_id.lower().endswith(".gguf")
            else model_id)
    return any(fnmatch.fnmatchcase(model_id, pattern)
               or fnmatch.fnmatchcase(stem, pattern)
               for pattern in patterns)


def model_selection_globs(manifest: dict[str, Any], selected: set[str],
                          patterns: list[str]) -> set[str]:
    """Relative-path globs whose catalog rows an explicit --models run needs.

    Bundle (non for-each-model) scenarios run whole when their id matches a
    pattern, so their covered globs join the selection; required_globs of
    every selected-task scenario are small shared dependencies and always
    join.
    """
    globs: set[str] = set()
    for scenario in manifest["scenarios"]:
        if scenario["task"] not in selected:
            continue
        globs.update(scenario.get("required_globs", []))
        if not scenario.get("for_each_model") and model_id_matches(
                scenario["id"], patterns):
            globs.update(scenario.get("covered_globs", []))
    return globs


def expand_specs(manifest: dict[str, Any], args: argparse.Namespace,
                 report_dir: Path) -> tuple[list[RunSpec], dict[str, list[str]]]:
    selected = selected_tasks(manifest, args.tasks)
    model_filter = [
        item.strip() for item in (getattr(args, "models", "") or "").split(",")
        if item.strip()]
    # An explicit model selection declares its own scope, so it bypasses the
    # cost-motivated light tier; the tier keeps guarding unscoped runs.
    # Programmatic callers that predate the CLI tier flag are treated as full
    # matrix requests. The command-line parser always supplies ``full`` and
    # therefore retains the explicit light-tier default.
    full = bool(getattr(args, "full", True)) or bool(model_filter)
    tiered = tiered_tasks(manifest)

    inputs = {
        "image": args.image,
        "image2": args.image2,
        "face_image": args.face_image,
        "sam_image": args.sam_image,
        "yolo_image": args.yolo_image,
    }
    specs: list[RunSpec] = []
    diagnostics: dict[str, list[str]] = {
        "missing_assets": [], "uncovered_assets": [],
        "local_only_scenarios": [], "skipped_model_assets": [],
        "skipped_scenarios": [],
    }
    covered: dict[str, set[Path]] = {task: set() for task in selected}
    owned: dict[str, set[Path]] = {task: set() for task in selected}
    # Model ids that matched the explicit --models selection, including ones
    # consumed inside bundle scenarios (whose spec id is the scenario id).
    filter_matched_ids: set[str] = set()

    for raw in manifest["scenarios"]:
        task = raw["task"]
        if task not in selected:
            continue
        if not scenario_runs_in_tier(raw, task, tiered, full):
            # Light tier: the task declared a lightweight subset and this
            # scenario is not part of it; the complete matrix needs --full.
            continue
        bundle_model_filter_active = False
        if model_filter and not raw.get("for_each_model"):
            if model_id_matches(raw["id"], model_filter):
                pass  # whole bundle scenario selected by id
            else:
                # Partial selection: the bundle runs, but only the covered
                # models matching the patterns (handed to the probe via the
                # {scenario_model_globs} token).
                covered_matches = [
                    path for pattern in raw.get("covered_globs", [])
                    for path in existing_files(args.assets, pattern)
                    if model_id_matches(path.name, model_filter)]
                if not covered_matches:
                    continue
                bundle_model_filter_active = True
        light_globs = raw.get("light_globs") or []
        light_active = not full and task in tiered and light_globs
        scenario_owned: set[Path] = set()
        if not model_filter:
            # "Every catalog model has a consumer" is a full-matrix contract;
            # an explicit --models selection narrows the audit to the user's
            # list, exactly like the tier narrows it.
            owned_patterns = light_globs if light_active else raw.get(
                "owned_globs",
                raw.get("covered_globs", [raw.get("model_glob", "")]))
            for pattern in filter(None, owned_patterns):
                scenario_owned.update(
                    path for path in existing_files(args.assets, pattern)
                    if not scenario_glob_excluded(path, raw))
        owned[task].update(scenario_owned)

        model_patterns = light_globs if light_active else (
            [raw["model_glob"]] if isinstance(raw.get("model_glob"), str)
            else list(raw.get("model_glob") or []))
        models: list[Path] = []
        for pattern in model_patterns:
            models.extend(existing_files(args.assets, pattern))
        models = [model for model in models
                  if not scenario_glob_excluded(model, raw)]
        if model_filter and raw.get("for_each_model"):
            models = [model for model in models
                      if model_id_matches(model.name, model_filter)]
        if raw.get("for_each_model") and not models:
            if not model_filter:
                detail = ", ".join(model_patterns) if model_patterns \
                    else "(no model glob)"
                diagnostics["missing_assets"].append(f"{raw['id']}: {detail}")
            continue
        required_models: set[Path] = set()
        for required in raw.get("required_globs", []):
            matches = existing_files(args.assets, required)
            if not matches:
                diagnostics["missing_assets"].append(f"{raw['id']}: {required}")
            covered[task].update(matches)
            required_models.update(matches)

        scenario_covered: set[Path] = set(required_models)
        local_only_missing: list[str] = []
        covered_source = light_globs if light_active else raw.get(
            "covered_globs", [])
        if not raw.get("for_each_model"):
            for pattern in covered_source:
                matches = [
                    path for path in existing_files(args.assets, pattern)
                    if not scenario_glob_excluded(path, raw)]
                if not matches:
                    if raw.get("local_only"):
                        local_only_missing.append(pattern)
                    elif not model_filter:
                        diagnostics["missing_assets"].append(
                            f"{raw['id']}: {pattern}")
                scenario_covered.update(matches)
            if bundle_model_filter_active:
                scenario_covered = required_models | {
                    path for path in scenario_covered
                    if model_id_matches(path.name, model_filter)}
                filter_matched_ids.update(
                    path.name for path in scenario_covered
                    if model_id_matches(path.name, model_filter))
            covered[task].update(scenario_covered)
        if local_only_missing:
            diagnostics["local_only_scenarios"].append(
                f"{raw['id']}: not installed ({', '.join(local_only_missing)})")
            continue

        rows: list[Path | None] = models if raw.get("for_each_model") else [None]
        for model in rows:
            model_id = model.name if model else raw["id"]
            scenario_report = (report_dir / "probes" / task / raw["id"] /
                               f"{task}-{model_id}.json")
            values = {
                "repo": str(args.repo),
                "build": str(args.build),
                "assets": str(args.assets),
                "backend": args.backend,
                "model": str(model) if model else "",
                "model_quant": model_quant(model) if model else "",
                "image": inputs["image"],
                "image2": inputs["image2"],
                "face_image": inputs["face_image"],
                "sam_image": inputs["sam_image"],
                "yolo_image": inputs["yolo_image"],
                "threads": str(args.threads),
                "warmup_runs": str(args.warmup_runs),
                "inference_runs": str(args.inference_runs),
                "gpu_idle_timeout": str(args.gpu_idle_timeout),
                "profile": getattr(args, "profile", "developer"),
                "trellis_steps": str(args.trellis_steps),
                "scenario_model_globs": (
                    ",".join(sorted(
                        path.name for path in scenario_covered
                        if model_id_matches(path.name, model_filter)))
                    if bundle_model_filter_active else
                    ",".join(sorted({
                        pattern.rsplit("/", 1)[-1]
                        for pattern in (light_globs if light_active
                                        else covered_source or model_patterns)
                    }))),
                "scenario_report": str(scenario_report),
            }
            executable = raw["executable"]
            references = getattr(args, "references", {})
            if executable == "python":
                command = [sys.executable]
            elif executable.startswith("binary:"):
                command = [str(find_binary(args.build, executable.split(":", 1)[1]))]
            else:
                command = [resolve_token(executable, values, args.build,
                                         references)]
            command.extend(resolve_token(item, values, args.build, references)
                           for item in raw.get("args", []))
            env = {key: resolve_token(value, values, args.build, references)
                   for key, value in raw.get("env", {}).items()}
            model_paths = tuple(sorted(({model} if model else set()) |
                                       required_models |
                                       (scenario_covered if model is None else set())))
            if model:
                covered[task].add(model)
            vram_estimate = raw.get("vram_estimate_mib")
            vram_overhead = raw.get("vram_overhead_mib")
            specs.append(RunSpec(
                scenario_id=raw["id"], task=task, model_id=model_id,
                model_paths=model_paths, command=tuple(command), env=env,
                accuracy_gate=raw["accuracy_gate"],
                metric_parser=raw.get("metric_parser", "generic"),
                fingerprint_policy=raw.get("fingerprint_policy",
                                           "exact"),
                require_fingerprint=raw.get("require_fingerprint", True),
                report_path=scenario_report,
                accuracy=normalize_accuracy_gate(raw),
                backend=args.backend,
                vram_estimate_mib=(float(vram_estimate)
                                   if vram_estimate is not None else None),
                vram_overhead_mib=(float(vram_overhead)
                                   if vram_overhead is not None else None),
            ))

    if model_filter:
        matched = {spec.model_id for spec in specs} | filter_matched_ids
        unmatched = [pattern for pattern in model_filter
                     if not any(model_id_matches(model_id, [pattern])
                                for model_id in matched)]
        if unmatched:
            raise ValueError(
                "--models matched nothing: " + ", ".join(unmatched))
    for task in selected:
        for path in sorted(owned[task] - covered[task]):
            diagnostics["uncovered_assets"].append(
                f"{task}: {path.relative_to(args.assets)}")
    return specs, diagnostics


def flatten_json(value: Any, prefix: str = "",
                 gate_metrics: frozenset[str] = frozenset()
                 ) -> tuple[dict[str, float], dict[str, str]]:
    # gate_metrics: leaf names the scenario's own structured accuracy gate
    # reads; they are metrics by definition, whatever the whitelist below.
    metrics: dict[str, float] = {}
    fingerprints: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            child_metrics, child_fingerprints = flatten_json(
                child, child_prefix, gate_metrics)
            metrics.update(child_metrics)
            fingerprints.update(child_fingerprints)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_metrics, child_fingerprints = flatten_json(
                child, f"{prefix}/{index}", gate_metrics)
            metrics.update(child_metrics)
            fingerprints.update(child_fingerprints)
    elif isinstance(value, bool):
        # Only explicit accuracy contracts consume booleans; informational flags are not timing/performance metrics.
        if prefix.rsplit("/", 1)[-1].lower() in gate_metrics:
            metrics[prefix] = float(value)
    elif isinstance(value, (int, float)):
        leaf = prefix.rsplit("/", 1)[-1].lower()
        path_parts = [part.lower() for part in prefix.split("/")]
        is_timing = any(part.endswith(TIMING_SUFFIXES) for part in path_parts)
        # Raw measured sequences may contain zero; scalar timing retains the original nonzero-sample rule.
        is_raw_sample = any(part.endswith("_samples_ms") for part in path_parts)
        if (is_raw_sample and float(value) >= 0.0) or (is_timing and float(value) > 0.0) or leaf in gate_metrics or leaf in {
                "mask_iou", "box_error", "score_error", "worst_rel_mae",
                "kpt_median_px", "descriptor_cosine_median",
                "max_diff", "mean_diff", "accuracy_max_diff", "upstream_cpu_max_diff", "probability_max_diff",
                "compatibility_accuracy_max_diff", "cpu_arithmetic_probability_max_diff",
                "cpu_arithmetic_probability_mean_diff",
                "probability_mean_diff", "argmax_match", "cross_f1",
                # OCR original CUDA truth and qualification must survive the
                # common report, including zero error and unqualified timing.
                "source_gpu_max_diff", "source_gpu_mean_diff",
                "source_gpu_argmax_match", "source_performance_qualified",
                "source_performance_pass", "source_gpu_audit_pass",
                "gpu_audit_pass", "gpu_sampled_process_peak_mib",
                "source_gpu_sampled_process_peak_mib",
                "cross_precision", "cross_recall", "upstream_cross_f1",
                "latency_over_upstream", "pytorch_probability_mae",
                "pytorch_probability_max_error", "pytorch_probability_error_excess",
                "pytorch_cross_f1_drop", "pytorch_annotation_error_excess",
                "upstream_pytorch_cross_f1", "upstream_pytorch_probability_mae",
                "upstream_pytorch_probability_max_error", "upstream_logit_max_error",
                "upstream_probability_max_error", "upstream_logit_mae",
                "compatibility_upstream_cross_f1", "compatibility_upstream_logit_max_error",
                "compatibility_upstream_probability_max_error",
                "compatibility_pytorch_probability_error_excess", "compatibility_pytorch_cross_f1_drop",
                "compatibility_pytorch_annotation_error_excess",
                "arithmetic_probability_max_error", "arithmetic_probability_mae",
                "upstream_probability_mae", "pipeline_images_per_second",
                "upstream_images_per_second", "weighted_latency_over_upstream", "map", "cp", "cr",
                "precision", "recall", "f1", "macro_precision", "macro_recall",
                "upstream_map_delta", "upstream_cp_delta", "upstream_cr_delta",
                "pytorch_map_delta", "pytorch_cp_delta", "pytorch_cr_delta",
                "caption_exact", "caption_word_f1", "pytorch_caption_exact",
                "pytorch_caption_word_f1",
                # lingbot upstream-parity probe (test_lingbot_stream_parity)
                "pose_t_rmse", "pose_quat_maxdiff", "intr_maxdiff",
                "depth_pool_rel_rmse", "conf_mean_maxdiff",
                "max_diff", "mean_diff", "argmax_match",
                # map_anything upstream-frames row: how many typed outputs
                # were actually compared (a 0-file pass is not evidence)
                "raw_files_compared"}:
            metrics[prefix] = float(value)
    elif isinstance(value, str):
        leaf = prefix.rsplit("/", 1)[-1].lower()
        if any(token in leaf for token in HASH_KEYS):
            fingerprints[prefix] = value
    return metrics, fingerprints


def gate_metric_names(gate: dict[str, Any] | None) -> frozenset[str]:
    """Lower-cased leaf names a structured accuracy gate reads."""
    gate = gate or {}
    names = set(gate.get("require_metrics", []))
    names.update(check["metric"] for check in gate.get("checks", []))
    return frozenset(name.rsplit("/", 1)[-1].lower() for name in names)


def parse_output(output: str, parser: str, report_path: Path,
                 gate_metrics: frozenset[str] = frozenset()
                 ) -> tuple[dict[str, float], dict[str, str]]:
    metrics: dict[str, float] = {}
    fingerprints: dict[str, str] = {}
    json_values: list[Any] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            json_values.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    if report_path.is_file():
        try:
            json_values.append(json.loads(report_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    for index, value in enumerate(json_values):
        parsed_metrics, parsed_fingerprints = flatten_json(
            value, f"json{index}", gate_metrics)
        metrics.update(parsed_metrics)
        fingerprints.update(parsed_fingerprints)

    if parser == "relation":
        # The relation probe keeps per-input diagnostics and exposes the worst
        # input for each gate in one explicit summary. Avoid resolving an
        # ambiguous leaf from several samples or silently dropping zero error.
        for value in json_values:
            if not isinstance(value, dict):
                continue
            for name, number in value.get("metrics", {}).items():
                if isinstance(number, (int, float)) and not isinstance(number, bool):
                    metrics[name] = float(number)
            if isinstance(value.get("fingerprint"), str):
                fingerprints["relation_output"] = value["fingerprint"]
    elif parser == "rfdetr":
        for name, number in re.findall(
                r"^\s*(forward \(A\+topK\+B\)|detect-total|detect-plugin-path)"
                r"\s*:\s*([0-9.]+)", output, re.MULTILINE):
            metrics[name.replace(" ", "_").replace("(", "").replace(")", "")
                    + "_ms"] = float(number)
    elif parser == "rmbg":
        match = re.search(r"median_ms=([0-9.]+)\s+p95_ms=([0-9.]+)\s+"
                          r"output_hash=([0-9]+)", output)
        if match:
            metrics["inference_p50_ms"] = float(match.group(1))
            metrics["inference_p95_ms"] = float(match.group(2))
            fingerprints["output_hash"] = match.group(3)
    elif parser == "trellis":
        for value in json_values:
            if isinstance(value, dict) and "geometry_sha12" in value:
                fingerprints[f"geometry/{value.get('device', 'unknown')}"] = str(
                    value["geometry_sha12"])
    elif parser == "sam3d":
        for value in json_values:
            if isinstance(value, dict) and "geometry_sha12" in value:
                fingerprints[f"geometry/{value.get('device', 'unknown')}"] = str(
                    value["geometry_sha12"])
                if value.get("finite") is not True or value.get("stable") is not True:
                    fingerprints["degenerate"] = "true"
    elif parser == "yolo":
        for value in json_values:
            if not isinstance(value, dict):
                continue
            measurements = value.get("measurements", {})
            if not isinstance(measurements, dict):
                continue
            for backend, rows in measurements.items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict) or "sanity" not in row:
                        continue
                    key = f"sanity/{backend}/{row.get('file', 'unknown')}"
                    fingerprints[key] = str(row["sanity"])
    match = re.search(r"match=([0-9.]+)ms", output)
    if match:
        metrics["match_ms"] = float(match.group(1))
    return metrics, fingerprints


def sweep_task_probe_outputs(task: str, report_root: Path) -> int:
    """Delete a finished task's per-probe output files.

    Every metric and fingerprint a probe reported is folded into the
    consolidated rows before this runs, so the raw probe JSONs under
    probes/ and baseline-probes/ are dead weight; the glob also removes
    stale files left by earlier runs of the same task. Model caches under
    the asset root are never touched: re-downloading them costs far more
    than the space they save. ``{task}-*.d`` directories are probe-private
    scratch trees named from ``{scenario_report}`` (e.g. the upstream
    parity row's retained failure tensors).
    """
    removed = 0
    for directory in (report_root / "probes",
                      report_root / "baseline-probes"):
        owned = directory / task
        if owned.is_dir():
            reports = len(list(owned.rglob(f"{task}-*.json")))
            try:
                # Task-owned paths contain the reports and their raw sidecars.
                shutil.rmtree(owned)
                removed += reports
            except OSError:
                pass
        # Also clean reports retained by the older flat/scenario-only layout.
        for path in directory.rglob(f"{task}-*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        for path in directory.glob(f"{task}-*.d"):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
    return removed


def _cache_model_relative(path: Path, assets_root: Path) -> PurePosixPath | None:
    """Relative destination of a cached model file, or None if unsafe.

    Mirrors the catalog's destination contract: exactly
    <folder ending in _models>/<name>.gguf, inside the assets root. This is
    the only shape --clean-model-cache may ever delete.
    """
    try:
        relative = path.resolve().relative_to(assets_root.resolve())
    except (OSError, ValueError):
        return None
    if (len(relative.parts) != 2 or
            not relative.parts[0].endswith("_models") or
            relative.suffix.lower() != ".gguf"):
        return None
    return relative


def prune_consumed_model_cache(task: str, model_paths,
                               holders_by_path: dict[Path, set[str]],
                               assets_root: Path) -> list[str]:
    """Delete consumed model files whose last referencing task just finished.

    holders_by_path maps every model path consumed by this run to the set of
    tasks whose specs reference it; finishing a task releases its claims, and
    a file is unlinked only when the last claim is gone (models shared across
    tasks, or across scenarios of one task, survive until then). Only files
    matching the pinned catalog layout inside the assets root are eligible;
    everything else is reported and left in place.
    """
    removed: list[str] = []
    for path in sorted(set(model_paths)):
        holders = holders_by_path.get(path)
        if holders is None:
            continue
        holders.discard(task)
        if holders:
            continue
        del holders_by_path[path]
        relative = _cache_model_relative(path, assets_root)
        if relative is None:
            print(f"model-cache cleanup skipped non-conforming path: {path}",
                  file=sys.stderr, flush=True)
            continue
        try:
            path.unlink()
        except OSError as error:
            print(f"model-cache cleanup could not remove {relative}: {error}",
                  file=sys.stderr, flush=True)
            continue
        removed.append(str(relative))
    return removed


def release_finished_task_outputs(
        task: str, specs_left_per_task: dict[str, int],
        consumed_by_task: dict[str, set[Path]],
        holders_by_path: dict[Path, set[str]],
        args: argparse.Namespace) -> tuple[int, list[str]]:
    """Finalize a finished spec's task once its last spec is summarized.

    Both cleanups are opt-in and off by default: --clean-probe-outputs sweeps
    the task's raw probe outputs (already folded into the report rows), and
    --clean-model-cache prunes the model-cache files this run consumed for
    the task. Model caches default to kept because re-downloading costs far
    more than the space; the prune exists for space-constrained hosts such
    as CI runners.
    """
    remaining = specs_left_per_task.get(task)
    if remaining is None:
        return 0, []
    specs_left_per_task[task] = remaining - 1
    if remaining > 1:
        return 0, []
    probe_removed = (sweep_task_probe_outputs(task, args.output.parent)
                     if args.clean_probe_outputs else 0)
    model_removed = (prune_consumed_model_cache(
                         task, consumed_by_task.get(task, ()),
                         holders_by_path, args.assets)
                     if args.clean_model_cache else [])
    return probe_removed, model_removed


def runtime_env(spec: RunSpec, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(spec.env)
    add_runtime_library_path(env, args.build)
    return env


def configured_qt_library_dir(build: Path) -> Path | None:
    """Recover the configured Qt runtime directory for direct CLI runs.

    CTest injects AICORE_QT_LIBRARY_DIR, but users also invoke this script
    directly. Reading CMakeCache keeps the spawned catalog/probe binaries on
    the same Qt major/minor build instead of mixing the host Qt with the one
    linked into AICore.
    """
    cache = build / "CMakeCache.txt"
    try:
        lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    cmake_dirs: list[Path] = []
    prefixes: list[Path] = []
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        variable = key.split(":", 1)[0]
        if variable in ("Qt5Core_DIR", "Qt6Core_DIR") and value:
            cmake_dirs.append(Path(value))
        elif variable == "CMAKE_PREFIX_PATH":
            prefixes.extend(Path(item) for item in value.split(";") if item)
    candidates: list[Path] = []
    for directory in cmake_dirs:
        if len(directory.parents) >= 2:
            qt_lib = directory.parents[1]
            candidates.extend(([qt_lib.parent / "bin", qt_lib]
                               if os.name == "nt" else [qt_lib]))
    for prefix in prefixes:
        candidates.append(prefix / ("bin" if os.name == "nt" else "lib"))
    return next((path.resolve() for path in candidates if path.is_dir()), None)


def _elf_runtime_dirs(binary: Path) -> list[Path]:
    """Extract DT_RUNPATH/DT_RPATH entries from an ELF binary (Linux)."""
    readelf = shutil.which("readelf")
    if not readelf:
        return []
    try:
        result = subprocess.run([readelf, "-d", str(binary)],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    dirs: list[Path] = []
    for line in result.stdout.splitlines():
        if "RPATH" not in line and "RUNPATH" not in line:
            continue
        match = re.search(r"\[(.*)\]", line)
        if not match:
            continue
        for entry in match.group(1).split(":"):
            entry = entry.replace("$ORIGIN", str(binary.parent))
            entry = entry.replace("${ORIGIN}", str(binary.parent))
            if entry:
                dirs.append(Path(entry))
    return dirs


def _macho_runtime_dirs(binary: Path) -> list[Path]:
    """Extract LC_RPATH entries from a Mach-O binary (macOS)."""
    otool = shutil.which("otool")
    if not otool:
        return []
    try:
        result = subprocess.run([otool, "-l", str(binary)],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    dirs: list[Path] = []
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH" and i + 1 < len(lines):
            match = re.search(r"path\s+(.+)\s*\(offset\s+\d+\)", lines[i + 1])
            if match:
                entry = match.group(1).replace("@loader_path", str(binary.parent))
                if entry:
                    dirs.append(Path(entry))
    return dirs


def _qt_core_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("Qt6Core.dll", "Qt5Core.dll")
    if sys.platform == "darwin":
        return ("libQt6Core.6.dylib", "libQt5Core.5.dylib", "libQt6Core.dylib",
                "libQt5Core.dylib")
    return ("libQt6Core.so.6", "libQt5Core.so.5")


def embedded_qt_library_dir(build: Path) -> Path | None:
    """Resolve the Qt runtime directory from the binaries' embedded paths.

    The loader resolves a shared library's own DT_NEEDED entries with that
    library's DT_RUNPATH, not the executable's, so fronting only the probe
    path still lets libAICore's Qt dependency fall through to a host Qt of a
    different minor version (Qt_5.15 symbol-version failure). The build
    bakes the absolute Qt runtime directory into every linked binary's
    RUNPATH/LC_RPATH, so reading it back from disk is machine-independent
    for in-place builds and needs no hardcoded path anywhere.
    """
    core_names = _qt_core_names()
    bin_dir = build / "bin"
    candidates: list[Path] = []
    if os.name == "nt":
        # PE has no runpath; Qt runtimes are deployed next to the binaries.
        candidates.append(bin_dir)
        tests_dir = bin_dir / "aicore_tests"
        if tests_dir.is_dir():
            candidates.append(tests_dir)
    else:
        binaries = [bin_dir / ("libAICore.dylib" if sys.platform == "darwin"
                               else "libAICore.so")]
        tests_dir = bin_dir / "aicore_tests"
        if tests_dir.is_dir():
            for probe in sorted(tests_dir.iterdir()):
                if probe.is_file() and os.access(probe, os.X_OK):
                    binaries.append(probe)
                    break
        for binary in binaries:
            if not binary.is_file():
                continue
            dirs = (_macho_runtime_dirs(binary) if sys.platform == "darwin"
                    else _elf_runtime_dirs(binary))
            candidates.extend(dirs)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        if any((resolved / name).is_file() for name in core_names):
            return resolved
    return None


def add_runtime_library_path(env: dict[str, str], build: Path) -> None:
    if os.name == "nt":
        variable = "PATH"
    elif sys.platform == "darwin":
        variable = "DYLD_LIBRARY_PATH"
    else:
        variable = "LD_LIBRARY_PATH"
    parts = [str(build / "bin")]
    qt_library_dir = env.get("AICORE_QT_LIBRARY_DIR")
    if not qt_library_dir:
        configured_qt = configured_qt_library_dir(build)
        qt_library_dir = str(configured_qt) if configured_qt else ""
    if not qt_library_dir:
        # Fall back to the paths baked into the binaries themselves: the
        # CMakeCache only exists for in-place builds, while the RUNPATH
        # resolution also covers relocated build trees on the build machine.
        embedded_qt = embedded_qt_library_dir(build)
        qt_library_dir = str(embedded_qt) if embedded_qt else ""
    if qt_library_dir:
        parts.append(qt_library_dir)
    # Same rationale as the ctest AICORE_TEST_ENV_EXTRA plumbing: a dlopen'ed
    # libggml-cuda.so needs the toolkit-matching cudart directory FIRST on the
    # loader path, otherwise the CUDA backend silently disappears (NDEBUG
    # swallows the loader failure) and cuda-requested probes fail to load.
    cuda_library_dir = env.get("AICORE_CUDA_LIBRARY_DIR")
    if cuda_library_dir:
        parts.append(cuda_library_dir)
    if env.get(variable):
        parts.append(env[variable])
    env[variable] = os.pathsep.join(parts)


def run_attempt(spec: RunSpec, args: argparse.Namespace) -> Attempt:
    started = time.perf_counter()
    spec.report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        spec.report_path.unlink()
    except FileNotFoundError:
        pass
    try:
        completed = subprocess.run(
            spec.command, env=runtime_env(spec, args), cwd=args.repo,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.timeout, check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
    except OSError as error:
        # A missing or non-executable probe (for example a plugin test that
        # was not built) is a row failure, not a runner crash.
        output = f"cannot execute probe {spec.command[0]}: {error}"
        returncode = 127
    except subprocess.TimeoutExpired as error:
        raw = error.stdout or ""
        output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        output += f"\nTIMEOUT after {args.timeout}s"
        returncode = 124
    duration_ms = (time.perf_counter() - started) * 1000.0
    metrics, fingerprints = parse_output(output, spec.metric_parser,
                                         spec.report_path,
                                         gate_metric_names(spec.accuracy))
    metrics.setdefault("process_wall_ms", duration_ms)
    return Attempt(returncode=returncode, duration_ms=duration_ms,
                   metrics=metrics, fingerprints=fingerprints,
                   output_tail=output[-8000:])


def median_metrics(attempts: list[Attempt]) -> dict[str, float]:
    keys = set().union(*(attempt.metrics for attempt in attempts))
    return {key: statistics.median(
                attempt.metrics[key] for attempt in attempts if key in attempt.metrics)
            for key in sorted(keys)}


def stable_fingerprints(attempts: list[Attempt],
                        spec: "RunSpec | None" = None
                        ) -> tuple[dict[str, str], list[str]]:
    # "stability_only" / "none": GPU generators whose output passes through
    # a parallel extractor (e.g. trellis mesh hashmap insertion) have no
    # run-stable vertex ORDER, so byte-level hashes are not a valid
    # cross-attempt gate — probe return codes plus the probe's own finite/
    # degeneracy checks carry the stability signal instead.
    if spec is not None and spec.fingerprint_policy in ("stability_only",
                                                        "none"):
        return {}, []
    keys = set().union(*(attempt.fingerprints for attempt in attempts))
    stable: dict[str, str] = {}
    failures: list[str] = []
    for key in sorted(keys):
        values = [attempt.fingerprints.get(key) for attempt in attempts]
        if any(value is None for value in values) or len(set(values)) != 1:
            failures.append(f"unstable fingerprint {key}: {values}")
        else:
            stable[key] = str(values[0])
    return stable, failures


def extract_error_summary(output_tail: str, limit: int = 5) -> str:
    """Compact root-cause evidence taken from the probe's own output.

    The tail already contains the failure text (test FAIL lines, backend
    errors, tracebacks, SKIP reasons); this only surfaces the last distinct
    lines so the report explains *why* a row failed without opening the log.
    """
    markers = re.compile(
        r"\b(?:FAIL|failed|error|assert\w*|Traceback|TIMEOUT|SKIP|[A-Za-z]*Error)\b"
        r"|GPU overlap persisted|GPU (?:did not become idle|preflight timed out)", re.IGNORECASE)
    evidence: list[str] = []
    for raw in output_tail.splitlines():
        line = raw.strip()
        if not line or not markers.search(line):
            continue
        if len(line) > 220:
            line = line[:217] + "..."
        if line not in evidence:
            evidence.append(line)
    return " | ".join(evidence[-limit:])


def summarize_attempts(spec: RunSpec, attempts: list[Attempt],
                       allow_incomplete: bool,
                       baseline: bool = False) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    codes = [attempt.returncode for attempt in attempts]
    fingerprints, stability_failures = stable_fingerprints(attempts, spec)
    if (spec.fingerprint_policy not in ("stability_only", "none") and
            spec.require_fingerprint and
            all(code == 0 for code in codes) and
            not fingerprints):
        stability_failures.append("probe emitted no output fingerprint")
    status = "pass"
    if any(code == 77 for code in codes):
        status = "skipped"
    if any(code not in (0, 77) for code in codes):
        status = "fail"
    if stability_failures:
        status = "unstable"
    metrics = median_metrics(attempts)
    accuracy_failures: list[str] = []
    attempt_rows: list[dict[str, Any]] = []
    error_summaries: list[str] = []
    for index, attempt in enumerate(attempts, 1):
        # A median is useful for timing, but must never hide a single bad
        # inference or a missing required metric in a successful probe.
        checks = (evaluate_accuracy_gate(
            spec.accuracy, attempt.metrics, backend=spec.backend,
            baseline=baseline) if attempt.returncode == 0 else [])
        prefix = f"attempt {index}: " if len(attempts) > 1 else ""
        accuracy_failures.extend(prefix + message for message in checks)
        error = " | ".join(checks) or extract_error_summary(attempt.output_tail)
        if error and (checks or attempt.returncode != 0):
            error_summaries.append(prefix + error)
        attempt_rows.append({
            "index": index, "return_code": attempt.returncode,
            "duration_ms": attempt.duration_ms, "metrics": attempt.metrics,
            "accuracy_failures": checks, "error_summary": error,
        })
    if accuracy_failures and status in ("pass", "skipped"):
        status = "fail"
    if status != "pass" and not (allow_incomplete and status == "skipped"):
        failures.append(f"{spec.key}: {status}, return codes={codes}")
    failures.extend(f"{spec.key}: {message}" for message in stability_failures)
    failures.extend(f"{spec.key}: {message}" for message in accuracy_failures)
    assets = [file_identity(path) for path in spec.model_paths if path.is_file()]
    return {
        "key": spec.key,
        "task": spec.task,
        "scenario_id": spec.scenario_id,
        "model_id": spec.model_id,
        "model_assets": assets,
        "accuracy_gate": spec.accuracy_gate,
        "accuracy": spec.accuracy,
        "accuracy_failures": accuracy_failures,
        "backend": spec.backend,
        "fingerprint_policy": spec.fingerprint_policy,
        "status": status,
        "return_codes": codes,
        "attempts": attempt_rows,
        "metrics": metrics,
        "fingerprints": fingerprints,
        "output_tail": attempts[-1].output_tail if attempts else "",
        "error_summary": " | ".join(error_summaries or stability_failures),
    }, failures


def summarize_attempt_pair(
        spec: RunSpec, attempts: list[Attempt],
        baseline_spec: RunSpec | None, baseline_attempts: list[Attempt],
        allow_incomplete: bool
        ) -> tuple[dict[str, Any], list[str], dict[str, Any] | None, list[str]]:
    """Summarize both build owners before task cache cleanup can unlink assets."""
    row, failures = summarize_attempts(spec, attempts, allow_incomplete)
    baseline_row = None
    baseline_failures: list[str] = []
    if baseline_spec is not None:
        baseline_row, baseline_failures = summarize_attempts(
            baseline_spec, baseline_attempts, allow_incomplete, baseline=True)
    return row, failures, baseline_row, baseline_failures


def file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def cpu_backend_module_score(module: Path, build: Path) -> tuple[int | None, str | None]:
    """Read the same host-compatibility score used by ggml CPU dispatch."""
    script = (
        "import ctypes, sys; "
        "backend = ctypes.CDLL(sys.argv[1]); "
        "print(int(backend.ggml_backend_score()))"
    )
    env = os.environ.copy()
    path_key = ("PATH" if os.name == "nt" else
                "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else
                "LD_LIBRARY_PATH")
    library_path = os.pathsep.join(
        [str(build / "bin"), env.get(path_key, "")]).rstrip(os.pathsep)
    env[path_key] = library_path
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(module)],
            capture_output=True, text=True, timeout=15, env=env)
    except subprocess.TimeoutExpired:
        return None, "timed out loading CPU backend module"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        return None, detail
    try:
        return int(completed.stdout.strip()), None
    except ValueError:
        return None, f"invalid score output: {completed.stdout.strip()!r}"


def cpu_backend_profile(build: Path) -> dict[str, Any]:
    """Describe the CPU dispatch modules that can run GPU fallback ops."""
    build = build.resolve()
    module_names: set[str] = set()
    module_paths: dict[str, Path] = {}
    for directory in (build / "bin", build / "bin" / "aicore_tests"):
        if not directory.is_dir():
            continue
        for path in directory.glob("*ggml-cpu*"):
            if not path.is_file():
                continue
            name = path.name
            for suffix in (".so", ".dylib", ".dll"):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            if name.startswith("libggml-cpu-"):
                module_name = name.removeprefix("libggml-cpu-")
            elif name.startswith("ggml-cpu-"):
                module_name = name.removeprefix("ggml-cpu-")
            elif name in {"libggml-cpu", "ggml-cpu"}:
                module_name = "generic"
            else:
                continue
            module_names.add(module_name)
            module_paths.setdefault(module_name, path)

    module_scores: dict[str, int | None] = {}
    score_errors: dict[str, str] = {}
    for module_name, module_path in sorted(module_paths.items()):
        score, error = cpu_backend_module_score(module_path, build)
        module_scores[module_name] = score
        if error:
            score_errors[module_name] = error
    supported_scores = [score for score in module_scores.values()
                        if score is not None and score > 0]

    settings: dict[str, str] = {}
    caches = [build / "CMakeCache.txt"]
    # A staged validation build may contain only selected installed binaries;
    # its llama ExternalProject cache lives beside that staging directory.
    for parent in build.parents:
        external_cache = parent / "llama/src/ext_llama-build/CMakeCache.txt"
        if external_cache.is_file():
            caches.append(external_cache)
            break
    for cache in caches:
        if not cache.is_file():
            continue
        for line in cache.read_text(errors="replace").splitlines():
            if ":" not in line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            name = key.split(":", 1)[0]
            if name in {"AICore_CPU_ALL_VARIANTS", "GGML_CPU_ALL_VARIANTS"}:
                settings[name] = value.strip().upper()
    effective = {"ON" if value in {"ON", "TRUE", "1"} else "OFF"
                 for value in settings.values()
                 if value in {"ON", "OFF", "TRUE", "FALSE", "1", "0"}}
    return {
        "modules": sorted(module_names),
        "module_scores": module_scores,
        "module_score_errors": score_errors,
        "best_dispatch_score": max(supported_scores, default=None),
        "best_dispatch_modules": sorted(
            name for name, score in module_scores.items()
            if score is not None and score > 0 and
            score == max(supported_scores, default=None)),
        "all_variants_settings": settings,
        "all_variants_consistent": len(effective) <= 1,
    }


def cpu_backend_profile_failures(candidate: dict[str, Any],
                                 baseline: dict[str, Any]) -> list[str]:
    failures = []
    if not candidate.get("all_variants_consistent", True):
        failures.append("candidate has conflicting AICore/GGML CPU variant settings")
    if not baseline.get("all_variants_consistent", True):
        failures.append("baseline has conflicting AICore/GGML CPU variant settings")
    label = "AICore_CPU_ALL_VARIANTS/GGML_CPU_ALL_VARIANTS"
    def effective_values(profile: dict[str, Any]) -> set[str]:
        values = profile.get("all_variants_settings", {}).values()
        return {"ON" if value in {"ON", "TRUE", "1"} else "OFF"
                for value in values
                if value in {"ON", "OFF", "TRUE", "FALSE", "1", "0"}}

    left_values = effective_values(baseline)
    right_values = effective_values(candidate)
    if left_values and right_values and left_values != right_values:
        failures.append(f"CPU backend configuration mismatch {label}: "
                        f"{sorted(left_values)} -> {sorted(right_values)}")
    baseline_score = baseline.get("best_dispatch_score")
    candidate_score = candidate.get("best_dispatch_score")
    if baseline_score is not None or candidate_score is not None:
        if baseline_score is None or candidate_score is None:
            failures.append("CPU backend dispatch score unavailable: "
                            f"baseline={baseline_score}, candidate={candidate_score}")
        elif baseline_score != candidate_score:
            failures.append("CPU backend dispatch capability mismatch: "
                            f"baseline score {baseline_score}, "
                            f"candidate score {candidate_score}")
    else:
        baseline_modules = set(baseline.get("modules", []))
        candidate_modules = set(candidate.get("modules", []))
        if baseline_modules != candidate_modules:
            missing = sorted(baseline_modules - candidate_modules)
            extra = sorted(candidate_modules - baseline_modules)
            failures.append("CPU backend module mismatch without runtime scores: "
                            f"candidate missing {missing or 'none'}, "
                            f"candidate extra {extra or 'none'}")
    return failures


def execution_artifacts(specs: list[RunSpec], build: Path) -> dict[str, Any]:
    """Identify command files and build runtimes, without claiming a loader trace."""
    build = build.resolve()
    paths = {Path(__file__).resolve()}
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        paths.add(cache)
    binary_dir = build / "bin"
    if binary_dir.is_dir():
        for path in binary_dir.iterdir():
            name = path.name.lower()
            if (path.is_file() and name.startswith(("aicore", "libaicore", "ggml", "libggml"))
                    and (".so" in name or name.endswith((".dll", ".dylib")))):
                paths.add(path)
    for spec in specs:
        for index, argument in enumerate(spec.command):
            path = Path(argument)
            if not path.is_absolute():
                continue
            if path.is_file() and (index == 0 or path.suffix == ".py" or
                                   path.is_relative_to(binary_dir)):
                # A model cache/report in build/Testing is not executable code.
                if index == 0 or path.suffix != ".gguf":
                    paths.add(path)
    return {str(path): {**file_identity(path), "resolved_path": str(path.resolve())}
            for path in sorted(paths)}


def execution_artifact_failures(before: dict[str, Any], after: dict[str, Any],
                                label: str) -> list[str]:
    return [f"{label} execution artifact changed during validation: {path}"
            for path in sorted(before.keys() | after.keys())
            if before.get(path) != after.get(path)]


def vram_skipped_row(spec: RunSpec, gate: dict[str, float]) -> dict[str, Any]:
    """Synthesize a result row for a scenario skipped by the VRAM gate.

    Model assets stay listed so the coverage audit remains stable across
    hosts; the row carries the measured free VRAM and the estimate so the
    report shows exactly why the probe did not run.
    """
    reason = (f"skipped before launch: estimated VRAM need "
              f"{gate['needed_mib']:.0f} MiB exceeds "
              f"{gate['free_mib']:.0f} MiB free (OOM guard)")
    return {
        "key": spec.key,
        "task": spec.task,
        "scenario_id": spec.scenario_id,
        "model_id": spec.model_id,
        "model_assets": [file_identity(path) for path in spec.model_paths
                         if path.is_file()],
        "accuracy_gate": spec.accuracy_gate,
        "accuracy": spec.accuracy,
        "fingerprint_policy": spec.fingerprint_policy,
        "status": VRAM_SKIP_STATUS,
        "return_codes": [],
        "metrics": {"vram_needed_mib": gate["needed_mib"],
                    "vram_free_mib": gate["free_mib"]},
        "fingerprints": {},
        "output_tail": reason,
        "error_summary": reason,
    }


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def is_candidate_performance_metric(key: str, backend: str) -> bool:
    """Compare candidate work, retaining reference/audit data as diagnostics.

    Source-framework runs and per-PID device samples are not measurements of
    the candidate runtime. Their own qualification gates remain mandatory.
    The summarized candidate process peak and native timings are comparable.
    """
    parts = key.lower().split("/")
    if any(part in {"audit", "sources"} for part in parts):
        return False
    if any(part.startswith("source_") for part in parts):
        return False
    # Sequences support tail-latency attribution; index pairs neither align run conditions nor constitute the original p50/p95 gate.
    if any(part.endswith("_samples_ms") for part in parts):
        return False
    return (any(part.endswith(TIMING_SUFFIXES) for part in parts)
            and not (backend != "cpu" and "cpu" in parts))


def compare_report(current: dict[str, Any], baseline: dict[str, Any],
                   threshold: float, absolute_noise_floor: float,
                   allow_incomplete: bool = False) -> list[str]:
    failures: list[str] = []
    for key in ("backend", "profile", "repeats", "warmup_runs", "inference_runs",
                "threads", "trellis_steps", "tier"):
        baseline_value = baseline.get(key, "developer" if key == "profile"
                                      else None)
        current_value = current.get(key, "developer" if key == "profile"
                                    else None)
        if baseline_value != current_value:
            failures.append(
                f"validation protocol mismatch {key}: "
                f"{baseline_value!r} -> {current_value!r}")
    current_cpu_profile = current.get("cpu_backend_profiles", {}).get("candidate")
    baseline_cpu_profile = baseline.get("cpu_backend_profiles", {}).get("candidate")
    if (current_cpu_profile is None) != (baseline_cpu_profile is None):
        failures.append("CPU backend dispatch profile changed or is missing")
    elif current_cpu_profile is not None:
        failures.extend(cpu_backend_profile_failures(
            current_cpu_profile, baseline_cpu_profile))
    if baseline.get("manifest", {}).get("sha256") != current.get(
            "manifest", {}).get("sha256"):
        failures.append("validation manifest identity changed")
    baseline_inputs = {
        name: (value.get("bytes"), value.get("sha256"))
        for name, value in baseline.get("inputs", {}).items()
    }
    current_inputs = {
        name: (value.get("bytes"), value.get("sha256"))
        for name, value in current.get("inputs", {}).items()
    }
    if baseline_inputs != current_inputs:
        failures.append("validation input identity changed")
    for key in ("machine", "cpu_count", "nvidia"):
        if baseline.get("host", {}).get(key) != current.get("host", {}).get(key):
            failures.append(f"validation host mismatch: {key}")

    base_rows = {row["key"]: row for row in baseline.get("results", [])}
    for row in current["results"]:
        if row.get("status") == VRAM_SKIP_STATUS:
            # Resource-conditional skip: there is no measurement to compare
            # and the skip is reported in the summary instead.
            continue
        if allow_incomplete and row.get("status", "pass") != "pass":
            continue
        base = base_rows.get(row["key"])
        if not base:
            if not allow_incomplete:
                failures.append(f"{row['key']}: missing baseline row")
            continue
        # Schema-1 snapshot reports written before status was added are still
        # comparable.  An explicitly incomplete baseline is never acceptable.
        if base.get("status", "pass") != "pass":
            if not allow_incomplete:
                failures.append(
                    f"{row['key']}: baseline status is "
                    f"{base.get('status', 'unknown')}")
            continue
        base_assets = [(item.get("bytes"), item.get("sha256"))
                       for item in base.get("model_assets", [])]
        current_assets = [(item.get("bytes"), item.get("sha256"))
                          for item in row.get("model_assets", [])]
        if base_assets != current_assets:
            failures.append(f"{row['key']}: model asset identity changed")
            continue
        performance_keys = {
            key for key in row["metrics"]
            if is_candidate_performance_metric(key, current.get("backend"))
        }
        base_performance_keys = {
            key for key in base.get("metrics", {})
            if is_candidate_performance_metric(key, current.get("backend"))
        }
        for key in sorted(base_performance_keys - performance_keys):
            failures.append(f"{row['key']}: missing candidate metric {key}")
        for key in sorted(performance_keys):
            candidate_value = row["metrics"][key]
            if key == "process_wall_ms" and len(row["metrics"]) > 1:
                continue
            baseline_value = base.get("metrics", {}).get(key)
            if baseline_value is None or baseline_value <= 0:
                failures.append(f"{row['key']}: missing metric baseline {key}")
                continue
            delta = (candidate_value - baseline_value) / baseline_value * 100.0
            row.setdefault("performance_deltas_pct", {})[key] = delta
            absolute_delta = candidate_value - baseline_value
            if delta > threshold and absolute_delta > absolute_noise_floor:
                failures.append(
                    f"{row['key']}: {key} regressed {delta:.2f}% "
                    f"({baseline_value:.3f} -> {candidate_value:.3f}, "
                    f"absolute +{absolute_delta:.3f})")
        if (row.get("fingerprint_policy") == "exact" and
                base.get("fingerprints") != row.get("fingerprints")):
            failures.append(f"{row['key']}: output fingerprint changed")
    extra = sorted(set(base_rows) - {row["key"] for row in current["results"]})
    if not allow_incomplete:
        failures.extend(f"missing candidate row: {key}" for key in extra)
    return failures


STATUS_ORDER = ("pass", "fail", "unstable", "skipped", "vram_skipped")
STATUS_LABELS = {"pass": "PASS", "fail": "FAIL", "unstable": "UNSTABLE",
                 "skipped": "SKIP", "vram_skipped": "VRAM-SKIP"}
BAR_WIDTH = 24


def _bar(count: int, total: int) -> str:
    filled = int(round(BAR_WIDTH * count / total)) if total else 0
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def render_summary_section(rows: list[dict[str, Any]]) -> list[str]:
    """Top-of-report totals: counts, rates, and a per-status distribution."""
    total = len(rows)
    counts = {status: 0 for status in STATUS_ORDER}
    for row in rows:
        counts.setdefault(row["status"], 0)
        counts[row["status"]] += 1
    ordered = [status for status in STATUS_ORDER if counts.get(status)] + \
              [status for status in counts if status not in STATUS_ORDER]
    bad = counts["fail"] + counts.get("unstable", 0)
    skipped = counts["skipped"] + counts.get("vram_skipped", 0)
    pass_rate = counts["pass"] / total * 100.0 if total else 0.0
    fail_rate = bad / total * 100.0 if total else 0.0
    skip_rate = skipped / total * 100.0 if total else 0.0
    lines = ["## Summary", "",
             f"- total scenarios: **{total}** — "
             f"PASS **{counts['pass']}**, FAIL **{counts['fail']}**, "
             f"UNSTABLE **{counts.get('unstable', 0)}**, "
             f"SKIP **{counts['skipped']}**, "
             f"VRAM-SKIP **{counts.get('vram_skipped', 0)}**",
             f"- pass rate: **{pass_rate:.1f}%** · fail rate: "
             f"**{fail_rate:.1f}%** · skip rate: **{skip_rate:.1f}%**",
             "",
             "| Status | Count | Share | Distribution |",
             "|---|---:|---:|---|"]
    for status in ordered:
        count = counts[status]
        share = f"{count / total * 100.0:.1f}%" if total else "0.0%"
        lines.append(f"| {STATUS_LABELS.get(status, status.upper())} | "
                     f"{count} | {share} | `{_bar(count, total)}` |")
    lines.append("")
    return lines


def render_per_task_section(rows: list[dict[str, Any]]) -> list[str]:
    """Per-task pass/fail/skip breakdown under the totals."""
    per_task: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = per_task.setdefault(row["task"], {})
        bucket[row["status"]] = bucket.get(row["status"], 0) + 1
    if not per_task:
        return []
    lines = ["| Task | Total | PASS | FAIL | UNSTABLE | SKIP | VRAM-SKIP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in sorted(per_task):
        bucket = per_task[task]
        total = sum(bucket.values())
        lines.append(f"| {task} | {total} | {bucket.get('pass', 0)} | "
                     f"{bucket.get('fail', 0)} | "
                     f"{bucket.get('unstable', 0)} | "
                     f"{bucket.get('skipped', 0)} | "
                     f"{bucket.get('vram_skipped', 0)} |")
    lines.append("")
    return lines


def render_failure_details(rows: list[dict[str, Any]]) -> list[str]:
    """One row per failed probe with the probe's own root-cause evidence."""
    failed = [row for row in rows
              if row.get("status") in ("fail", "unstable")]
    lines = ["## Failure details", "",
             "Evidence lines come from the probe output tail (last "
             "distinct error lines).", ""]
    if not failed:
        lines.append("- none")
        return lines
    lines.extend(["| Task | Scenario/model | Status | Evidence |",
                  "|---|---|---|---|"])
    for row in failed:
        evidence = (row.get("error_summary") or "(no error line captured)")
        evidence = evidence.replace("|", "\\|")
        lines.append(f"| {row['task']} | `{row['model_id']}` | "
                     f"{row['status'].upper()} | {evidence} |")
    lines.append("")
    return lines


def render_vram_section(rows: list[dict[str, Any]]) -> list[str]:
    gated = [row for row in rows if row.get("status") == VRAM_SKIP_STATUS]
    lines = ["## VRAM-gated skips", "",
             "Probes skipped before launch because the estimated working "
             "set exceeds the currently free VRAM (OOM guard). Free VRAM "
             "is queried per probe; raise it (close other processes), use a "
             "larger GPU, or lower --vram-overhead-mib to run them.", ""]
    if not gated:
        lines.append("- none")
        return lines
    lines.extend(["| Task | Scenario/model | Needed (MiB) | Free (MiB) |",
                  "|---|---|---:|---:|"])
    for row in gated:
        needed = row["metrics"].get("vram_needed_mib", 0.0)
        free = row["metrics"].get("vram_free_mib", 0.0)
        lines.append(f"| {row['task']} | `{row['model_id']}` | "
                     f"{needed:.0f} | {free:.0f} |")
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = report["results"]
    lines = ["# AICore model validation", "",
             f"- verdict: **{report['verdict']}**",
             f"- backend: `{report['backend']}`",
             f"- profile: `{report.get('profile', 'developer')}`",
             f"- scenarios: {len(rows)}",
             f"- repeated process runs: {report['repeats']}",
             f"- revision: `{report['revision'] or 'unknown'}`", "",
             *render_summary_section(rows),
             *render_per_task_section(rows)]
    lines.extend(["| Task | Scenario/model | Accuracy gate | Accuracy contract | Stability | Metrics |",
                  "|---|---|---|---|---|---|"])
    for row in rows:
        metric_text = ", ".join(f"{key}={value:.3f}"
                                for key, value in row["metrics"].items())
        stable = "PASS" if row["status"] == "pass" else \
            STATUS_LABELS.get(row["status"], row["status"].upper())
        accuracy = row.get("accuracy", {})
        accuracy_text = ("structured" if accuracy.get("structured")
                         else accuracy.get("reference", "probe"))
        lines.append(f"| {row['task']} | `{row['model_id']}` | "
                     f"{row['accuracy_gate']} | {accuracy_text} | {stable} | "
                     f"{metric_text} |")
    lines.extend(["", *render_failure_details(rows),
                  *render_vram_section(rows)])
    lines.extend(["## All failures (raw)", ""])
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- none")
    lines.extend(["", "## Incomplete reasons", ""])
    if report.get("incomplete_reasons"):
        lines.extend(f"- {reason}" for reason in report["incomplete_reasons"])
    else:
        lines.append("- none")
    lines.extend(["", "## Coverage", ""])
    for task, count in sorted(report["coverage"]["models_by_task"].items()):
        lines.append(f"- `{task}`: {count} model asset(s) consumed")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=REPO_ROOT / "build_app")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--assets", type=Path,
                        default=Path(os.environ.get(
                            "AICORE_TEST_ASSET_ROOT",
                            # Mirror the C++ data-root override
                            # (core/AICore/src/common/data_root_util.cpp):
                            # CLOUDVIEWER_DATA_ROOT relocates the whole
                            # cloudViewer_data tree, enabling cold-cache
                            # download verification in an isolated root.
                            str(Path(os.environ.get(
                                "CLOUDVIEWER_DATA_ROOT",
                                Path.home() / "cloudViewer_data"))
                                / "extract"))))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--catalog-binary", type=Path,
                        help="override test_catalog_dump_urls executable")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--full", action="store_true",
                        help="run the complete matrix; the default light tier "
                             "runs only the lightweight subset declared by "
                             "tiered tasks (sam3/trellis) and skips their "
                             "heavy scenarios")
    parser.add_argument("--tasks", default="",
                        help="comma-separated subset; empty means every task")
    parser.add_argument("--scenarios", default="",
                        help="comma-separated exact scenario ids; empty means "
                             "every scenario selected by --tasks")
    parser.add_argument("--models", default="",
                        help="comma-separated fnmatch patterns selecting "
                             "individual models: model file names (with or "
                             "without the .gguf suffix) or bundle scenario ids "
                             "(e.g. trellis-coarse-q8, yolo). Implies full "
                             "expansion for the selected tasks and narrows "
                             "the coverage audit to the selection; a pattern "
                             "that matches nothing is an error")
    parser.add_argument("--profile", choices=("developer", "release"),
                        default="developer",
                        help="developer is a fast diagnostic loop; release "
                             "requires a controlled baseline and stronger sampling")
    parser.add_argument("--repeats", type=int, default=None,
                        help="process-level stability repeats (profile default: "
                             "developer=2, release=3)")
    parser.add_argument("--warmup-runs", type=int, default=None,
                        help="per-process warmup forwards (profile default: "
                             "developer=1, release=2)")
    parser.add_argument("--inference-runs", type=int, default=None,
                        help="timed forwards per process (profile default: "
                             "developer=2, release=10)")
    parser.add_argument("--trellis-steps", type=int, default=12)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--gpu-idle-timeout", type=float, default=60.0,
                        help="maximum seconds a probe waits for a clean GPU "
                             "sample before failing; shared hosts with "
                             "intermittent CUDA workloads raise it")
    parser.add_argument("--absolute-noise-floor", "--noise-floor",
                        dest="absolute_noise_floor", type=float, default=3.0,
                        help="minimum absolute increase in metric units")
    parser.add_argument("--vram-overhead-mib", type=float,
                        default=DEFAULT_VRAM_OVERHEAD_MIB,
                        help="default activation/working-set MiB assumed on "
                             "top of model bytes for scenarios that do not "
                             "declare vram_overhead_mib")
    parser.add_argument("--baseline", type=Path,
                        help="JSON report captured before the optimization")
    parser.add_argument("--baseline-build", type=Path,
                        help="preferred controlled A/B: interleave this build "
                             "with --build")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "build_app" / "Testing" /
                                "aicore_validation.json")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="development-only: skip unavailable model downloads "
                             "and probe SKIPs; the report is INCOMPLETE")
    parser.add_argument("--offline", action="store_true",
                        help="verify complete model and test-input caches without "
                             "downloading; missing or corrupt files fail")
    parser.add_argument("--clean-probe-outputs", action="store_true",
                        help="after each task's rows are summarized, delete its raw "
                             "per-probe output files under Testing/probes/ and "
                             "Testing/baseline-probes/ (metrics and fingerprints "
                             "are already folded into the report); off by default, "
                             "intended for space-constrained hosts such as CI "
                             "runners. Model caches are never deleted.")
    parser.add_argument("--clean-model-cache", action="store_true",
                        help="after each task's rows are summarized, delete the "
                             "model files this run consumed for that task from "
                             "the asset cache root (~/cloudViewer_data/extract "
                             "by default); off by default because re-downloading "
                             "costs bandwidth and time, intended for "
                             "space-constrained hosts such as CI runners. Only "
                             "files this run's tier consumed are removed, and "
                             "shared models survive until their last "
                             "referencing task finishes. Combine with --offline "
                             "only if you re-download separately.")
    parser.add_argument("--download-jobs", type=int, default=3)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-timeout", type=int, default=120,
                        help="per-socket model download timeout in seconds")
    parser.add_argument("--list", action="store_true",
                        help="ensure model and test-input caches and print expanded "
                             "commands without executing inference")
    parser.add_argument("--image", default="")
    parser.add_argument("--image2", default="")
    parser.add_argument("--face-image", default="")
    parser.add_argument("--sam-image", default="")
    parser.add_argument("--yolo-image", default="")
    parser.add_argument("--reference", action="append", default=[],
                        metavar="NAME=PATH",
                        help="external truth program root consumed by "
                             "{reference:NAME} manifest tokens (repeatable), "
                             "e.g. map_anything_ggml=<map-anything-ggml "
                             "checkout>; rows whose reference is absent are "
                             "skipped and fail the gate unless "
                             "--allow-incomplete")
    args = parser.parse_args()
    try:
        args.references = parse_references(args.reference)
    except ValueError as exc:
        parser.error(str(exc))
    profile_defaults = {
        "developer": (2, 1, 2),
        "release": (3, 2, 10),
    }[args.profile]
    if args.repeats is None:
        args.repeats = profile_defaults[0]
    if args.warmup_runs is None:
        args.warmup_runs = profile_defaults[1]
    if args.inference_runs is None:
        args.inference_runs = profile_defaults[2]
    args.repo = args.repo.resolve()
    args.build = args.build.resolve()
    args.assets = args.assets.resolve()
    args.manifest = args.manifest.resolve()
    if args.catalog_binary:
        args.catalog_binary = args.catalog_binary.resolve()
    args.output = args.output.resolve()
    if args.baseline_build:
        args.baseline_build = args.baseline_build.resolve()
    if args.baseline and args.baseline_build:
        parser.error("--baseline and --baseline-build are mutually exclusive")
    if args.profile == "release":
        if not args.baseline and not args.baseline_build:
            parser.error("--profile release requires --baseline or --baseline-build")
        if args.allow_incomplete:
            parser.error("--profile release is incompatible with --allow-incomplete")
        if args.repeats < 3 or args.warmup_runs < 2 or args.inference_runs < 5:
            parser.error("release profile requires repeats >= 3, warmup-runs >= 2, "
                         "and inference-runs >= 5")
    default_images = args.assets / "lightglue_test_images"
    if not args.image:
        args.image = str(default_images / "sacre_coeur1.jpg")
    if not args.image2:
        args.image2 = str(default_images / "sacre_coeur2.jpg")
    if not args.face_image:
        args.face_image = str(args.assets / "friends_faces/query/friends1.jpg")
    if not args.sam_image:
        args.sam_image = str(args.assets / "sam_test_data/images/cand1.jpg")
    if not args.yolo_image:
        args.yolo_image = str(args.assets / "objects_detection_data/images/bus.jpg")
    if (args.repeats < 1 or args.inference_runs < 1 or args.warmup_runs < 0 or
            args.download_jobs < 1 or args.download_retries < 1 or
            args.download_timeout < 1):
        parser.error("repeat/download values must be positive; warmup-runs "
                     "must be non-negative")
    if args.gpu_idle_timeout <= 0:
        parser.error("--gpu-idle-timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    cpu_backend_profiles = {"candidate": cpu_backend_profile(args.build)}
    if args.baseline_build:
        cpu_backend_profiles["baseline"] = cpu_backend_profile(args.baseline_build)
        profile_failures = cpu_backend_profile_failures(
            cpu_backend_profiles["candidate"], cpu_backend_profiles["baseline"])
        if profile_failures and not args.list:
            print("controlled A/B CPU backend preflight failed:", file=sys.stderr)
            for failure in profile_failures:
                print(f"- {failure}", file=sys.stderr)
            print("Rebuild the candidate until its CPU fallback dispatch "
                  "capability matches the baseline.", file=sys.stderr)
            return 1
    manifest_identity: dict[str, Any] = {}
    manifest = select_scenarios(load_manifest(args.manifest, identity=manifest_identity), args.scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = selected_tasks(manifest, args.tasks)
    try:
        catalog = load_model_catalog(args, manifest, selected)
        if args.profile == "release":
            contract_failures = release_manifest_accuracy_contract_failures(
                manifest, args, selected, catalog)
            if contract_failures:
                print("release accuracy contract failed:", file=sys.stderr)
                for failure in contract_failures:
                    print(f"- {failure}", file=sys.stderr)
                return 1
        cache_summary, cache_failures = ensure_model_assets(
            catalog, args.assets, offline=args.offline,
            timeout=args.download_timeout, retries=args.download_retries,
            jobs=args.download_jobs)
        specs, diagnostics = expand_specs(manifest, args, args.output.parent)
        input_catalog = parse_input_assets(
            manifest, selected, {spec.scenario_id for spec in specs})
        input_summary, input_failures = ensure_input_assets(
            input_catalog, args.assets, offline=args.offline,
            timeout=args.download_timeout, retries=args.download_retries,
            jobs=args.download_jobs)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"validation cache preflight failed: {error}", file=sys.stderr)
        return 1
    if cache_failures:
        for failure in cache_failures:
            prefix = ("model cache incomplete" if args.allow_incomplete else
                      "model cache preflight failed")
            print(f"{prefix}: {failure}", file=sys.stderr)
        if not args.allow_incomplete:
            return 1
    cache_state = ("incomplete" if cache_summary["unavailable"] else "ready")
    print(f"model cache {cache_state}: "
          f"{cache_summary['expected']} expected, "
          f"{cache_summary['verified_cached']} cached, "
          f"{cache_summary['downloaded']} downloaded",
          flush=True)
    if input_failures:
        for failure in input_failures:
            print(f"input cache preflight failed: {failure}", file=sys.stderr)
        return 1
    input_cache_state = "incomplete" if input_summary["unavailable"] else "ready"
    print(f"input cache {input_cache_state}: "
          f"{input_summary['expected']} expected, "
          f"{input_summary['verified_cached']} cached, "
          f"{input_summary['downloaded']} downloaded",
          flush=True)
    if args.allow_incomplete and cache_summary["unavailable"]:
        diagnostics["skipped_model_assets"].extend(
            cache_summary["unavailable"])
        specs, skipped = filter_unavailable_specs(
            specs, args.assets, cache_summary["unavailable"])
        diagnostics["skipped_scenarios"].extend(skipped)
    release_contract_failures = (
        release_accuracy_contract_failures(specs)
        if args.profile == "release" else [])
    if release_contract_failures:
        print("release accuracy contract failed:", file=sys.stderr)
        for failure in release_contract_failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    baseline_args = None
    baseline_specs: dict[str, RunSpec] = {}
    if args.baseline_build:
        baseline_args = copy.copy(args)
        baseline_args.build = args.baseline_build
        expanded, baseline_diagnostics = expand_specs(
            manifest, baseline_args, args.output.parent / "baseline-probes")
        if args.allow_incomplete and cache_summary["unavailable"]:
            expanded, _ = filter_unavailable_specs(
                expanded, args.assets, cache_summary["unavailable"])
        baseline_specs = {spec.key: spec for spec in expanded}
        for category, entries in baseline_diagnostics.items():
            diagnostics[category].extend(
                entry for entry in entries if entry not in diagnostics[category])
    if args.list:
        for spec in specs:
            print(spec.key)
            print("  " + " ".join(spec.command))
        for category, entries in diagnostics.items():
            for entry in entries:
                print(f"[{category}] {entry}")
        return 0

    try:
        artifact_snapshots = {"candidate": execution_artifacts(specs, args.build)}
        if baseline_args:
            artifact_snapshots["baseline"] = execution_artifacts(
                list(baseline_specs.values()), baseline_args.build)
    except OSError as error:
        print(f"execution artifact preflight failed: {error}", file=sys.stderr)
        return 1
    results: list[dict[str, Any]] = []
    baseline_results: list[dict[str, Any]] = []
    failures: list[str] = []
    if not args.allow_incomplete:
        failures.extend(f"missing asset: {item}"
                        for item in diagnostics["missing_assets"])
        failures.extend(f"uncovered asset: {item}"
                        for item in diagnostics["uncovered_assets"])

    # Optional per-task cleanup (--clean-probe-outputs / --clean-model-cache):
    # once a task's last spec has been summarized, its probe outputs are
    # folded into the rows above and its consumed models have no further run
    # consumer, so both can be released instead of accumulating. Off by
    # default; opt in on space-constrained hosts.
    specs_left_per_task = {task: 0 for task in selected}
    consumed_by_task: dict[str, set[Path]] = {task: set() for task in selected}
    for spec in specs:
        specs_left_per_task[spec.task] += 1
        consumed_by_task[spec.task].update(
            path for path in spec.model_paths if path.is_file())
    holders_by_path: dict[Path, set[str]] = {}
    for task, paths in consumed_by_task.items():
        for path in paths:
            holders_by_path.setdefault(path, set()).add(task)

    def release_task_outputs(task: str) -> None:
        probe_removed, model_removed = release_finished_task_outputs(
            task, specs_left_per_task, consumed_by_task, holders_by_path,
            args)
        if probe_removed:
            print(f"  cleaned {probe_removed} probe output file(s) for task "
                  f"{task}", flush=True)
        if model_removed:
            print(f"  pruned {len(model_removed)} cached model file(s) for "
                  f"task {task}", flush=True)

    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec.key}", flush=True)
        # OOM guard: skip probes whose estimated working set cannot fit in
        # the currently free VRAM instead of crashing mid-run.
        gate = vram_gate_decision(spec, args)
        if gate is not None:
            row = vram_skipped_row(spec, gate)
            results.append(row)
            release_task_outputs(spec.task)
            print(f"  {VRAM_SKIP_STATUS}: need {gate['needed_mib']:.0f} MiB, "
                  f"free {gate['free_mib']:.0f} MiB", flush=True)
            continue
        attempts: list[Attempt] = []
        baseline_attempts: list[Attempt] = []
        baseline_spec = baseline_specs.get(spec.key)
        for repeat in range(args.repeats):
            order = ("baseline", "candidate") if repeat % 2 == 0 else (
                "candidate", "baseline")
            for side in order:
                if side == "baseline" and baseline_spec and baseline_args:
                    attempt = run_attempt(baseline_spec, baseline_args)
                    baseline_attempts.append(attempt)
                    print(f"  baseline {repeat + 1}/{args.repeats}: "
                          f"rc={attempt.returncode} "
                          f"wall={attempt.duration_ms:.1f} ms", flush=True)
                elif side == "candidate":
                    attempt = run_attempt(spec, args)
                    attempts.append(attempt)
                    print(f"  candidate {repeat + 1}/{args.repeats}: "
                          f"rc={attempt.returncode} "
                          f"wall={attempt.duration_ms:.1f} ms", flush=True)
        row, row_failures, baseline_row, baseline_row_failures = (
            summarize_attempt_pair(spec, attempts, baseline_spec,
                                   baseline_attempts, args.allow_incomplete))
        results.append(row)
        failures.extend(row_failures)
        if args.baseline_build:
            if not baseline_spec:
                failures.append(f"{spec.key}: missing baseline build scenario")
            else:
                assert baseline_row is not None
                baseline_results.append(baseline_row)
                failures.extend(f"baseline: {failure}"
                                for failure in baseline_row_failures)
        release_task_outputs(spec.task)

    assets_by_task: dict[str, set[str]] = {}
    for row in results:
        task_assets = assets_by_task.setdefault(row["task"], set())
        task_assets.update(item["path"] for item in row["model_assets"])
    models_by_task = {task: len(paths)
                      for task, paths in sorted(assets_by_task.items())}
    try:
        if file_identity(args.manifest) != manifest_identity:
            failures.append("validation manifest changed during execution")
    except OSError as error:
        failures.append(f"validation manifest unavailable after execution: {error}")
    artifact_failures = []
    for label, side_specs, side_build in [
            ("candidate", specs, args.build),
            *([("baseline", list(baseline_specs.values()), baseline_args.build)]
              if baseline_args else [])]:
        try:
            artifact_failures.extend(execution_artifact_failures(
                artifact_snapshots[label], execution_artifacts(side_specs, side_build), label))
        except OSError as error:
            artifact_failures.append(f"{label} execution artifact unavailable after validation: {error}")
    failures.extend(artifact_failures)
    report: dict[str, Any] = {
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "revision": git_output(args.repo, "rev-parse", "HEAD"),
        "worktree_dirty": bool(git_output(args.repo, "status", "--porcelain")),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "cpu_count": os.cpu_count(),
                 "nvidia": command_output([
                     "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"])},
        "build": str(args.build), "assets": str(args.assets),
        "cpu_backend_profiles": cpu_backend_profiles,
        "backend": args.backend,
        "profile": args.profile,
        "tier": "full" if args.full else "light",
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs, "inference_runs": args.inference_runs,
        "threads": args.threads, "trellis_steps": args.trellis_steps,
        "gpu_idle_timeout": args.gpu_idle_timeout,
        "threshold_pct": args.threshold,
        "absolute_noise_floor": args.absolute_noise_floor,
        "baseline": str(args.baseline.resolve()) if args.baseline else "",
        "baseline_build": str(args.baseline_build) if args.baseline_build else "",
        "model_cache": cache_summary,
        "input_cache": input_summary,
        "manifest": manifest_identity,
        "execution_artifacts": {"before": artifact_snapshots,
            "unchanged": not artifact_failures,
            "scope": "selected command files, runner, build AICore/ggml libraries and CMake cache; not a transitive loader trace"},
        "references": dict(getattr(args, "references", {})),
        "inputs": {
            name: file_identity(Path(path))
            for name, path in {
                "image": args.image, "image2": args.image2,
                "face_image": args.face_image, "sam_image": args.sam_image,
                "yolo_image": args.yolo_image,
            }.items() if Path(path).is_file()
        },
        "coverage": {"models_by_task": models_by_task, **diagnostics},
        "results": results,
        "failures": failures,
    }
    incomplete_reasons: list[str] = []
    if args.allow_incomplete:
        incomplete_reasons.extend(
            f"model unavailable: {item}"
            for item in diagnostics["skipped_model_assets"])
        incomplete_reasons.extend(
            f"scenario skipped: {item}"
            for item in diagnostics["skipped_scenarios"])
        incomplete_reasons.extend(
            f"probe skipped: {row['key']}"
            for row in results if row.get("status") == "skipped")
        incomplete_reasons.extend(
            f"missing asset: {item}" for item in diagnostics["missing_assets"])
        incomplete_reasons.extend(
            f"uncovered asset: {item}"
            for item in diagnostics["uncovered_assets"])
    # VRAM-gated skips are resource-conditional, never silent: list them
    # explicitly regardless of --allow-incomplete so the summary explains
    # why the affected probes did not run on this host.
    incomplete_reasons.extend(
        f"vram insufficient: {row['key']} ({row['error_summary']})"
        for row in results if row.get("status") == VRAM_SKIP_STATUS)
    report["incomplete_reasons"] = incomplete_reasons
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        failures.extend(compare_report(report, baseline, args.threshold,
                                       args.absolute_noise_floor,
                                       args.allow_incomplete))
    elif args.baseline_build:
        baseline = dict(report)
        baseline["build"] = str(args.baseline_build)
        baseline["results"] = baseline_results
        baseline["failures"] = []
        failures.extend(compare_report(report, baseline, args.threshold,
                                       args.absolute_noise_floor,
                                       args.allow_incomplete))
        report["interleaved_baseline_results"] = baseline_results
    report["verdict"] = ("FAIL" if failures else
                         "INCOMPLETE" if incomplete_reasons else "PASS")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"report: {args.output}")
    print(f"summary: {markdown}")
    print(f"verdict: {report['verdict']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
