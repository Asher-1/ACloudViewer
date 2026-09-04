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
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
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
        if previous and previous != asset:
            raise ValueError(f"conflicting model catalog rows for {relative}")
        assets[relative] = asset
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
            completed.stdout, selected,
            manifest_model_patterns(
                manifest, selected, full=True, tiered=tiered),
            task_selection=task_selection)
    light_patterns = {
        task: manifest_model_patterns(
            manifest, {task}, full=False, tiered=tiered)
        for task in tiered}
    return parse_catalog_output(
        completed.stdout, selected,
        manifest_model_patterns(manifest, selected, full=full, tiered=tiered),
        task_light_patterns=None if full else light_patterns)


def parse_input_assets(manifest: dict[str, Any], selected: set[str]
                       ) -> list[InputAsset]:
    assets: list[InputAsset] = []
    for index, raw in enumerate(manifest.get("input_assets", []), 1):
        asset_id = str(raw.get("id", ""))
        tasks = frozenset(str(task) for task in raw.get("tasks", []))
        if not tasks & selected:
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
        selected_scenarios = [scenario for scenario in manifest["scenarios"]
                              if scenario["task"] in selected]
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
        with zipfile.ZipFile(archive) as stream:
            names = set(stream.namelist())
            for relative, digest, size in asset.extracted_files:
                member = str(relative)
                if member not in names:
                    raise RuntimeError(f"archive is missing {member}")
                destination = root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(member) as source:
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
    except (OSError, zipfile.BadZipFile) as error:
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


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != 1:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")
    if (not manifest.get("tasks") or not manifest.get("scenarios") or
            not manifest.get("input_assets")):
        raise ValueError("manifest must declare tasks, scenarios, and input assets")
    return manifest


def find_binary(build: Path, name: str) -> Path:
    suffixes = (".exe", "") if os.name == "nt" else ("",)
    for directory in (build / "bin" / "aicore_tests", build / "bin"):
        for suffix in suffixes:
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    return (build / "bin" / "aicore_tests" / name).resolve()


def model_quant(path: Path) -> str:
    match = re.search(r"-(f32|f16|q8_0|q8|q4_0|q4_k)\.gguf$", path.name,
                      re.IGNORECASE)
    return match.group(1).lower() if match else "f16"


def resolve_token(text: str, values: dict[str, str], build: Path) -> str:
    def binary_replace(match: re.Match[str]) -> str:
        return str(find_binary(build, match.group(1)))

    text = re.sub(r"\{binary:([^}]+)\}", binary_replace, text)
    return text.format_map(values)


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
    full = bool(getattr(args, "full", False)) or bool(model_filter)
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
                scenario_owned.update(existing_files(args.assets, pattern))
        owned[task].update(scenario_owned)

        model_patterns = light_globs if light_active else (
            [raw["model_glob"]] if raw.get("model_glob") else [])
        models: list[Path] = []
        for pattern in model_patterns:
            models.extend(existing_files(args.assets, pattern))
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
                matches = existing_files(args.assets, pattern)
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
            scenario_report = report_dir / "probes" / f"{task}-{model_id}.json"
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
            if executable == "python":
                command = [sys.executable]
            elif executable.startswith("binary:"):
                command = [str(find_binary(args.build, executable.split(":", 1)[1]))]
            else:
                command = [resolve_token(executable, values, args.build)]
            command.extend(resolve_token(item, values, args.build)
                           for item in raw.get("args", []))
            env = {key: resolve_token(value, values, args.build)
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


def flatten_json(value: Any, prefix: str = "") -> tuple[dict[str, float], dict[str, str]]:
    metrics: dict[str, float] = {}
    fingerprints: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            child_metrics, child_fingerprints = flatten_json(child, child_prefix)
            metrics.update(child_metrics)
            fingerprints.update(child_fingerprints)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_metrics, child_fingerprints = flatten_json(child, f"{prefix}/{index}")
            metrics.update(child_metrics)
            fingerprints.update(child_fingerprints)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        leaf = prefix.rsplit("/", 1)[-1].lower()
        path_parts = (part.lower() for part in prefix.split("/"))
        is_timing = any(part.endswith(TIMING_SUFFIXES) for part in path_parts)
        if (is_timing and float(value) > 0.0) or leaf in {
                "mask_iou", "box_error", "score_error", "worst_rel_mae",
                "kpt_median_px", "descriptor_cosine_median"}:
            metrics[prefix] = float(value)
    elif isinstance(value, str):
        leaf = prefix.rsplit("/", 1)[-1].lower()
        if any(token in leaf for token in HASH_KEYS):
            fingerprints[prefix] = value
    return metrics, fingerprints


def parse_output(output: str, parser: str, report_path: Path) -> tuple[dict[str, float], dict[str, str]]:
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
        parsed_metrics, parsed_fingerprints = flatten_json(value, f"json{index}")
        metrics.update(parsed_metrics)
        fingerprints.update(parsed_fingerprints)

    if parser == "rfdetr":
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
    than the space they save.
    """
    removed = 0
    for directory in (report_root / "probes",
                      report_root / "baseline-probes"):
        for path in directory.glob(f"{task}-*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
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


def add_runtime_library_path(env: dict[str, str], build: Path) -> None:
    if os.name == "nt":
        variable = "PATH"
    elif sys.platform == "darwin":
        variable = "DYLD_LIBRARY_PATH"
    else:
        variable = "LD_LIBRARY_PATH"
    parts = [str(build / "bin")]
    qt_library_dir = env.get("AICORE_QT_LIBRARY_DIR")
    if qt_library_dir:
        parts.append(qt_library_dir)
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
    except subprocess.TimeoutExpired as error:
        raw = error.stdout or ""
        output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        output += f"\nTIMEOUT after {args.timeout}s"
        returncode = 124
    duration_ms = (time.perf_counter() - started) * 1000.0
    metrics, fingerprints = parse_output(output, spec.metric_parser,
                                         spec.report_path)
    metrics.setdefault("process_wall_ms", duration_ms)
    return Attempt(returncode=returncode, duration_ms=duration_ms,
                   metrics=metrics, fingerprints=fingerprints,
                   output_tail=output[-8000:])


def median_metrics(attempts: list[Attempt]) -> dict[str, float]:
    keys = set().union(*(attempt.metrics for attempt in attempts))
    return {key: statistics.median(
                attempt.metrics[key] for attempt in attempts if key in attempt.metrics)
            for key in sorted(keys)}


def stable_fingerprints(attempts: list[Attempt]) -> tuple[dict[str, str], list[str]]:
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
    markers = ("FAIL", "failed", "error", "assert", "Traceback",
               "TIMEOUT", "SKIP:")
    evidence: list[str] = []
    for raw in output_tail.splitlines():
        line = raw.strip()
        if not line or not any(marker in line for marker in markers):
            continue
        if len(line) > 220:
            line = line[:217] + "..."
        if line not in evidence:
            evidence.append(line)
    return " | ".join(evidence[-limit:])


def summarize_attempts(spec: RunSpec, attempts: list[Attempt],
                       allow_incomplete: bool) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    codes = [attempt.returncode for attempt in attempts]
    fingerprints, stability_failures = stable_fingerprints(attempts)
    if (spec.require_fingerprint and all(code == 0 for code in codes) and
            not fingerprints):
        stability_failures.append("probe emitted no output fingerprint")
    status = "pass"
    if any(code == 77 for code in codes):
        status = "skipped"
    if any(code not in (0, 77) for code in codes):
        status = "fail"
    if stability_failures:
        status = "unstable"
    if status != "pass" and not (allow_incomplete and status == "skipped"):
        failures.append(f"{spec.key}: {status}, return codes={codes}")
    failures.extend(f"{spec.key}: {message}" for message in stability_failures)
    assets = [file_identity(path) for path in spec.model_paths if path.is_file()]
    return {
        "key": spec.key,
        "task": spec.task,
        "scenario_id": spec.scenario_id,
        "model_id": spec.model_id,
        "model_assets": assets,
        "accuracy_gate": spec.accuracy_gate,
        "fingerprint_policy": spec.fingerprint_policy,
        "status": status,
        "return_codes": codes,
        "metrics": median_metrics(attempts),
        "fingerprints": fingerprints,
        "output_tail": attempts[-1].output_tail if attempts else "",
        "error_summary": extract_error_summary(
            attempts[-1].output_tail if attempts else ""),
    }, failures


def file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


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


def compare_report(current: dict[str, Any], baseline: dict[str, Any],
                   threshold: float, absolute_noise_floor: float,
                   allow_incomplete: bool = False) -> list[str]:
    failures: list[str] = []
    for key in ("backend", "repeats", "warmup_runs", "inference_runs",
                "threads", "trellis_steps", "tier"):
        if baseline.get(key) != current.get(key):
            failures.append(
                f"validation protocol mismatch {key}: "
                f"{baseline.get(key)!r} -> {current.get(key)!r}")
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
            if any(part.lower().endswith(TIMING_SUFFIXES)
                   for part in key.split("/"))
            and not (current.get("backend") != "cpu" and "/cpu/" in key.lower())
        }
        base_performance_keys = {
            key for key in base.get("metrics", {})
            if any(part.lower().endswith(TIMING_SUFFIXES)
                   for part in key.split("/"))
            and not (current.get("backend") != "cpu" and "/cpu/" in key.lower())
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
             f"- scenarios: {len(rows)}",
             f"- repeated process runs: {report['repeats']}",
             f"- revision: `{report['revision'] or 'unknown'}`", "",
             *render_summary_section(rows),
             *render_per_task_section(rows)]
    lines.extend(["| Task | Scenario/model | Accuracy gate | Stability | Metrics |",
                  "|---|---|---|---|---|"])
    for row in rows:
        metric_text = ", ".join(f"{key}={value:.3f}"
                                for key, value in row["metrics"].items())
        stable = "PASS" if row["status"] == "pass" else \
            STATUS_LABELS.get(row["status"], row["status"].upper())
        lines.append(f"| {row['task']} | `{row['model_id']}` | "
                     f"{row['accuracy_gate']} | {stable} | {metric_text} |")
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
                            Path.home() / "cloudViewer_data" / "extract")))
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
    parser.add_argument("--models", default="",
                        help="comma-separated fnmatch patterns selecting "
                             "individual models: model file names (with or "
                             "without the .gguf suffix) or bundle scenario ids "
                             "(e.g. trellis-coarse-q8, yolo). Implies full "
                             "expansion for the selected tasks and narrows "
                             "the coverage audit to the selection; a pattern "
                             "that matches nothing is an error")
    parser.add_argument("--repeats", type=int, default=2,
                        help="process-level stability repeats; 2 is the minimum "
                             "needed to compare output fingerprints across "
                             "processes and detect the 'unstable' status, so "
                             "lower it only for throwaway diagnostics")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="per-process warmup forwards before timing; one "
                             "suffices because the performance gate requires "
                             "both a relative and an absolute increase")
    parser.add_argument("--inference-runs", type=int, default=5,
                        help="timed forwards per process; 5 keeps the p50 "
                             "stable for the default gate. For release-grade "
                             "A/B verdicts pass --inference-runs 10 for "
                             "tighter p95 statistics")
    parser.add_argument("--trellis-steps", type=int, default=12)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=float, default=5.0)
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
    args = parser.parse_args()
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
    return args


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = selected_tasks(manifest, args.tasks)
    try:
        catalog = load_model_catalog(args, manifest, selected)
        cache_summary, cache_failures = ensure_model_assets(
            catalog, args.assets, offline=args.offline,
            timeout=args.download_timeout, retries=args.download_retries,
            jobs=args.download_jobs)
        input_catalog = parse_input_assets(manifest, selected)
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
    specs, diagnostics = expand_specs(manifest, args, args.output.parent)
    if args.allow_incomplete and cache_summary["unavailable"]:
        diagnostics["skipped_model_assets"].extend(
            cache_summary["unavailable"])
        specs, skipped = filter_unavailable_specs(
            specs, args.assets, cache_summary["unavailable"])
        diagnostics["skipped_scenarios"].extend(skipped)
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
        row, row_failures = summarize_attempts(
            spec, attempts, args.allow_incomplete)
        results.append(row)
        release_task_outputs(spec.task)
        failures.extend(row_failures)
        if args.baseline_build:
            if not baseline_spec:
                failures.append(f"{spec.key}: missing baseline build scenario")
            else:
                baseline_row, baseline_row_failures = summarize_attempts(
                    baseline_spec, baseline_attempts, args.allow_incomplete)
                baseline_results.append(baseline_row)
                failures.extend(f"baseline: {failure}"
                                for failure in baseline_row_failures)

    assets_by_task: dict[str, set[str]] = {}
    for row in results:
        task_assets = assets_by_task.setdefault(row["task"], set())
        task_assets.update(item["path"] for item in row["model_assets"])
    models_by_task = {task: len(paths)
                      for task, paths in sorted(assets_by_task.items())}
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
        "backend": args.backend,
        "tier": "full" if args.full else "light",
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs, "inference_runs": args.inference_runs,
        "threads": args.threads, "trellis_steps": args.trellis_steps,
        "threshold_pct": args.threshold,
        "absolute_noise_floor": args.absolute_noise_floor,
        "baseline": str(args.baseline.resolve()) if args.baseline else "",
        "baseline_build": str(args.baseline_build) if args.baseline_build else "",
        "model_cache": cache_summary,
        "input_cache": input_summary,
        "manifest": file_identity(args.manifest),
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
