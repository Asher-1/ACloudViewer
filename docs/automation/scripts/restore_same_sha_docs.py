#!/usr/bin/env python3
"""Restore a verified native main build for catalogue-only Pages publication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile

REPOSITORY = "Asher-1/ACloudViewer"


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def pages(path, key, request=api):
    rows = []
    for page in range(1, 101):
        payload = request(
            f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
        batch = payload[key]
        if not isinstance(batch, list):
            raise ValueError("Invalid paginated artifact provenance")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
    raise ValueError("Artifact provenance pagination exceeded its bound")


def select_artifact(sha, request=api):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected a full source SHA")
    base = f"repos/{REPOSITORY}"
    runs = pages(
        f"{base}/actions/workflows/documentation.yml/runs?head_sha={sha}&branch=main&event=push&status=success",
        "workflow_runs", request)
    for run in sorted(runs, key=lambda r: r["id"], reverse=True):
        if (run["head_sha"] != sha or run["head_branch"] != "main" or
                run["event"] != "push" or run["status"] != "completed" or
                run["conclusion"] != "success" or
                run["repository"]["full_name"] != REPOSITORY or
                run["head_repository"]["full_name"] != REPOSITORY or
                run["path"] != ".github/workflows/documentation.yml"):
            raise ValueError("Native documentation run identity mismatch")
        jobs = pages(
            f"{base}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs",
            "jobs", request)
        native = [j for j in jobs if j["name"] == "build-docs"]
        if len(native) != 1 or native[0]["status"] != "completed" or native[0][
                "conclusion"] != "success":
            raise ValueError("Native documentation build job did not succeed")
        native_steps = {
            step['name']: step for step in native[0].get('steps', [])
        }
        for name in ('Build documentation Docker image',
                     'Extract documentation from Docker'):
            if native_steps.get(name, {}).get('conclusion') != 'success':
                raise ValueError(
                    'Source job did not execute the native documentation build')
        artifacts = pages(f"{base}/actions/runs/{run['id']}/artifacts",
                          "artifacts", request)
        matches = [
            a for a in artifacts if a["name"] == f"acloudviewer-{sha}-docs"
        ]
        if len(matches) > 1:
            raise ValueError("Ambiguous native documentation artifact")
        if not matches or matches[0]["expired"]:
            continue
        artifact = matches[0]
        source = artifact["workflow_run"]
        if (source["id"] != run["id"] or source["head_sha"] != sha or
                source["head_branch"] != "main" or
                source["head_repository_id"] != run["head_repository"]["id"] or
                source["repository_id"] != run["repository"]["id"] or
                not isinstance(artifact["id"], int) or artifact["id"] <= 0 or
                not isinstance(artifact["size_in_bytes"], int) or
                artifact["size_in_bytes"] <= 0 or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}", artifact.get("digest", ""))):
            raise ValueError("Artifact identity or digest mismatch")
        return run, artifact
    raise ValueError(
        "No verified same-SHA native main documentation artifact is available")


def download(artifact, output):
    # gh follows the signed redirect without exposing credentials in arguments.
    with output.open("xb") as stream:
        subprocess.run([
            "gh", "api",
            f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"
        ],
                       stdout=stream,
                       check=True)


def restore_archive(archive, artifact, sha, destination):
    if destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to replace existing documentation output")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive.stat().st_size != artifact[
            "size_in_bytes"] or "sha256:" + digest != artifact["digest"]:
        raise ValueError("Downloaded Actions ZIP size or SHA-256 mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="same-sha-docs-",
                                     dir=destination.parent) as temp:
        temp = Path(temp)
        tar_path = temp / "native.tar.gz"
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            if (len(members) != 1 or
                    members[0].filename != f"acloudviewer-{sha}-docs.tar.gz" or
                    members[0].is_dir() or members[0].flag_bits & 1 or
                    members[0].file_size >= 10_000_000_000 or
                    stat.S_IFMT(members[0].external_attr >> 16)
                    not in (0, stat.S_IFREG)):
                raise ValueError(
                    "Expected only the native documentation tar in Actions ZIP")
            with zipped.open(
                    members[0]) as source, tar_path.open("xb") as target:
                shutil.copyfileobj(source, target)
        tree = temp / "tree"
        tree.mkdir()
        with tarfile.open(tar_path, "r:gz") as tar:
            entries = tar.getmembers()
            seen = set()
            total = 0
            for entry in entries:
                path = PurePosixPath(entry.name)
                if (not entry.name or "\\" in entry.name or
                        path.is_absolute() or ".." in path.parts or
                        not (entry.isdir() or entry.isfile()) or
                        entry.size < 0):
                    raise ValueError("Unsafe native documentation tar member")
                key = str(path)
                if key in seen:
                    raise ValueError("Duplicate native documentation tar path")
                seen.add(key)
                total += entry.size
                if total >= 10_000_000_000:
                    raise ValueError(
                        "Native documentation exceeds the tar publication bound"
                    )
            for entry in entries:
                target = tree.joinpath(*PurePosixPath(entry.name).parts)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(entry) as source, target.open(
                            "xb") as output:
                        shutil.copyfileobj(source, output)
            index = tree / "index.html"
            if not index.is_file():
                raise ValueError("Native documentation index.html is missing")
            identities = re.findall(r"main-([0-9a-f]{7,40})(?![0-9a-f])",
                                    index.read_text(encoding="utf-8"))
            if not identities or any(
                    not sha.startswith(prefix) for prefix in identities):
                raise ValueError(
                    "Native documentation entry has another build identity")
        os.replace(tree, destination)


def restore(sha, destination, request=api, transfer=download):
    if destination.is_symlink():
        raise ValueError("Documentation output must not be a symlink")
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Documentation output already exists")
    run, artifact = select_artifact(sha, request)
    with tempfile.TemporaryDirectory(prefix="same-sha-download-") as temp:
        archive = Path(temp) / "artifact.zip"
        transfer(artifact, archive)
        restore_archive(archive, artifact, sha, destination)
    return {
        "source_sha": sha,
        "source_run_id": run["id"],
        "source_run_attempt": run["run_attempt"],
        "artifact_id": artifact["id"],
        "artifact_name": artifact["name"],
        "artifact_digest": artifact["digest"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (os.environ.get("GITHUB_REPOSITORY") != REPOSITORY or
            os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or
            os.environ.get("GITHUB_REF") != "refs/heads/main" or
            os.environ.get("GITHUB_SHA") != args.sha):
        raise ValueError(
            "Same-SHA reuse is restricted to an exact main dispatch")
    print(json.dumps(restore(args.sha, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
