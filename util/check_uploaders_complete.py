#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Report whether every asset-uploading workflow has finished for a commit.

WHY THIS EXISTS
---------------
`npm publish` and `Clean release` both trigger on `workflow_run` for five
uploaders (Windows, MacOS, Ubuntu, Ubuntu CUDA, Ubuntu Wheel). GitHub treats
that list as **OR, not AND**: the run fires once for *each* workflow that
completes. GitHub offers no native "all of these finished" trigger.

The result, measured on commit 1cf1fa06: 20 `npm publish` runs and 20
`Clean release` runs for one commit, each spending ~35s of runner time to
recompute the same answer. Only the final one could do anything useful.

Downstream logic is already idempotent and data-driven (it inspects the
release's asset matrix), so correctness was never at risk -- only cost. This
gate lets 19 of those 20 runs exit in a couple of seconds: "are all five
uploaders done for this commit yet?" is answerable from the API before any
checkout, install or publish work happens.

Absence is treated as completion: GitHub creates every matching run at push
time, so a workflow with no run for this SHA was never triggered (path
filter, disabled workflow) and waiting for it would block forever.

Exit codes:
  0  every uploader has finished  -> caller may proceed
  2  some uploader is still running -> caller should stop cheaply
  1  the API call failed -> caller should surface the error

Usage:
    python3 util/check_uploaders_complete.py                 # current SHA
    python3 util/check_uploaders_complete.py --sha <sha>
    python3 util/check_uploaders_complete.py --repo owner/name --sha <sha>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# The workflows that upload assets to the main-devel release. Kept in sync with
# the `workflow_run.workflows` lists in npm-publish.yml and update-release.yml;
# a name that does not match a real workflow would silently never be seen.
UPLOADERS = ("Windows", "MacOS", "Ubuntu", "Ubuntu CUDA", "Ubuntu Wheel")


def gh_api(path: str) -> dict:
    """Call `gh api` and return parsed JSON. Raises on failure."""
    result = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"gh api {path} failed")
    return json.loads(result.stdout)


class ShaLookupError(RuntimeError):
    """The commit has no visible workflow runs at all."""


def uploader_states(repo: str, sha: str) -> dict[str, str]:
    """Return {workflow name: status|'absent'} for this commit.

    Raises ShaLookupError when the query finds no runs whatsoever.

    That case must not be folded into "absent = nothing to wait for": the
    API matches `head_sha` exactly, so a short SHA (or a SHA the token cannot
    see) returns an empty list, and treating emptiness as "everything
    finished" would let the gate pass unconditionally -- silently disabling
    it. GitHub always creates a run per triggered workflow at push time, so a
    commit with zero runs means the lookup itself is wrong.
    """
    # 100 is the API maximum for one page and comfortably covers ~20 workflows
    # per commit; there is at most one run per workflow per SHA after grouping.
    payload = gh_api(f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100")
    runs = payload.get("workflow_runs", [])

    if not runs:
        raise ShaLookupError(
            f"no workflow runs found for {sha!r}; the API matches head_sha "
            f"exactly, so pass the full 40-character SHA")

    # Later entries are newer; a re-run of a workflow keeps its name, so the
    # last occurrence is the authoritative status for that workflow.
    latest: dict[str, str] = {}
    for run in runs:
        name = run.get("name")
        if name in UPLOADERS:
            latest[name] = run.get("status") or "unknown"

    return {name: latest.get(name, "absent") for name in UPLOADERS}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=
        "Check whether all asset-uploading workflows finished for a commit")
    # $GITHUB_SHA is NOT the triggering commit under `workflow_run`: it points
    # at the head of the default branch. Callers pass the real SHA explicitly
    # (github.event.workflow_run.head_sha); this default only helps for local use.
    parser.add_argument("--sha",
                        default=os.environ.get("UPLOADER_SHA", ""),
                        help="full 40-char commit SHA (required)")
    parser.add_argument("--repo",
                        default=os.environ.get("GITHUB_REPOSITORY", ""),
                        help="owner/name (defaults to $GITHUB_REPOSITORY)")
    parser.add_argument("--quiet",
                        action="store_true",
                        help="print only the verdict, not per-workflow detail")
    args = parser.parse_args(argv[1:])

    if not args.sha or not args.repo:
        print(
            "error: --sha and --repo (or GITHUB_SHA/GITHUB_REPOSITORY) "
            "are required",
            file=sys.stderr)
        return 1

    try:
        states = uploader_states(args.repo, args.sha)
    except ShaLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: could not query workflow runs: {exc}", file=sys.stderr)
        return 1

    pending = [
        name for name, state in states.items()
        if state not in ("completed", "absent")
    ]

    if not args.quiet:
        for name in UPLOADERS:
            state = states[name]
            mark = "ok" if state in ("completed", "absent") else ".."
            note = " (not triggered)" if state == "absent" else ""
            print(f"  {mark:2s} {name}: {state}{note}")

    if pending:
        print(f"waiting on {len(pending)} uploader(s): {', '.join(pending)}")
        return 2

    print("all uploaders finished for this commit")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
