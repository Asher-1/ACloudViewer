#!/usr/bin/env python3
"""Fail CI when a workflow file would be rejected by GitHub before it runs.

WHY THIS EXISTS
---------------
GitHub validates a workflow file *before* creating any jobs. When validation
fails, the run is recorded with the file path as its name and **zero jobs**:
no logs, no step output, nothing to read. GitHub's only explanation is an
annotation inside the run's HTML page, e.g.

    Invalid workflow file: .github/workflows/documentation.yml#L1
    (Line: 587, Col: 14): Exceeded max expression length 21000

That failure mode is invisible to `actionlint`, to a YAML schema check, and
to `bash -n`, because the file is *valid YAML* -- it just exceeds a limit
GitHub imposes on how a `run:` scalar is treated as an expression template.
This checker enforces the limits that are silent when violated.

Exit 0 when every workflow is safe to submit; 1 otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

# GitHub rejects a single `run:` scalar whose expression expansion is deemed
# to exceed this many characters. The limit applies to the scalar as a whole.
MAX_RUN_SCALAR = 21000

# Headroom below the hard limit. A step within a few hundred characters of the
# ceiling is one small edit away from breaking the entire workflow, and the
# breakage is silent, so fail well before the cliff.
SAFE_RUN_SCALAR = 14000

# Matches POSIX heredoc redirections: `<<EOF`, `<<-EOF`, `<<'EOF'`, `<<"EOF"`.
# Deliberately anchored to a redirection context (start of line or after
# whitespace/`|`/`;`/`&`/`(`) so a `<<` bit-shift inside e.g. `$((a << 2))`
# is not reported.
HEREDOC_RE = re.compile(r"(?:^|[|&;(]\s*|\s)(<<-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2", re.M)


def iter_run_scalars(doc: dict):
    """Yield (job_id, step_index, step_name, length, text) for every run step."""
    jobs = doc.get("jobs") or {}
    if not isinstance(jobs, dict):
        return
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        for index, step in enumerate(job.get("steps") or []):
            if not isinstance(step, dict):
                continue
            body = step.get("run")
            if isinstance(body, str):
                yield job_id, index, step.get("name") or "(unnamed)", len(body), body


def check_file(path: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors fail CI; warnings inform."""
    problems: list[str] = []
    warnings: list[str] = []
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{path}: YAML does not parse: {exc}"], warnings

    if not isinstance(doc, dict):
        return [f"{path}: top level is not a mapping"], warnings

    if not doc.get("jobs"):
        problems.append(f"{path}: no jobs defined")

    for job_id, index, name, length, body in iter_run_scalars(doc):
        location = f"{path}:{job_id}.steps[{index}] {name!r}"
        if length > MAX_RUN_SCALAR:
            problems.append(
                f"{location}: run scalar is {length} chars, over GitHub's "
                f"{MAX_RUN_SCALAR} limit -- move the body into a script file "
                f"and call it (the whole workflow will otherwise be rejected "
                f"with no jobs and no logs)")
        elif length > SAFE_RUN_SCALAR:
            problems.append(
                f"{location}: run scalar is {length} chars, past the safe "
                f"budget of {SAFE_RUN_SCALAR} (limit {MAX_RUN_SCALAR}); "
                f"extract it to docs/automation/scripts/ before it crosses")

        # Heredocs inside block scalars are legal and do work, so this is a
        # warning, not a failure. It cannot fail the build without banning a
        # pattern this repo legitimately uses today.
        heredoc = HEREDOC_RE.search(body)
        if heredoc:
            line_no = body[:heredoc.start()].count("\n") + 1
            warnings.append(
                f"{location}: heredoc on script line {line_no} "
                f"({heredoc.group(0).strip()!r}); an extracted script is "
                f"easier to review and test")

    return problems, warnings


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or sorted(
        Path(".github/workflows").glob("*.yml"))
    if not targets:
        print("check_workflow_limits: no workflow files found", file=sys.stderr)
        return 1

    problems: list[str] = []
    warnings: list[str] = []
    for path in targets:
        errors, warns = check_file(path)
        problems.extend(errors)
        warnings.extend(warns)

    for warning in warnings:
        print(f"⚠️  {warning}", file=sys.stderr)

    for problem in problems:
        print(f"❌ {problem}", file=sys.stderr)

    if problems:
        print(f"\n{len(problems)} workflow problem(s) found.", file=sys.stderr)
        return 1

    print(f"✅ {len(targets)} workflow file(s) within safe limits")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
