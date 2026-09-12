#!/usr/bin/env python3
"""Verify that every AICore model-catalog URL is reachable (HTTP 200).

Catches catalog/release drift before it reaches users: the catalog can be
self-consistent (and its contract tests green) while individual entries still
404 on the release — e.g. the historical rmbg_q8_0.gguf / rmbg_q4_K.gguf
entries that only failed at download time.

Usage: check_catalog_assets.py <catalog_dump_binary>
Exit code 0 iff every URL answers HTTP 200.
"""

import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_USER_AGENT = {"User-Agent": "ACloudViewer-catalog-check"}
_TIMEOUT_SECONDS = 30


def _check(url: str):
    try:
        req = urllib.request.Request(url, method="HEAD", headers=_USER_AGENT)
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            return url, resp.status
    except Exception as exc:  # noqa: BLE001 - report any reachability failure
        return url, f"{type(exc).__name__}: {exc}"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <catalog_dump_binary>", file=sys.stderr)
        return 2
    try:
        dump = subprocess.run(
            [sys.argv[1]], capture_output=True, text=True, check=True, timeout=60
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"catalog dump failed: {exc}", file=sys.stderr)
        return 1
    urls = [line.strip() for line in dump.stdout.splitlines() if line.strip()]
    if not urls:
        print("catalog dump produced no URLs", file=sys.stderr)
        return 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_check, urls))
    failed = [(url, status) for url, status in results if status != 200]
    for url, status in failed:
        print(f"FAIL {status}: {url}", file=sys.stderr)
    print(f"{len(results) - len(failed)}/{len(results)} catalog URLs OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
