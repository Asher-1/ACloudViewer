#!/usr/bin/env python3
"""Verify that every AICore model-catalog URL is reachable (HTTP 200).

Catches catalog/release drift before it reaches users: the catalog can be
self-consistent (and its contract tests green) while individual entries still
404 on the release — e.g. the historical rmbg_q8_0.gguf / rmbg_q4_K.gguf
entries that only failed at download time.

Rate limiting is not drift. Model hosts (Hugging Face) throttle anonymous CI
traffic with HTTP 429; a renamed or deleted asset answers 404, never 429.
Transient statuses (429/5xx) are retried with backoff and, when still
throttled, reported as SKIPPED instead of failing the build so that
shared-runner throttling cannot turn the catalog check red. Drift statuses
(404/403/...) remain hard failures.

Usage: check_catalog_assets.py <catalog_dump_binary>
Exit code 0 iff every URL answers HTTP 200 or was skipped as rate-limited.
"""

import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_USER_AGENT = {"User-Agent": "ACloudViewer-catalog-check"}
_TIMEOUT_SECONDS = 30
_ATTEMPTS = 3
_RETRY_SLEEP_SECONDS = (1.0, 2.0)
_MAX_RETRY_AFTER_SECONDS = 5.0
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _check(url: str):
    """Return (url, detail, skipped); detail is None on HTTP 200.

    skipped=True marks a transient rate-limited/unavailable URL that could not
    be verified on this run — reported, but not a catalog failure.
    """
    last_status = None
    for attempt in range(_ATTEMPTS):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=_USER_AGENT)
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                return url, None, False
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code not in _TRANSIENT_STATUS:
                return url, f"HTTPError: HTTP Error {exc.code}: {exc.reason}", False
            last_status = exc.code
            if attempt + 1 < _ATTEMPTS:
                try:
                    retry_after = float(exc.headers.get("Retry-After", 0.0))
                except (TypeError, ValueError):
                    retry_after = 0.0
                base = _RETRY_SLEEP_SECONDS[min(attempt, len(_RETRY_SLEEP_SECONDS) - 1)]
                time.sleep(min(max(retry_after, base), _MAX_RETRY_AFTER_SECONDS))
        except Exception as exc:  # noqa: BLE001 - report any reachability failure
            return url, f"{type(exc).__name__}: {exc}", False
    return (
        url,
        f"HTTP {last_status} after {_ATTEMPTS} attempts (rate-limited)",
        True,
    )


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
    failed = [(url, detail) for url, detail, skip in results if skip is False and detail]
    skipped = [(url, detail) for url, detail, skip in results if skip]
    for url, detail in failed:
        print(f"FAIL {detail}: {url}", file=sys.stderr)
    for url, detail in skipped:
        print(f"SKIP {detail}: {url}", file=sys.stderr)
    ok = len(results) - len(failed) - len(skipped)
    print(
        f"{ok}/{len(results)} catalog URLs OK "
        f"({len(skipped)} skipped as rate-limited)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
