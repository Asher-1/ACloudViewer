#!/usr/bin/env python3
"""Fetch the prior download catalogue without mistaking outages for absence."""
from __future__ import annotations

import argparse
from http.client import HTTPException
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "Asher-1/ACloudViewer"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/main-devel"
PAGES_URL = "https://asher-1.github.io/ACloudViewer/downloads_data.json"
RETRIES = 3


class FetchError(RuntimeError):
    pass


def _request(url: str,
             token: str,
             *,
             opener: Callable = urlopen,
             sleep: Callable = time.sleep) -> tuple[int, bytes]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "acloudviewer-release-scanner"
    }
    last_error = "no response"
    for attempt in range(RETRIES):
        request = Request(url, headers=headers)
        # urllib copies ordinary headers on redirects, including Authorization.
        # Authenticate only the original GitHub API request, never its redirect.
        if token:
            request.add_unredirected_header("Authorization", f"Bearer {token}")
        try:
            with opener(request, timeout=30) as response:
                status = response.status
                body = response.read()
        except HTTPError as error:
            status, body = error.code, b""
            error.close()
        except (URLError, TimeoutError, OSError, HTTPException) as error:
            status, body = 0, b""
            last_error = str(error)
        else:
            last_error = f"HTTP {status}"

        if status == 200 or status == 404:
            return status, body
        if status not in (0, 429) and not (500 <= status <= 599):
            raise FetchError(f"GET {url} returned non-retryable HTTP {status}")
        last_error = f"GET {url} returned HTTP {status}" if status else f"GET {url} failed: {last_error}"
        if attempt + 1 < RETRIES:
            sleep(2**attempt)
    raise FetchError(f"GET {url} failed after {RETRIES} attempts: {last_error}")


def _validate(raw: bytes, source: str) -> dict:
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FetchError(f"{source} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise FetchError(f"{source} must be a JSON object")
    metadata, downloads = data.get("version_metadata"), data.get(
        "download_data")
    if not isinstance(metadata, list) or not isinstance(downloads, dict):
        raise FetchError(
            f"{source} must contain version_metadata[] and download_data{{}}")
    seen = set()
    for row in metadata:
        if not isinstance(row, dict) or not isinstance(row.get("value"),
                                                       str) or not row["value"]:
            raise FetchError(
                f"{source} contains a malformed version_metadata entry")
        if row["value"] in seen:
            raise FetchError(
                f"{source} contains a duplicate version {row['value']}")
        seen.add(row["value"])
    if set(downloads) != seen or any(
            not isinstance(value, dict) for value in downloads.values()):
        raise FetchError(
            f"{source} has mismatched versions or malformed download_data entries"
        )
    return data


def fetch_existing(output: Path,
                   *,
                   token: str = "",
                   request: Callable = _request,
                   run: Callable = subprocess.run,
                   sleep: Callable = time.sleep) -> str:
    """Atomically install valid prior data; initialize only if both sources are absent."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    api_status, api_body = request(API_URL, token)
    raw = None
    source = ""
    if api_status == 200:
        try:
            release = json.loads(api_body)
            assets = release["assets"]
            if not isinstance(assets, list) or any(
                    not isinstance(x, dict) or
                    not isinstance(x.get("name"), str) for x in assets):
                raise ValueError("assets is not a list of named assets")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError,
                ValueError) as error:
            raise FetchError(
                f"main-devel release response is malformed: {error}") from error
        matches = sum(
            asset["name"] == "downloads_data.json" for asset in assets)
        if matches > 1:
            raise FetchError(
                "main-devel has duplicate downloads_data.json assets")
        if matches == 1:
            with tempfile.TemporaryDirectory(prefix="downloads-fetch-") as temp:
                for attempt in range(RETRIES):
                    # Failed gh transfers can leave a partial file. Each retry
                    # gets its own directory, so it never reuses partial JSON.
                    directory = Path(temp) / str(attempt)
                    directory.mkdir()
                    path = directory / "downloads_data.json"
                    try:
                        result = run([
                            "gh", "release", "download", "main-devel",
                            "--pattern", "downloads_data.json", "--dir",
                            str(directory)
                        ],
                                     capture_output=True,
                                     text=True,
                                     timeout=60)
                    except subprocess.TimeoutExpired:
                        result = None
                    except OSError as error:
                        raise FetchError(
                            "Could not execute gh release download") from error
                    if result is not None and result.returncode == 0 and path.is_file(
                    ):
                        raw = path.read_bytes()
                        break
                    if attempt + 1 == RETRIES:
                        raise FetchError(
                            "main-devel declares downloads_data.json but its download failed after 3 attempts"
                        )
                    sleep(2**attempt)
            source = "main-devel release asset"
    elif api_status != 404:
        raise FetchError(
            f"main-devel release lookup returned unexpected HTTP {api_status}")

    if raw is None:
        page_status, page_body = request(PAGES_URL, "")
        if page_status == 200:
            raw, source = page_body, "existing GitHub Pages file"
        elif page_status == 404:
            raw = b'{"generated_at":"","version_metadata":[],"download_data":{}}\n'
            source = "initial empty catalogue (release asset absent and Pages returned 404)"
        else:
            raise FetchError(
                f"GitHub Pages fallback returned unexpected HTTP {page_status}")

    _validate(raw, source)
    fd, staging = tempfile.mkstemp(prefix=output.name + ".",
                                   suffix=".tmp",
                                   dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, output)
    finally:
        try:
            os.unlink(staging)
        except FileNotFoundError:
            pass
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        source = fetch_existing(args.output,
                                token=os.environ.get("GH_TOKEN", "") or
                                os.environ.get("GITHUB_TOKEN", ""))
    except FetchError as error:
        print(f"download catalogue fetch failed: {error}", file=sys.stderr)
        return 1
    print(f"Using {source}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
