# ----------------------------------------------------------------------------
# -                        CloudViewer: asher-1.github.io                    -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 asher-1.github.io
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

"""Minimal data download helpers for tests and examples.

This module provides a lightweight replacement for Open3D's `o3d.data` API
that is sufficient for internal tests which expect:

- `cloudViewer.data.DataDescriptor`
- `cloudViewer.data.DownloadDataset`
- `cloudViewer.data.cloudViewer_downloads_prefix`

It supports downloading a single ZIP file and extracting it into an
`extract_dir` directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# Base URL for sample datasets used in tests. Can be overridden via env var.
cloudViewer_downloads_prefix: str = os.environ.get(
    "CLOUDVIEWER_DOWNLOADS_PREFIX",
    # Default to GitHub raw content for the downloads repo.
    "https://github.com/Asher-1/cloudViewer_downloads/raw/main/",
)


@dataclasses.dataclass
class DataDescriptor:
    """Descriptor for a downloadable dataset artifact.

    Attributes:
        url: Fully qualified URL to the resource.
        md5: Optional MD5 checksum for integrity verification.
    """

    url: str
    md5: Optional[str] = None


def _download(url: str, dst_path: Path) -> None:
    """Download a file from URL to dst_path."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst_path.as_posix())


def _assert_md5(path: Path, expected_md5: str) -> None:
    """Validate file against expected MD5, raising on mismatch."""
    hasher = hashlib.md5()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    md5sum = hasher.hexdigest()
    if md5sum.lower() != expected_md5.lower():
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"MD5 mismatch: expected {expected_md5}, got {md5sum} for {path}"
        )


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract a ZIP archive to target_dir; ignore non-zip files."""
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(target_dir)


class DownloadDataset:
    """Downloads and extracts a dataset described by DataDescriptor.

    This creates a temporary extraction directory that can be removed by the
    caller. The directory path is available via the `extract_dir` attribute.
    """

    def __init__(self, prefix: str, data_descriptor: DataDescriptor) -> None:
        self.prefix = prefix
        self.data_descriptor = data_descriptor

        # Use a temporary directory for extraction. Caller cleans it up.
        base_tmp = os.environ.get("CLOUDVIEWER_DATA_TMPDIR", tempfile.gettempdir())
        self.extract_dir = tempfile.mkdtemp(prefix=f"{prefix}-", dir=base_tmp)

        # Download into a temporary file under the extract_dir, then extract.
        tmp_archive_path = Path(self.extract_dir) / "dataset.zip"
        _download(self.data_descriptor.url, tmp_archive_path)
        if self.data_descriptor.md5:
            _assert_md5(tmp_archive_path, self.data_descriptor.md5)
        _extract_archive(tmp_archive_path, Path(self.extract_dir))

    # Optional helper for explicit cleanup if desired by callers.
    def cleanup(self) -> None:
        """Remove the temporary extraction directory, if it exists."""
        if os.path.isdir(self.extract_dir):
            try:
                shutil.rmtree(self.extract_dir)
            except OSError:
                # Ignore cleanup failure in test helper.
                pass

    def get_extract_dir(self) -> str:
        """Return the dataset extraction directory path."""
        return self.extract_dir
