// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace colmap {

// Detect if given string is a URI
// (i.e., starts with http://, https://, file://).
bool IsURI(const std::string& uri);

#ifdef COLMAP_DOWNLOAD_ENABLED

// Progress callback function type: (downloaded_bytes, total_bytes) -> void
// If total_bytes is 0, the total size is unknown.
// Using int64_t instead of curl_off_t to avoid including curl headers in the public interface
using DownloadProgressCallback = std::function<void(int64_t downloaded, int64_t total)>;

// Download file from server. Supports any protocol supported by Curl.
// Automatically follows redirects. Returns null in case of failure. Notice that
// this function is not suitable for large files that don't fit easily into
// memory. If such a use case emerges in the future, we want to instead stream
// the downloaded data chunks to disk instead of accumulating them in memory.
// progress_callback: Optional callback function to report download progress.
std::optional<std::string> DownloadFile(
    const std::string& url,
    DownloadProgressCallback progress_callback = nullptr);

// Computes SHA256 digest for given string.
std::string ComputeSHA256(const std::string_view& str);

// Downloads and caches file from given URI. The URI must take the format
// "<url>;<name>;<sha256>". The file will be cached under
// $HOME/.cache/colmap/<sha256>-<name>. File integrity is checked against the
// provided SHA256 digest. Throws exception if the digest does not match.
// Returns the path to the cached file.
// progress_callback: Optional callback function to report download progress.
std::string DownloadAndCacheFile(
    const std::string& uri,
    DownloadProgressCallback progress_callback = nullptr);

// Overwrites the default download cache directory at $HOME/.cache/colmap/.
void OverwriteDownloadCacheDir(std::filesystem::path path);

#endif  // COLMAP_DOWNLOAD_ENABLED

// Check if a URI would be cached (i.e., if the cached file already exists).
// Returns the cache path if the file exists, empty path otherwise.
// Only works for URI format: "<url>;<name>;<sha256>"
std::filesystem::path GetCachedFilePath(const std::string& uri);

// If the given URI is a local filesystem path, returns the input path. If the
// URI matches the "<url>;<name>;<sha256>" format, calls DownloadAndCacheFile().
// Throws runtime exception if download is not supported.
std::filesystem::path MaybeDownloadAndCacheFile(const std::string& uri);

}  // namespace colmap
