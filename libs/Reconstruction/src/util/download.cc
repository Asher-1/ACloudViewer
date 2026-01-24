// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "util/download.h"

#ifdef COLMAP_DOWNLOAD_ENABLED

#include "util/logging.h"
#include "util/misc.h"
#include "util/string.h"

// BoringSSL is currently compatible with OpenSSL 1.1.0
#define OPENSSL_API_COMPAT 10100
// clang-format off
// Must include openssl before curl to build on Windows.
#include <openssl/sha.h>

// https://stackoverflow.com/a/41873190/1255535
#ifdef _MSC_VER
#pragma comment(lib, "wldap32.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "Ws2_32.lib")
#define USE_SSLEAY
#define USE_OPENSSL
#endif

// CURL_STATICLIB is only needed on Windows when using static curl library
#ifdef _MSC_VER
#define CURL_STATICLIB
#endif

#include <curl/curl.h>
#include <curl/easy.h>
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>
#include <vector>

#ifndef _MSC_VER
extern "C" {
extern char** environ;
}
#endif

namespace colmap {

namespace {

size_t WriteCurlData(char* buf,
                     size_t size,
                     size_t nmemb,
                     std::ostringstream* data_stream) {
  *data_stream << std::string_view(buf, size * nmemb);
  return size * nmemb;
}

// Structure to hold progress callback and cancel flag
struct ProgressData {
  DownloadProgressCallback callback;
  bool* canceled;
};

int CurlProgressCallback(void* clientp,
                         curl_off_t dltotal,
                         curl_off_t dlnow,
                         curl_off_t /* ultotal */,
                         curl_off_t /* ulnow */) {
  ProgressData* progress_data = static_cast<ProgressData*>(clientp);
  
  // Check if download was canceled
  if (progress_data && progress_data->canceled && *progress_data->canceled) {
    return 1;  // Return non-zero to cancel download
  }
  
  if (progress_data && progress_data->callback) {
    // Call the user-provided callback (convert curl_off_t to int64_t)
    progress_data->callback(static_cast<int64_t>(dlnow), static_cast<int64_t>(dltotal));
    
    // Check again after callback (user might have canceled)
    if (progress_data->canceled && *progress_data->canceled) {
      return 1;  // Return non-zero to cancel download
    }
  } else {
    // Default behavior: print progress to stdout (for non-GUI applications)
    if (dltotal <= 0) {
      return 0;  // Continue download
    }

    const double percent = 100.0 * static_cast<double>(dlnow) / static_cast<double>(dltotal);
    const double dlnow_mb = static_cast<double>(dlnow) / (1024.0 * 1024.0);
    const double dltotal_mb = static_cast<double>(dltotal) / (1024.0 * 1024.0);

    // Print progress bar: [====>    ] 45.2% (12.3/27.2 MB)
    const int bar_width = 40;
    const int filled = static_cast<int>(bar_width * percent / 100.0);
    
    // Use \r to overwrite the same line (single-line progress bar)
    // Use std::cout for normal log output
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
      if (i < filled) {
        std::cout << "=";
      } else if (i == filled) {
        std::cout << ">";
      } else {
        std::cout << " ";
      }
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << percent << "% ("
              << dlnow_mb << "/" << dltotal_mb << " MB)" << std::flush;
  }
  
  return 0;  // Continue download
}

struct CurlHandle {
  CurlHandle() {
    static std::once_flag global_curl_init;
    std::call_once(global_curl_init, []() {
      curl_global_init(CURL_GLOBAL_ALL);
    });

    ptr = curl_easy_init();
  }

  ~CurlHandle() { curl_easy_cleanup(ptr); }

  CURL* ptr;
};

std::string SHA256DigestToHex(const unsigned char* digest, size_t length) {
  std::ostringstream hex;
  for (size_t i = 0; i < length; ++i) {
    hex << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned int>(digest[i]);
  }
  return hex.str();
}

std::optional<std::string> GetEnvSafe(const char* key) {
#ifdef _MSC_VER
  size_t size = 0;
  getenv_s(&size, nullptr, 0, key);
  if (size == 0) {
    return std::nullopt;
  }
  std::string value(size, ' ');
  getenv_s(&size, value.data(), size, key);
  // getenv_s returns a null-terminated string, so we need to remove the
  // trailing null character in our std::string.
  CHECK_EQ(value.back(), '\0');
  return value.substr(0, size - 1);
#else
  // Non-MSVC replacement for std::getenv_s. The safe variant
  // std::getenv_s is not available on all platforms, unfortunately.
  // Stores environment variables as: "key1=value1", "key2=value2", ..., null
  char** env = environ;
  const std::string_view key_sv(key);
  for (; *env; ++env) {
    const std::string_view key_value(*env);
    if (key_sv.size() < key_value.size() &&
        key_value.substr(0, key_sv.size()) == key_sv &&
        key_value[key_sv.size()] == '=') {
      return std::string(key_value.substr(
          key_sv.size() + 1, key_value.size() - key_sv.size() - 1));
    }
  }
  return std::nullopt;
#endif
}

}  // namespace

std::optional<std::string> DownloadFile(
    const std::string& url,
    DownloadProgressCallback progress_callback) {
  std::cout << "Downloading file from: " << url << std::endl;

  CurlHandle handle;
  CHECK_NOTNULL(handle.ptr);

  curl_easy_setopt(handle.ptr, CURLOPT_URL, url.c_str());
  curl_easy_setopt(handle.ptr, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(handle.ptr, CURLOPT_WRITEFUNCTION, &WriteCurlData);
  std::ostringstream data_stream;
  curl_easy_setopt(handle.ptr, CURLOPT_WRITEDATA, &data_stream);

  // Enable progress callback for download progress display
  bool canceled = false;
  ProgressData progress_data{progress_callback, &canceled};
  curl_easy_setopt(handle.ptr, CURLOPT_XFERINFOFUNCTION, CurlProgressCallback);
  curl_easy_setopt(handle.ptr, CURLOPT_XFERINFODATA, &progress_data);
  curl_easy_setopt(handle.ptr, CURLOPT_NOPROGRESS, 0L);

  // Respect SSL_CERT_FILE and SSL_CERT_DIR environment variables for
  // cross-distribution compatibility (e.g., Ubuntu vs RHEL-based systems).
  const std::optional<std::string> ssl_cert_file = GetEnvSafe("SSL_CERT_FILE");
  if (ssl_cert_file.has_value() && !ssl_cert_file->empty()) {
    VLOG(2) << "Using SSL_CERT_FILE: " << *ssl_cert_file;
    curl_easy_setopt(handle.ptr, CURLOPT_CAINFO, ssl_cert_file->c_str());
  }
  const std::optional<std::string> ssl_cert_dir = GetEnvSafe("SSL_CERT_DIR");
  if (ssl_cert_dir.has_value() && !ssl_cert_dir->empty()) {
    VLOG(2) << "Using SSL_CERT_DIR: " << *ssl_cert_dir;
    curl_easy_setopt(handle.ptr, CURLOPT_CAPATH, ssl_cert_dir->c_str());
  }

  const CURLcode code = curl_easy_perform(handle.ptr);
  
  // Print newline after progress bar (only if using default callback)
  // Use std::cout for normal log output
  if (!progress_callback) {
    std::cout << std::endl;  // Move to next line after progress bar
  }
  
  // Check if download was canceled
  if (canceled || code == CURLE_ABORTED_BY_CALLBACK) {
    std::cerr << "WARNING: Download was canceled by user" << std::endl;
    return std::nullopt;
  }
  
  if (code != CURLE_OK) {
    if (code == CURLE_SSL_CACERT_BADFILE || code == CURLE_SSL_CACERT) {
      std::cerr << "ERROR: Curl SSL certificate error (code " << code
                << "). Try setting SSL_CERT_FILE to point to your system's "
                   "CA certificate bundle (e.g., "
                   "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt on "
                   "Ubuntu/Debian)." << std::endl;
    } else {
      std::cerr << "ERROR: Curl failed to perform request with code: " << code
                << " (" << curl_easy_strerror(code) << ")" << std::endl;
    }
    return std::nullopt;
  }

  long response_code = 0;
  curl_easy_getinfo(handle.ptr, CURLINFO_RESPONSE_CODE, &response_code);
  if (response_code != 0 && (response_code < 200 || response_code >= 300)) {
    std::cerr << "ERROR: Request failed with status: " << response_code << std::endl;
    return std::nullopt;
  }

  std::string data_str = data_stream.str();
  std::cout << "Downloaded " << data_str.size() << " bytes ("
            << std::fixed << std::setprecision(2)
            << static_cast<double>(data_str.size()) / (1024.0 * 1024.0) << " MB)" << std::endl;

  return data_str;
}

std::string ComputeSHA256(const std::string_view& str) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(str.data()), str.size(), digest);
  return SHA256DigestToHex(digest, SHA256_DIGEST_LENGTH);
}

namespace {

std::optional<std::filesystem::path> download_cache_dir_overwrite;

std::optional<std::filesystem::path> HomeDir() {
#ifdef _MSC_VER
  std::optional<std::string> userprofile = GetEnvSafe("USERPROFILE");
  if (userprofile.has_value()) {
    return *userprofile;
  }
  const std::optional<std::string> homedrive = GetEnvSafe("HOMEDRIVE");
  const std::optional<std::string> homepath = GetEnvSafe("HOMEPATH");
  if (!homedrive.has_value() || !homepath.has_value()) {
    return std::nullopt;
  }
  return std::filesystem::path(*homedrive) / std::filesystem::path(*homepath);
#else
  std::optional<std::string> home = GetEnvSafe("HOME");
  if (!home.has_value()) {
    return std::nullopt;
  }
  return *home;
#endif
}

}  // namespace

std::string DownloadAndCacheFile(
    const std::string& uri,
    DownloadProgressCallback progress_callback) {
  const std::vector<std::string> parts = StringSplit(uri, ";");
  CHECK_EQ(parts.size(), 3)
      << "Invalid URI format. Expected: <url>;<name>;<sha256>";

  const std::string& url = parts[0];
  CHECK(!url.empty());
  const std::string& name = parts[1];
  CHECK(!name.empty());
  const std::string& sha256 = parts[2];
  CHECK_EQ(sha256.size(), 64);

  std::filesystem::path download_cache_dir;
  if (download_cache_dir_overwrite.has_value()) {
    download_cache_dir = *download_cache_dir_overwrite;
  } else {
    const std::optional<std::filesystem::path> home_dir = HomeDir();
    CHECK(home_dir.has_value());
    download_cache_dir = *home_dir / ".cache" / "cloudViewer";
  }

  if (!std::filesystem::exists(download_cache_dir)) {
    VLOG(2) << "Creating download cache directory: " << download_cache_dir;
    CHECK(std::filesystem::create_directories(download_cache_dir));
  }

  const auto path = download_cache_dir / (sha256 + "-" + name);

  if (std::filesystem::exists(path)) {
    std::cout << "File already cached. Using cached file at: " << path << std::endl;
    std::vector<char> blob;
    ReadBinaryBlob(path.string(), &blob);
    CHECK_EQ(ComputeSHA256(std::string_view(blob.data(), blob.size())), sha256)
        << "The cached file does not match the expected SHA256";
  } else {
    std::cout << "File not found in cache. Downloading from: " << url << std::endl;
    std::cout << "Cache directory: " << download_cache_dir << std::endl;
    std::cout << "Cache file path: " << path << std::endl;
    const std::optional<std::string> blob = DownloadFile(url, progress_callback);
    CHECK(blob.has_value()) << "Failed to download file";
    CHECK_EQ(ComputeSHA256(std::string_view(blob->data(), blob->size())), sha256)
        << "The downloaded file does not match the expected SHA256";
    std::cout << "Caching file at: " << path << std::endl;
    std::vector<char> blob_vec(blob->begin(), blob->end());
    WriteBinaryBlob(path.string(), blob_vec);
    std::cout << "File successfully cached at: " << path << std::endl;
  }

  return path.string();
}

void OverwriteDownloadCacheDir(std::filesystem::path path) {
  download_cache_dir_overwrite = std::move(path);
}

bool IsURI(const std::string& uri) {
  return StringStartsWith(uri, "http://") ||
         StringStartsWith(uri, "https://") || StringStartsWith(uri, "file://");
}

std::filesystem::path GetCachedFilePath(const std::string& uri) {
  if (!IsURI(uri)) {
    return std::filesystem::path();  // Not a URI, return empty
  }
  
  const std::vector<std::string> parts = StringSplit(uri, ";");
  if (parts.size() != 3) {
    return std::filesystem::path();  // Invalid URI format
  }

  const std::string& sha256 = parts[2];
  if (sha256.size() != 64) {
    return std::filesystem::path();  // Invalid SHA256
  }

  const std::string& name = parts[1];
  if (name.empty()) {
    return std::filesystem::path();
  }

  std::filesystem::path download_cache_dir;
#ifdef COLMAP_DOWNLOAD_ENABLED
  if (download_cache_dir_overwrite.has_value()) {
    download_cache_dir = *download_cache_dir_overwrite;
  } else {
#endif
    const std::optional<std::filesystem::path> home_dir = HomeDir();
    if (!home_dir.has_value()) {
      return std::filesystem::path();
    }
    download_cache_dir = *home_dir / ".cache" / "cloudViewer";
#ifdef COLMAP_DOWNLOAD_ENABLED
  }
#endif

  const auto path = download_cache_dir / (sha256 + "-" + name);
  if (std::filesystem::exists(path)) {
    return path;
  }
  
  return std::filesystem::path();  // File doesn't exist
}

std::filesystem::path MaybeDownloadAndCacheFile(const std::string& uri) {
  if (!IsURI(uri)) {
    return uri;
  }
#ifdef COLMAP_DOWNLOAD_ENABLED
  return DownloadAndCacheFile(uri);
#else
  throw std::runtime_error("COLMAP was compiled without download support");
#endif
}

}  // namespace colmap

#endif  // COLMAP_DOWNLOAD_ENABLED
