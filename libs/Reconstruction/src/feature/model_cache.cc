// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "feature/model_cache.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <unordered_set>

#include "aicore/asset_digests.h"
#include "util/download.h"

namespace colmap {
namespace {

using aicore::AssetDigestForFile;

std::mutex& ModelCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_set<std::string>& VerifiedModels() {
    static std::unordered_set<std::string> paths;
    return paths;
}

bool VerifyModelFile(const std::filesystem::path& path,
                     const char* expected_digest) {
#ifdef COLMAP_DOWNLOAD_ENABLED
    return expected_digest != nullptr &&
           std::filesystem::is_regular_file(path) &&
           ComputeFileSHA256(path) == expected_digest;
#else
    (void)expected_digest;
    return std::filesystem::is_regular_file(path);
#endif
}

}  // namespace

const aicore_model_entry* FindCatalogModel(
        const aicore_model_family family,
        const std::function<bool(const aicore_model_entry*)>& matches) {
    const int count = aicore_model_count(family);
    for (int index = 0; index < count; ++index) {
        const aicore_model_entry* entry = aicore_model_at(family, index);
        if (entry != nullptr && entry->filename != nullptr && matches(entry)) {
            return entry;
        }
    }
    return nullptr;
}

std::string CatalogModelPath(const aicore_model_entry* entry,
                             char* (*cache_dir_fn)(void)) {
    if (entry == nullptr || entry->filename == nullptr) return "";
    char* cache_dir = cache_dir_fn();
    if (cache_dir == nullptr) return "";
    std::string path =
            (std::filesystem::path(cache_dir) / entry->filename).string();
    std::free(cache_dir);
    return path;
}

bool EnsureCachedModel(const aicore_model_entry* entry,
                       const std::filesystem::path& target,
                       const std::string& engine_name) {
    if (entry == nullptr || entry->filename == nullptr ||
        entry->download_url == nullptr) {
        return false;
    }
    const char* expected_digest = AssetDigestForFile(entry->filename);
    if (expected_digest == nullptr) {
        std::cerr << "ERROR: " << engine_name
                  << " catalog entry has no SHA-256: " << entry->filename
                  << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(ModelCacheMutex());
    if (VerifiedModels().count(target.string()) != 0) {
        return true;
    }
    if (VerifyModelFile(target, expected_digest)) {
        VerifiedModels().insert(target.string());
        return true;
    }
#ifdef COLMAP_DOWNLOAD_ENABLED
    std::error_code error;
    std::filesystem::remove(target, error);
    const std::filesystem::path temporary =
            target.string() + ".partial." +
            std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count());
    const std::string downloaded =
            DownloadAndCacheFile(entry->download_url, temporary);
    if (downloaded.empty() || !VerifyModelFile(temporary, expected_digest)) {
        std::filesystem::remove(temporary, error);
        std::cerr << "ERROR: " << engine_name
                  << " model download or SHA-256 verification failed: "
                  << entry->filename << std::endl;
        return false;
    }
    std::filesystem::rename(temporary, target, error);
    if (error) {
        // A concurrent process may have published the same verified model.
        if (!VerifyModelFile(target, expected_digest)) {
            std::filesystem::remove(temporary, error);
            std::cerr << "ERROR: cannot publish " << engine_name
                      << " model in shared cache: " << target << " ("
                      << error.message() << ")" << std::endl;
            return false;
        }
        std::filesystem::remove(temporary, error);
    }
    VerifiedModels().insert(target.string());
    return true;
#else
    std::cerr << "ERROR: " << engine_name
              << " model is absent from shared cache and reconstruction was "
                 "built with DOWNLOAD_ENABLED=OFF: "
              << target << std::endl;
    return false;
#endif
}

}  // namespace colmap
