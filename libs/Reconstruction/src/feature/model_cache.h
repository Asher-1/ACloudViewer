// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Shared-catalog model resolution and provisioning for the learned feature
// engines (ALIKED, LightGlue). Upstream resolves model files through
// feature/resources.h URIs and util/file.h download caching; the fork's
// equivalent source of truth is the AICore model catalog, whose entries carry
// the published URL and whose SHA-256 values live in aicore/asset_digests.h.

#include <filesystem>
#include <functional>
#include <string>

#include "aicore/model_catalog_capi.h"

namespace colmap {

// Returns the first catalog entry of the family that satisfies `matches`, or
// nullptr.
const aicore_model_entry* FindCatalogModel(
        aicore_model_family family,
        const std::function<bool(const aicore_model_entry*)>& matches);

// Returns the shared-cache path for the entry (cache_dir/filename), or "" if
// the entry or the cache directory is unavailable. `cache_dir_fn` is the
// task-specific AICore cache directory accessor; the returned buffer is freed
// here.
std::string CatalogModelPath(const aicore_model_entry* entry,
                             char* (*cache_dir_fn)(void));

// Ensures that `target` holds the catalog entry with a verified SHA-256,
// downloading it into the shared cache when absent or corrupt. Returns false
// if the entry is unknown, has no pinned digest, or cannot be provisioned.
bool EnsureCachedModel(const aicore_model_entry* entry,
                       const std::filesystem::path& target,
                       const std::string& engine_name);

}  // namespace colmap
