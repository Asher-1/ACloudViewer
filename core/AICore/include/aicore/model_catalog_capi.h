// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared model catalog contract for plugins.  AICore owns the rows; plugins
// only consume this stable, read-only C ABI for presentation and download
// requests.

#pragma once

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum aicore_model_family {
    AICORE_MODEL_FAMILY_DEPTH = 0,
    AICORE_MODEL_FAMILY_DEEPLSD = 1,
    AICORE_MODEL_FAMILY_LIGHTGLUE = 2,
    AICORE_MODEL_FAMILY_GAUSSIAN = 3,
    AICORE_MODEL_FAMILY_ALIKED = 4,
} aicore_model_family;

typedef struct aicore_model_entry {
    const char* filename;
    const char* display_name;
    const char* download_url;
    // Task-specific role.  Depth uses "depth" or "metric"; other families
    // may leave it empty or use their own stable role string.
    const char* role;
    // LightGlue matcher type: 1=SIFT, 2=ALIKED, 0=not applicable.
    int matcher_type;
    // Pinned content identity owned by AICore.  Empty only for an explicitly
    // unpinned development asset; published rows must provide 64 hex chars.
    const char* sha256;
} aicore_model_entry;

/** Number of published entries for a task family. */
AICORE_CAPI int aicore_model_count(aicore_model_family family);

/** Borrowed catalog entry.  The entry and its strings remain valid for the
 *  process lifetime. */
AICORE_CAPI const aicore_model_entry* aicore_model_at(
        aicore_model_family family, int index);

/** Borrowed entry selected by its exact GGUF filename, or NULL. */
AICORE_CAPI const aicore_model_entry* aicore_model_by_filename(
        aicore_model_family family, const char* filename);

#ifdef __cplusplus
}
#endif
