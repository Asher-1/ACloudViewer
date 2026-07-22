// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Unified ggml backend abstraction C API.
// Centralises platform-specific device enumeration and backend lifecycle so
// that downstream consumers (plugins, reconstruction, Python bindings) never
// need to include ggml headers or carry #ifdef __APPLE__ logic.

#pragma once

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- device enumeration ---- */

typedef struct {
    const char* id;    /* device string passed to load/warmup: "auto" "cpu" … */
    const char* label; /* human-readable: "Auto (Metal → CUDA → CPU)" */
    int is_default;    /* 1 if this entry should be pre-selected in UI */
} aicore_device_info;

/* Number of entries returned by aicore_device_at.
   Includes "auto", GPU options for the current platform, and "cpu". */
AICORE_CAPI int aicore_device_count(void);

/* Device info at 0-based index. Returns NULL when index is out of range.
   Returned pointer is valid for the lifetime of the process. */
AICORE_CAPI const aicore_device_info* aicore_device_at(int index);

/* Human-readable auto-pick order for the current platform, e.g.
   "Metal → CUDA → CPU".  Returned pointer is a static string. */
AICORE_CAPI const char* aicore_auto_device_order(void);

/* ---- backend lifecycle ---- */

/* Register ggml backends and clear sticky GPU errors on the calling thread.
   Call on the UI thread before spawning a worker for safe CUDA/Metal init.
   Returns 0 on success, non-zero on failure. */
AICORE_CAPI int aicore_warmup_backend(const char* device);

/* Returns 1 if the device string names a GPU target (auto, gpu, cuda, metal,
   opencl, vulkan), 0 if it is "cpu" or empty/NULL. */
AICORE_CAPI int aicore_is_gpu_device(const char* device);

#ifdef __cplusplus
}
#endif
