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
    const char* id; /* device string passed to load/warmup: "auto" "cpu" … */
    const char* label; /* human-readable: "Auto (Metal -> CPU)" */
    int is_default;    /* 1 if this entry should be pre-selected in UI */
} aicore_device_info;

/* Backend API ABI version. Increment when this header's binary contract
   changes. */
AICORE_CAPI int aicore_backend_abi_version(void);

/* Number of entries returned by aicore_device_at. Only devices successfully
   discovered at runtime are listed, plus the synthetic "auto" entry. */
AICORE_CAPI int aicore_device_count(void);

/* Device info at 0-based index. Returns NULL when index is out of range.
   Returned pointer is valid for the lifetime of the process. */
AICORE_CAPI const aicore_device_info* aicore_device_at(int index);

/* Human-readable auto-pick order for the current platform, e.g.
   "Metal -> CPU". Returned pointer is a static string. */
AICORE_CAPI const char* aicore_auto_device_order(void);

/* Returns 1 when the requested runtime device is available, otherwise 0.
   device accepts "auto", "cpu", "gpu", or a backend id such as
   "vulkan:1". */
AICORE_CAPI int aicore_device_available(const char* device);

/* ---- backend lifecycle ---- */

/* Register ggml backends and verify the requested device. Call on the UI thread
   before spawning a worker for safe GPU initialization. Returns 0 on success
   and -1 when the requested backend is unavailable. */
AICORE_CAPI int aicore_warmup_backend(const char* device);

/* Last backend error on the calling thread. The returned pointer remains valid
   until the next backend API call on that thread. */
AICORE_CAPI const char* aicore_backend_last_error(void);

/* Returns 1 if the device string names a GPU target (auto, gpu, sycl, cuda,
   metal, opencl, vulkan), 0 for "cpu" or empty/NULL. */
AICORE_CAPI int aicore_is_gpu_device(const char* device);

#ifdef __cplusplus
}
#endif
