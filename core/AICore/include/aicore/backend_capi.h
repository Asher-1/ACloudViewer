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

#include <stdint.h>

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

enum aicore_backend_capability {
    AICORE_BACKEND_CAP_COMPUTE = 1u << 0,
    AICORE_BACKEND_CAP_GPU = 1u << 1,
    AICORE_BACKEND_CAP_TASK_CANCEL = 1u << 2,
    AICORE_BACKEND_CAP_MULTI_DEVICE = 1u << 3,
};

/* Model/device capability query. This is deliberately model-level rather
 * than a backend-wide guess: callers can disable an unsupported UI choice
 * before allocating a context. Memory figures are conservative working-set
 * estimates for admission control, not a replacement for allocation errors. */
enum aicore_model_kind {
    AICORE_MODEL_DEPTH = 1,
    AICORE_MODEL_GAUSSIAN = 2,
    AICORE_MODEL_ALIKED = 3,
    AICORE_MODEL_LIGHTGLUE = 4,
    AICORE_MODEL_DEEPLSD = 5,
    AICORE_MODEL_FACEDETECT = 6,
    AICORE_MODEL_RFDETR = 7,
    AICORE_MODEL_RMBG = 8,
};

enum aicore_model_capability {
    AICORE_MODEL_CAP_FULL_GRAPH = 1u << 0,
    AICORE_MODEL_CAP_TASK_CANCEL = 1u << 1,
    AICORE_MODEL_CAP_CPU_FALLBACK = 1u << 2,
    AICORE_MODEL_CAP_DYNAMIC_INPUT = 1u << 3,
};

enum aicore_model_precision {
    AICORE_MODEL_PRECISION_FP32 = 1u << 0,
    AICORE_MODEL_PRECISION_QUANTIZED_WEIGHTS = 1u << 1,
};

typedef struct {
    uint32_t struct_size;  /* caller sets sizeof(aicore_model_device_info) */
    uint32_t abi_version;  /* currently 1 */
    uint32_t capabilities; /* enum aicore_model_capability */
    uint32_t precision;    /* enum aicore_model_precision */
    uint64_t min_working_set_bytes;
    uint64_t recommended_working_set_bytes;
    uint32_t max_input_width;  /* 0 = constrained by available memory */
    uint32_t max_input_height; /* 0 = constrained by available memory */
} aicore_model_device_info;

/* Backend API ABI version. Increment when this header's binary contract
   changes. This is the single source of truth: the implementation returns it
   (see aicore_backend_abi_version) and runtime checkers must stay in sync
   (tests/depth/test_depth_capi_contract.cpp, util/check_aicore_runtime.py). */
#define AICORE_BACKEND_ABI_VERSION 2
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

/* Capability bitmask for a concrete device or "auto". Returns 0 when the
   request cannot be resolved. */
AICORE_CAPI unsigned int aicore_device_capabilities(const char* device);

/* Describe whether a complete graph for model can run on device, together
 * with its precision and working-set contract. Returns 0 on success and -1
 * for an unknown model, unavailable device, or invalid output struct. */
AICORE_CAPI int aicore_model_device_info_query(
        enum aicore_model_kind model,
        const char* device,
        aicore_model_device_info* out_info);

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
