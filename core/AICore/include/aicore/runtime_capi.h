// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Process-wide AICore inference runtime: cooperative cancellation and task
// scheduling. Long-running workers should acquire the device task lock for
// their requested backend and bind a caller-owned cancel token.

#pragma once

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aicore_cancel_token aicore_cancel_token;

#if defined(_MSC_VER)
#define AICORE_LEGACY_API  \
    __declspec(deprecated( \
            "use task-owned cancellation tokens or device_task_lock"))
#elif defined(__GNUC__) || defined(__clang__)
#define AICORE_LEGACY_API      \
    __attribute__((deprecated( \
            "use task-owned cancellation tokens or device_task_lock")))
#else
#define AICORE_LEGACY_API
#endif

/** Allocate an independent task cancellation token. */
AICORE_CAPI aicore_cancel_token* aicore_cancel_token_new(void);
AICORE_CAPI void aicore_cancel_token_free(aicore_cancel_token* token);
AICORE_CAPI void aicore_cancel_token_reset(aicore_cancel_token* token);
AICORE_CAPI void aicore_cancel_token_request(aicore_cancel_token* token);
AICORE_CAPI int aicore_cancel_token_requested(const aicore_cancel_token* token);

/** Bind a token to the calling inference thread without clearing a pending
 *  request. Backend checks use the innermost token until matching scope_end. */
AICORE_CAPI void aicore_cancel_scope_begin(aicore_cancel_token* token);
AICORE_CAPI void aicore_cancel_scope_end(aicore_cancel_token* token);

/* ---- cooperative cancel (checked between graph runs / batch items) ---- */

/** Legacy process-wide token compatibility API. New tasks should own a token.
 */
AICORE_LEGACY_API AICORE_CAPI void aicore_cancel_begin(void);

/** End the current cancellable session (clears cancel flag). */
AICORE_LEGACY_API AICORE_CAPI void aicore_cancel_end(void);

/** Request cooperative cancel for the active session (thread-safe). */
AICORE_LEGACY_API AICORE_CAPI void aicore_cancel_request(void);

/** Returns 1 when cancel was requested for the active session, else 0. */
AICORE_CAPI int aicore_cancel_requested(void);

/* ---- global serial inference lock (one AICore job at a time) ---- */

/** Block until the global inference lock is acquired. Returns 0 on success. */
AICORE_LEGACY_API AICORE_CAPI int aicore_inference_lock(void);

/** Release the global inference lock. No-op if not held by this thread. */
AICORE_LEGACY_API AICORE_CAPI void aicore_inference_unlock(void);

/** Try to acquire the lock without blocking. Returns 0 if acquired, -1 if busy.
 */
AICORE_LEGACY_API AICORE_CAPI int aicore_inference_try_lock(void);

/* ---- device task scheduling ---- */

/** Acquire the queue for a resolved runtime device. "auto" is resolved once
 * when the task starts, so independent CPU and GPU jobs are no longer forced
 * through one process-wide mutex. Nested calls on the same thread are not
 * supported. */
AICORE_CAPI int aicore_device_task_lock(const char* device);

/**
 * Acquire a resolved-device queue while observing \p token. Returns 0 after
 * acquisition, 1 when the token was cancelled while waiting, and -1 on an
 * invalid nested acquisition. The caller owns the matching unlock only after
 * a zero return value.
 */
AICORE_CAPI int aicore_device_task_lock_cancelable(
        const char* device, const aicore_cancel_token* token);

/** Try to acquire a device queue. Returns 0 if acquired, -1 when busy. */
AICORE_CAPI int aicore_device_task_try_lock(const char* device);

/** Release the device queue acquired by this thread. */
AICORE_CAPI void aicore_device_task_unlock(void);

#ifdef __cplusplus
}
#endif

#undef AICORE_LEGACY_API
