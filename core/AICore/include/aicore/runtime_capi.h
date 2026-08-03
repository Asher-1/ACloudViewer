// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Process-wide AICore inference runtime: cooperative cancel and serial
// execution. Plugins call aicore_inference_lock() before inference and
// aicore_cancel_begin() for each task; cancelTask() calls
// aicore_cancel_request().

#pragma once

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- cooperative cancel (checked between graph runs / batch items) ---- */

/** Begin a cancellable inference session; clears any prior cancel flag. */
AICORE_CAPI void aicore_cancel_begin(void);

/** End the current cancellable session (clears cancel flag). */
AICORE_CAPI void aicore_cancel_end(void);

/** Request cooperative cancel for the active session (thread-safe). */
AICORE_CAPI void aicore_cancel_request(void);

/** Returns 1 when cancel was requested for the active session, else 0. */
AICORE_CAPI int aicore_cancel_requested(void);

/* ---- global serial inference lock (one AICore job at a time) ---- */

/** Block until the global inference lock is acquired. Returns 0 on success. */
AICORE_CAPI int aicore_inference_lock(void);

/** Release the global inference lock. No-op if not held by this thread. */
AICORE_CAPI void aicore_inference_unlock(void);

/** Try to acquire the lock without blocking. Returns 0 if acquired, -1 if busy.
 */
AICORE_CAPI int aicore_inference_try_lock(void);

#ifdef __cplusplus
}
#endif
