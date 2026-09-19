// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Header-only C++ RAII helpers over the process runtime primitives in
// aicore/runtime_capi.h. Qt-free on purpose: every AICore consumer
// (plugins, reconstruction, Python bindings, tests) may use them without
// pulling Qt into the build. Purely inline — nothing here changes the
// libAICore export map or the C ABI surface.
//
// Scope notes (from runtime_capi.h, keep in sync):
//   - aicore_device_task_lock() rejects nested acquisition on the same
//     thread; never nest DeviceTaskLock guards.
//   - aicore_device_task_unlock() must only run after a successful (zero)
//     lock acquisition; isLocked() reports whether the guard owns the
//     queue.

#pragma once

#if defined(__cplusplus)

#include "aicore/runtime_capi.h"

namespace aicore {
namespace runtime {

/** Owns one resolved-device task queue for the current scope.
 *
 *  Construction acquires aicore_device_task_lock(device); destruction
 *  releases it exactly when acquisition succeeded. Check is_locked()
 *  before running inference: a failed acquisition means the queue was
 *  busy or the device request was invalid.
 */
class DeviceTaskLock {
public:
    explicit DeviceTaskLock(const char* device)
        : locked_(device != nullptr && aicore_device_task_lock(device) == 0) {}

    ~DeviceTaskLock() {
        if (locked_) {
            aicore_device_task_unlock();
        }
    }

    DeviceTaskLock(const DeviceTaskLock&) = delete;
    DeviceTaskLock& operator=(const DeviceTaskLock&) = delete;

    /** True only when the queue was acquired for this scope. */
    bool isLocked() const { return locked_; }

private:
    bool locked_;
};

/** Binds a caller-owned cancel token to the calling thread for the
 *  current scope (aicore_cancel_scope_begin/end). The token must outlive
 *  the guard; pass nullptr to make the guard a no-op.
 */
class CancelScope {
public:
    explicit CancelScope(aicore_cancel_token* token) : token_(token) {
        if (token_) {
            aicore_cancel_scope_begin(token_);
        }
    }

    ~CancelScope() {
        if (token_) {
            aicore_cancel_scope_end(token_);
        }
    }

    CancelScope(const CancelScope&) = delete;
    CancelScope& operator=(const CancelScope&) = delete;

private:
    aicore_cancel_token* token_;
};

}  // namespace runtime
}  // namespace aicore

#endif  // defined(__cplusplus)
