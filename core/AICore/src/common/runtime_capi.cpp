// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/runtime_capi.h"

#include <atomic>
#include <mutex>

namespace {

std::mutex g_inference_mutex;
std::atomic<bool> g_cancel_requested{false};
thread_local bool g_inference_lock_held = false;

}  // namespace

AICORE_CAPI void aicore_cancel_begin(void) {
    g_cancel_requested.store(false, std::memory_order_release);
}

AICORE_CAPI void aicore_cancel_end(void) {
    g_cancel_requested.store(false, std::memory_order_release);
}

AICORE_CAPI void aicore_cancel_request(void) {
    g_cancel_requested.store(true, std::memory_order_release);
}

AICORE_CAPI int aicore_cancel_requested(void) {
    return g_cancel_requested.load(std::memory_order_acquire) ? 1 : 0;
}

AICORE_CAPI int aicore_inference_lock(void) {
    g_inference_mutex.lock();
    g_inference_lock_held = true;
    return 0;
}

AICORE_CAPI void aicore_inference_unlock(void) {
    if (!g_inference_lock_held) return;
    g_inference_lock_held = false;
    g_inference_mutex.unlock();
}

AICORE_CAPI int aicore_inference_try_lock(void) {
    if (!g_inference_mutex.try_lock()) return -1;
    g_inference_lock_held = true;
    return 0;
}
