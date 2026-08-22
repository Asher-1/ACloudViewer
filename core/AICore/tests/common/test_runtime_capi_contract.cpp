// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"

namespace {

int Fail(const char* message) {
    std::fprintf(stderr, "runtime contract: %s\n", message);
    return 1;
}

}  // namespace

// The legacy process-wide cancel / inference-lock entry points are marked
// AICORE_LEGACY_API (deprecated). This contract test intentionally verifies
// they still work, so suppress the deprecation warnings for the whole file.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

int main() {
    aicore_cancel_token* outer = aicore_cancel_token_new();
    aicore_cancel_token* inner = aicore_cancel_token_new();
    if (!outer || !inner) return Fail("token allocation failed");

    // A request issued before the worker binds its scope must not be lost.
    aicore_cancel_token_request(outer);
    aicore_cancel_scope_begin(outer);
    if (!aicore_cancel_requested()) return Fail("early cancel was cleared");

    aicore_cancel_token_reset(outer);
    if (aicore_cancel_requested()) return Fail("token reset failed");

    aicore_cancel_scope_begin(inner);
    aicore_cancel_token_request(inner);
    if (!aicore_cancel_requested()) return Fail("inner cancel not visible");
    aicore_cancel_scope_end(inner);
    if (aicore_cancel_requested()) return Fail("outer scope not restored");
    aicore_cancel_scope_end(outer);

    aicore_cancel_begin();
    if (aicore_cancel_requested()) return Fail("legacy begin not reset");
    aicore_cancel_request();
    if (!aicore_cancel_requested()) return Fail("legacy cancel not visible");
    aicore_cancel_end();
    if (aicore_cancel_requested()) return Fail("legacy end not reset");

    if (aicore_device_task_lock("cpu") != 0) {
        return Fail("CPU task queue acquisition failed");
    }
    if (aicore_device_task_try_lock("cpu") == 0) {
        aicore_device_task_unlock();
        return Fail("nested task queue acquisition unexpectedly succeeded");
    }
    aicore_device_task_unlock();
    if (aicore_device_task_try_lock("cpu") != 0) {
        return Fail("CPU task queue was not released");
    }
    aicore_device_task_unlock();

    // A queued task must be cancelable before it obtains the physical device
    // lock. This prevents Stop from being held hostage by another session.
    if (aicore_device_task_lock("cpu") != 0) {
        return Fail("CPU task queue acquisition for cancel test failed");
    }
    aicore_cancel_token* queued = aicore_cancel_token_new();
    if (!queued) return Fail("queued token allocation failed");
    std::atomic<int> queuedResult{-2};
    std::thread waiter([&]() {
        queuedResult.store(aicore_device_task_lock_cancelable("cpu", queued));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    aicore_cancel_token_request(queued);
    waiter.join();
    aicore_device_task_unlock();
    if (queuedResult.load() != 1) {
        aicore_cancel_token_free(queued);
        return Fail("queued task did not observe cancellation");
    }
    aicore_cancel_token_free(queued);

    aicore_cancel_token_free(inner);
    aicore_cancel_token_free(outer);

    // Global serial inference lock: acquire, hold, try, release.
    if (aicore_inference_lock() != 0) return Fail("inference lock failed");
    if (aicore_inference_try_lock() != -1) {
        aicore_inference_unlock();
        return Fail("nested inference lock unexpectedly succeeded");
    }
    aicore_inference_unlock();
    if (aicore_inference_try_lock() != 0) {
        return Fail("inference try-lock failed after unlock");
    }
    aicore_inference_unlock();

    // Device capability bitmask: "cpu" is compute+cancel (no GPU bits).
    // "gpu" may resolve to a real accelerator when present, or fall back to
    // cpu on runners without one — either way the mask must be non-zero.
    // nullptr is treated as "auto" by the runtime (documented contract), so
    // it must also resolve to a non-zero mask.
    if (aicore_device_capabilities("cpu") == 0) {
        return Fail("cpu capabilities unresolved");
    }
    if (aicore_device_capabilities("gpu") == 0) {
        return Fail("gpu capabilities unresolved");
    }
    if (aicore_device_capabilities(nullptr) == 0) {
        return Fail("null (auto) capabilities unresolved");
    }
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
