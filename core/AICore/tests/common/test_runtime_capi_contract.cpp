// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>

#include "aicore/runtime_capi.h"

namespace {

int Fail(const char* message) {
    std::fprintf(stderr, "runtime contract: %s\n", message);
    return 1;
}

}  // namespace

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

    aicore_cancel_token_free(inner);
    aicore_cancel_token_free(outer);
    return 0;
}
