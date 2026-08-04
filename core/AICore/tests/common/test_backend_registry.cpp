// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <string>
#include <vector>

#include "../../src/facedetect/backend.hpp"
#include "ggml_backend_registry.hpp"

namespace {

int Fail(const char* message) {
    std::fprintf(stderr, "backend registry: %s\n", message);
    return 1;
}

}  // namespace

int main() {
    std::string error;
    aicore::runtime::BackendLease first =
            aicore::runtime::acquire_backend_lease("cpu", 1, &error);
    if (!first)
        return Fail(error.empty() ? "first lease failed" : error.c_str());

    aicore::runtime::BackendLease second =
            aicore::runtime::acquire_backend_lease("cpu", 1, &error);
    if (!second)
        return Fail(error.empty() ? "second lease failed" : error.c_str());
    if (first.handle() != second.handle()) {
        return Fail("compatible sessions did not share a backend");
    }

    fd::Backend face_session(1, "cpu");
    if (face_session.handle() != second.handle()) {
        return Fail("FaceDetect session did not share the physical backend");
    }

    first.reset();
    if (!second || second.handle() == nullptr) {
        return Fail("remaining lease lost its backend after peer release");
    }
    const auto lock = second.lock();
    if (!lock.owns_lock())
        return Fail("remaining lease could not lock backend");

    aicore::runtime::BackendLease different_threads =
            aicore::runtime::acquire_backend_lease("cpu", 2, &error);
    if (!different_threads) {
        return Fail(error.empty() ? "different-thread lease failed"
                                  : error.c_str());
    }
    if (second.handle() == different_threads.handle()) {
        return Fail("CPU thread configurations shared one backend");
    }
    const std::vector<aicore::runtime::BackendLease> group = {
            second, different_threads, second};
    const auto group_lock = aicore::runtime::lock_backend_leases(group);
    if (!group_lock.owns_lock()) {
        return Fail("multi-backend lease group could not lock");
    }
    return 0;
}
