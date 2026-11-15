// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// MapAnything ggml runtime - backend facade implementation over the shared
// AICore lease registry (template: tasks/gkd/gkd_backend.cpp).

#include "tasks/map_anything/map_anything_backend.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <thread>

#include "common/ggml_backend_utils.hpp"
#include "ggml-backend.h"
#include "ggml.h"
#include "tasks/map_anything/map_anything_common.hpp"

namespace map_anything {

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

}  // namespace

Backend init_backend(const std::string& device_request,
                     int n_threads,
                     std::string* error) {
    Backend be;
    // threads <= 0 means the backend default: the upstream cpp_ggml runtime
    // sizes the CPU threadpool from hardware_concurrency() (the graphs are
    // large ViT transformer stacks, so single-thread CPU inference is ~Nx
    // slower than the upstream reference). GPU leases ignore the count.
    be.n_threads =
            (n_threads > 0)
                    ? n_threads
                    : (int)std::max(1u, std::thread::hardware_concurrency());

    ggml_common::load_backends_once();

    const std::string want =
            to_lower(device_request.empty() ? "auto" : device_request);

    if (want != "cpu") {
        ggml_common::GpuBackendGroup group =
                ggml_common::resolve_gpu_group(want);
        if (group.primary()) {
            be.lease = aicore::runtime::adopt_backend_lease(
                    group.gpus[0], group.names[0], be.n_threads);
            group.gpus[0] = nullptr;
            group.release();
            if (be.lease) {
                be.handle = be.lease.handle();
                be.name = be.lease.device();
                log_info("GPU backend: %s", be.name.c_str());
                return be;
            }
            log_warn("failed to acquire GPU backend lease; using CPU");
        } else if (want != "auto") {
            log_warn("requested device %s not found; falling back to CPU",
                     want.c_str());
        }
    }

    std::string cpu_error;
    be.lease = aicore::runtime::acquire_backend_lease("cpu", be.n_threads,
                                                      &cpu_error);
    be.handle = be.lease.handle();
    if (!be.handle) {
        log_error("CPU backend init failed: %s", cpu_error.c_str());
        if (error) *error = cpu_error;
        return be;
    }
    // The graph builders stage F16 weights differently on the CPU backend;
    // the lease device string is lowercase, the graph test expects "CPU".
    be.name = "CPU";
    return be;
}

void free_backend(Backend& be) {
    be.handle = nullptr;
    be.name.clear();
    be.lease.reset();
}

}  // namespace map_anything
