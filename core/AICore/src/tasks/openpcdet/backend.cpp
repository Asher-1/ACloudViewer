// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AICore adaptation of OpenPCDet-GGML's engine backend bundle. See backend.h.
#include "backend.h"

#include <cstring>
#include <string>

#include "common/ggml_backend_utils.hpp"
#include "ggml-alloc.h"
#include "ggml.h"
#include "models/bev_backbone.h"

namespace openpcdet {

bool engine_backend::init(const std::string& device_req, int n_threads) {
    release();

    ggml_common::load_backends_once();

    const std::string want =
            ggml_common::to_lower(device_req.empty() ? "auto" : device_req);

    if (want != "cpu") {
        ggml_common::GpuBackendGroup group =
                ggml_common::resolve_gpu_group(want);
        if (group.primary()) {
            aicore::runtime::BackendLease lease =
                    aicore::runtime::adopt_backend_lease(
                            group.gpus[0], group.names[0], n_threads);
            group.gpus[0] = nullptr;
            group.release();
            if (lease) {
                gpu_lease_ = std::move(lease);
                be = gpu_lease_.handle();
                device = gpu_lease_.device();
            }
        } else if (want != "auto" && want != "gpu") {
            error = "requested device '" + device_req + "' not found";
            release();
            return false;
        }
    }

    if (!be) {
        std::string cpu_error;
        cpu_lease_ = aicore::runtime::acquire_backend_lease("cpu", n_threads,
                                                            &cpu_error);
        be = cpu_lease_.handle();
        if (!be) {
            error = "cpu backend init failed: " + cpu_error;
            release();
            return false;
        }
        if (device.empty()) device = cpu_lease_.device();
    }

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!galloc) {
        error = "graph allocator init failed";
        release();
        return false;
    }

    // CPU direct-conv work-buffer guard: route large batch convs through
    // im2col+GEMM (see bev_backbone.cpp). GPU keeps the direct path.
    if (is_cpu()) {
        set_conv_direct_max_patches(4096);
    }
    return true;
}

void engine_backend::release() {
    if (galloc) {
        ggml_gallocr_free(galloc);
        galloc = nullptr;
    }
    be = nullptr;
    gpu_lease_.reset();
    cpu_lease_.reset();
    device.clear();
    error.clear();
}

bool engine_backend::is_cpu() const {
    return be && ggml_common::is_cpu_backend(be);
}

bool engine_backend::is_cuda() const {
    // Backend names are "CUDA0", "Vulkan0", "CPU" — match the upstream
    // engine's prefix checks (the resolved lease device string is the same
    // ggml backend name, but stay independent of its case/registry).
    const char* name = be ? ggml_backend_name(be) : nullptr;
    return name && strncmp(name, "CUDA", 4) == 0;
}

bool engine_backend::is_vulkan() const {
    const char* name = be ? ggml_backend_name(be) : nullptr;
    return name && strncmp(name, "Vulkan", 6) == 0;
}

}  // namespace openpcdet
