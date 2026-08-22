// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Backend initialization for LightGlue — uses shared AICore ggml utilities.

#include "tasks/lightglue/backend.hpp"

#include <ggml-backend.h>

#include "common/ggml_backend_utils.hpp"
#include "tasks/lightglue/common.hpp"

#if !defined(AICORE_BACKEND_DL)
#include <ggml-cpu.h>
#endif

#include <cstdlib>

namespace aicore {
namespace lightglue {

bool engine_backend::init(const std::string& device_req, int n_threads) {
    release();
    if (n_threads <= 0) {
        n_threads = static_cast<int>(ggml_common::default_cpu_threads());
    }
    // The historical LIGHTGLUE_NTHREADS default override was an env fallback
    // and is removed; explicit threads/options win.
    lease = aicore::runtime::acquire_backend_lease(device_req, n_threads,
                                                   &error);
    if (!lease) return false;
    be = lease.handle();
    device = lease.device();

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!galloc) {
        error = "gallocr init failed";
        release();
        return false;
    }
    LG_LOG("ggml backend initialized: device=%s", device.c_str());
    return true;
}

void engine_backend::release() {
    if (galloc) {
        ggml_gallocr_free(galloc);
        galloc = nullptr;
    }
    if (be) {
        be = nullptr;
    }
    lease.reset();
    device.clear();
    error.clear();
}

bool engine_backend::is_cpu() const { return lease.is_cpu(); }

bool engine_backend::supports_fused_attention() const {
    if (be == nullptr || is_cpu()) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char* name = reg ? ggml_backend_reg_name(reg) : nullptr;
    // ggml-vulkan's fused flash-attention path currently produces invalid
    // LightGlue assignment scores for the ALIKED model. Keep the mathematically
    // equivalent manual attention graph until that backend advertises a
    // numerically conformant implementation.
    return name != nullptr && ggml_common::to_lower(name) != "vulkan";
}

}  // namespace lightglue
}  // namespace aicore
