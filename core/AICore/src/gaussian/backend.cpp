// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Backend initialization for FreeSplatter.
// Adapted from free-splatter.cpp/src/backend.cpp to use ggml_common utilities
// and the project's 3rdparty_ggml static build.

#include "backend.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <ggml_common/ggml_backend_utils.hpp>
#include <thread>

#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace aicore {
namespace gaussian {

namespace {

void clear_sticky_cuda_errors() {
#if defined(GGML_USE_CUDA)
    // Other subsystems (BEV remap, SIBR, etc.) may leave the CUDA context in an
    // error state; clear it before ggml touches the device.
    cudaGetLastError();
#endif
}

}  // namespace

bool engine_backend::init(const std::string& device_req, int n_threads) {
    release();
    ggml_common::load_backends_once();

    std::string name;
    int want_idx = 0;
    ggml_common::parse_device(device_req, name, want_idx);

    // Auto: Linux/Win CUDA -> OpenCL -> Vulkan -> CPU; macOS Metal -> Vulkan ->
    // CUDA -> CPU.
    if (name.empty() || name == "auto") {
        static const char* kAutoOrder[] = {"cuda", "opencl", "vulkan", nullptr};
        for (const char* try_name : kAutoOrder) {
            if (init(std::string(try_name), n_threads)) return true;
            release();
        }
        return init("cpu", n_threads);
    }

    if (name == "cpu") {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!be) {
            error = "CPU backend init failed";
            return false;
        }
        device = "cpu";
        if (const char* env = std::getenv("FREE_SPLATTER_NTHREADS")) {
            if (int v = std::atoi(env)) n_threads = v;
        }
        if (n_threads <= 0) {
            n_threads = (int)ggml_common::default_cpu_threads();
        }
        ggml_common::set_cpu_threads(be, n_threads);
    } else if (name == "gpu" || name == "cuda" || name == "vulkan" ||
               name == "opencl") {
        clear_sticky_cuda_errors();
        be = ggml_common::find_gpu_backend(name, want_idx, device);
        if (!be) {
            error = "no usable '" + name +
                    "' device (built with "
                    "GGML_USE_CUDA/GGML_USE_OPENCL/GGML_USE_VULKAN? driver "
                    "present?)";
            return false;
        }
    } else {
        error = "unknown device '" + device_req +
                "' (want auto|cpu|gpu|cuda|opencl|vulkan)";
        return false;
    }

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!galloc) {
        error = "gallocr init failed";
        release();
        return false;
    }
    return true;
}

void engine_backend::release() {
    if (galloc) {
        ggml_gallocr_free(galloc);
        galloc = nullptr;
    }
    if (be) {
        ggml_backend_free(be);
        be = nullptr;
    }
    device.clear();
}

bool engine_backend::is_cpu() const { return ggml_common::is_cpu_backend(be); }

}  // namespace gaussian
}  // namespace aicore
