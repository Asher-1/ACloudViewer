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

#ifdef AICore_HAS_CVLOG
#include <CVLog.h>
#define FS_LOG(...) CVLog::Print("[FS] " __VA_ARGS__)
#else
#define FS_LOG(...)                                \
    do {                                           \
        std::fprintf(stderr, "[FS] " __VA_ARGS__); \
        std::fprintf(stderr, "\n");                \
    } while (0)
#endif

#if defined(GGML_USE_CUDA) && !defined(GGML_BACKEND_DL)
#include <cuda_runtime.h>
#endif

namespace aicore {
namespace gaussian {

namespace {

void clear_sticky_cuda_errors() {
#if defined(GGML_USE_CUDA) && !defined(GGML_BACKEND_DL)
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

    FS_LOG("init: device_req='%s' parsed_name='%s' want_idx=%d dev_count=%zu",
           device_req.c_str(), name.c_str(), want_idx,
           ggml_backend_dev_count());

    // Auto: platform order from find_auto_gpu_backend (CUDA/OpenCL/Metal, no
    // Vulkan).
    if (name.empty() || name == "auto") {
        std::string resolved;
        if (ggml_backend_t gpu = ggml_common::find_auto_gpu_backend(resolved)) {
            be = gpu;
            device = resolved;
        } else {
            return init("cpu", n_threads);
        }
    } else if (name == "cpu") {
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
    } else if (name == "gpu" || name == "cuda" || name == "opencl" ||
               name == "metal") {
        clear_sticky_cuda_errors();
        be = ggml_common::find_gpu_backend(name, want_idx, device);
        if (!be) {
            error = "no usable '" + name +
                    "' device (built with GGML_USE_CUDA/GGML_USE_OPENCL? "
                    "driver "
                    "present?)";
            return false;
        }
    } else if (name == "vulkan") {
        error = "Vulkan backend is disabled in this build (use "
                "auto|cpu|cuda|opencl)";
        return false;
    } else {
        error = "unknown device '" + device_req +
                "' (want auto|cpu|gpu|cuda|opencl|metal)";
        return false;
    }

    FS_LOG("ggml backend initialized: device=%s", device.c_str());

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
