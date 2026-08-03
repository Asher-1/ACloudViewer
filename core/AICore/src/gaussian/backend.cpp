// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Backend initialization for FreeSplatter.
// Adapted from free-splatter.cpp/src/backend.cpp to use the shared AICore
// backend registry.

#include "backend.hpp"

#include <ggml-backend.h>
#if !defined(AICORE_BACKEND_DL)
#include <ggml-cpu.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

#include "aicore/runtime_capi.h"
#include "common.hpp"
#include "ggml_backend_utils.hpp"

#if defined(AICORE_CUDA_STATIC_LINKED)
#include <cuda_runtime.h>
#endif

namespace aicore {
namespace gaussian {

namespace {

void clear_sticky_cuda_errors() {
#if defined(AICORE_CUDA_STATIC_LINKED)
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
    if (n_threads <= 0) {
        n_threads = static_cast<int>(ggml_common::default_cpu_threads());
    }

    FS_LOG("init: device_req='%s' parsed_name='%s' want_idx=%d dev_count=%zu",
           device_req.c_str(), name.c_str(), want_idx,
           ggml_backend_dev_count());

    // Auto / GPU: resolve all matching devices (multi-GPU when auto).
    if (name.empty() || name == "auto" || name == "gpu" || name == "cuda" ||
        name == "opencl" || name == "metal" || name == "sycl" ||
        name == "vulkan") {
        clear_sticky_cuda_errors();
#ifdef __APPLE__
        const bool disable_metal_opt =
                (name == "metal" || name == "gpu" || name == "auto");
        const char* saved_opt =
                disable_metal_opt ? getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE")
                                  : nullptr;
        const char* saved_fuse = disable_metal_opt
                                         ? getenv("GGML_METAL_FUSION_DISABLE")
                                         : nullptr;
        if (disable_metal_opt) {
            setenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE", "1", 1);
            setenv("GGML_METAL_FUSION_DISABLE", "1", 1);
        }
#endif
        ggml_common::GpuBackendGroup group =
                ggml_common::resolve_gpu_group(device_req);
#ifdef __APPLE__
        if (disable_metal_opt) {
            if (saved_opt)
                setenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE", saved_opt, 1);
            else
                unsetenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE");
            if (saved_fuse)
                setenv("GGML_METAL_FUSION_DISABLE", saved_fuse, 1);
            else
                unsetenv("GGML_METAL_FUSION_DISABLE");
        }
#endif
        if (!group.primary()) {
            if (name.empty() || name == "auto") {
                return init("cpu", n_threads);
            }
            error = "no usable '" + name +
                    "' device (backend built and runtime driver present?)";
            return false;
        }
        gpu_backends = std::move(group.gpus);
        be = gpu_backends.front();
        device = group.primary_name();
        if (gpu_backends.size() > 1) {
            device += " (x" + std::to_string(gpu_backends.size()) + " GPUs)";
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
    } else {
        error = "unknown device '" + device_req +
                "' (want auto|cpu|gpu|sycl|vulkan|cuda|metal)";
        return false;
    }

    FS_LOG("ggml backend initialized: device=%s", device.c_str());

    if (ggml_backend_dev_type(ggml_backend_get_device(be)) !=
        GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,
                                           nullptr);
        if (!cpu_be) {
            error = "CPU fallback init failed";
            release();
            return false;
        }
        ggml_common::set_cpu_threads(cpu_be, n_threads);
        use_sched = true;
    } else {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!galloc) {
            error = "gallocr init failed";
            release();
            return false;
        }
    }
    return true;
}

void engine_backend::release() {
    if (sched) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (galloc) {
        ggml_gallocr_free(galloc);
        galloc = nullptr;
    }
    for (ggml_backend_t gpu : gpu_backends) {
        if (gpu) ggml_backend_free(gpu);
    }
    gpu_backends.clear();
    be = nullptr;
    if (cpu_be) {
        ggml_backend_free(cpu_be);
        cpu_be = nullptr;
    }
    use_sched = false;
    device.clear();
}

bool engine_backend::is_cpu() const { return ggml_common::is_cpu_backend(be); }

bool engine_backend::alloc_graph(ggml_cgraph* graph, size_t graph_size) {
    if (!use_sched) return galloc && ggml_gallocr_alloc_graph(galloc, graph);
    if (!sched) {
        sched = ggml_common::new_gpu_sched(gpu_backends, cpu_be, graph_size);
        if (!sched) return false;
    }
    ggml_backend_sched_reset(sched);
    return ggml_backend_sched_alloc_graph(sched, graph);
}

enum ggml_status engine_backend::compute_graph(ggml_cgraph* graph) {
    try {
        if (aicore_cancel_requested()) return GGML_STATUS_FAILED;
        return use_sched ? ggml_backend_sched_graph_compute(sched, graph)
                         : ggml_backend_graph_compute(be, graph);
    } catch (const std::exception& e) {
        FS_ERR("backend threw exception during graph_compute: %s "
               "(device=%s). Try switching to a different device.",
               e.what(), device.c_str());
        return GGML_STATUS_FAILED;
    } catch (...) {
        FS_ERR("unknown exception during graph_compute (device=%s). "
               "Try switching to a different device.",
               device.c_str());
        return GGML_STATUS_FAILED;
    }
}

}  // namespace gaussian
}  // namespace aicore
