// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Backend initialization for FreeSplatter.
// Adapted from free-splatter.cpp/src/backend.cpp to use the shared AICore
// backend registry.

#include "tasks/gaussian/backend.hpp"

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
#include "common/ggml_backend_utils.hpp"
#include "common/ggml_env_bridge.hpp"
#include "tasks/gaussian/common.hpp"

#if defined(AICORE_CUDA_STATIC_LINKED)
#include <cuda_runtime.h>
#endif

namespace aicore {
namespace gaussian {

using aicore::apply_ggml_env_overrides;
using aicore::GgmlEnvOverrides;
using aicore::GgmlEnvSnapshot;
using aicore::restore_ggml_env_snapshot;
using aicore::take_ggml_env_snapshot;

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
        // ggml-metal's graph optimizer/fusion mis-handles the FreeSplatter
        // graph; scope GGML_METAL_*_DISABLE to the backend-creation window
        // via the ggml env bridge (the ONLY place AICore touches those
        // variables), then restore the shell's values.
        const bool disable_metal_opt =
                (name == "metal" || name == "gpu" || name == "auto");
        GgmlEnvSnapshot metal_env_snapshot;
        if (disable_metal_opt) {
            metal_env_snapshot =
                    take_ggml_env_snapshot({"GGML_METAL_GRAPH_OPTIMIZE_DISABLE",
                                            "GGML_METAL_FUSION_DISABLE"});
            GgmlEnvOverrides disable;
            disable.metal_graph_optimize_disable = true;
            disable.metal_fusion_disable = true;
            apply_ggml_env_overrides(disable);
        }
#endif
        ggml_common::GpuBackendGroup group =
                ggml_common::resolve_gpu_group(device_req);
#ifdef __APPLE__
        if (disable_metal_opt) {
            restore_ggml_env_snapshot(metal_env_snapshot);
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
        device = group.primary_name();
        for (size_t i = 0; i < group.gpus.size(); ++i) {
            aicore::runtime::BackendLease lease =
                    aicore::runtime::adopt_backend_lease(
                            group.gpus[i], group.names[i], n_threads);
            group.gpus[i] = nullptr;
            if (!lease) {
                error = "failed to acquire GPU backend lease";
                release();
                return false;
            }
            gpu_backends.push_back(lease.handle());
            gpu_leases.push_back(std::move(lease));
        }
        be = gpu_backends.front();
        if (gpu_backends.size() > 1) {
            device += " (x" + std::to_string(gpu_backends.size()) + " GPUs)";
        }
    } else if (name == "cpu") {
        // The historical FREE_SPLATTER_NTHREADS default override was an env
        // fallback and is removed; explicit threads/options win.
        if (n_threads <= 0) {
            n_threads = (int)ggml_common::default_cpu_threads();
        }
        cpu_lease = aicore::runtime::acquire_backend_lease("cpu", n_threads,
                                                           &error);
        if (!cpu_lease) return false;
        be = cpu_lease.handle();
        device = cpu_lease.device();
    } else {
        error = "unknown device '" + device_req +
                "' (want auto|cpu|gpu|sycl|vulkan|cuda|metal)";
        return false;
    }

    FS_LOG("ggml backend initialized: device=%s", device.c_str());

    if (ggml_backend_dev_type(ggml_backend_get_device(be)) !=
        GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_lease = aicore::runtime::acquire_backend_lease("cpu", n_threads,
                                                           &error);
        cpu_be = cpu_lease.handle();
        if (!cpu_be) {
            if (error.empty()) error = "CPU fallback init failed";
            release();
            return false;
        }
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
    {
        const auto backend_lock = lock();
        if (sched) {
            ggml_backend_sched_free(sched);
            sched = nullptr;
        }
        if (galloc) {
            ggml_gallocr_free(galloc);
            galloc = nullptr;
        }
    }
    gpu_backends.clear();
    be = nullptr;
    cpu_be = nullptr;
    gpu_leases.clear();
    cpu_lease.reset();
    use_sched = false;
    device.clear();
}

bool engine_backend::is_cpu() const { return cpu_lease && gpu_leases.empty(); }

aicore::runtime::BackendLeaseLock engine_backend::lock() const {
    std::vector<aicore::runtime::BackendLease> leases = gpu_leases;
    if (cpu_lease) leases.push_back(cpu_lease);
    return aicore::runtime::lock_backend_leases(leases);
}

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
