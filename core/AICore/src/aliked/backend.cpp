// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "backend.h"

#include "../common/ggml_backend_utils.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#if defined(AICORE_VULKAN_ALIKED)
#include "gpu_sync.hpp"
#include "vulkan/vulkan_aliked_dispatch.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>

namespace lightglue::internal {
namespace {

std::string Lower(std::string value) {
    for (char &c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

int DefaultThreadCount() {
    unsigned logical = std::max(1u, std::thread::hardware_concurrency());
    std::ifstream smt("/sys/devices/system/cpu/smt/active");
    int active = 0;
    if (smt >> active && active == 1) {
        logical = std::max(1u, logical / 2);
    }
    return static_cast<int>(logical);
}

bool VulkanSchedEnabledByEnv() {
    const char *env = std::getenv("LIGHTGLUE_ALIKED_VULKAN_SCHED");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return std::strcmp(env, "0") != 0 && Lower(env) != "false" &&
           Lower(env) != "off";
}

void SetCpuThreads(ggml_backend_t backend, int count) {
    ggml_backend_reg_t registry =
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(registry,
                                              "ggml_backend_set_n_threads"));
    if (fn != nullptr) {
        fn(backend, count);
    }
}

void LoadBackends() {
    static const bool loaded = [] {
        ggml_backend_load_all();
        return true;
    }();
    (void)loaded;
}

}  // namespace

bool Backend::Init(const std::string &request, int num_threads) {
    Release();
    LoadBackends();

    const size_t colon = request.find(':');
    const std::string name = Lower(
            colon == std::string::npos ? request : request.substr(0, colon));
    const int wanted_index = colon == std::string::npos
                                     ? 0
                                     : std::atoi(request.c_str() + colon + 1);

    if (name.empty() || name == "cpu") {
        handle = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,
                                           nullptr);
        if (handle == nullptr) {
            error = "failed to initialize the ggml CPU backend";
            return false;
        }
        device = "cpu";
        SetCpuThreads(handle,
                      num_threads > 0 ? num_threads : DefaultThreadCount());

    } else if (name == "gpu" || name == "cuda" || name == "vulkan") {
        int index = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            if (name != "gpu") {
                const char *registry = ggml_backend_reg_name(
                        ggml_backend_dev_backend_reg(dev));
                if (registry == nullptr || Lower(registry) != name) {
                    continue;
                }
            }
            if (index++ != wanted_index) {
                continue;
            }
            handle = ggml_backend_dev_init(dev, nullptr);
            if (handle != nullptr) {
                device = ggml_backend_dev_name(dev);
                break;
            }
        }
        if (handle == nullptr) {
            error = "no usable '" + name +
                    "' backend; enable the corresponding CMake option and "
                    "check the "
                    "driver";
            return false;
        }
    } else {
        error = "unknown device '" + request +
                "' (expected cpu, gpu, cuda, or vulkan)";
        return false;
    }

    allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(handle));
    if (allocator == nullptr) {
        error = "failed to create the ggml graph allocator";
        Release();
        return false;
    }

#if defined(AICORE_VULKAN_ALIKED)
    if (IsVulkan()) {
        lightglue::aliked_internal::ApplyVulkanAlikedPerfDefaults();
    }
#endif

    if (IsVulkan() && VulkanSchedEnabledByEnv()) {
        cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,
                                                nullptr);
        if (cpu_backend == nullptr) {
            error = "failed to initialize CPU backend for Vulkan scheduler";
            Release();
            return false;
        }
        SetCpuThreads(cpu_backend,
                      num_threads > 0 ? num_threads : DefaultThreadCount());
        std::vector<ggml_backend_t> backs = {handle, cpu_backend};
        std::vector<ggml_backend_buffer_type_t> bufts = {
                ggml_backend_get_default_buffer_type(handle),
                ggml_backend_get_default_buffer_type(cpu_backend),
        };
        sched = ggml_backend_sched_new(
                backs.data(), bufts.data(), static_cast<int>(backs.size()),
                /*graph_size=*/512, /*parallel=*/false,
                /*op_offload=*/false);
        if (sched == nullptr) {
            error = "failed to create ggml backend scheduler for Vulkan";
            Release();
            return false;
        }
        use_sched = true;
    }
    return true;
}

bool Backend::SchedRunGraph(ggml_cgraph *graph,
                            const std::function<void()> &set_inputs,
                            std::string *error,
                            const std::function<void()> &before_alloc) {
    if (!HasSched() || graph == nullptr) {
        if (error) {
            *error = "Vulkan scheduler is not initialized";
        }
        return false;
    }
    ggml_backend_sched_reset(sched);
    if (before_alloc) {
        before_alloc();
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        if (error) {
            *error = "ggml_backend_sched_alloc_graph failed";
        }
        return false;
    }
    if (set_inputs) {
        set_inputs();
    }
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        if (error) {
            *error = "ggml_backend_sched_graph_compute failed";
        }
        return false;
    }
    ggml_backend_sched_synchronize(sched);
#if defined(AICORE_VULKAN_ALIKED)
    if (IsVulkan()) {
        lightglue::aliked_internal::VkAlikedQueueIdle(handle);
    }
#endif
    return true;
}

bool Backend::IsCpu() const {
    return handle != nullptr &&
           ggml_backend_dev_type(ggml_backend_get_device(handle)) ==
                   GGML_BACKEND_DEVICE_TYPE_CPU;
}

bool Backend::IsGpu() const {
    return handle != nullptr &&
           ggml_backend_dev_type(ggml_backend_get_device(handle)) ==
                   GGML_BACKEND_DEVICE_TYPE_GPU;
}

bool Backend::IsCuda() const {
    if (!IsGpu() || handle == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(handle);
    const char *registry =
            ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    return registry != nullptr && Lower(registry) == "cuda";
}

bool Backend::IsVulkan() const {
    if (!IsGpu() || handle == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(handle);
    const char *registry =
            ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    return registry != nullptr && Lower(registry) == "vulkan";
}

void Backend::Release() {
#if defined(AICORE_VULKAN_ALIKED)
    if (IsVulkan() && handle != nullptr) {
        lightglue::aliked_internal::VkAlikedQueueIdle(handle);
        ggml_backend_synchronize(handle);
    }
#endif
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
        allocator = nullptr;
    }
    if (cpu_backend != nullptr) {
        ggml_backend_free(cpu_backend);
        cpu_backend = nullptr;
    }
    if (handle != nullptr) {
        ggml_backend_free(handle);
        handle = nullptr;
    }
    use_sched = false;
    device.clear();
    error.clear();
}

}  // namespace lightglue::internal
