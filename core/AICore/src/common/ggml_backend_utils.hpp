// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Private ggml backend initialization and discovery utilities.
// Extracted from common patterns in DA3 backend.cpp and free-splatter backend.cpp.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#if !defined(AICORE_BACKEND_DL)
#include "ggml-cpu.h"
#endif

#include "common/ggml_env_bridge.hpp"


#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ggml_common {

// Convert string to lowercase (for device name comparison).
inline std::string to_lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Default CPU threads = physical cores. SMT siblings only add contention for
// matmul-heavy work, so on x86 (logical == 2x physical) we halve -- but ONLY
// when SMT is actually on. ARM / Apple silicon have no SMT.
inline unsigned default_cpu_threads() {
    unsigned logical = std::max(1u, std::thread::hardware_concurrency());
#ifdef __linux__
    std::ifstream smt("/sys/devices/system/cpu/smt/active");
    int on = 0;
    if (smt >> on && on == 1) {
        return std::max(1u, logical / 2);
    }
#endif
    return logical;
}

// Parse a device request string: "cuda:1" -> ("cuda", 1); "vulkan" -> ("vulkan", 0).
inline void parse_device(const std::string& req, std::string& name, int& index) {
    const size_t colon = req.find(':');
    name  = to_lower(colon == std::string::npos ? req : req.substr(0, colon));
    index = (colon != std::string::npos) ? std::atoi(req.c_str() + colon + 1) : 0;
}

// Discover dynamically-loadable backends. For a GGML_BACKEND_DL build this
// loads every libggml-cpu-<isa>.so/dylib from the library directory.
// On macOS app bundles the executable is inside Contents/MacOS/ while backend
// dylibs live alongside libAICore.dylib, so we resolve our own dylib's
// directory and pass it to ggml_backend_load_all_from_path().
// Registers that the backends are loaded so later ggml env overrides (see
// aicore::apply_ggml_env_overrides) can warn about the snapshot semantics.
inline void load_backends_once() {
    static const bool done = [] {
#if defined(AICORE_BACKEND_DL)
        const char* search_dir = nullptr;
        static std::string dir;
        // Resolve directory containing this shared library (libAICore).
#if defined(__APPLE__) || defined(__linux__)
        Dl_info info;
        if (dladdr((void*)&load_backends_once, &info) && info.dli_fname) {
            dir = info.dli_fname;
            auto pos = dir.find_last_of('/');
            if (pos != std::string::npos) {
                dir.resize(pos);
                search_dir = dir.c_str();
            }
        }
#elif defined(_WIN32)
        HMODULE module = nullptr;
        if (GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCWSTR>(&load_backends_once), &module)) {
            std::wstring module_path(32768, L'\0');
            const DWORD size = GetModuleFileNameW(
                    module, module_path.data(),
                    static_cast<DWORD>(module_path.size()));
            if (size > 0 && size < module_path.size()) {
                module_path.resize(size);
                dir = std::filesystem::path(module_path)
                              .parent_path()
                              .u8string();
                search_dir = dir.c_str();
            }
        }
#endif
#ifndef NDEBUG
        if (search_dir) {
            fprintf(stderr, "[AICore] loading ggml backends from '%s'\n",
                    search_dir);
        }
#endif
        ggml_backend_load_all_from_path(search_dir);
#else
        ggml_backend_load_all();
#endif
#ifndef NDEBUG
        fprintf(stderr, "[AICore] %zu ggml device(s) available\n",
                ggml_backend_dev_count());
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            fprintf(stderr, "[AICore] device[%zu]: name='%s' type=%d backend='%s'\n",
                    i, ggml_backend_dev_name(dev),
                    (int)ggml_backend_dev_type(dev),
                    ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
        }
#endif
        aicore::mark_ggml_backends_loaded();
        return true;
    }();
    (void)done;
}

// Set CPU thread count through the backend registry.
// In a DL build the symbol lives in the variant .so, not the linked base;
// in a static build ggml_backend_cpu_set_n_threads is directly available.
inline void set_cpu_threads(ggml_backend_t be, int n_threads) {
#if defined(AICORE_BACKEND_DL)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
    auto set_fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_fn) set_fn(be, n_threads);
#else
    ggml_backend_cpu_set_n_threads(be, n_threads);
#endif
}

// Check if a backend is CPU type.
inline bool is_cpu_backend(ggml_backend_t be) {
    if (!be) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

// Normalize user-facing device names to ggml registry names.
// ggml registers backends under short names (e.g. "Metal" → "MTL") that differ
// from the human-readable names used in our UI and config strings.
inline std::string normalize_backend_name(const std::string& name) {
    if (name == "metal") return "mtl";
    return name;
}

// Find an accelerator backend by name and optional index. Both discrete and
// integrated GPUs are valid (OpenCL commonly exposes only an iGPU).
// "gpu" picks the first available accelerator of any loaded backend.
inline ggml_backend_t find_gpu_backend(const std::string& want_name, int want_idx, std::string& resolved_name) {
    const std::string want_reg = (want_name == "gpu") ? "" : normalize_backend_name(want_name);
    int gpu_idx = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        if (!want_reg.empty()) {
            const char* reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
            if (!reg || to_lower(reg) != want_reg) continue;
        }
        if (gpu_idx++ != want_idx) continue;
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (be) {
            resolved_name = ggml_backend_dev_name(dev);
            return be;
        }
    }
    return nullptr;
}

// First integrated GPU (iGPU), when no discrete GPU is available.
inline ggml_backend_t find_integrated_gpu_backend(std::string& resolved_name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (be) {
            resolved_name = ggml_backend_dev_name(dev);
            return be;
        }
    }
    return nullptr;
}

// Runtime auto-pick follows the build's backend order. When a CUDA backend
// was built (AICORE_CUDA_BUILT), CUDA precedes Vulkan on Linux/Windows so Auto
// and explicit "cuda" agree on the same backend family.
inline const char* const* auto_backend_ids() {
#if defined(__APPLE__)
    static const char* kOrder[] = {"metal", nullptr};
#elif defined(AICORE_CUDA_BUILT)
    static const char* kOrder[] = {"cuda", "vulkan", nullptr};
#else
    static const char* kOrder[] = {"vulkan", nullptr};
#endif
    return kOrder;
}

inline bool auto_includes_backend(const std::string& backend) {
    for (const char* const* p = auto_backend_ids(); *p; ++p) {
        if (backend == *p) return true;
    }
    return false;
}

inline std::string registry_backend_id(const char* reg_name) {
    const std::string name = to_lower(reg_name ? reg_name : "");
    if (name == "mtl" || name == "metal") return "metal";
    if (name.find("cuda") != std::string::npos) return "cuda";
    if (name.find("opencl") != std::string::npos) return "opencl";
    if (name.find("vulkan") != std::string::npos) return "vulkan";
    if (name.find("cpu") != std::string::npos) return "cpu";
    return name;
}

// All GPU backends for a device request. "auto" collects every GPU of the first
// auto-priority backend family (e.g. both cuda:0 and cuda:1); "cuda:1" selects one.
struct GpuBackendGroup {
    std::vector<ggml_backend_t> gpus;
    std::vector<std::string> names;

    ggml_backend_t primary() const {
        return gpus.empty() ? nullptr : gpus.front();
    }
    const std::string& primary_name() const {
        static const std::string kEmpty;
        return names.empty() ? kEmpty : names.front();
    }
    void release() {
        for (ggml_backend_t be : gpus) {
            if (be) ggml_backend_free(be);
        }
        gpus.clear();
        names.clear();
    }
};

inline GpuBackendGroup resolve_gpu_group(const std::string& device_req) {
    load_backends_once();
    GpuBackendGroup group;
    std::string name;
    int want_idx = 0;
    parse_device(device_req, name, want_idx);

    auto append_gpu = [&](ggml_backend_dev_t dev) {
        if (ggml_backend_t be = ggml_backend_dev_init(dev, nullptr)) {
            group.gpus.push_back(be);
            const char* dev_name = ggml_backend_dev_name(dev);
            group.names.push_back(dev_name ? dev_name : "gpu");
        }
    };

    const bool want_all_of_family =
            (name.empty() || name == "auto" || name == "gpu") && want_idx == 0;

    if (want_all_of_family) {
        for (const char* const* p = auto_backend_ids(); *p; ++p) {
            const std::string family = *p;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                const auto type = ggml_backend_dev_type(dev);
                if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
                    type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    continue;
                }
                const char* reg = ggml_backend_reg_name(
                        ggml_backend_dev_backend_reg(dev));
                if (registry_backend_id(reg) != family) continue;
                append_gpu(dev);
            }
            if (!group.gpus.empty()) break;
        }
        if (group.gpus.empty()) {
            std::string resolved;
            if (ggml_backend_t be = find_integrated_gpu_backend(resolved)) {
                group.gpus.push_back(be);
                group.names.push_back(resolved);
            }
        }
    } else if (name == "cpu") {
        return group;
    } else {
        std::string resolved;
        if (ggml_backend_t be = find_gpu_backend(name, want_idx, resolved)) {
            group.gpus.push_back(be);
            group.names.push_back(resolved);
        }
    }
    return group;
}

inline ggml_backend_sched_t new_gpu_sched(
        const std::vector<ggml_backend_t>& gpus,
        ggml_backend_t cpu_backend,
        size_t graph_size) {
    if (gpus.empty()) return nullptr;
    std::vector<ggml_backend_t> backs = gpus;
    if (cpu_backend) backs.push_back(cpu_backend);
    if (backs.size() < 2) return nullptr;
    return ggml_backend_sched_new(backs.data(), nullptr,
                                  static_cast<int>(backs.size()), graph_size,
                                  /*parallel=*/false, /*op_offload=*/true);
}

inline ggml_backend_t find_auto_backend(std::string& resolved_name) {
    for (const char* const* p = auto_backend_ids(); *p; ++p) {
        ggml_backend_t be = find_gpu_backend(*p, 0, resolved_name);
        if (be) return be;
    }
    return nullptr;
}

// Resolve a user/device UI string to a ggml backend registry id understood by
// simple CreateBackend helpers (cpu | cuda | vulkan | metal | …).
inline std::string resolve_device_request(const std::string& device_req) {
    load_backends_once();
    std::string name;
    int want_idx = 0;
    parse_device(device_req, name, want_idx);
    const auto with_index = [want_idx](const std::string& family) {
        return want_idx > 0 ? family + ":" + std::to_string(want_idx)
                            : family;
    };
    if (name.empty() || name == "auto") {
        for (const char* const* p = auto_backend_ids(); *p; ++p) {
            std::string resolved;
            if (find_gpu_backend(*p, want_idx, resolved)) return with_index(*p);
        }
        return "cpu";
    }
    if (name == "gpu") {
        for (const char* const* p = auto_backend_ids(); *p; ++p) {
            std::string resolved;
            if (find_gpu_backend(*p, 0, resolved)) return *p;
        }
        return "cpu";
    }
    if (name == "cpu") return "cpu";
    std::string resolved;
    if (find_gpu_backend(name, want_idx, resolved)) return with_index(name);
    return "cpu";
}

}  // namespace ggml_common
