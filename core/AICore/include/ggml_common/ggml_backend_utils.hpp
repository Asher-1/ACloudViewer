// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared ggml backend initialization and discovery utilities.
// Extracted from common patterns in DA3 backend.cpp and free-splatter backend.cpp.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#if !defined(GGML_BACKEND_DL)
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
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
inline void load_backends_once() {
    static const bool done = [] {
#if defined(GGML_BACKEND_DL) && (defined(__APPLE__) || defined(__linux__))
        const char* search_dir = nullptr;
        // Resolve directory containing this shared library (libAICore).
        Dl_info info;
        if (dladdr((void*)&load_backends_once, &info) && info.dli_fname) {
            static std::string dir;
            dir = info.dli_fname;
            auto pos = dir.find_last_of('/');
            if (pos != std::string::npos) {
                dir.resize(pos);
                search_dir = dir.c_str();
                fprintf(stderr, "[ggml_common] load_backends_once: search_dir='%s'\n", search_dir);
            }
        }
        ggml_backend_load_all_from_path(search_dir);
#else
        ggml_backend_load_all();
#endif
        fprintf(stderr, "[ggml_common] load_backends_once: %zu devices available\n",
                ggml_backend_dev_count());
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            fprintf(stderr, "[ggml_common]   dev[%zu]: name='%s' type=%d reg='%s'\n",
                    i, ggml_backend_dev_name(dev),
                    (int)ggml_backend_dev_type(dev),
                    ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
        }
        return true;
    }();
    (void)done;
}

// Set CPU thread count through the backend registry.
// In a DL build the symbol lives in the variant .so, not the linked base;
// in a static build ggml_backend_cpu_set_n_threads is directly available.
inline void set_cpu_threads(ggml_backend_t be, int n_threads) {
#if defined(GGML_BACKEND_DL)
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
    return be && ggml_backend_dev_type(ggml_backend_get_device(be)) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

// Normalize user-facing device names to ggml registry names.
// ggml registers backends under short names (e.g. "Metal" → "MTL") that differ
// from the human-readable names used in our UI and config strings.
inline std::string normalize_backend_name(const std::string& name) {
    if (name == "metal") return "mtl";
    return name;
}

// Find a GPU backend by name (e.g. "cuda", "metal", "vulkan") and optional index.
// "gpu" picks the first available GPU of any compiled-in backend.
inline ggml_backend_t find_gpu_backend(const std::string& want_name, int want_idx, std::string& resolved_name) {
    const std::string want_reg = (want_name == "gpu") ? "" : normalize_backend_name(want_name);
    int gpu_idx = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
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

// Runtime auto-pick. Order is platform-specific because macOS OpenCL is unusable
// and Metal is the primary GPU path on Apple. Vulkan is not used (opt-in build only).
inline ggml_backend_t find_auto_gpu_backend(std::string& resolved_name) {
#if defined(__APPLE__)
    static const char* kAutoOrder[] = {"metal", "cuda", nullptr};
#else
    static const char* kAutoOrder[] = {"cuda", "opencl", nullptr};
#endif
    for (const char** p = kAutoOrder; *p; ++p) {
        ggml_backend_t be = find_gpu_backend(*p, 0, resolved_name);
        if (be) return be;
    }
    return nullptr;
}

}  // namespace ggml_common
