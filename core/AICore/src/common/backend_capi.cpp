// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/backend_capi.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ggml_backend_utils.hpp"

#if defined(AICORE_CUDA_STATIC_LINKED)
#include <cuda_runtime.h>
#endif

namespace {

struct OwnedDevice {
    std::string id;
    std::string label;
    aicore_device_info view{};
};

thread_local std::string g_last_error;

std::string canonical_backend_name(const char* raw) {
    const std::string name = ggml_common::to_lower(raw ? raw : "");
    if (name == "mtl" || name == "metal") return "metal";
    if (name.find("cuda") != std::string::npos) return "cuda";
    if (name.find("opencl") != std::string::npos) return "opencl";
    if (name.find("vulkan") != std::string::npos) return "vulkan";
    if (name.find("cpu") != std::string::npos) return "cpu";
    return name;
}

const char* device_type_label(enum ggml_backend_dev_type type) {
    if (type == GGML_BACKEND_DEVICE_TYPE_CPU) return "CPU";
    if (type == GGML_BACKEND_DEVICE_TYPE_ACCEL) return "CPU Accelerator";
    if (type == GGML_BACKEND_DEVICE_TYPE_IGPU) return "Integrated GPU";
    return "GPU";
}

std::string backend_display_name(const std::string& backend) {
    if (backend.empty()) return backend;
    std::string out = backend;
    out[0] = (char)std::toupper((unsigned char)out[0]);
    return out;
}

std::string format_device_label(const std::string& backend,
                                enum ggml_backend_dev_type type,
                                const char* description) {
    std::string label =
            backend_display_name(backend) + " — " + device_type_label(type);
    if (description && description[0] != '\0') {
        label += " (";
        label += description;
        label += ")";
    }
    return label;
}

std::vector<OwnedDevice> build_devices() {
    ggml_common::load_backends_once();

    std::vector<OwnedDevice> result;
    result.reserve(ggml_backend_dev_count() + 1);
    result.push_back({"auto", "Auto", {}});

    std::map<std::string, int> backend_indices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU &&
            type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        const char* reg_name =
                ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        std::string backend = canonical_backend_name(reg_name);
        if (backend.empty()) continue;

        int& index = backend_indices[backend];
        std::string id = backend;
        if (index > 0) id += ":" + std::to_string(index);
        ++index;

        std::string label = format_device_label(
                backend, type, ggml_backend_dev_description(dev));
        result.push_back({std::move(id), std::move(label), {}});
    }

    std::string order;
    for (const char* const* p = ggml_common::auto_backend_ids(); *p; ++p) {
        std::string resolved;
        if (!ggml_common::find_gpu_backend(*p, 0, resolved)) continue;
        if (!order.empty()) order += " -> ";
        order += backend_display_name(*p);
    }
    if (!order.empty()) order += " -> ";
    order += "CPU";
    result.front().label = "Auto (" + order + ")";

    for (size_t i = 0; i < result.size(); ++i) {
        result[i].view.id = result[i].id.c_str();
        result[i].view.label = result[i].label.c_str();
        result[i].view.is_default = (i == 0) ? 1 : 0;
    }
    return result;
}

const std::vector<OwnedDevice>& devices() {
    static const std::vector<OwnedDevice> value = build_devices();
    return value;
}

const std::string& auto_order() {
    static const std::string value = [] {
        const std::string& label = devices().front().label;
        return label.size() > 7 ? label.substr(6, label.size() - 7)
                                : std::string("unavailable");
    }();
    return value;
}

bool has_device(const char* device) {
    ggml_common::load_backends_once();
    const std::string requested =
            ggml_common::to_lower(device && device[0] ? device : "auto");
    if (requested == "cpu") return true;
    if (requested == "auto") {
        std::string resolved;
        if (ggml_common::find_auto_backend(resolved)) return true;
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,
                                         nullptr) != nullptr;
    }
    if (requested == "gpu") {
        std::string resolved;
        return ggml_common::find_gpu_backend("gpu", 0, resolved) != nullptr;
    }

    std::string parsed_name;
    int want_idx = 0;
    ggml_common::parse_device(requested, parsed_name, want_idx);
    if (parsed_name.empty() || parsed_name == "auto") {
        std::string resolved;
        return ggml_common::find_auto_backend(resolved) != nullptr;
    }
    std::string resolved;
    return ggml_common::find_gpu_backend(parsed_name, want_idx, resolved) !=
           nullptr;
}

}  // namespace

AICORE_CAPI int aicore_backend_abi_version(void) { return 1; }

AICORE_CAPI int aicore_device_count(void) {
    try {
        g_last_error.clear();
        return static_cast<int>(devices().size());
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "backend discovery failed";
        return 0;
    }
}

AICORE_CAPI const aicore_device_info* aicore_device_at(int index) {
    try {
        g_last_error.clear();
        const auto& available = devices();
        if (index < 0 || static_cast<size_t>(index) >= available.size()) {
            g_last_error = "device index out of range";
            return nullptr;
        }
        return &available[static_cast<size_t>(index)].view;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        g_last_error = "backend discovery failed";
        return nullptr;
    }
}

AICORE_CAPI const char* aicore_auto_device_order(void) {
    try {
        g_last_error.clear();
        return auto_order().c_str();
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return "unavailable";
    } catch (...) {
        g_last_error = "backend discovery failed";
        return "unavailable";
    }
}

AICORE_CAPI int aicore_device_available(const char* device) {
    try {
        g_last_error.clear();
        return has_device(device) ? 1 : 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "backend discovery failed";
        return 0;
    }
}

AICORE_CAPI int aicore_warmup_backend(const char* device) {
    if (aicore_device_available(device)) {
#if defined(AICORE_CUDA_STATIC_LINKED)
        cudaGetLastError();
#endif
        return 0;
    }
    const std::string requested = device && device[0] ? device : "auto";
    g_last_error = "requested AICore device is unavailable: " + requested;
    return -1;
}

AICORE_CAPI const char* aicore_backend_last_error(void) {
    return g_last_error.c_str();
}

AICORE_CAPI int aicore_is_gpu_device(const char* device) {
    if (!device || device[0] == '\0') return 0;
    const std::string requested = ggml_common::to_lower(device);
    if (requested == "cpu") return 0;
    if (requested == "auto") return aicore_device_available("gpu");
    return requested == "gpu" || requested.rfind("cuda", 0) == 0 ||
           requested.rfind("sycl", 0) == 0 ||
           requested.rfind("metal", 0) == 0 ||
           requested.rfind("opencl", 0) == 0 ||
           requested.rfind("vulkan", 0) == 0;
}
