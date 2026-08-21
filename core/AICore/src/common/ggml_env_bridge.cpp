// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/ggml_env_bridge.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>

#include "common/aicore_log.hpp"

namespace aicore {
namespace {

std::mutex g_mutex;
bool g_backends_loaded = false;

void set_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

void clear_env(const char* key) {
#ifdef _WIN32
    _putenv_s(key, "");
#else
    unsetenv(key);
#endif
}

void apply_bool(const char* key, const std::optional<bool>& value) {
    if (!value.has_value()) return;  // untouched: the user shell wins
    if (*value) {
        set_env(key, "1");
    } else {
        clear_env(key);
    }
}

}  // namespace

void apply_ggml_env_overrides(const GgmlEnvOverrides& overrides) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_backends_loaded) {
        // Existing backend instances keep their snapshot of these variables;
        // only instances created after this point see the new values. The
        // historical setenv code had exactly the same (silent) limitation.
        AICORE_LOG_WARN("[AICore] ",
                        "warning: ggml env overrides applied after the "
                        "backends were loaded; existing instances keep their "
                        "snapshot");
    }
    apply_bool("GGML_VK_DISABLE_F16", overrides.vk_disable_f16);
    apply_bool("GGML_VK_DISABLE_COOPMAT", overrides.vk_disable_coopmat);
    apply_bool("GGML_VK_DISABLE_COOPMAT2", overrides.vk_disable_coopmat2);
    apply_bool("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT",
               overrides.vk_disable_integer_dot_product);
    apply_bool("GGML_METAL_GRAPH_OPTIMIZE_DISABLE",
               overrides.metal_graph_optimize_disable);
    apply_bool("GGML_METAL_FUSION_DISABLE", overrides.metal_fusion_disable);
    apply_bool("RMBG_VK_SCALAR_DIRECT_CONV",
               overrides.rmbg_vk_scalar_direct_conv);
    apply_bool("RMBG_CUDA_CONV_TF32", overrides.rmbg_cuda_conv_tf32);
    if (overrides.rmbg_vk_coopmat_matmul.has_value()) {
        if (overrides.rmbg_vk_coopmat_matmul->empty()) {
            clear_env("RMBG_VK_COOPMAT_MATMUL");
        } else {
            set_env("RMBG_VK_COOPMAT_MATMUL",
                    overrides.rmbg_vk_coopmat_matmul->c_str());
        }
    }
    apply_bool("NVIDIA_TF32_OVERRIDE", overrides.nvidia_tf32_override);
}

GgmlEnvSnapshot take_ggml_env_snapshot(const std::vector<std::string>& keys) {
    std::lock_guard<std::mutex> lock(g_mutex);
    GgmlEnvSnapshot snapshot;
    snapshot.saved.reserve(keys.size());
    for (const std::string& key : keys) {
        const char* value = std::getenv(key.c_str());
        snapshot.saved.emplace_back(
                key, std::make_pair(value != nullptr,
                                    value != nullptr ? std::string(value)
                                                     : std::string()));
    }
    return snapshot;
}

void restore_ggml_env_snapshot(const GgmlEnvSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& entry : snapshot.saved) {
        if (entry.second.first) {
            set_env(entry.first.c_str(), entry.second.second.c_str());
        } else {
            clear_env(entry.first.c_str());
        }
    }
}

void mark_ggml_backends_loaded() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_backends_loaded = true;
}

}  // namespace aicore
