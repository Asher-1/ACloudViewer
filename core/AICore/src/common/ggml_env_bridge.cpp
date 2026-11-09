// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/ggml_env_bridge.hpp"

#include <FileSystem.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "common/aicore_log.hpp"
#include "ggml.h"

namespace aicore {
namespace {

std::mutex g_mutex;
bool g_backends_loaded = false;

void set_env(const char* key, const char* value) {
    cloudViewer::utility::filesystem::SetEnvVar(key, value);
}

void clear_env(const char* key) {
    cloudViewer::utility::filesystem::SetEnvVar(key, {});
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
    apply_bool("GGML_VK_DISABLE_DOT2", overrides.vk_disable_dot2);
    apply_bool("GGML_METAL_GRAPH_OPTIMIZE_DISABLE",
               overrides.metal_graph_optimize_disable);
    apply_bool("GGML_METAL_FUSION_DISABLE", overrides.metal_fusion_disable);
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

void apply_vulkan_runtime_defaults() {
    std::lock_guard<std::mutex> lock(g_mutex);
    // Shell wins: an explicit GGML_VK_ALLOW_SYSMEM_FALLBACK setting (either
    // direction) is the user's choice and is left untouched.
    const char* shell = std::getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
    if (shell != nullptr) {
        AICORE_LOG_INFO(
                "[AICore] ",
                "GGML_VK_ALLOW_SYSMEM_FALLBACK already set by the shell "
                "('%s') — keeping it\n",
                shell);
        return;
    }
    set_env("GGML_VK_ALLOW_SYSMEM_FALLBACK", "1");
    AICORE_LOG_INFO("[AICore] ",
                    "Vulkan sysmem fallback enabled by default "
                    "(GGML_VK_ALLOW_SYSMEM_FALLBACK=1): a device allocation "
                    "that no longer fits VRAM degrades to host memory "
                    "instead of failing the run\n");
}

void apply_vulkan_f32_accuracy_profile() {
    GgmlEnvOverrides vo;
    vo.vk_disable_f16 = true;
    vo.vk_disable_coopmat = true;
    vo.vk_disable_coopmat2 = true;
    vo.vk_disable_integer_dot_product = true;
    vo.vk_disable_dot2 = true;
    apply_ggml_env_overrides(vo);
}

void mark_ggml_backends_loaded() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_backends_loaded = true;
}

namespace {

void ggml_log_forward_cb(enum ggml_log_level level,
                         const char* text,
                         void* user) {
    (void)user;
    if (!text || !*text) return;
    // The bridge routes ggml/llama internals below WARN through the
    // thread-local AICore level: the llama.cpp model loader alone prints
    // ~160 INFO lines per GGUF (full metadata dump, tensor plan, context
    // buffers) and the graph builder hundreds of DEBUG lines per token.
    // The facts the console needs (device, precision, readiness) are
    // logged by the task itself through the AICore logger, so both DEBUG
    // and INFO from this bridge are dropped unless the runtime level is
    // lowered to DEBUG. CONT fragments follow their parent message so a
    // dropped header cannot leave orphaned continuations.
    static thread_local bool dropping_message = false;
    if (level == GGML_LOG_LEVEL_CONT) {
        if (dropping_message) return;
        fputs(text, stderr);
        return;
    }
    dropping_message = false;
    if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_INFO) {
        dropping_message = aicore_get_log_level() > AICORE_LOG_LEVEL_DEBUG;
        if (dropping_message) return;
    }
    if (level == GGML_LOG_LEVEL_WARN) {
        // Trim ggml's trailing newline: the AICORE_LOG macros append one.
        std::string msg(text);
        while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
            msg.pop_back();
        }
        if (msg.empty()) return;
        AICORE_LOG_WARN("[ggml] ", "%s", msg.c_str());
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::string msg(text);
        while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
            msg.pop_back();
        }
        if (msg.empty()) return;
        AICORE_LOG_ERROR("[ggml] ", "%s", msg.c_str());
        return;
    }
    fputs(text, stderr);
}

}  // namespace

void install_ggml_log_bridge() {
    static const bool installed = [] {
        ggml_log_set(ggml_log_forward_cb, nullptr);
        return true;
    }();
    (void)installed;
}

ggml_log_callback ggml_log_bridge_callback() { return ggml_log_forward_cb; }

}  // namespace aicore
