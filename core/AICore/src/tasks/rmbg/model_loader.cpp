// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// AICore adaptation of RMBG-2.0-GGML model_loader.cpp. Device selection goes
// through the AICore runtime (process-shared BackendLease) instead of
// ggml_backend_init_by_type so "auto" follows the platform order and other
// AICore tasks share the same physical backends.

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <memory>

#include "common/ggml_backend_utils.hpp"
#include "common/ggml_env_bridge.hpp"
#include "ggml-backend.h"
#include "gguf.h"
#include "tasks/rmbg/rmbg.hpp"
#include "tasks/rmbg/rmbg_graph.hpp"
#include "tasks/rmbg/swin_backbone.hpp"

namespace rmbg {

static std::string lower(std::string value);

namespace {

// Coopmat matmul whitelist of the "optimized" profile (historical default
// written into RMBG_VK_COOPMAT_MATMUL before the env bridge existed).
constexpr const char *kOptimizedCoopmatWhitelist =
        "bb_layers_0,bb_layers_1,bb_layers_2,bb_layers_3,sq0_,db4_,db3_,db2_"
        ",db1_";

// Apply the ggml-side environment overrides implied by a math profile.
// Device-aware, mirroring the historical configure_backend_profile():
// the Vulkan switches are only meaningful when a Vulkan backend may load,
// the cuBLAS TF32 switch only for CUDA-bound requests. Runs BEFORE
// pick_backend() so instances created during device resolution see the
// values.
void apply_profile_env(const std::string &profile,
                       const std::string &requested) {
    aicore::GgmlEnvOverrides env;
    const bool generic_gpu = requested == "gpu";
    const bool may_vulkan = requested == "auto" || generic_gpu ||
                            requested.rfind("vulkan", 0) == 0;
    const bool may_cuda = requested == "auto" || generic_gpu ||
                          requested.rfind("cuda", 0) == 0;
    if (may_vulkan) {
        if (profile == "strict") {
            env.vk_disable_f16 = true;
            env.vk_disable_coopmat = true;
            env.vk_disable_coopmat2 = true;
            env.vk_disable_integer_dot_product = true;
            env.rmbg_vk_scalar_direct_conv = false;
            env.rmbg_vk_coopmat_matmul = std::string("");
        } else if (profile == "fast" || profile == "unsafe-fast") {
            env.vk_disable_f16 = false;
            env.vk_disable_coopmat = false;
            env.vk_disable_coopmat2 = false;
            env.vk_disable_integer_dot_product = false;
            env.rmbg_vk_scalar_direct_conv = false;
            env.rmbg_vk_coopmat_matmul = std::string("");
        } else {  // "optimized" (default)
            env.vk_disable_f16 = true;
            env.vk_disable_coopmat = false;
            env.vk_disable_coopmat2 = true;
            env.vk_disable_integer_dot_product = true;
            env.rmbg_vk_scalar_direct_conv = true;
            env.rmbg_vk_coopmat_matmul =
                    std::string(kOptimizedCoopmatWhitelist);
        }
    }
    if (profile == "strict" && may_cuda) {
        // Bit-stable FP32 GEMMs. The default keeps cuBLAS TF32 enabled; its
        // measured alpha error remains below 1.4e-3.
        env.nvidia_tf32_override = false;
    }
    aicore::apply_ggml_env_overrides(env);
}

// Profile-driven graph fields. The caller's fine-tuning fields (qkv layout,
// flash attention, cuda f16/nn gemm, ...) are kept; the profile always wins
// for the flow-defining switches, exactly like the historical configure
// step overwriting the RMBG_VK_DIRECT_CONV variable.
void apply_profile_to_graph(const std::string &profile, GraphOptions &opts) {
    if (profile == "strict") {
        opts.vulkan_direct_conv = false;
        opts.vk_f16_disabled = true;
        opts.strict_math = true;
    } else if (profile == "fast" || profile == "unsafe-fast") {
        opts.vulkan_direct_conv = false;
        opts.vk_f16_disabled = false;
        opts.strict_math = false;
    } else if (profile == "optimized") {
        opts.vulkan_direct_conv = true;
        opts.vk_f16_disabled = true;
        opts.strict_math = false;
    } else {  // "default" (no Vulkan/CUDA acceleration)
        opts.vulkan_direct_conv = false;
        opts.vk_f16_disabled = false;
        opts.strict_math = false;
    }
}

}  // namespace

// Acquire a process-shared backend lease for the requested device. "auto"
// resolves through the AICore runtime (CUDA -> Vulkan -> CPU on
// Linux/Windows, Metal -> CPU on macOS).
static bool pick_backend(const char *device,
                         int n_threads,
                         aicore::runtime::BackendLease &lease,
                         std::string &backend_name,
                         std::string &err) {
    ggml_common::load_backends_once();
    const std::string requested = lower(device && device[0] ? device : "auto");
    if (requested == "cpu") {
        lease = aicore::runtime::acquire_backend_lease(
                "cpu", n_threads > 0 ? n_threads : 0, &err);
        if (!lease) return false;
        backend_name = ggml_backend_name(lease.handle());
        return true;
    }
    if (requested == "auto" || requested == "gpu") {
        lease = aicore::runtime::acquire_backend_lease(
                "auto", n_threads > 0 ? n_threads : 0, &err);
        if (!lease) return false;
        backend_name = ggml_backend_name(lease.handle());
        return true;
    }
    // Explicit backend id ("cuda", "vulkan", "metal", "cuda:0", ...).
    lease = aicore::runtime::acquire_backend_lease(
            requested, n_threads > 0 ? n_threads : 0, &err);
    if (!lease) return false;
    backend_name = ggml_backend_name(lease.handle());
    return true;
}

static std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

std::string normalize_math_profile(const char *profile) {
    std::string mode = lower(profile ? profile : "");
    if (mode.empty()) mode = "optimized";
    if (mode != "strict" && mode != "unsafe-fast" && mode != "fast" &&
        mode != "optimized" && mode != "default") {
        mode = "optimized";
    }
    return mode;
}

bool load_gguf(const char *path,
               const char *device,
               int n_threads,
               const char *math_profile,
               const GraphOptions &user_graph_options,
               Model &out,
               std::string &err) {
    if (!path || !path[0]) {
        err = "empty path";
        return false;
    }
    free_model(out);

    const std::string profile = normalize_math_profile(math_profile);
    const std::string requested = lower(device && device[0] ? device : "auto");
    // The ggml env overrides must be applied before the backend instances
    // are created (pick_backend below resolves the device).
    apply_profile_env(profile, requested);

    WeightMap weights;
    if (!weights.load_gguf(path, err)) return false;
    if (!weights.get_f32("sq0_conv_in_weight")) {
        const std::filesystem::path decoder =
                std::filesystem::path(path).parent_path() /
                "decoder_alpha_f16.gguf";
        if (!std::filesystem::exists(decoder)) {
            err = "GGUF has no decoder weights; use a unified GGUF or place " +
                  decoder.string() + " beside the encoder";
            return false;
        }
        if (!weights.merge_gguf(decoder.string().c_str(), err)) return false;
    }

    // Metadata is optional for split compatibility files.
    ggml_context *meta = nullptr;
    struct gguf_init_params params = {.no_alloc = true, .ctx = &meta};
    gguf_context *ctx = gguf_init_from_file(path, params);
    if (ctx) {
        const int k_size = gguf_find_key(ctx, "rmbg.input_size");
        if (k_size >= 0)
            out.cfg.input_size = (int)gguf_get_val_u32(ctx, k_size);
        const int k_mean = gguf_find_key(ctx, "rmbg.img.mean");
        if (k_mean >= 0) {
            const float *mean = (const float *)gguf_get_arr_data(ctx, k_mean);
            for (int i = 0; i < 3; ++i) out.cfg.mean[i] = mean[i];
        }
        const int k_std = gguf_find_key(ctx, "rmbg.img.std");
        if (k_std >= 0) {
            const float *st = (const float *)gguf_get_arr_data(ctx, k_std);
            for (int i = 0; i < 3; ++i) out.cfg.std[i] = st[i];
        }
        gguf_free(ctx);
        ggml_free(meta);
    }

    std::string backend_err;
    if (!pick_backend(device, n_threads, out.lease, out.backend_name,
                      backend_err)) {
        err = "requested ggml backend unavailable (" + backend_err +
              "): " + (device ? device : "auto");
        return false;
    }
    out.backend = out.lease.handle();
    const std::string resolved_backend = lower(out.backend_name);
    if (resolved_backend.find("vulkan") != std::string::npos) {
        out.math_profile = profile;
    } else if (resolved_backend.find("cuda") != std::string::npos) {
        out.math_profile = profile == "strict" ? "strict" : "optimized";
    } else {
        out.math_profile = "default";
    }
    out.n_threads = n_threads;

    GraphOptions graph_options = user_graph_options;
    apply_profile_to_graph(profile, graph_options);

    std::unique_ptr<RmbgDeviceGraph> graph(new RmbgDeviceGraph);
    if (!graph->init(out.backend, weights, out.cfg.input_size, graph_options,
                     err)) {
        graph.reset();
        out.lease.reset();
        out.backend = nullptr;
        return false;
    }
    out.graph = graph.release();
    out.graph_ready = true;
    return true;
}

void free_model(Model &m) {
    delete m.graph;
    m.graph = nullptr;
    if (m.lease) {
        m.lease.reset();
    }
    m.backend = nullptr;
    m.backend_name.clear();
    m.math_profile.clear();
    m.graph_ready = false;
}

}  // namespace rmbg
