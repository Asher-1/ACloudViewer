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

#include "rmbg.hpp"
#include "rmbg_graph.hpp"
#include "swin_backbone.hpp"

#include "ggml-backend.h"
#include "ggml_backend_utils.hpp"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>

namespace rmbg {

static std::string lower(std::string value);

static void set_env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

static void clear_env(const char * key) {
#ifdef _WIN32
    _putenv_s(key, "");
#else
    unsetenv(key);
#endif
}

static void configure_strict_math(const char * device) {
    const std::string requested = lower(device ? device : "auto");
    const char * strict = std::getenv("RMBG_STRICT_MATH");
    const bool strict_math = strict && strict[0] && std::strcmp(strict, "0") != 0;
    if (strict_math && (requested == "auto" || requested.rfind("cuda", 0) == 0)) {
        // Set RMBG_STRICT_MATH=1 for bit-stable FP32 GEMMs. The default keeps
        // cuBLAS TF32 enabled; its measured alpha error remains below 1.4e-3.
        set_env("NVIDIA_TF32_OVERRIDE", "0");
    }
    if (requested == "auto" || requested.rfind("vulkan", 0) == 0) {
        const char * fast = std::getenv("RMBG_VULKAN_FAST");
        const bool fast_math = fast && fast[0] && std::strcmp(fast, "0") != 0;
        if (!fast_math) {
            // Strict Vulkan is the default because cooperative matrix shaders
            // exceed the alpha parity threshold for the F32 reference model.
            set_env("GGML_VK_DISABLE_F16", "1");
            set_env("GGML_VK_DISABLE_COOPMAT", "1");
            set_env("GGML_VK_DISABLE_COOPMAT2", "1");
            set_env("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1");
        } else {
            // A parent shell may previously have run strict mode. Explicitly
            // clear the opt-out flags so RMBG_VULKAN_FAST is deterministic.
            clear_env("GGML_VK_DISABLE_F16");
            clear_env("GGML_VK_DISABLE_COOPMAT");
            clear_env("GGML_VK_DISABLE_COOPMAT2");
            clear_env("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT");
        }
    }
}

static std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return value;
}

// Acquire a process-shared backend lease for the requested device. "auto"
// resolves through the AICore runtime (CUDA -> Vulkan -> CPU on
// Linux/Windows, Metal -> CPU on macOS).
static bool pick_backend(const char * device, int n_threads,
                         aicore::runtime::BackendLease & lease,
                         std::string & backend_name, std::string & err) {
    configure_strict_math(device);

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

bool load_gguf(const char * path, const char * device, int n_threads,
               Model & out, std::string & err) {
    if (!path || !path[0]) { err = "empty path"; return false; }
    free_model(out);

    WeightMap weights;
    if (!weights.load_gguf(path, err)) return false;
    if (!weights.get_f32("sq0_conv_in_weight")) {
        const std::filesystem::path decoder =
            std::filesystem::path(path).parent_path() / "decoder_alpha_f16.gguf";
        if (!std::filesystem::exists(decoder)) {
            err = "GGUF has no decoder weights; use a unified GGUF or place " +
                  decoder.string() + " beside the encoder";
            return false;
        }
        if (!weights.merge_gguf(decoder.string().c_str(), err)) return false;
    }

    // Metadata is optional for split compatibility files.
    ggml_context * meta = nullptr;
    struct gguf_init_params params = { .no_alloc = true, .ctx = &meta };
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx) {
        const int k_size = gguf_find_key(ctx, "rmbg.input_size");
        if (k_size >= 0) out.cfg.input_size = (int) gguf_get_val_u32(ctx, k_size);
        const int k_mean = gguf_find_key(ctx, "rmbg.img.mean");
        if (k_mean >= 0) {
            const float * mean = (const float *) gguf_get_arr_data(ctx, k_mean);
            for (int i = 0; i < 3; ++i) out.cfg.mean[i] = mean[i];
        }
        const int k_std = gguf_find_key(ctx, "rmbg.img.std");
        if (k_std >= 0) {
            const float * st = (const float *) gguf_get_arr_data(ctx, k_std);
            for (int i = 0; i < 3; ++i) out.cfg.std[i] = st[i];
        }
        gguf_free(ctx);
        ggml_free(meta);
    }

    std::string backend_err;
    if (!pick_backend(device, n_threads, out.lease, out.backend_name,
                      backend_err)) {
        err = "requested ggml backend unavailable (" + backend_err + "): " +
              (device ? device : "auto");
        return false;
    }
    out.backend = out.lease.handle();
    out.n_threads = n_threads;

    std::unique_ptr<RmbgDeviceGraph> graph(new RmbgDeviceGraph);
    if (!graph->init(out.backend, weights, out.cfg.input_size, err)) {
        graph.reset();
        out.lease.reset();
        out.backend = nullptr;
        return false;
    }
    out.graph = graph.release();
    out.graph_ready = true;
    return true;
}

void free_model(Model & m) {
    delete m.graph;
    m.graph = nullptr;
    if (m.lease) {
        m.lease.reset();
    }
    m.backend = nullptr;
    m.backend_name.clear();
    m.graph_ready = false;
}

} // namespace rmbg
