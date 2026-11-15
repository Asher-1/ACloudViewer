// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AnyCalib C API implementation: the session wraps the static ggml graph
// (anycalib_graph.cpp) with the task contract from aicore/anycalib_capi.h.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

#include "aicore/anycalib_capi.h"
#include "aicore/pipeline_timing.h"
#include "aicore/runtime_capi.h"
#include "common/capi_utils.hpp"
#include "tasks/anycalib/anycalib_graph.hpp"

constexpr int kAnycalibAbiVersion = 1;

struct aicore_anycalib_ctx {
    aicore::anycalib::Session* session = nullptr;
    std::string last_error;
    aicore_pipeline_timings timings{};
};

struct aicore_anycalib_options {
    aicore::capi::CommonOptions common;
};

AICORE_CAPI int aicore_anycalib_abi_version(void) {
    return kAnycalibAbiVersion;
}

AICORE_CAPI aicore_anycalib_options* aicore_anycalib_options_new(void) {
    return new (std::nothrow) aicore_anycalib_options();
}

AICORE_CAPI void aicore_anycalib_options_free(aicore_anycalib_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_anycalib_options_set_device(
        aicore_anycalib_options* opts, const char* device) {
    if (opts != nullptr) aicore::capi::set_device(opts->common, device);
}

AICORE_CAPI void aicore_anycalib_options_set_threads(
        aicore_anycalib_options* opts, int n_threads) {
    if (opts != nullptr) aicore::capi::set_threads(opts->common, n_threads);
}

AICORE_CAPI aicore_anycalib_ctx* aicore_anycalib_load_opts(
        const char* gguf_path, const aicore_anycalib_options* opts) {
    auto* ctx = new (std::nothrow) aicore_anycalib_ctx();
    if (ctx == nullptr) return nullptr;
    if (gguf_path == nullptr || gguf_path[0] == '\0') {
        ctx->last_error = "model path is empty";
        return ctx;
    }
    aicore::anycalib::SessionOptions sopts;
    sopts.device = opts != nullptr ? opts->common.device : std::string("auto");
    sopts.num_threads = opts != nullptr ? opts->common.threads : 0;
    auto* session = new (std::nothrow) aicore::anycalib::Session();
    if (session == nullptr) {
        ctx->last_error = "anycalib out of memory for session";
        return ctx;
    }
    ctx->session = session;
    if (!session->Load(gguf_path, sopts)) {
        ctx->last_error = session->error();
    }
    return ctx;
}

AICORE_CAPI void aicore_anycalib_free(aicore_anycalib_ctx* ctx) {
    if (ctx == nullptr) return;
    delete ctx->session;
    delete ctx;
}

AICORE_CAPI int aicore_anycalib_is_ready(const aicore_anycalib_ctx* ctx) {
    return ctx != nullptr && ctx->session != nullptr && ctx->session->ready()
                   ? 1
                   : 0;
}

AICORE_CAPI const char* aicore_anycalib_last_error(
        const aicore_anycalib_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI void aicore_anycalib_free_buffer(void* p) { std::free(p); }

AICORE_CAPI int aicore_anycalib_run(aicore_anycalib_ctx* ctx,
                                    const float* image_rgb_chw,
                                    float** out_rays,
                                    float** out_tangent,
                                    int32_t* out_count) {
    if (out_rays != nullptr) *out_rays = nullptr;
    if (out_tangent != nullptr) *out_tangent = nullptr;
    if (out_count != nullptr) *out_count = 0;
    if (ctx == nullptr || out_rays == nullptr || out_tangent == nullptr ||
        out_count == nullptr) {
        return -1;
    }
    if (ctx->session == nullptr || !ctx->session->ready()) {
        ctx->last_error = "anycalib context is not ready";
        return -1;
    }
    if (image_rgb_chw == nullptr) {
        ctx->last_error = "network input is required";
        return -1;
    }
    try {
        aicore::anycalib::RunResult result;
        const auto t0 = std::chrono::steady_clock::now();
        if (!ctx->session->Run(image_rgb_chw, result)) {
            ctx->last_error = ctx->session->error();
            return -1;
        }
        constexpr int32_t kCount = 322 * 322;
        auto* rays = (float*)std::malloc(result.rays.size() * sizeof(float));
        auto* tangent =
                (float*)std::malloc(result.tangent.size() * sizeof(float));
        if (rays == nullptr || tangent == nullptr) {
            std::free(rays);
            std::free(tangent);
            ctx->last_error = "anycalib out of memory for typed results";
            return -1;
        }
        std::memcpy(rays, result.rays.data(),
                    result.rays.size() * sizeof(float));
        std::memcpy(tangent, result.tangent.data(),
                    result.tangent.size() * sizeof(float));
        *out_rays = rays;
        *out_tangent = tangent;
        *out_count = kCount;
        ctx->timings = {};
        ctx->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
        ctx->timings.valid_fields =
                AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
                AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
        // Stage split: preprocess = graph build + input/constant upload
        // (no image decode: the caller owns geometry), inference = backend
        // graph compute, postprocess = typed-result readback + the exact
        // f32 ray normalization.
        ctx->timings.preprocess_ms = result.preprocess_ms;
        ctx->timings.inference_ms = result.inference_ms;
        ctx->timings.postprocess_ms = result.postprocess_ms;
        ctx->timings.e2e_ms = result.e2e_ms;
        return 0;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "anycalib out of memory";
        return -1;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("anycalib error: ") + e.what();
        return -1;
    }
}

AICORE_CAPI int aicore_anycalib_last_pipeline_timings(
        const aicore_anycalib_ctx* ctx, aicore_pipeline_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr ||
        ctx->timings.valid_fields == 0) {
        return -1;
    }
    *out_timings = ctx->timings;
    return 0;
}

AICORE_CAPI void aicore_anycalib_shutdown(void) {
    // AnyCalib shares the process-wide backend registry with the other
    // tasks; the common shutdown is idempotent and never destroys live
    // contexts.
    aicore_runtime_shutdown();
}
