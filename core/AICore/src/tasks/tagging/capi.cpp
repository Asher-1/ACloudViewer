// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>

#include "aicore/runtime_capi.h"
#include "aicore/tagging_capi.h"
#include "common/data_root_util.hpp"
#include "common/ggml_backend_utils.hpp"
#include "session.hpp"

struct aicore_tagging_ctx {
    std::unique_ptr<aicore::tagging::Session> session;
    std::string error;
    aicore_pipeline_timings timings{};
    bool has_timings = false;
};
namespace {
struct OwnedResult {
    aicore_tagging_result
            public_result{};  // first member, released by its address
    std::vector<std::string> names;
    std::vector<aicore_tagging_tag> tags;
    std::string caption;
};
template <class T>
bool valid(const T* value) {
    return value && value->struct_size >= sizeof(T) &&
           value->abi_version == AICORE_TAGGING_ABI_VERSION;
}
}  // namespace
aicore_tagging_options aicore_tagging_default_options(void) {
    aicore_tagging_options result{};
    result.struct_size = sizeof(result);
    result.abi_version = AICORE_TAGGING_ABI_VERSION;
    result.device = "auto";
    result.max_length = 30;
    return result;
}
aicore_tagging_request aicore_tagging_default_request(void) {
    aicore_tagging_request result{};
    result.struct_size = sizeof(result);
    result.abi_version = AICORE_TAGGING_ABI_VERSION;
    result.threshold = -1;
    return result;
}
aicore_tagging_ctx* aicore_tagging_create(
        const aicore_tagging_options* options) {
    std::unique_ptr<aicore_tagging_ctx> ctx(new (std::nothrow)
                                                    aicore_tagging_ctx);
    if (!ctx) return nullptr;
    try {
        if (!valid(options))
            throw std::runtime_error("invalid options ABI/size");
        if (options->cpu_precision != AICORE_TAGGING_CPU_AUTO &&
            options->cpu_precision != AICORE_TAGGING_CPU_UPSTREAM)
            throw std::runtime_error("invalid CPU precision mode");
        if (options->max_length < 10 || options->max_length > 512 ||
            (options->beam_batch != 0 && options->beam_batch != 1 &&
             options->beam_batch != 3))
            throw std::runtime_error("invalid caption length/beam batch");
        auto effective = *options;
        if (effective.threads <= 0)
            effective.threads = int(ggml_common::default_cpu_threads());
        ctx->session = std::make_unique<aicore::tagging::Session>(effective);
    } catch (const std::exception& e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unknown tagging load error";
    }
    return ctx.release();
}
void aicore_tagging_free(aicore_tagging_ctx* ctx) { delete ctx; }
int aicore_tagging_ready(const aicore_tagging_ctx* ctx) {
    return ctx && bool(ctx->session);
}
const char* aicore_tagging_last_error(const aicore_tagging_ctx* ctx) {
    return ctx ? ctx->error.c_str() : "null context";
}
const char* aicore_tagging_context_device(const aicore_tagging_ctx* ctx) {
    return aicore_tagging_ready(ctx) ? ctx->session->device().c_str() : "";
}
int aicore_tagging_infer_image(aicore_tagging_ctx* ctx,
                               const aicore_image_view* image,
                               const aicore_tagging_request* request,
                               aicore_tagging_result** output) {
    if (output) *output = nullptr;
    if (!ctx) return -1;
    ctx->has_timings = false;
    ctx->timings = {};
    const auto start = std::chrono::steady_clock::now();
    try {
        if (!output || !image || !valid(request) || !ctx->session)
            throw std::runtime_error(
                    "invalid request, image, output or context");
        if (!std::isfinite(request->threshold) ||
            (request->threshold != -1 &&
             (request->threshold < 0 || request->threshold > 1)))
            throw std::runtime_error("threshold must be -1 or 0..1");
        if (!std::isfinite(request->threshold_epsilon) ||
            request->threshold_epsilon < 0 || request->threshold_epsilon > 1)
            throw std::runtime_error("threshold epsilon must be 0..1");
        aicore::img::Image borrowed;
        if (!aicore::img::image_from_view(*image, borrowed))
            throw std::runtime_error("invalid image view/stride/format");
        auto native = ctx->session->infer(borrowed, *request);
        const auto processed = std::chrono::steady_clock::now();
        auto result = std::make_unique<OwnedResult>();
        result->names = ctx->session->names();
        result->caption = std::move(native.caption);
        result->tags.resize(native.logits.size());
        for (size_t i = 0; i < native.logits.size(); ++i) {
            float probability = 1.f / (1.f + std::exp(-native.logits[i]));
            const float threshold =
                    ctx->session->selectionThreshold(i, *request);
            if (!std::isfinite(native.logits[i]) || !std::isfinite(probability))
                throw std::runtime_error("non-finite tagging output");
            result->tags[i] = {
                    int32_t(i),
                    result->names[i].c_str(),
                    native.logits[i],
                    probability,
                    threshold,
                    probability > threshold && !ctx->session->excluded(i)};
        }
        const auto end = std::chrono::steady_clock::now();
        ctx->timings = native.timings;
        ctx->timings.valid_fields |=
                AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
        ctx->timings.postprocess_ms =
                std::chrono::duration<double, std::milli>(end - processed)
                        .count();
        ctx->timings.e2e_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
        result->public_result = {sizeof(aicore_tagging_result),
                                 AICORE_TAGGING_ABI_VERSION,
                                 result->tags.size(),
                                 result->tags.data(),
                                 result->caption.c_str(),
                                 ctx->timings,
                                 result.get()};
        *output = &result.release()->public_result;
        ctx->error.clear();
        ctx->has_timings = true;
        return 0;
    } catch (const std::exception& e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unknown tagging inference error";
    }
    return aicore_cancel_requested() ? 1 : -1;
}
void aicore_tagging_result_free(aicore_tagging_result* result) {
    if (result) delete static_cast<OwnedResult*>(result->private_owner);
}
int aicore_tagging_last_pipeline_timings(const aicore_tagging_ctx* ctx,
                                         aicore_pipeline_timings* output) {
    if (!ctx || !output || !ctx->has_timings) return -1;
    *output = ctx->timings;
    return 0;
}
char* aicore_tagging_model_cache_dir(void) {
    try {
        auto path = aicore::extract_model_dir("ram_models");
        auto* result = static_cast<char*>(std::malloc(path.size() + 1));
        if (result) std::memcpy(result, path.c_str(), path.size() + 1);
        return result;
    } catch (...) {
        return nullptr;
    }
}
void aicore_tagging_free_buffer(void* buffer) { std::free(buffer); }
void aicore_tagging_shutdown(void) { aicore_runtime_shutdown(); }
