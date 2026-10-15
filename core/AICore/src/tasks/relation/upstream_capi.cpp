// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "relateanything/c_api.h"
#include "relateanything/full_model.hpp"

namespace {
struct Names;
}
struct ra_context {
    std::unique_ptr<relateanything::FullModel> model;
    relateanything::ModelOptions options;
    std::array<char, 1024> error{};
    std::shared_ptr<const Names> names;
};
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
            .count();
}
struct Names {
    std::vector<std::string> values;
    std::vector<const char *> views;
};
struct ResultOwner {
    ra_result api{};
    relateanything::ImageOutput tensors;
    std::vector<ra_relation> relations, spatial, semantic;
    std::shared_ptr<const Names> names;
};
std::vector<ra_relation> convert(
        const std::vector<relateanything::Relation> &input) {
    std::vector<ra_relation> result;
    result.reserve(input.size());
    for (const auto &r : input)
        result.push_back({static_cast<uint32_t>(r.subject),
                          static_cast<uint32_t>(r.object),
                          static_cast<uint32_t>(r.predicate), r.score});
    return result;
}
void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}
void set_error(ra_context *ctx, const char *message) noexcept {
    if (!ctx) return;
    const auto size = std::min(std::strlen(message), ctx->error.size() - 1);
    std::memcpy(ctx->error.data(), message, size);
    ctx->error[size] = 0;
}
}  // namespace
extern "C" {
ra_options ra_default_options(void) {
    ra_options result{};
    result.struct_size = sizeof(result);
    result.abi_version = RA_ABI_VERSION;
    result.flash_attention = result.compact_pairs = result.fused_vocabulary =
            result.cache_positions = result.retain_features = 1;
    result.backend = "cpu";
    result.max_boxes = 60;
    result.topk = 128;
    result.strict_f32 = RA_STRICT_F32_AUTO;
    return result;
}
ra_request ra_default_request(void) {
    ra_request result{};
    result.struct_size = sizeof(result);
    result.abi_version = RA_ABI_VERSION;
    return result;
}
ra_context *ra_create(const ra_options *options) {
    ra_context *ctx = nullptr;
    try {
        ctx = new ra_context;
        require(options && options->struct_size == sizeof(*options) &&
                        options->abi_version == RA_ABI_VERSION,
                "invalid options ABI");
        require(options->model_path && options->model_path[0],
                "model_path is required");
        require(options->max_boxes > 0 && options->max_boxes <= 10000 &&
                        options->topk > 0,
                "invalid object or result limit");
        require(options->threads <=
                        static_cast<uint32_t>(std::numeric_limits<int>::max()),
                "thread count exceeds integer range");
        auto &o = ctx->options;
        if (options->backend_binding)
            o.backend_binding = *options->backend_binding;
        o.model = options->model_path;
        o.backend = options->backend ? options->backend : "cpu";
        if (options->diagnostic_directory)
            o.dump_directory = options->diagnostic_directory;
        require(o.backend == "cpu" || o.backend == "cuda" ||
                        o.backend == "vulkan",
                "invalid backend");
        require(options->strict_f32 >= RA_STRICT_F32_AUTO,
                "invalid strict_f32 mode");
        require(!options->quant_f32_input || o.backend == "cuda",
                "quant_f32_input requires CUDA");
        o.device_index = options->device_index;
        o.threads = options->threads;
        o.max_boxes = options->max_boxes;
        o.topk = options->topk;
        o.strict_f32 = options->strict_f32 > 0;
        o.strict_f32_auto = options->strict_f32 < 0;
        o.quant_f32_input = options->quant_f32_input != 0;
        o.raw_logits = options->raw_logits != 0;
        o.decompose = options->decompose != 0;
        if (options->vocabulary_path) o.vocabulary = options->vocabulary_path;
        if (options->vocabulary_names_path)
            o.vocabulary_names = options->vocabulary_names_path;
        if (options->predicate_type_map_path)
            o.predicate_type_map = options->predicate_type_map_path;
        o.flash_attention = options->flash_attention != 0;
        o.compact_pairs = options->compact_pairs != 0;
        o.fused_vocabulary = options->fused_vocabulary != 0;
        o.cache_positions = options->cache_positions != 0;
        o.retain_features = options->retain_features != 0;
        if (options->text_student_path)
            o.text_student = options->text_student_path;
        if (options->tokenizer_path) o.tokenizer = options->tokenizer_path;
        require(options->predicate_text_count <= 10000 &&
                        (!options->predicate_text_count ||
                         options->predicate_texts),
                "invalid predicate text array");
        for (size_t i = 0; i < options->predicate_text_count; ++i) {
            require(options->predicate_texts[i] &&
                            options->predicate_texts[i][0],
                    "empty predicate text");
            o.predicate_texts.emplace_back(options->predicate_texts[i]);
        }
        ctx->model = std::make_unique<relateanything::FullModel>(o);
        auto names = std::make_shared<Names>();
        names->values = ctx->model->predicates();
        for (const auto &name : names->values)
            names->views.push_back(name.c_str());
        ctx->names = std::move(names);
    } catch (const std::exception &e) {
        if (ctx) ctx->model.reset();
        set_error(ctx, e.what());
    } catch (...) {
        if (ctx) ctx->model.reset();
        set_error(ctx, "unknown model creation error");
    }
    return ctx;
}
void ra_free(ra_context *ctx) {
    try {
        delete ctx;
    } catch (...) {
    }
}
int ra_ready(const ra_context *ctx) { return ctx && ctx->model ? 1 : 0; }
int ra_calibration(const ra_context *ctx, float *a, float *b) {
    if (!ctx || !ctx->model || !a || !b) return -1;
    ctx->model->calibration(*a, *b);
    return 0;
}
const char *ra_last_error(const ra_context *ctx) {
    return ctx ? ctx->error.data() : "null context";
}
int ra_infer_image(ra_context *ctx,
                   const ra_image_view *image,
                   const ra_request *request,
                   ra_result **output) {
    if (output) *output = nullptr;
    if (!ctx) return -1;
    const auto begin = Clock::now();
    try {
        require(output && image && request,
                "image, request and output are required");
        require(ctx->model != nullptr, "context is not ready");
        require(request->struct_size == sizeof(*request) &&
                        request->abi_version == RA_ABI_VERSION,
                "invalid request ABI");
        require(request->object_count <= ctx->options.max_boxes,
                "object count exceeds max_boxes");
        require(!request->object_count || request->boxes_xyxy,
                "boxes are required");
        if (request->cancelled && request->cancelled(request->user)) {
            set_error(ctx, "cancelled");
            return 1;
        }
        relateanything::ImageOptions regions;
        if (request->object_count)
            regions.boxes_xyxy.assign(
                    request->boxes_xyxy,
                    request->boxes_xyxy + request->object_count * 4);
        if (request->box_scores)
            regions.box_scores.assign(
                    request->box_scores,
                    request->box_scores + request->object_count);
        if (request->masks) {
            require(image->width > 0 && image->height > 0 &&
                            image->width <= 20000 && image->height <= 20000,
                    "invalid mask image dimensions");
            const size_t area =
                    static_cast<size_t>(image->width) * image->height;
            require(request->object_count <= SIZE_MAX / area &&
                            request->mask_bytes == request->object_count * area,
                    "invalid mask capacity");
            regions.masks.assign(request->masks,
                                 request->masks + request->mask_bytes);
        } else
            require(request->mask_bytes == 0,
                    "mask capacity supplied without data");
        const auto input = relateanything::prepare_image(
                {image->data, static_cast<size_t>(image->width),
                 static_cast<size_t>(image->height), image->row_stride_bytes,
                 image->data_bytes,
                 static_cast<relateanything::ImageFormat>(image->format)},
                regions);
        const double preprocess = ms(begin);
        auto owner = std::make_unique<ResultOwner>();
        owner->tensors = ctx->model->infer(input);
        if (request->cancelled && request->cancelled(request->user)) {
            set_error(ctx, "cancelled");
            return 1;
        }
        owner->relations = convert(owner->tensors.relations);
        owner->spatial = convert(owner->tensors.spatial_relations);
        owner->semantic = convert(owner->tensors.semantic_relations);
        owner->names = ctx->names;
        auto &r = owner->api;
        const auto &t = owner->tensors;
        r.struct_size = sizeof(r);
        r.abi_version = RA_ABI_VERSION;
        r.relation_count = owner->relations.size();
        r.spatial_count = owner->spatial.size();
        r.semantic_count = owner->semantic.size();
        r.pair_count = t.pair_count;
        r.predicate_count = t.predicate_count;
        r.relations = owner->relations.data();
        r.spatial_relations = owner->spatial.data();
        r.semantic_relations = owner->semantic.data();
        r.predicate_logits = t.predicate_logits.empty()
                                     ? nullptr
                                     : t.predicate_logits.data();
        r.pair_logits = t.pair_logits.data();
        r.subject_indices = t.subject_indices.data();
        r.object_indices = t.object_indices.data();
        r.valid_mask = t.valid_mask.data();
        r.predicate_spatial = t.predicate_spatial.empty()
                                      ? nullptr
                                      : t.predicate_spatial.data();
        r.predicate_names = owner->names->views.data();
        r.backend_buffer_bytes = t.backend_buffer_bytes;
        r.forward_ms = t.milliseconds;
        r.timings.abi_version = RA_ABI_VERSION;
        r.timings.valid_fields = RA_TIMING_PREPROCESS | RA_TIMING_E2E;
        r.timings.preprocess_ms = preprocess;
        r.timings.e2e_ms = ms(begin);
        r.private_owner = owner.get();
        *output = &owner.release()->api;
        ctx->error[0] = 0;
        return 0;
    } catch (const std::exception &e) {
        set_error(ctx, e.what());
        return -1;
    } catch (...) {
        set_error(ctx, "unknown inference error");
        return -1;
    }
}
void ra_result_free(ra_result *result) {
    try {
        if (result) delete static_cast<ResultOwner *>(result->private_owner);
    } catch (...) {
    }
}
}
