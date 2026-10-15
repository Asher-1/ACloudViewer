// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "aicore/relation_capi.h"
#include "aicore/runtime_capi.h"
#include "common/data_root_util.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "gguf.h"
#include "relateanything/c_api.h"
#include "relateanything/full_model.hpp"

struct aicore_relation_ctx {
    aicore::runtime::BackendLease accelerator, cpu;
    ra_context *native = nullptr;
    relateanything::ModelOptions text_options;
    std::unique_ptr<relateanything::PredicateEncoder> text_encoder;
    std::array<char, 1024> error{};
    std::string device;
    aicore_pipeline_timings timings{};
    bool text_only = false, loaded = false, has_timings = false;
    ~aicore_relation_ctx() {
        auto lock = aicore::runtime::lock_backend_leases({accelerator, cpu});
        text_encoder.reset();
        ra_free(native);
    }
};
namespace {
using Clock = std::chrono::steady_clock;
void require(bool value, const char *message) {
    if (!value) throw std::invalid_argument(message);
}
void error(aicore_relation_ctx *ctx, const char *message) noexcept {
    if (!ctx) return;
    const auto n = std::min(std::strlen(message), ctx->error.size() - 1);
    std::memcpy(ctx->error.data(), message, n);
    ctx->error[n] = 0;
}
template <class T>
bool valid(const T *v) {
    return v && v->struct_size >= sizeof(T) &&
           v->abi_version == AICORE_RELATION_ABI_VERSION;
}
void retain(void *) {
}  // ctx owns leases until all native session objects are gone
int cancelled(void *) { return aicore_cancel_requested(); }
struct Result {
    aicore_relation_result api{};
    std::unique_ptr<ra_result, decltype(&ra_result_free)> native{
            nullptr, ra_result_free};
    std::vector<aicore_relation_edge> edges, spatial, semantic;
};
struct Embeddings {
    aicore_relation_embeddings api{};
    std::vector<float> values;
};
std::vector<aicore_relation_edge> edges(const ra_relation *p, size_t n) {
    std::vector<aicore_relation_edge> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        require(std::isfinite(p[i].score), "non-finite relation score");
        out.push_back({p[i].subject, p[i].object, p[i].predicate, p[i].score});
    }
    return out;
}
}  // namespace
aicore_relation_options aicore_relation_default_options(void) {
    aicore_relation_options o{};
    o.struct_size = sizeof(o);
    o.abi_version = 1;
    o.device = "auto";
    o.max_boxes = 60;
    o.topk = 128;
    o.strict_f32 = -1;
    o.flash_attention = o.compact_pairs = o.fused_vocabulary =
            o.cache_positions = o.retain_features = 1;
    return o;
}
aicore_relation_request aicore_relation_default_request(void) {
    aicore_relation_request r{};
    r.struct_size = sizeof(r);
    r.abi_version = 1;
    return r;
}
aicore_relation_ctx *aicore_relation_create(
        const aicore_relation_options *options) {
    auto ctx = std::unique_ptr<aicore_relation_ctx>(
            new (std::nothrow) aicore_relation_ctx);
    if (!ctx) return nullptr;
    try {
        require(valid(options), "invalid options size/version");
        require(options->strict_f32 >= -1 && options->strict_f32 <= 1,
                "strict_f32 must be -1 (automatic), 0 or 1");
        require(options->model_path && *options->model_path,
                "model_path is required");
        require(options->max_boxes && options->max_boxes <= 10000 &&
                        options->topk,
                "invalid object/output limits");
        const int threads = options->threads > 0
                                    ? options->threads
                                    : int(ggml_common::default_cpu_threads());
        std::string failure;
        ctx->accelerator = aicore::runtime::acquire_backend_lease(
                options->device ? options->device : "auto", threads, &failure);
        if (!ctx->accelerator) throw std::runtime_error(failure);
        ctx->cpu = aicore::runtime::acquire_backend_lease("cpu", threads,
                                                          &failure);
        if (!ctx->cpu) throw std::runtime_error(failure);
        ctx->device = ctx->accelerator.device();
        std::string backend =
                ctx->accelerator.is_cpu()
                        ? "cpu"
                        : (ctx->device.rfind("cuda", 0) == 0
                                   ? "cuda"
                                   : (ctx->device.rfind("vulkan", 0) == 0
                                              ? "vulkan"
                                              : ""));
        if (backend.empty() &&
            (!options->device || std::strcmp(options->device, "auto") == 0)) {
            ctx->accelerator = ctx->cpu;
            ctx->device = ctx->cpu.device();
            backend = "cpu";
        }
        require(!backend.empty(),
                "RelateAnything supports CPU, CUDA and Vulkan");
        require(!options->quant_f32_input || backend == "cuda",
                "quant_f32_input requires CUDA");
        ra_backend_binding binding{};
        binding.struct_size = sizeof(binding);
        binding.abi_version = 1;
        binding.cpu_handle = ctx->cpu.handle();
        binding.accelerator_handle = ctx->accelerator.handle();
        binding.lease_user = ctx.get();
        binding.retain_lease = binding.release_lease = retain;
        auto lock = aicore::runtime::lock_backend_leases(
                {ctx->accelerator, ctx->cpu});
        ctx->text_only = options->text_only != 0;
        auto &text = ctx->text_options;
        text.backend_binding = binding;
        text.backend = backend;
        text.threads = threads;
        text.strict_f32 = options->strict_f32 > 0;
        text.strict_f32_auto = options->strict_f32 < 0;
        text.text_student = options->text_student_path
                                    ? options->text_student_path
                                    : options->model_path;
        if (options->tokenizer_path) text.tokenizer = options->tokenizer_path;
        if (options->diagnostic_directory && *options->diagnostic_directory) {
            std::filesystem::create_directories(options->diagnostic_directory);
            text.token_output =
                    (std::filesystem::path(options->diagnostic_directory) /
                     "text-token-ids.json")
                            .string();
        }
        if (ctx->text_only) {
            gguf_init_params params{};
            params.no_alloc = true;
            auto *gguf = gguf_init_from_file(options->model_path, params);
            require(gguf != nullptr, "cannot read text student GGUF");
            const auto key = gguf_find_key(gguf, "ra.graph_kind");
            const bool is_student =
                    key >= 0 &&
                    gguf_get_kv_type(gguf, key) == GGUF_TYPE_STRING &&
                    std::strcmp(gguf_get_val_str(gguf, key),
                                "relateanything_text_student_v1") == 0;
            gguf_free(gguf);
            require(is_student, "text context requires a native text student");
            require(!text.tokenizer.empty() &&
                            std::filesystem::is_regular_file(text.tokenizer),
                    "official CLIP tokenizer is required");
            ctx->loaded = true;
        } else {
            auto o = ra_default_options();
            o.backend_binding = &binding;
            o.model_path = options->model_path;
            o.backend = backend.c_str();
            o.threads = threads;
            o.max_boxes = options->max_boxes;
            o.topk = options->topk;
            o.strict_f32 = options->strict_f32;
            o.quant_f32_input = options->quant_f32_input;
            o.raw_logits = options->raw_logits;
            o.decompose = options->decompose;
            o.flash_attention = options->flash_attention;
            o.compact_pairs = options->compact_pairs;
            o.fused_vocabulary = options->fused_vocabulary;
            o.cache_positions = options->cache_positions;
            o.retain_features = options->retain_features;
            o.vocabulary_path = options->vocabulary_path;
            o.vocabulary_names_path = options->vocabulary_names_path;
            o.predicate_type_map_path = options->predicate_type_map_path;
            o.text_student_path = options->text_student_path;
            o.tokenizer_path = options->tokenizer_path;
            o.predicate_texts = options->predicate_texts;
            o.predicate_text_count = options->predicate_text_count;
            o.diagnostic_directory = options->diagnostic_directory;
            ctx->native = ra_create(&o);
            if (!ra_ready(ctx->native))
                throw std::runtime_error(ra_last_error(ctx->native));
            ctx->loaded = true;
        }
    } catch (const std::exception &e) {
        error(ctx.get(), e.what());
    } catch (...) {
        error(ctx.get(), "unknown relation load error");
    }
    return ctx.release();
}
void aicore_relation_free(aicore_relation_ctx *ctx) { delete ctx; }
int aicore_relation_ready(const aicore_relation_ctx *ctx) {
    return ctx && ctx->loaded;
}
const char *aicore_relation_last_error(const aicore_relation_ctx *ctx) {
    return ctx ? ctx->error.data() : "null context";
}
const char *aicore_relation_context_device(const aicore_relation_ctx *ctx) {
    return ctx ? ctx->device.c_str() : "";
}
int aicore_relation_calibration(const aicore_relation_ctx *ctx,
                                float *a,
                                float *b) {
    if (!ctx || !ctx->loaded || ctx->text_only) return -1;
    return ra_calibration(ctx->native, a, b);
}
int aicore_relation_infer_image(aicore_relation_ctx *ctx,
                                const aicore_image_view *image,
                                const aicore_relation_request *request,
                                aicore_relation_result **output) {
    if (output) *output = nullptr;
    if (!ctx) return -1;
    ctx->has_timings = false;
    ctx->timings = {};
    try {
        require(output && image && valid(request) && ctx->loaded &&
                        !ctx->text_only,
                "invalid image/request/output/context");
        require(image->data && image->width > 0 && image->height > 0 &&
                        image->width <= 20000 && image->height <= 20000,
                "invalid image dimensions");
        const int channels =
                image->format == AICORE_IMAGE_GRAY8
                        ? 1
                        : (image->format == AICORE_IMAGE_RGB8 ||
                                           image->format == AICORE_IMAGE_BGR8
                                   ? 3
                                   : (image->format == AICORE_IMAGE_RGBA8 ||
                                                      image->format ==
                                                              AICORE_IMAGE_BGRA8
                                              ? 4
                                              : 0));
        require(channels &&
                        image->row_stride_bytes >=
                                size_t(image->width) * channels &&
                        image->row_stride_bytes <=
                                SIZE_MAX / size_t(image->height),
                "invalid image stride/format");
        if (aicore_cancel_requested()) {
            error(ctx, "cancelled");
            return 1;
        }
        auto lock = aicore::runtime::lock_backend_leases(
                {ctx->accelerator, ctx->cpu});
        ra_image_view v{image->data,
                        image->width,
                        image->height,
                        image->row_stride_bytes,
                        image->row_stride_bytes * size_t(image->height),
                        static_cast<ra_image_format>(image->format)};
        auto r = ra_default_request();
        r.boxes_xyxy = request->boxes_xyxy;
        r.object_count = request->object_count;
        r.box_scores = request->box_scores;
        r.masks = request->masks;
        r.mask_bytes = request->mask_bytes;
        r.cancelled = cancelled;
        ra_result *raw = nullptr;
        const int rc = ra_infer_image(ctx->native, &v, &r, &raw);
        if (rc) {
            error(ctx, ra_last_error(ctx->native));
            return rc;
        }
        std::unique_ptr<ra_result, decltype(&ra_result_free)> owned_raw(
                raw, ra_result_free);
        auto owner = std::make_unique<Result>();
        owner->native = std::move(owned_raw);
        owner->edges = edges(raw->relations, raw->relation_count);
        owner->spatial = edges(raw->spatial_relations, raw->spatial_count);
        owner->semantic = edges(raw->semantic_relations, raw->semantic_count);
        auto &out = owner->api;
        out.struct_size = sizeof(out);
        out.abi_version = 1;
        out.relation_count = raw->relation_count;
        out.spatial_count = raw->spatial_count;
        out.semantic_count = raw->semantic_count;
        out.pair_count = raw->pair_count;
        out.predicate_count = raw->predicate_count;
        out.relations = owner->edges.data();
        out.spatial_relations = owner->spatial.data();
        out.semantic_relations = owner->semantic.data();
        out.predicate_logits = raw->predicate_logits;
        out.pair_logits = raw->pair_logits;
        out.subject_indices = raw->subject_indices;
        out.object_indices = raw->object_indices;
        out.valid_mask = raw->valid_mask;
        out.predicate_spatial = raw->predicate_spatial;
        out.predicate_names = raw->predicate_names;
        out.backend_buffer_bytes = raw->backend_buffer_bytes;
        out.forward_ms = raw->forward_ms;
        out.timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
        out.timings.valid_fields = raw->timings.valid_fields;
        out.timings.preprocess_ms = raw->timings.preprocess_ms;
        out.timings.e2e_ms = raw->timings.e2e_ms;
        out.private_owner = owner.get();
        ctx->timings = out.timings;
        ctx->has_timings = true;
        *output = &owner.release()->api;
        ctx->error[0] = 0;
        return 0;
    } catch (const std::exception &e) {
        error(ctx, e.what());
    } catch (...) {
        error(ctx, "unknown relation inference error");
    }
    return -1;
}
void aicore_relation_result_free(aicore_relation_result *r) {
    if (r) delete static_cast<Result *>(r->private_owner);
}
int aicore_relation_encode_text(aicore_relation_ctx *ctx,
                                const char *const *texts,
                                size_t count,
                                aicore_relation_embeddings **output) {
    if (output) *output = nullptr;
    if (!ctx) return -1;
    ctx->has_timings = false;
    try {
        require(output && ctx->loaded && texts && count && count <= 10000,
                "invalid text/context/output");
        std::vector<std::string> names;
        names.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            require(texts[i] != nullptr, "null predicate text");
            names.emplace_back(texts[i]);
        }
        if (aicore_cancel_requested()) {
            error(ctx, "cancelled");
            return 1;
        }
        auto lock = aicore::runtime::lock_backend_leases(
                {ctx->accelerator, ctx->cpu});
        auto owner = std::make_unique<Embeddings>();
        const auto start = Clock::now();
        if (!ctx->text_encoder)
            ctx->text_encoder =
                    std::make_unique<relateanything::PredicateEncoder>(
                            ctx->text_options);
        owner->values = ctx->text_encoder->encode(names);
        if (aicore_cancel_requested()) {
            error(ctx, "cancelled");
            return 1;
        }
        for (float v : owner->values)
            require(std::isfinite(v), "non-finite text embedding");
        auto &out = owner->api;
        out.struct_size = sizeof(out);
        out.abi_version = 1;
        out.count = count;
        out.dimension = 512;
        out.values = owner->values.data();
        out.timings.abi_version = 1;
        out.timings.valid_fields = AICORE_TIMING_E2E;
        out.timings.e2e_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - start)
                        .count();
        out.private_owner = owner.get();
        ctx->timings = out.timings;
        ctx->has_timings = true;
        *output = &owner.release()->api;
        ctx->error[0] = 0;
        return 0;
    } catch (const std::exception &e) {
        error(ctx, e.what());
    } catch (...) {
        error(ctx, "unknown text encoding error");
    }
    return -1;
}
void aicore_relation_embeddings_free(aicore_relation_embeddings *r) {
    if (r) delete static_cast<Embeddings *>(r->private_owner);
}
int aicore_relation_last_pipeline_timings(const aicore_relation_ctx *ctx,
                                          aicore_pipeline_timings *out) {
    if (!ctx || !ctx->has_timings || !out) return -1;
    *out = ctx->timings;
    return 0;
}
char *aicore_relation_model_cache_dir(void) {
    try {
        const auto s = aicore::extract_model_dir("relation_models");
        auto *p = static_cast<char *>(std::malloc(s.size() + 1));
        if (p) std::memcpy(p, s.c_str(), s.size() + 1);
        return p;
    } catch (...) {
        return nullptr;
    }
}
void aicore_relation_free_buffer(void *p) { std::free(p); }
void aicore_relation_shutdown(void) { aicore_runtime_shutdown(); }
