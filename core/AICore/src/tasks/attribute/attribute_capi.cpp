// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "aicore/attribute_capi.h"
#include "aicore/runtime_capi.h"
#include "common/ggml_backend_registry.hpp"
#include "common/capi_utils.hpp"
#include "common/data_root_util.hpp"
#include "common/image_utils.hpp"
#include "common/paddle_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

struct Group {
    const char* name;
    std::vector<const char*> labels;
    float threshold;
};
struct Recipe {
    aicore_attribute_preset_entry preset;
    std::vector<Group> groups;
};

// Preserve pinned PULC order, strict thresholds and group argmax; no softmax.
const Recipe recipes[] = {
    {{sizeof(aicore_attribute_preset_entry), AICORE_ATTRIBUTE_ABI_VERSION,
      "pulc-person-f32.gguf", "PULC Person (local F32)", "person",
      "3694c7f269585096eb378d1183bbdc4e8fa0e3dc0c89cb4591738cf538fd2827", 26, 21},
     {{"Hat", {"Yes", "No"}, .5f}, {"Glasses", {"Yes", "No"}, .3f},
      {"Sleeve", {"ShortSleeve", "LongSleeve"}, -1.f},
      {"UpperStride", {"Yes", "No"}, .5f}, {"UpperLogo", {"Yes", "No"}, .5f},
      {"UpperPlaid", {"Yes", "No"}, .5f}, {"UpperSplice", {"Yes", "No"}, .5f},
      {"LowerStripe", {"Yes", "No"}, .5f}, {"LowerPattern", {"Yes", "No"}, .5f},
      {"LongCoat", {"Yes", "No"}, .5f}, {"Trousers", {"Yes", "No"}, .5f},
      {"Shorts", {"Yes", "No"}, .5f}, {"Skirt&Dress", {"Yes", "No"}, .5f},
      {"Shoe", {"Boots", "No boots"}, .5f}, {"HandBag&Dress", {"Yes", "No"}, .5f},
      {"ShoulderBag&Dress", {"Yes", "No"}, .5f}, {"Backpack", {"Yes", "No"}, .5f},
      {"HoldObjectsInFront", {"Yes", "No"}, .6f},
      {"Age", {"AgeLess18", "Age18-60", "AgeOver60"}, -1.f},
      {"Gender", {"Female", "Male"}, .5f}, {"Direction", {"Front", "Side", "Back"}, -1.f}}},
    {{sizeof(aicore_attribute_preset_entry), AICORE_ATTRIBUTE_ABI_VERSION,
      "pulc-vehicle-f32.gguf", "PULC Vehicle (local F32)", "vehicle",
      "78362592b60c19b07cd3e41ad680dcc1fdff246e3fe450da8e46d45737a6e07c", 19, 2},
     {{"Color", {"yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black"}, -1.f},
      {"Type", {"sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck", "estate"}, -1.f}}}
};

struct OwnedResult {
    aicore_attribute_result result{};
    std::vector<float> probabilities;
    std::vector<aicore_attribute_item> attributes;
};
}  // namespace

struct aicore_attribute_ctx {
    aicore::runtime::BackendLease primary, cpu;
    std::unique_ptr<aicore::paddle::OcrWeights> weights;
    aicore::paddle::OcrExecPlan plan;
    aicore::paddle::OcrGraphExec execution;
    std::shared_ptr<aicore::paddle::OcrFoldPool> fold;
    ggml_context* graphContext = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *input = nullptr, *output = nullptr;
    const Recipe* recipe = nullptr;
    int width = 0, height = 0;
    std::string error;
    aicore_pipeline_timings timings{};
    bool hasTimings = false;

    ~aicore_attribute_ctx() {
        const auto lock = aicore::runtime::lock_backend_leases({primary, cpu});
        execution.free();
        if (graphContext) ggml_free(graphContext);
        fold.reset();
        weights.reset();
    }
    bool ready() const { return recipe && graph && input && output && primary; }
};

extern "C" {
AICORE_CAPI int aicore_attribute_preset_count(void) {
    return static_cast<int>(sizeof(recipes) / sizeof(recipes[0]));
}
AICORE_CAPI const aicore_attribute_preset_entry* aicore_attribute_preset_at(int index) {
    return index >= 0 && index < aicore_attribute_preset_count() ? &recipes[index].preset : nullptr;
}
AICORE_CAPI char* aicore_attribute_model_cache_dir(void) {
    try { return aicore::capi::dup_cstr(aicore::extract_model_dir("attribute_models")); }
    catch (...) { return nullptr; }
}
AICORE_CAPI void aicore_attribute_free_buffer(char* buffer) { std::free(buffer); }

AICORE_CAPI aicore_attribute_options aicore_attribute_default_options(void) {
    return {sizeof(aicore_attribute_options), AICORE_ATTRIBUTE_ABI_VERSION, nullptr, "auto", 0};
}

AICORE_CAPI aicore_attribute_ctx* aicore_attribute_create(const aicore_attribute_options* options) {
    std::unique_ptr<aicore_attribute_ctx> ctx;
    try {
        ctx = std::make_unique<aicore_attribute_ctx>();
        if (!options || options->struct_size < sizeof(*options) ||
            options->abi_version != AICORE_ATTRIBUTE_ABI_VERSION)
            throw std::runtime_error("attribute: invalid options ABI");
        if (!options->model_path || !*options->model_path)
            throw std::runtime_error("attribute: local model file required");
        ctx->weights = std::make_unique<aicore::paddle::OcrWeights>();
        if (!ctx->weights->load(options->model_path, &ctx->error))
            throw std::runtime_error(ctx->error);
        const int key = gguf_find_key(ctx->weights->gguf, "ocr.source.sha256");
        if (key < 0 || gguf_get_kv_type(ctx->weights->gguf, key) != GGUF_TYPE_STRING ||
            ctx->weights->role != "cls")
            throw std::runtime_error("attribute: missing source identity or classification graph contract");
        const char* source = gguf_get_val_str(ctx->weights->gguf, key);
        for (const auto& recipe : recipes)
            if (std::strcmp(source, recipe.preset.source_sha256) == 0) ctx->recipe = &recipe;
        if (!ctx->recipe) throw std::runtime_error("attribute: unsupported attribute model source");
        const auto shape = ctx->weights->program.staticShapes.find(ctx->weights->program.feedInput);
        std::string shapeInfo = ctx->weights->program.feedInput;
        if (shape != ctx->weights->program.staticShapes.end())
            for (int64_t dim : shape->second) shapeInfo += "/" + std::to_string(dim);
        if (shape == ctx->weights->program.staticShapes.end() || shape->second.size() != 4 ||
            (shape->second[0] != 1 && shape->second[0] != -1) || shape->second[1] != 3 ||
            shape->second[2] <= 0 || shape->second[2] > 4096 ||
            shape->second[3] <= 0 || shape->second[3] > 4096 ||
            shape->second[2] * shape->second[3] > 4 * 1024 * 1024)
            throw std::runtime_error("attribute: invalid NCHW input geometry " + shapeInfo);
        ctx->height = static_cast<int>(shape->second[2]);
        ctx->width = static_cast<int>(shape->second[3]);
        ctx->primary = aicore::runtime::acquire_backend_lease(
            options->device ? options->device : "auto", options->threads, &ctx->error);
        if (!ctx->primary) throw std::runtime_error(ctx->error);
        ctx->cpu = ctx->primary.is_cpu() ? ctx->primary :
            aicore::runtime::acquire_backend_lease("cpu", options->threads, &ctx->error);
        if (!ctx->cpu) throw std::runtime_error(ctx->error);
        const auto lock = aicore::runtime::lock_backend_leases({ctx->primary, ctx->cpu});
        ctx->plan = aicore::paddle::makeExecPlan(
            ctx->primary.is_cpu() ? nullptr : ctx->primary.handle(), ctx->cpu.handle(), true);
        auto build = aicore::paddle::execBuildOptions(ctx->plan);
        ctx->graphContext = ggml_init({64 * 1024 * 1024, nullptr, true});
        if (!ctx->graphContext) throw std::runtime_error("attribute: graph metadata allocation failed");
        ctx->input = ggml_new_tensor_4d(ctx->graphContext, GGML_TYPE_F32, ctx->width, ctx->height, 3, 1);
        ggml_set_name(ctx->input, ctx->weights->program.feedInput.c_str());
        ggml_set_input(ctx->input);
        ctx->weights->resetFoldPool();
        ctx->output = aicore::paddle::buildGraph(ctx->graphContext, *ctx->weights, ctx->input,
                                               &ctx->error, nullptr, nullptr, &build);
        ctx->fold = ctx->weights->foldPool();
        if (!ctx->output || ctx->output->type != GGML_TYPE_F32 ||
            ggml_nelements(ctx->output) != static_cast<int64_t>(ctx->recipe->preset.probability_count))
            throw std::runtime_error("attribute: attribute output shape mismatch: " + ctx->error);
        if (!aicore::paddle::prepareGraph(*ctx->weights, ctx->graphContext, ctx->plan,
                {ctx->output}, &ctx->graph, ctx->execution, ctx->error))
            throw std::runtime_error(ctx->error);
        ctx->error.clear();
    } catch (const std::exception& e) {
        if (!ctx) return nullptr;
        ctx->error = e.what();
        ctx->graph = nullptr;
    } catch (...) {
        if (!ctx) return nullptr;
        ctx->error = "attribute: unknown load error";
        ctx->graph = nullptr;
    }
    return ctx.release();
}

AICORE_CAPI void aicore_attribute_free(aicore_attribute_ctx* ctx) { delete ctx; }
AICORE_CAPI int aicore_attribute_ready(const aicore_attribute_ctx* ctx) { return ctx && ctx->ready(); }
AICORE_CAPI const char* aicore_attribute_last_error(const aicore_attribute_ctx* ctx) {
    return ctx ? ctx->error.c_str() : "attribute: NULL context";
}
AICORE_CAPI const char* aicore_attribute_context_device(const aicore_attribute_ctx* ctx) {
    return ctx && ctx->primary ? ctx->primary.device().c_str() : "";
}

AICORE_CAPI int aicore_attribute_run(aicore_attribute_ctx* ctx,
                                     const aicore_image_view* image,
                                     aicore_attribute_result** output) {
    if (output) *output = nullptr;
    if (!ctx) return -1;
    try {
        const auto begin = Clock::now();
        if (!ctx->ready() || !image || !output)
            throw std::runtime_error("attribute: context, input or output unavailable");
        if (aicore_cancel_requested()) throw std::runtime_error("attribute: cancelled");
        aicore::img::Image view;
        if (!aicore::img::image_from_view(*image, view))
            throw std::runtime_error("attribute: invalid image stride/format");
        const size_t count = static_cast<size_t>(ctx->width) * ctx->height;
        // Fixed source-model input: 8U RGB plus NCHW F32 scratch, about 15 bytes/pixel.
        std::vector<uint8_t> resized(count * 3);
        std::vector<float> pixels(count * 3);
        aicore::img::cv_resize_linear_u8(view, resized.data(), ctx->width, ctx->height, 3);
        constexpr float mean[] = {.485f, .456f, .406f};
        constexpr float stddev[] = {.229f, .224f, .225f};
        for (size_t i = 0; i < count; ++i)
            for (size_t c = 0; c < 3; ++c)
                pixels[c * count + i] = (resized[i * 3 + c] * (1.f / 255.f) - mean[c]) / stddev[c];
        const auto prepared = Clock::now();
        auto result = std::make_unique<OwnedResult>();
        result->probabilities.resize(ctx->recipe->preset.probability_count);
        {
            const auto lock = aicore::runtime::lock_backend_leases({ctx->primary, ctx->cpu});
            ggml_backend_tensor_set(ctx->input, pixels.data(), 0, pixels.size() * sizeof(float));
            aicore::paddle::computeGraph(ctx->plan, ctx->execution, ctx->graph);
            ggml_backend_tensor_get(ctx->output, result->probabilities.data(), 0,
                                    result->probabilities.size() * sizeof(float));
        }
        if (aicore_cancel_requested()) throw std::runtime_error("attribute: cancelled");
        const auto computed = Clock::now();
        for (float value : result->probabilities)
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::runtime_error("attribute: invalid attribute probability");
        size_t offset = 0;
        for (const auto& group : ctx->recipe->groups) {
            const bool binary = group.threshold >= 0;
            const size_t width = binary ? 1 : group.labels.size();
            if (offset + width > result->probabilities.size())
                throw std::runtime_error("attribute: attribute group out of bounds");
            size_t selected = 0;
            if (binary) selected = result->probabilities[offset] > group.threshold ? 0 : 1;
            else for (size_t i = 1; i < width; ++i)
                if (result->probabilities[offset + i] > result->probabilities[offset + selected]) selected = i;
            const size_t index = offset + (binary ? 0 : selected);
            const float probability = result->probabilities[index];
            result->attributes.push_back({group.name, group.labels[selected],
                binary && selected == 1 ? 1.f - probability : probability, static_cast<int32_t>(index)});
            offset += width;
        }
        if (offset != result->probabilities.size()) throw std::runtime_error("attribute: attribute probabilities not fully consumed");
        result->result = {sizeof(aicore_attribute_result), AICORE_ATTRIBUTE_ABI_VERSION,
            ctx->recipe->preset.label, image->width, image->height, result->attributes.data(),
            result->attributes.size(), result->probabilities.data(), result->probabilities.size(), result.get()};
        const auto end = Clock::now();
        ctx->timings = {AICORE_PIPELINE_TIMINGS_ABI_VERSION,
            AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE | AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E,
            elapsed(begin, prepared), elapsed(prepared, computed), elapsed(computed, end), 0, elapsed(begin, end)};
        ctx->hasTimings = true;
        ctx->error.clear();
        *output = &result.release()->result;
        return 0;
    } catch (const std::exception& e) { ctx->error = e.what(); }
      catch (...) { ctx->error = "attribute: unknown inference error"; }
    return -1;
}

AICORE_CAPI void aicore_attribute_result_free(aicore_attribute_result* result) {
    if (result) delete static_cast<OwnedResult*>(result->private_owner);
}
AICORE_CAPI int aicore_attribute_last_pipeline_timings(
        const aicore_attribute_ctx* ctx, aicore_pipeline_timings* output) {
    if (!ctx || !output || !ctx->hasTimings || output->abi_version != AICORE_PIPELINE_TIMINGS_ABI_VERSION)
        return -1;
    *output = ctx->timings;
    return 0;
}
AICORE_CAPI void aicore_attribute_shutdown(void) { aicore_runtime_shutdown(); }
}  // extern "C"
