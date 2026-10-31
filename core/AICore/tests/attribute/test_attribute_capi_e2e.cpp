// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <json/json.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "aicore/attribute_capi.h"
#include "aicore/map_anything_capi.h"
#include "aicore/runtime_capi.h"

namespace {
int failures = 0;
void check(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
const char* env(const char* key) {
    const char* value = std::getenv(key);
    return value && *value ? value : nullptr;
}
using Context =
        std::unique_ptr<aicore_attribute_ctx, decltype(&aicore_attribute_free)>;
using Result = std::unique_ptr<aicore_attribute_result,
                               decltype(&aicore_attribute_result_free)>;
Result infer(aicore_attribute_ctx* ctx, const aicore_image_view& view) {
    aicore_attribute_result* result = nullptr;
    check(aicore_attribute_run(ctx, &view, &result) == 0,
          aicore_attribute_last_error(ctx));
    return Result(result, aicore_attribute_result_free);
}
void verify(const aicore_attribute_result* result,
            const Json::Value& truth,
            const aicore_attribute_preset_entry& preset,
            int width,
            int height) {
    check(result != nullptr, "Missing real typed result");
    if (!result) return;
    check(result->struct_size == sizeof(*result) && result->abi_version == 1,
          "Result ABI identity");
    check(std::string(result->label) == preset.label &&
                  result->width == width && result->height == height,
          "Original full-image geometry and task class");
    check(result->probability_count == preset.probability_count &&
                  result->attribute_count == preset.attribute_count,
          "Complete probability and attribute-group counts");
    check(truth["probabilities"].size() == result->probability_count,
          "Independent reference output count");
    if (truth["probabilities"].size() != result->probability_count) return;
    for (size_t i = 0; i < result->probability_count; ++i)
        check(std::isfinite(result->probabilities[i]) &&
                      std::abs(result->probabilities[i] -
                               truth["probabilities"][Json::ArrayIndex(i)]
                                       .asFloat()) <= .001f,
              "Original F32 independent-source per-output gate");
    for (size_t i = 0; i < result->attribute_count; ++i) {
        const auto& item = result->attributes[i];
        check(truth["attributes"][item.name].asString() == item.value,
              "Original source attribute selection");
        check(item.probability_index >= 0 &&
                      size_t(item.probability_index) <
                              result->probability_count &&
                      std::isfinite(item.score) && item.score >= 0 &&
                      item.score <= 1,
              "Typed attribute probability index and score");
    }
}
}  // namespace

// The GGUF, original-source reference and image must be supplied explicitly;
// a missing-asset skip must not replace real acceptance.
int main() {
    const char* imagePath = env("AICORE_TEST_ATTRIBUTE_IMAGE");
    const char* device = env("AICORE_TEST_DEVICE");
    if (!imagePath) {
        std::fprintf(stderr, "Missing AICORE_TEST_ATTRIBUTE_IMAGE\n");
        return 2;
    }
    uint8_t* rgb = nullptr;
    int width = 0, height = 0;
    if (aicore_map_anything_load_path_rgb(imagePath, &rgb, &width, &height) !=
                0 ||
        !rgb)
        return 2;
    const std::vector<uint8_t> original(rgb, rgb + size_t(width) * height * 3);
    aicore_map_anything_free_buffer(rgb);
    const size_t stride = size_t(width) * 3 + 13;
    std::vector<uint8_t> padded(stride * height, 0x5a);
    for (int y = 0; y < height; ++y)
        std::copy_n(original.data() + size_t(y) * width * 3, size_t(width) * 3,
                    padded.data() + size_t(y) * stride);
    const auto immutable = padded;
    aicore_image_view view{padded.data(), width, height, stride,
                           AICORE_IMAGE_RGB8};
    for (int model = 0; model < 2; ++model) {
        const auto* preset = aicore_attribute_preset_at(model);
        const char* path = env(model ? "AICORE_TEST_ATTRIBUTE_VEHICLE_GGUF"
                                     : "AICORE_TEST_ATTRIBUTE_PERSON_GGUF");
        const char* reference =
                env(model ? "AICORE_TEST_ATTRIBUTE_VEHICLE_REFERENCE"
                          : "AICORE_TEST_ATTRIBUTE_PERSON_REFERENCE");
        if (!preset || !path || !reference) {
            std::fprintf(stderr,
                         "Missing the real model or the independent source "
                         "reference\n");
            return 2;
        }
        Json::Value truth;
        std::ifstream file(reference);
        if (!file || !(file >> truth)) return 2;
        check(truth["source_sha256"].asString() == preset->source_sha256,
              "Independent reference source identity");
        std::vector<float> first;
        for (int lifecycle = 0; lifecycle < 2; ++lifecycle) {
            auto options = aicore_attribute_default_options();
            options.model_path = path;
            options.device = device ? device : "cpu";
            options.threads = 4;
            Context ctx(aicore_attribute_create(&options),
                        aicore_attribute_free);
            check(aicore_attribute_ready(ctx.get()) == 1,
                  aicore_attribute_last_error(ctx.get()));
            if (!aicore_attribute_ready(ctx.get())) continue;
            options.model_path = "invalid-after-create.gguf";
            check(std::string(aicore_attribute_context_device(ctx.get()))
                                  .find(options.device) != std::string::npos,
                  "Must not silently fall back to another backend");
            auto output = infer(ctx.get(), view);
            verify(output.get(), truth, *preset, width, height);
            if (!output) continue;
            std::vector<float> values(
                    output->probabilities,
                    output->probabilities + output->probability_count);
            if (!lifecycle)
                first = values;
            else
                check(values == first,
                      "Numeric stability across context recreation");
            aicore_pipeline_timings timings{};
            timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
            check(aicore_attribute_last_pipeline_timings(ctx.get(), &timings) ==
                                  0 &&
                          timings.valid_fields == 23 &&
                          timings.preprocess_ms >= 0 &&
                          timings.inference_ms >= 0 &&
                          timings.postprocess_ms >= 0 &&
                          timings.e2e_ms >= timings.inference_ms,
                  "Common stage timing");
            aicore_attribute_result* rejected = output.get();
            auto invalid = view;
            invalid.row_stride_bytes = 1;
            check(aicore_attribute_run(ctx.get(), &invalid, &rejected) == -1 &&
                          !rejected && *aicore_attribute_last_error(ctx.get()),
                  "An invalid stride clears the output and reports an error");
            auto* token = aicore_cancel_token_new();
            check(token != nullptr, "Independent cancel-token allocation");
            if (token) {
                aicore_cancel_token_request(token);
                aicore_cancel_scope_begin(token);
                check(aicore_attribute_run(ctx.get(), &view, &rejected) == -1 &&
                              !rejected,
                      "A pre-cancelled run must not produce a successful "
                      "result");
                aicore_cancel_token_reset(token);
                auto recovered = infer(ctx.get(), view);
                verify(recovered.get(), truth, *preset, width, height);
                aicore_cancel_scope_end(token);
                aicore_cancel_token_free(token);
            }
            aicore_attribute_shutdown();
            auto repeated = infer(ctx.get(), view);
            if (repeated)
                check(std::vector<float>(repeated->probabilities,
                                         repeated->probabilities +
                                                 repeated->probability_count) ==
                              first,
                      "shutdown must not destroy a live context");
            ctx.reset();
            verify(output.get(), truth, *preset, width, height);
        }
    }
    check(padded == immutable,
          "Borrowed RGB pixels and padding must not be modified");
    aicore_attribute_shutdown();
    std::printf("attribute real C ABI: %d failures\n", failures);
    return failures ? 1 : 0;
}
