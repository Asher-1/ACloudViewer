// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <aicore/ocr_capi.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

bool errorContains(const char* needle) {
    const char* error = aicore_ocr_last_error();
    return error && std::strstr(error, needle);
}

}  // namespace

int main() {
    int failures = 0;

    const auto* tiny =
            aicore_ocr_preset_by_rec_filename("ppocrv6-tiny-rec-f32.gguf");
    const auto* small =
            aicore_ocr_preset_by_rec_filename("ppocrv6-small-rec-f32.gguf");
    const auto* japan =
            aicore_ocr_preset_by_rec_filename("ppocrv3-japan-rec-f32.gguf");
    const auto* sourceV4 =
            aicore_ocr_preset_by_rec_filename("ppocrv4-source-rec-f32.gguf");
    if (!japan || !sourceV4 ||
        std::strcmp(japan->det_stem, sourceV4->det_stem) ||
        std::strcmp(japan->display_name, "japan_PP-OCRv3") || !tiny || !small ||
        tiny->det_thresh != .2f || tiny->det_box_thresh != .4f ||
        small->det_box_thresh != .45f || tiny->det_max_candidates != 3000 ||
        tiny->cls_thresh != .9f || tiny->rec_batch != 6 ||
        aicore_ocr_preset_at(-1) ||
        aicore_ocr_preset_at(aicore_ocr_preset_count()) ||
        aicore_ocr_preset_by_rec_filename("unknown.gguf") ||
        // Recipes are not fictitious published download assets.
        aicore_ocr_model_by_filename("ppocrv6-tiny-rec-f32.gguf"))
        ++failures;

    aicore_ocr_options options{};
    options.struct_size = offsetof(aicore_ocr_options, device);
    options.abi_version = AICORE_OCR_ABI_VERSION;
    options.det_model_path = "/does/not/exist-det.gguf";
    options.rec_model_path = "/does/not/exist-rec.gguf";

    // The original ABI ended before `device`. Its struct_size must remain
    // accepted; model loading, rather than size validation, owns this failure.
    aicore_ocr_ctx* ctx = aicore_ocr_context_create(&options);
    if (ctx || !errorContains("det model load failed")) {
        std::fprintf(stderr, "legacy OCR options struct_size was rejected\n");
        ++failures;
    }
    aicore_ocr_context_destroy(ctx);

    // Options added after det_limit_type must not be read from older callers.
    options.struct_size = offsetof(aicore_ocr_options, cls_model_path);
    options.cls_thresh = std::numeric_limits<float>::quiet_NaN();
    ctx = aicore_ocr_context_create(&options);
    if (ctx || !errorContains("det model load failed")) ++failures;
    aicore_ocr_context_destroy(ctx);
    options.struct_size = sizeof(options);
    ctx = aicore_ocr_context_create(&options);
    if (ctx || !errorContains("invalid classifier")) ++failures;
    aicore_ocr_context_destroy(ctx);
    options.cls_thresh = 0;

    options.struct_size = offsetof(aicore_ocr_options, rec_max_batch);
    ctx = aicore_ocr_context_create(&options);
    if (ctx || !errorContains("struct_size")) {
        std::fprintf(stderr, "truncated OCR options were not rejected\n");
        ++failures;
    }
    aicore_ocr_context_destroy(ctx);

    options.struct_size = sizeof(options);
    options.abi_version = AICORE_OCR_ABI_VERSION + 1;
    ctx = aicore_ocr_context_create(&options);
    if (ctx || !errorContains("abi version mismatch")) {
        std::fprintf(stderr, "OCR ABI mismatch was not rejected\n");
        ++failures;
    }
    aicore_ocr_context_destroy(ctx);

    options.abi_version = AICORE_OCR_ABI_VERSION;
    options.kie_model_path = "/does/not/exist-kie.gguf";
    ctx = aicore_ocr_context_create(&options);
    if (ctx || !errorContains("KIE models are not supported")) {
        std::fprintf(stderr,
                     "unsupported OCR KIE option was silently accepted\n");
        ++failures;
    }
    aicore_ocr_context_destroy(ctx);

    if (aicore_ocr_recognize_regions(nullptr, nullptr, nullptr, 0) != nullptr ||
        aicore_ocr_context_ready(nullptr) != 0 ||
        std::strcmp(aicore_ocr_context_device(nullptr), "") != 0 ||
        aicore_ocr_label_count(nullptr) != 0 ||
        aicore_ocr_label_at(nullptr, 0) != nullptr ||
        aicore_ocr_box_count(nullptr) != 0 ||
        aicore_ocr_box_at(nullptr, 0) != nullptr ||
        aicore_ocr_region_count(nullptr) != 0 ||
        aicore_ocr_region_at(nullptr, 0) != nullptr ||
        aicore_ocr_relation_count(nullptr) != 0 ||
        aicore_ocr_relation_at(nullptr, 0) != nullptr ||
        aicore_ocr_orientation_at(nullptr, 0) != nullptr) {
        std::fprintf(stderr, "OCR null accessor contract failed\n");
        ++failures;
    }

    aicore_ocr_result* result = nullptr;
    const aicore_pipeline_timings timings = aicore_ocr_timings(result);
    const aicore_ocr_post_timing_breakdown post =
            aicore_ocr_post_timings(result);
    if (timings.valid_fields != 0 || post.valid_fields != 0) {
        std::fprintf(stderr, "OCR null timing contract failed\n");
        ++failures;
    }
    aicore_ocr_result_release(result);

    // Task shutdown delegates to the idempotent process runtime shutdown and
    // must remain safe when no contexts exist.
    aicore_ocr_runtime_shutdown();
    aicore_ocr_runtime_shutdown();

    if (failures == 0) std::printf("OCR C API contract OK\n");
    return failures == 0 ? 0 : 1;
}
