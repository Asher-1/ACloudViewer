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

namespace {

bool errorContains(const char* needle) {
    const char* error = aicore_ocr_last_error();
    return error && std::strstr(error, needle);
}

}  // namespace

int main() {
    int failures = 0;

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

    if (aicore_ocr_context_ready(nullptr) != 0 ||
        std::strcmp(aicore_ocr_context_device(nullptr), "") != 0 ||
        aicore_ocr_label_count(nullptr) != 0 ||
        aicore_ocr_label_at(nullptr, 0) != nullptr ||
        aicore_ocr_box_count(nullptr) != 0 ||
        aicore_ocr_box_at(nullptr, 0) != nullptr ||
        aicore_ocr_region_count(nullptr) != 0 ||
        aicore_ocr_region_at(nullptr, 0) != nullptr ||
        aicore_ocr_relation_count(nullptr) != 0 ||
        aicore_ocr_relation_at(nullptr, 0) != nullptr) {
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
