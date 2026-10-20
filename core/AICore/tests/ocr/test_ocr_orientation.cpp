// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <aicore/map_anything_capi.h>
#include <aicore/ocr_capi.h>
#include <json/json.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "tests/common/test_macros.hpp"

static int failures = 0;

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: %s DET REC CLS IMAGE REFERENCE_JSON DEVICE\n",
                     argv[0]);
        return 2;
    }
    Json::Value reference;
    std::ifstream file(argv[5]);
    if (!file || !(file >> reference) || reference["rows"].size() < 7) return 2;
    uint8_t* pixels = nullptr;
    int width = 0, height = 0;
    if (aicore_map_anything_load_path_rgb(argv[4], &pixels, &width, &height) ||
        !pixels)
        return 2;
    // Deliberately padded BGR input exercises the borrowed stride contract.
    const size_t stride = width * 3 + 19;
    std::vector<uint8_t> bgr(stride * height, 0x7f);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
                bgr[y * stride + x * 3 + c] =
                        pixels[(y * width + x) * 3 + 2 - c];
    aicore_map_anything_free_buffer(pixels);
    aicore_image_view image{bgr.data(), width, height, stride,
                            AICORE_IMAGE_BGR8};
    std::vector<float> quads;
    for (const auto& row : reference["rows"])
        for (const auto& corner : row["quad"])
            for (const auto& value : corner) quads.push_back(value.asFloat());
    const int count = reference["rows"].size();
    for (float threshold : {.9f, 1.f}) {
        aicore_ocr_options options{};
        options.struct_size = sizeof(options);
        options.abi_version = AICORE_OCR_ABI_VERSION;
        options.det_model_path = argv[1];
        options.rec_model_path = argv[2];
        options.cls_model_path = argv[3];
        options.device = argv[6];
        options.cls_thresh = threshold;
        options.cls_max_batch = 6;
        options.rec_max_batch = 6;
        auto* ctx = aicore_ocr_context_create(&options);
        if (!ctx) {
            std::fprintf(stderr, "%s\n", aicore_ocr_last_error());
            return 1;
        }
        for (int repeat = 0; repeat < 2; ++repeat) {
            auto* result =
                    aicore_ocr_recognize(ctx, &image, quads.data(), count);
            AICORE_CHECK(result != nullptr);
            if (!result) {
                std::fprintf(stderr, "%s\n", aicore_ocr_last_error());
                aicore_ocr_context_destroy(ctx);
                return 1;
            }
            AICORE_CHECK(aicore_ocr_box_count(result) == count);
            for (int i = 0; i < count; ++i) {
                const auto* orientation = aicore_ocr_orientation_at(result, i);
                const auto* box = aicore_ocr_box_at(result, i);
                const auto& row = reference["rows"][i];
                AICORE_CHECK(orientation && box);
                if (!orientation || !box) continue;
                const float score = row["score"].asFloat();
                AICORE_CHECK(orientation->angle == row["angle"].asInt());
                AICORE_CHECK(std::isfinite(orientation->score));
                AICORE_CHECK(std::fabs(orientation->score - score) < .001f);
                AICORE_CHECK(
                        orientation->rotated ==
                        int(orientation->angle == 180 && score > threshold));
                AICORE_CHECK(std::memcmp(box->quad, quads.data() + i * 8,
                                         sizeof(box->quad)) == 0);
                if (threshold == .9f && row.isMember("text"))
                    AICORE_CHECK(std::string(box->text) ==
                                 row["text"].asString());
            }
            AICORE_CHECK(!aicore_ocr_orientation_at(result, -1));
            AICORE_CHECK(!aicore_ocr_orientation_at(result, count));
            const auto timing = aicore_ocr_timings(result);
            AICORE_CHECK(timing.valid_fields & AICORE_TIMING_INFERENCE);
            AICORE_CHECK(timing.e2e_ms >= timing.inference_ms);
            aicore_ocr_result_release(result);
        }
        aicore_ocr_context_destroy(ctx);
    }
    std::printf(
            "{\"orientation_rows\":%d,\"thresholds\":2,\"repeats\":2,"
            "\"failures\":%d}\n",
            count, failures);
    return failures ? 1 : 0;
}
