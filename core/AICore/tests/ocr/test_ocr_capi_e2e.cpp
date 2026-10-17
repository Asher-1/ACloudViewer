// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// End-to-end `ocr` C ABI test on real det/rec models: the recognize-only
// path (aicore_ocr_recognize) against the det path (aicore_ocr_run) on the
// PaddleOCR table demo (ppocr_testdata/test_files/table.jpg).
//
// Invariants: feeding back the run() quads reproduces run() bit-exactly
// (same crop, same ratio-group pad width, same chunks); results are one per
// input quad in input order, so a reversed input yields the reversed texts;
// degenerate or non-finite quads yield an empty text with score 0 without
// disturbing their neighbours; a BGR view equals the RGB view; invalid
// arguments fail with an error and no result.
//
// Env: AICORE_TEST_OCR_DET_GGUF, AICORE_TEST_OCR_REC_GGUF,
// AICORE_TEST_TABLE_IMAGE, AICORE_TEST_DEVICE. Exits 77 when the assets are
// not fetched.

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define access _access
#define R_OK 4
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"
#include "aicore/ocr_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

namespace {

const char* env(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] ? v : nullptr;
}

bool readable(const char* path) { return path && access(path, R_OK) == 0; }

struct Image {
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
};

bool load(const char* path, Image& out) {
    uint8_t* data = nullptr;
    if (aicore_map_anything_load_path_rgb(path, &data, &out.w, &out.h) != 0 ||
        !data)
        return false;
    out.rgb.assign(data, data + static_cast<size_t>(out.w) * out.h * 3);
    aicore_map_anything_free_buffer(data);
    return true;
}

aicore_image_view view(const std::vector<uint8_t>& px,
                       const Image& img,
                       aicore_image_format format) {
    aicore_image_view v{};
    v.data = px.data();
    v.width = img.w;
    v.height = img.h;
    v.row_stride_bytes = static_cast<size_t>(img.w) * 3;
    v.format = format;
    return v;
}

struct Box {
    float quad[4][2];
    std::string text;
    float score;
};

std::vector<Box> boxes(const aicore_ocr_result* r) {
    std::vector<Box> out;
    for (int i = 0; i < aicore_ocr_box_count(r); ++i) {
        const aicore_ocr_box* b = aicore_ocr_box_at(r, i);
        Box x;
        std::memcpy(x.quad, b->quad, sizeof(x.quad));
        x.text = b->text ? b->text : "";
        x.score = b->score;
        out.push_back(x);
    }
    return out;
}

std::vector<Box> recognize(aicore_ocr_ctx* ctx,
                           const aicore_image_view& v,
                           const std::vector<float>& quads) {
    aicore_ocr_result* r = aicore_ocr_recognize(
            ctx, &v, quads.data(), static_cast<int32_t>(quads.size() / 8));
    AICORE_CHECK(r != nullptr);
    if (!r) {
        std::fprintf(stderr, "recognize: %s\n", aicore_ocr_last_error());
        return {};
    }
    const aicore_pipeline_timings tm = aicore_ocr_timings(r);
    AICORE_CHECK((tm.valid_fields & AICORE_TIMING_INFERENCE) != 0);
    AICORE_CHECK(tm.e2e_ms >= tm.inference_ms);
    std::vector<Box> out = boxes(r);
    aicore_ocr_result_release(r);
    return out;
}

void append(std::vector<float>& quads, const float (&q)[4][2]) {
    for (int i = 0; i < 4; ++i) {
        quads.push_back(q[i][0]);
        quads.push_back(q[i][1]);
    }
}

}  // namespace

int main() {
    const char* det = env("AICORE_TEST_OCR_DET_GGUF");
    const char* rec = env("AICORE_TEST_OCR_REC_GGUF");
    const char* imagePath = env("AICORE_TEST_TABLE_IMAGE");
    const char* device =
            env("AICORE_TEST_DEVICE") ? env("AICORE_TEST_DEVICE") : "cpu";
    if (!readable(det) || !readable(rec) || !readable(imagePath)) {
        std::fprintf(stderr,
                     "test_ocr_capi_e2e: assets not fetched, skipping\n");
        return 77;
    }
    Image img;
    if (!load(imagePath, img)) {
        std::fprintf(stderr, "cannot decode %s\n", imagePath);
        return 1;
    }
    aicore_ocr_options o{};
    o.struct_size = sizeof(o);
    o.abi_version = AICORE_OCR_ABI_VERSION;
    o.det_model_path = det;
    o.rec_model_path = rec;
    o.device = device;
    aicore_ocr_ctx* ctx = aicore_ocr_context_create(&o);
    if (!ctx) {
        std::fprintf(stderr, "context create failed: %s\n",
                     aicore_ocr_last_error());
        return 1;
    }
    const aicore_image_view rgb = view(img.rgb, img, AICORE_IMAGE_RGB8);

    aicore_ocr_result* runResult = aicore_ocr_run(ctx, &rgb);
    AICORE_CHECK(runResult != nullptr);
    const std::vector<Box> ref =
            runResult ? boxes(runResult) : std::vector<Box>{};
    aicore_ocr_result_release(runResult);
    AICORE_CHECK(ref.size() >= 8);
    bool sawDemoText = false;
    for (const Box& b : ref) sawDemoText |= b.text == "CRuncover";
    AICORE_CHECK(sawDemoText);
    const size_t n = ref.size();

    // 1. run() quads fed back: identical texts and bit-identical scores.
    std::vector<float> quads;
    for (const Box& b : ref) append(quads, b.quad);
    const std::vector<Box> same = recognize(ctx, rgb, quads);
    AICORE_CHECK(same.size() == n);
    for (size_t i = 0; i < same.size() && i < n; ++i) {
        AICORE_CHECK(same[i].text == ref[i].text);
        AICORE_CHECK(same[i].score == ref[i].score);
        AICORE_CHECK(std::memcmp(same[i].quad, ref[i].quad,
                                 sizeof(same[i].quad)) == 0);
    }

    // 2. Input order is output order. The ratio-group pad widths depend on
    // the quad set, not its order, so the texts follow the permutation.
    std::vector<float> reversed;
    for (size_t i = n; i-- > 0;) append(reversed, ref[i].quad);
    const std::vector<Box> rev = recognize(ctx, rgb, reversed);
    AICORE_CHECK(rev.size() == n);
    for (size_t i = 0; i < rev.size() && i < n; ++i) {
        AICORE_CHECK(rev[i].text == ref[n - 1 - i].text);
        AICORE_CHECK(std::fabs(rev[i].score - ref[n - 1 - i].score) < 1e-4f);
    }

    // 3. Degenerate quads in between valid ones: empty text, score 0, the
    // quad echoed, and the valid neighbours unchanged (a lone crop forms
    // its own pad group, so it is compared to a lone recognize).
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float point[4][2] = {{5, 5}, {5, 5}, {5, 5}, {5, 5}};
    const float outside[4][2] = {
            {-90, -90}, {-10, -90}, {-10, -10}, {-90, -10}};
    const float nanQuad[4][2] = {{nan, 0}, {10, 0}, {10, 10}, {0, 10}};
    std::vector<float> lone;
    append(lone, ref[0].quad);
    const std::vector<Box> loneRef = recognize(ctx, rgb, lone);
    std::vector<float> mixed;
    append(mixed, point);
    append(mixed, ref[0].quad);
    append(mixed, nanQuad);
    append(mixed, outside);
    const std::vector<Box> mix = recognize(ctx, rgb, mixed);
    AICORE_CHECK(mix.size() == 4);
    AICORE_CHECK(loneRef.size() == 1);
    if (mix.size() == 4 && loneRef.size() == 1) {
        for (int i : {0, 2, 3}) {
            AICORE_CHECK(mix[i].text.empty());
            AICORE_CHECK(mix[i].score == 0.0f);
        }
        AICORE_CHECK(std::memcmp(mix[3].quad, outside, sizeof(outside)) == 0);
        AICORE_CHECK(mix[1].text == loneRef[0].text);
        AICORE_CHECK(mix[1].score == loneRef[0].score);
        AICORE_CHECK(!mix[1].text.empty());
    }

    // 4. A BGR view of the same pixels is the same input.
    std::vector<uint8_t> bgrPx(img.rgb.size());
    for (size_t i = 0; i < bgrPx.size(); i += 3) {
        bgrPx[i] = img.rgb[i + 2];
        bgrPx[i + 1] = img.rgb[i + 1];
        bgrPx[i + 2] = img.rgb[i];
    }
    const std::vector<Box> bgr =
            recognize(ctx, view(bgrPx, img, AICORE_IMAGE_BGR8), quads);
    AICORE_CHECK(bgr.size() == n);
    for (size_t i = 0; i < bgr.size() && i < n; ++i) {
        AICORE_CHECK(bgr[i].text == ref[i].text);
        AICORE_CHECK(bgr[i].score == ref[i].score);
    }

    // 5. Invalid arguments: no result, an error message.
    AICORE_CHECK(aicore_ocr_recognize(ctx, &rgb, quads.data(), 0) == nullptr);
    AICORE_CHECK(aicore_ocr_last_error()[0] != '\0');
    AICORE_CHECK(aicore_ocr_recognize(ctx, &rgb, nullptr, 1) == nullptr);
    AICORE_CHECK(aicore_ocr_recognize(nullptr, &rgb, quads.data(), 1) ==
                 nullptr);

    aicore_ocr_context_destroy(ctx);
    aicore_ocr_runtime_shutdown();
    if (failures) {
        std::fprintf(stderr, "test_ocr_capi_e2e: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_ocr_capi_e2e: OK (%zu boxes on %s)\n", n, device);
    return 0;
}
