// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// End-to-end `table` C ABI test on real models: OCR det/rec + SLANet_plus
// on the PaddleOCR table demo (ppocr_testdata/test_files/table.jpg), plus
// the layout -> table composition on a document page when the layout model
// and page are configured.
//
// Invariants: the 4 x 4 grid with the colspan-4 header, the OCR text in the
// HTML/CSV, structure-only runs, BGR/stride view equivalence, repeated runs
// and a recreated context are byte-identical, timings, and the layout
// contract (error without layout labels; layout-sourced tables otherwise).
//
// Env: AICORE_TEST_TABLE_GGUF, AICORE_TEST_OCR_DET_GGUF,
// AICORE_TEST_OCR_REC_GGUF, AICORE_TEST_TABLE_IMAGE, AICORE_TEST_DEVICE;
// optional AICORE_TEST_OCR_LAYOUT_GGUF + AICORE_TEST_TABLE_PAGE. Exits 77
// when the assets are not fetched.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"
#include "aicore/ocr_capi.h"
#include "aicore/table_capi.h"
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

aicore_image_view rgbView(const Image& img) {
    aicore_image_view v{};
    v.data = img.rgb.data();
    v.width = img.w;
    v.height = img.h;
    v.row_stride_bytes = static_cast<size_t>(img.w) * 3;
    v.format = AICORE_IMAGE_RGB8;
    return v;
}

aicore_ocr_ctx* makeOcr(const char* det,
                        const char* rec,
                        const char* layout,
                        const char* device) {
    aicore_ocr_options o{};
    o.struct_size = sizeof(o);
    o.abi_version = AICORE_OCR_ABI_VERSION;
    o.det_model_path = det;
    o.rec_model_path = rec;
    o.layout_model_path = layout;
    o.device = device;
    // Official table pipeline text detection: max 960 long side.
    o.det_limit_type = 1;
    o.det_max_side = 960;
    return aicore_ocr_context_create(&o);
}

aicore_table_ctx* makeTable(const char* model, const char* device) {
    aicore_table_options o{};
    o.struct_size = sizeof(o);
    o.abi_version = AICORE_TABLE_ABI_VERSION;
    o.model_path = model;
    o.device = device;
    return aicore_table_context_create(&o);
}

// Everything a caller can observe, serialized for byte comparison.
std::string fingerprint(const aicore_table_result* r) {
    std::string s;
    for (int i = 0; i < aicore_table_count(r); ++i) {
        const aicore_table* t = aicore_table_at(r, i);
        s += std::to_string(t->rows) + "x" + std::to_string(t->cols) + "|" +
             t->html + "|" + t->csv + "|";
        for (int c = 0; c < t->cell_count; ++c) {
            const aicore_table_cell* cell =
                    aicore_table_cell_at(r, t->cell_offset + c);
            char buf[160];
            std::snprintf(buf, sizeof(buf), "%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f;",
                          cell->row, cell->col, cell->row_span, cell->col_span,
                          cell->box[0], cell->box[1], cell->box[2],
                          cell->box[3]);
            s += buf;
        }
    }
    return s;
}

void checkDemoTable(const aicore_table_result* r, bool withText) {
    AICORE_CHECK(aicore_table_count(r) == 1);
    const aicore_table* t = aicore_table_at(r, 0);
    AICORE_CHECK(t != nullptr);
    if (!t) return;
    AICORE_CHECK(t->layout_region == -1);
    AICORE_CHECK(t->rows == 4 && t->cols == 4);
    AICORE_CHECK(t->cell_count == 13);
    AICORE_CHECK(aicore_table_cell_count(r) == 13);
    AICORE_CHECK(t->structure_score > 0.9f && t->structure_score <= 1.0f);
    int tokens = 0;
    while (aicore_table_token_at(r, 0, tokens)) ++tokens;
    AICORE_CHECK(tokens == t->token_count);
    AICORE_CHECK(aicore_table_token_at(r, 1, 0) == nullptr);
    const aicore_table_cell* head = aicore_table_cell_at(r, 0);
    AICORE_CHECK(head && head->row == 0 && head->col == 0 &&
                 head->col_span == 4 && head->row_span == 1);
    const aicore_table_cell* last = aicore_table_cell_at(r, 12);
    AICORE_CHECK(last && last->row == 3 && last->col == 3);
    const std::string html = t->html;
    AICORE_CHECK(html.rfind("<html><body><table><tr><td colspan=\"4\">", 0) ==
                 0);
    if (withText) {
        AICORE_CHECK(std::string(t->csv).rfind("CRuncover,,,\r\n", 0) == 0);
        AICORE_CHECK(html.find("$744.78") != std::string::npos);
        AICORE_CHECK(head && std::strcmp(head->text, "CRuncover") == 0 &&
                     head->ocr_box_count == 1);
    } else {
        AICORE_CHECK(std::string(t->csv).rfind(",,,\r\n", 0) == 0);
        AICORE_CHECK(head && head->text && head->text[0] == '\0' &&
                     head->ocr_box_count == 0);
    }
    const aicore_pipeline_timings tm = aicore_table_timings(r);
    AICORE_CHECK((tm.valid_fields & AICORE_TIMING_INFERENCE) != 0);
    AICORE_CHECK(tm.inference_ms > 0.0 && tm.e2e_ms >= tm.inference_ms);
}

}  // namespace

int main() {
    const char* tableModel = env("AICORE_TEST_TABLE_GGUF");
    const char* det = env("AICORE_TEST_OCR_DET_GGUF");
    const char* rec = env("AICORE_TEST_OCR_REC_GGUF");
    const char* imagePath = env("AICORE_TEST_TABLE_IMAGE");
    const char* device =
            env("AICORE_TEST_DEVICE") ? env("AICORE_TEST_DEVICE") : "cpu";
    if (!readable(tableModel) || !readable(det) || !readable(rec) ||
        !readable(imagePath)) {
        std::fprintf(stderr,
                     "test_table_capi_e2e: assets not fetched, skipping\n");
        return 77;
    }
    Image img;
    if (!load(imagePath, img)) {
        std::fprintf(stderr, "cannot decode %s\n", imagePath);
        return 1;
    }
    aicore_table_ctx* table = makeTable(tableModel, device);
    aicore_ocr_ctx* ocr = makeOcr(det, rec, nullptr, device);
    if (!table || !ocr) {
        std::fprintf(stderr, "context create failed: %s / %s\n",
                     aicore_table_last_error(), aicore_ocr_last_error());
        return 1;
    }
    AICORE_CHECK(aicore_table_context_ready(table) == 1);
    AICORE_CHECK(aicore_table_context_device(table)[0] != '\0');
    const aicore_image_view view = rgbView(img);

    // Structure only (no OCR result, NULL options = whole image).
    aicore_table_result* bare =
            aicore_table_run(table, &view, nullptr, nullptr, nullptr);
    AICORE_CHECK(bare != nullptr);
    if (bare) checkDemoTable(bare, false);

    aicore_ocr_result* boxes = aicore_ocr_run(ocr, &view);
    AICORE_CHECK(boxes != nullptr && aicore_ocr_box_count(boxes) > 0);
    aicore_table_run_options ro{};
    ro.struct_size = sizeof(ro);
    ro.abi_version = AICORE_TABLE_ABI_VERSION;
    aicore_table_result* first =
            aicore_table_run(table, &view, ocr, boxes, &ro);
    AICORE_CHECK(first != nullptr);
    std::string reference;
    if (first) {
        checkDemoTable(first, true);
        reference = fingerprint(first);
        // Structure (and cell geometry) does not depend on the OCR input.
        if (bare) {
            const aicore_table* a = aicore_table_at(bare, 0);
            const aicore_table* b = aicore_table_at(first, 0);
            AICORE_CHECK(a && b && a->token_count == b->token_count);
            for (int c = 0; a && b && c < b->cell_count; ++c)
                AICORE_CHECK(std::memcmp(aicore_table_cell_at(bare, c)->box,
                                         aicore_table_cell_at(first, c)->box,
                                         sizeof(float) * 4) == 0);
        }
    }

    // Repeated run on the same context.
    aicore_table_result* again =
            aicore_table_run(table, &view, ocr, boxes, &ro);
    AICORE_CHECK(again && fingerprint(again) == reference);
    aicore_table_result_release(again);

    // BGR8 storage and a padded row stride describe the same pixels.
    std::vector<uint8_t> bgr(img.rgb.size());
    for (size_t i = 0; i < img.rgb.size(); i += 3) {
        bgr[i] = img.rgb[i + 2];
        bgr[i + 1] = img.rgb[i + 1];
        bgr[i + 2] = img.rgb[i];
    }
    aicore_image_view bgrView = view;
    bgrView.data = bgr.data();
    bgrView.format = AICORE_IMAGE_BGR8;
    aicore_table_result* fromBgr =
            aicore_table_run(table, &bgrView, ocr, boxes, &ro);
    AICORE_CHECK(fromBgr && fingerprint(fromBgr) == reference);
    aicore_table_result_release(fromBgr);
    const size_t stride = static_cast<size_t>(img.w) * 3 + 13;
    std::vector<uint8_t> padded(stride * img.h, 0xA5);
    for (int y = 0; y < img.h; ++y)
        std::memcpy(padded.data() + y * stride,
                    img.rgb.data() + static_cast<size_t>(y) * img.w * 3,
                    static_cast<size_t>(img.w) * 3);
    aicore_image_view stridedView = view;
    stridedView.data = padded.data();
    stridedView.row_stride_bytes = stride;
    aicore_table_result* fromStride =
            aicore_table_run(table, &stridedView, ocr, boxes, &ro);
    AICORE_CHECK(fromStride && fingerprint(fromStride) == reference);
    aicore_table_result_release(fromStride);

    // Invalid run options and layout without layout labels are errors.
    aicore_table_run_options badRo = ro;
    badRo.abi_version = AICORE_TABLE_ABI_VERSION + 1;
    AICORE_CHECK(aicore_table_run(table, &view, ocr, boxes, &badRo) == nullptr);
    aicore_table_run_options layoutRo = ro;
    layoutRo.use_layout_regions = 1;
    AICORE_CHECK(aicore_table_run(table, &view, ocr, boxes, &layoutRo) ==
                 nullptr);
    AICORE_CHECK(std::strstr(aicore_table_last_error(), "layout") != nullptr);

    // A recreated context reproduces the result.
    aicore_table_context_destroy(table);
    table = makeTable(tableModel, device);
    AICORE_CHECK(table != nullptr);
    if (table) {
        aicore_table_result* recreated =
                aicore_table_run(table, &view, ocr, boxes, &ro);
        AICORE_CHECK(recreated && fingerprint(recreated) == reference);
        aicore_table_result_release(recreated);
    }
    aicore_table_result_release(first);
    aicore_table_result_release(bare);
    aicore_ocr_result_release(boxes);

    // layout -> table on a document page.
    const char* layout = env("AICORE_TEST_OCR_LAYOUT_GGUF");
    const char* pagePath = env("AICORE_TEST_TABLE_PAGE");
    Image page;
    if (table && readable(layout) && readable(pagePath) &&
        load(pagePath, page)) {
        aicore_ocr_ctx* layoutOcr = makeOcr(det, rec, layout, device);
        AICORE_CHECK(layoutOcr != nullptr);
        if (layoutOcr) {
            const aicore_image_view pageView = rgbView(page);
            aicore_ocr_result* pageBoxes = aicore_ocr_run(layoutOcr, &pageView);
            AICORE_CHECK(pageBoxes && aicore_ocr_region_count(pageBoxes) > 0);
            aicore_table_result* tables = aicore_table_run(
                    table, &pageView, layoutOcr, pageBoxes, &layoutRo);
            AICORE_CHECK(tables && aicore_table_count(tables) >= 1);
            for (int i = 0; tables && i < aicore_table_count(tables); ++i) {
                const aicore_table* t = aicore_table_at(tables, i);
                AICORE_CHECK(t->layout_region >= 0 &&
                             t->layout_region <
                                     aicore_ocr_region_count(pageBoxes));
                AICORE_CHECK(t->box[0] >= 0 && t->box[2] <= page.w &&
                             t->box[1] >= 0 && t->box[3] <= page.h);
                AICORE_CHECK(t->rows > 0 && t->cols > 0 && t->cell_count > 0);
            }
            std::printf("layout page: %d regions, %d tables\n",
                        aicore_ocr_region_count(pageBoxes),
                        aicore_table_count(tables));
            aicore_table_result_release(tables);
            aicore_ocr_result_release(pageBoxes);
            aicore_ocr_context_destroy(layoutOcr);
        }
    } else {
        std::printf("layout page: not configured, composition check skipped\n");
    }

    aicore_ocr_context_destroy(ocr);
    aicore_table_context_destroy(table);
    aicore_table_runtime_shutdown();
    if (failures) {
        std::fprintf(stderr, "test_table_capi_e2e: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_table_capi_e2e: OK (%s)\n", device);
    return 0;
}
