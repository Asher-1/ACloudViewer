// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// YOLO C API contract test — fast, no GGUF assets required. Covers ABI,
// options lifecycle (setters/getters), catalog queries, null guards and the
// model-free introspection/error paths. Inference itself is exercised by
// test_yolo_capi_performance (needs AICORE_TEST_YOLO_* assets).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/yolo_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_yolo_abi_version() >= 1);

    // Null-safe teardown / lifecycle.
    aicore_yolo_free(nullptr);
    aicore_yolo_options_free(nullptr);
    aicore_yolo_free_buffer(nullptr);
    aicore_yolo_free_buffer(nullptr);
    aicore_yolo_seg_result_free(nullptr);

    AICORE_CHECK(aicore_yolo_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_yolo_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_yolo_last_error(nullptr) == nullptr);

    // Options lifecycle: every setter must be a no-op on NULL (no crash),
    // and the getters must report the values we set.
    aicore_yolo_options* opts = aicore_yolo_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_yolo_options_set_device(opts, "cpu");
    aicore_yolo_options_set_threads(opts, 4);
    aicore_yolo_options_set_conf_thres(opts, 0.5f);
    aicore_yolo_options_set_iou_thres(opts, 0.6f);
    aicore_yolo_options_set_top_k(opts, 42);
    aicore_yolo_options_set_input_size(opts, 640, 480);
    aicore_yolo_options_set_log_level(opts, 1);
    aicore_yolo_options_set_keep_all_ops(opts, 1);
    aicore_yolo_options_set_profile_ops(opts, 1);
    aicore_yolo_options_set_profile_gaps(opts, 1);

    AICORE_CHECK(aicore_yolo_options_get_conf_thres(opts) == 0.5f);
    AICORE_CHECK(aicore_yolo_options_get_iou_thres(opts) == 0.6f);

    // Null guards on every setter (must be safe no-ops).
    aicore_yolo_options_set_device(nullptr, "cpu");
    aicore_yolo_options_set_threads(nullptr, 4);
    aicore_yolo_options_set_conf_thres(nullptr, 0.5f);
    aicore_yolo_options_set_iou_thres(nullptr, 0.6f);
    aicore_yolo_options_set_top_k(nullptr, 42);
    aicore_yolo_options_set_input_size(nullptr, 640, 480);
    aicore_yolo_options_set_log_level(nullptr, 1);
    aicore_yolo_options_set_keep_all_ops(nullptr, 1);
    aicore_yolo_options_set_profile_ops(nullptr, 1);
    aicore_yolo_options_set_profile_gaps(nullptr, 1);
    // NULL options expose the documented defaults (not zero).
    AICORE_CHECK(aicore_yolo_options_get_conf_thres(nullptr) == 0.25f);
    AICORE_CHECK(aicore_yolo_options_get_iou_thres(nullptr) == 0.7f);

    // Loading a nonexistent file must fail cleanly and report an error.
    aicore_yolo_ctx* ctx =
            aicore_yolo_load_opts("/nonexistent/yolo.gguf", opts);
    AICORE_CHECK(ctx != nullptr);  // ctx is allocated; model inside is null
    AICORE_CHECK(aicore_yolo_is_ready(ctx) == 0);
    AICORE_CHECK(aicore_yolo_last_error(ctx) != nullptr);
    AICORE_CHECK(std::strcmp(aicore_yolo_context_task(ctx), "") == 0);
    AICORE_CHECK(aicore_yolo_context_model_name(ctx) != nullptr);
    AICORE_CHECK(aicore_yolo_context_image_size(ctx) == 0);
    AICORE_CHECK(aicore_yolo_context_num_classes(ctx) == 0);
    AICORE_CHECK(aicore_yolo_context_end2end(ctx) == 0);
    AICORE_CHECK(aicore_yolo_context_device(ctx) != nullptr);
    // No model loaded -> no engine -> threads must read as 0.
    AICORE_CHECK(aicore_yolo_context_threads(ctx) == 0);

    aicore_yolo_timings timings{};
    AICORE_CHECK(aicore_yolo_last_timings(ctx, &timings) == -1);
    AICORE_CHECK(aicore_yolo_last_timings(ctx, nullptr) == -1);

    // Inference entry points must reject a ctx with no loaded model.
    static const uint8_t kRgb[3 * 3 * 3] = {
            255, 0,   0, 0,   255, 0,   0,   0,  255, 255, 255, 0,  0, 255,
            255, 255, 0, 255, 128, 128, 128, 64, 64,  64,  32,  32, 32};
    char* json = aicore_yolo_detect_rgb_json(ctx, kRgb, 3, 3);
    AICORE_CHECK(json == nullptr);
    char* json_path = aicore_yolo_detect_path_json(ctx, "/nonexistent/x.png");
    AICORE_CHECK(json_path == nullptr);

    int32_t dw = 0, dh = 0;
    float* depth = aicore_yolo_depth_rgb(ctx, kRgb, 3, 3, &dw, &dh);
    AICORE_CHECK(depth == nullptr);
    float* depth_path =
            aicore_yolo_depth_path(ctx, "/nonexistent/x.png", &dw, &dh);
    AICORE_CHECK(depth_path == nullptr);
    AICORE_CHECK(aicore_yolo_last_depth_json(ctx) == nullptr);

    aicore_yolo_segment_result* seg = aicore_yolo_seg_rgb(ctx, kRgb, 3, 3);
    AICORE_CHECK(seg == nullptr);
    AICORE_CHECK(aicore_yolo_seg_det_count(nullptr) == 0);
    AICORE_CHECK(aicore_yolo_seg_det_at(nullptr, 0).score == 0.0f);
    AICORE_CHECK(aicore_yolo_seg_mask_at(nullptr, 0).data == nullptr);

    // Host-weight memory management: no engine loaded -> both must fail
    // cleanly (release/ensure need a live session).
    AICORE_CHECK(aicore_yolo_release_host_weights(nullptr) == -1);
    AICORE_CHECK(aicore_yolo_ensure_host_weights(nullptr) == -1);
    AICORE_CHECK(aicore_yolo_release_host_weights(ctx) == -1);
    AICORE_CHECK(aicore_yolo_ensure_host_weights(ctx) == -1);

    // set_detect_thresholds no-ops on NULL and accepts normal values.
    aicore_yolo_set_detect_thresholds(nullptr, 0.5f, 0.6f, 100);  // no-op
    aicore_yolo_set_detect_thresholds(ctx, 0.5f, 0.6f, 100);  // no ctx->engine

    aicore_yolo_free(ctx);
    aicore_yolo_options_free(opts);

    // Introspection on null ctx must not crash and return empty values.
    AICORE_CHECK(aicore_yolo_context_task(nullptr) == nullptr ||
                 std::strcmp(aicore_yolo_context_task(nullptr), "") == 0);
    AICORE_CHECK(aicore_yolo_context_device(nullptr) != nullptr);
    AICORE_CHECK(aicore_yolo_context_threads(nullptr) == 0);

    // Image loader: nonexistent path must fail and zero the outputs.
    uint8_t* rgb = nullptr;
    int32_t iw = 0, ih = 0;
    AICORE_CHECK(aicore_yolo_load_path_rgb("/nonexistent/x.png", &rgb, &iw,
                                           &ih) == -1);
    AICORE_CHECK(rgb == nullptr && iw == 0 && ih == 0);
    AICORE_CHECK(aicore_yolo_load_path_rgb(nullptr, &rgb, &iw, &ih) == -1);

    // Model catalog contract: task-tagged entries (63 = 3 quant x
    // (10 detect + 10 segment + 1 depth); counts must be stable).
    const int n_total = aicore_yolo_model_count(AICORE_YOLO_ROLE_ANY);
    AICORE_CHECK(n_total > 0);
    AICORE_CHECK(aicore_yolo_model_at(-1, AICORE_YOLO_ROLE_ANY) == nullptr);
    AICORE_CHECK(aicore_yolo_model_at(n_total, AICORE_YOLO_ROLE_ANY) ==
                 nullptr);
    for (int i = 0; i < n_total; ++i) {
        const aicore_yolo_model_entry* e =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_ANY);
        AICORE_CHECK(e != nullptr && e->filename != nullptr &&
                     e->download_url != nullptr && e->display_name != nullptr &&
                     e->license_note != nullptr && e->task != nullptr);
    }
    const int n_det = aicore_yolo_model_count(AICORE_YOLO_ROLE_DETECTION);
    const int n_dep = aicore_yolo_model_count(AICORE_YOLO_ROLE_DEPTH);
    const int n_seg = aicore_yolo_model_count(AICORE_YOLO_ROLE_SEGMENT);
    AICORE_CHECK(n_det > 0 && n_dep > 0 && n_seg > 0);
    // Classification is by task tag: depth_capable marks the depth variant,
    // so the detection view (everything that is not depth) already contains
    // the segmentation entries.
    AICORE_CHECK(n_det + n_dep == n_total);
    AICORE_CHECK(n_seg < n_det);
    AICORE_CHECK(aicore_yolo_model_at(0, AICORE_YOLO_ROLE_DETECTION) !=
                 nullptr);
    AICORE_CHECK(aicore_yolo_model_at(n_det, AICORE_YOLO_ROLE_DETECTION) ==
                 nullptr);
    AICORE_CHECK(aicore_yolo_model_at(0, AICORE_YOLO_ROLE_DEPTH) != nullptr);
    AICORE_CHECK(aicore_yolo_model_at(n_dep, AICORE_YOLO_ROLE_DEPTH) ==
                 nullptr);
    AICORE_CHECK(aicore_yolo_model_at(0, AICORE_YOLO_ROLE_SEGMENT) != nullptr);
    AICORE_CHECK(aicore_yolo_model_at(n_seg, AICORE_YOLO_ROLE_SEGMENT) ==
                 nullptr);

    const aicore_yolo_model_entry* first =
            aicore_yolo_model_at(0, AICORE_YOLO_ROLE_ANY);
    AICORE_CHECK(aicore_yolo_model_by_filename(first->filename) == first);
    AICORE_CHECK(aicore_yolo_model_by_filename("nope.gguf") == nullptr);
    AICORE_CHECK(aicore_yolo_model_download_base() != nullptr &&
                 std::strstr(aicore_yolo_model_download_base(),
                             "yolo_gguf_models") != nullptr);

    // verify_model contract (no model assets needed): NULL safety, unknown
    // basename and open failure layers.
    aicore_yolo_verify_report vr{};
    AICORE_CHECK(aicore_yolo_verify_model(nullptr, &vr) == -1);
    AICORE_CHECK(aicore_yolo_verify_model("", &vr) == -1);
    AICORE_CHECK(aicore_yolo_verify_model(nullptr, nullptr) == -1);
    AICORE_CHECK(aicore_yolo_verify_model("/nonexistent/model.gguf", &vr) ==
                 -1);
    AICORE_CHECK(vr.filename_ok == 0);  // basename not in the catalog

    // Catalog-named file that does not exist: basename matches, open fails
    // before the size layer runs.
    aicore_yolo_verify_report vr2{};
    AICORE_CHECK(aicore_yolo_verify_model("/nonexistent-dir/yolov8n-f16.gguf",
                                          &vr2) == -1);
    AICORE_CHECK(vr2.filename_ok == 1);
    AICORE_CHECK(vr2.size_ok == 0);
    AICORE_CHECK(vr2.hash_ok == 0);
    AICORE_CHECK(vr2.magic_ok == 0);
    AICORE_CHECK(vr2.task_ok == 0);

    // Cache dir / info json.
    char* dir = aicore_yolo_model_cache_dir();
    AICORE_CHECK(dir != nullptr && std::strlen(dir) > 0);
    aicore_yolo_free_buffer(dir);

    // info_json on a ctx with no model must not crash (returns NULL because
    // there is no model metadata to serialize).
    aicore_yolo_ctx* empty =
            aicore_yolo_load_opts("/nonexistent/y.gguf", nullptr);
    AICORE_CHECK(empty != nullptr);
    char* info = aicore_yolo_info_json(empty);
    aicore_yolo_free_buffer(info);  // NULL is a valid no-op input
    aicore_yolo_free(empty);

    // Warmup on cpu must succeed without any model loaded.
    AICORE_CHECK(aicore_yolo_warmup_backend("cpu") == 0);
    aicore_yolo_shutdown();
    aicore_yolo_shutdown();  // idempotent

    if (failures == 0) {
        std::printf("[yolo] contract test passed\n");
    }
    return failures;
}
