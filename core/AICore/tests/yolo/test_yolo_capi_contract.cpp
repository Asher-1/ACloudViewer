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

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "aicore/yolo_capi.h"

TEST(YoloCApiContract, AbiAndNullReleaseQueries) {
    EXPECT_GE(aicore_yolo_abi_version(), 5);

    // Null-safe teardown / lifecycle.
    aicore_yolo_free(nullptr);
    aicore_yolo_options_free(nullptr);
    aicore_yolo_free_buffer(nullptr);
    aicore_yolo_free_buffer(nullptr);
    aicore_yolo_seg_result_free(nullptr);
    aicore_yolo_pose_result_free(nullptr);
    aicore_yolo_obb_result_free(nullptr);
    aicore_yolo_semantic_result_free(nullptr);
    aicore_yolo_classify_result_free(nullptr);

    EXPECT_EQ(aicore_yolo_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_yolo_is_ready(nullptr), 0);
    EXPECT_EQ(aicore_yolo_last_error(nullptr), nullptr);
}

TEST(YoloCApiContract, AttentionAccumulationValidatesModes) {
    aicore_yolo_options* opts = aicore_yolo_options_new();
    EXPECT_NE(opts, nullptr);
    EXPECT_EQ(aicore_yolo_options_get_attention_accumulation(opts),
              AICORE_YOLO_ATTENTION_UPSTREAM);
    for (int mode : {AICORE_YOLO_ATTENTION_F32, AICORE_YOLO_ATTENTION_UPSTREAM,
                     AICORE_YOLO_ATTENTION_F32}) {
        EXPECT_EQ(aicore_yolo_options_set_attention_accumulation(opts, mode),
                  0);
        EXPECT_EQ(aicore_yolo_options_get_attention_accumulation(opts), mode);
    }
    for (int invalid : {-1, 2, std::numeric_limits<int>::max()}) {
        EXPECT_EQ(aicore_yolo_options_set_attention_accumulation(opts, invalid),
                  -1);
        EXPECT_EQ(aicore_yolo_options_get_attention_accumulation(opts),
                  AICORE_YOLO_ATTENTION_F32);
    }
    EXPECT_EQ(aicore_yolo_options_set_attention_accumulation(
                      nullptr, AICORE_YOLO_ATTENTION_F32),
              -1);
    EXPECT_EQ(aicore_yolo_options_get_attention_accumulation(nullptr), -1);
    EXPECT_EQ(aicore_yolo_context_attention_accumulation(nullptr), -1);
    aicore_yolo_options_free(opts);
}

TEST(YoloCApiContract, ProbabilityThresholdsAcceptEndpointsAndRejectNonFinite) {
    aicore_yolo_options* opts = aicore_yolo_options_new();
    EXPECT_NE(opts, nullptr);
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

    EXPECT_EQ(aicore_yolo_options_get_conf_thres(opts), 0.5f);
    EXPECT_EQ(aicore_yolo_options_get_iou_thres(opts), 0.6f);

    // Both ends of the UI/source [0,1] range are valid probabilities.
    // Invalid values, including NaN, must preserve the last accepted value.
    for (float endpoint : {0.0f, 1.0f}) {
        aicore_yolo_options_set_conf_thres(opts, endpoint);
        aicore_yolo_options_set_iou_thres(opts, endpoint);
        EXPECT_EQ(aicore_yolo_options_get_conf_thres(opts), endpoint);
        EXPECT_EQ(aicore_yolo_options_get_iou_thres(opts), endpoint);
    }
    for (float invalid :
         {-0.01f, 1.01f, std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::infinity()}) {
        aicore_yolo_options_set_conf_thres(opts, invalid);
        aicore_yolo_options_set_iou_thres(opts, invalid);
        EXPECT_EQ(aicore_yolo_options_get_conf_thres(opts), 1.0f);
        EXPECT_EQ(aicore_yolo_options_get_iou_thres(opts), 1.0f);
    }
    aicore_yolo_options_set_conf_thres(opts, 0.5f);
    aicore_yolo_options_set_iou_thres(opts, 0.6f);
    aicore_yolo_options_free(opts);
}

TEST(YoloCApiContract, NullSettersOpenVocabularyAndDefaultThresholds) {
    aicore_yolo_options* opts = aicore_yolo_options_new();
    EXPECT_NE(opts, nullptr);
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
    // Open-vocabulary setters: NULL options / NULL array must be safe no-ops;
    // an empty entry inside the array is preserved (background prompt).
    aicore_yolo_options_set_classes(nullptr, nullptr, 0);
    const char* kClasses[] = {"person", "", "bus"};
    aicore_yolo_options_set_classes(opts, kClasses, 3);
    aicore_yolo_options_set_classes(opts, nullptr, 3);    // clears
    aicore_yolo_options_set_classes(opts, kClasses, -1);  // clears
    aicore_yolo_options_set_text_model(nullptr, "x.gguf");
    aicore_yolo_options_set_text_model(opts, nullptr);  // clears
    // NULL options expose the documented defaults (not zero).
    EXPECT_EQ(aicore_yolo_options_get_conf_thres(nullptr), 0.25f);
    EXPECT_EQ(aicore_yolo_options_get_iou_thres(nullptr), 0.7f);
    aicore_yolo_options_free(opts);
}

TEST(YoloCApiContract, FailedLoadIntrospectionInferenceAndRecovery) {
    aicore_yolo_options* opts = aicore_yolo_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_yolo_options_set_device(opts, "cpu");
    // Loading a nonexistent file must fail cleanly and report an error.
    aicore_yolo_ctx* ctx =
            aicore_yolo_load_opts("/nonexistent/yolo.gguf", opts);
    EXPECT_NE(ctx, nullptr);  // ctx is allocated; model inside is null
    EXPECT_EQ(aicore_yolo_is_ready(ctx), 0);
    EXPECT_NE(aicore_yolo_last_error(ctx), nullptr);
    EXPECT_EQ(std::strcmp(aicore_yolo_context_task(ctx), ""), 0);
    EXPECT_NE(aicore_yolo_context_model_name(ctx), nullptr);
    EXPECT_EQ(aicore_yolo_context_image_size(ctx), 0);
    EXPECT_EQ(aicore_yolo_context_num_classes(ctx), 0);
    EXPECT_EQ(aicore_yolo_context_end2end(ctx), 0);
    EXPECT_NE(aicore_yolo_context_device(ctx), nullptr);
    // No model loaded -> no engine -> threads must read as 0.
    EXPECT_EQ(aicore_yolo_context_threads(ctx), 0);
    EXPECT_EQ(aicore_yolo_context_attention_accumulation(ctx), -1);

    aicore_yolo_timings timings{};
    EXPECT_EQ(aicore_yolo_last_timings(ctx, &timings), -1);
    EXPECT_EQ(aicore_yolo_last_timings(ctx, nullptr), -1);
    aicore_pipeline_timings pipeline_timings{};
    EXPECT_EQ(aicore_yolo_last_pipeline_timings(ctx, &pipeline_timings), -1);
    EXPECT_EQ(aicore_yolo_last_pipeline_timings(ctx, nullptr), -1);
    aicore_yolo_depth_stats depth_stats{};
    EXPECT_EQ(aicore_yolo_last_depth_stats(ctx, &depth_stats), -1);
    EXPECT_EQ(aicore_yolo_last_depth_stats(ctx, nullptr), -1);

    // Inference entry points must reject a ctx with no loaded model.
    static const uint8_t kRgb[3 * 3 * 3] = {
            255, 0,   0, 0,   255, 0,   0,   0,  255, 255, 255, 0,  0, 255,
            255, 255, 0, 255, 128, 128, 128, 64, 64,  64,  32,  32, 32};
    const aicore_image_view image{kRgb, 3, 3, 9, AICORE_IMAGE_RGB8};
    char* json = aicore_yolo_detect_rgb_json(ctx, kRgb, 3, 3);
    EXPECT_EQ(json, nullptr);
    char* json_path = aicore_yolo_detect_path_json(ctx, "/nonexistent/x.png");
    EXPECT_EQ(json_path, nullptr);

    int32_t dw = 0, dh = 0;
    float* depth = aicore_yolo_depth_rgb(ctx, kRgb, 3, 3, &dw, &dh);
    EXPECT_EQ(depth, nullptr);
    float* depth_path =
            aicore_yolo_depth_path(ctx, "/nonexistent/x.png", &dw, &dh);
    EXPECT_EQ(depth_path, nullptr);
    EXPECT_EQ(aicore_yolo_last_depth_json(ctx), nullptr);
    EXPECT_EQ(aicore_yolo_depth_image(ctx, &image, &dw, &dh), nullptr);
    EXPECT_EQ(aicore_yolo_depth_image(ctx, nullptr, &dw, &dh), nullptr);

    aicore_yolo_segment_result* seg = aicore_yolo_seg_rgb(ctx, kRgb, 3, 3);
    EXPECT_EQ(seg, nullptr);
    EXPECT_EQ(aicore_yolo_seg_image(ctx, &image), nullptr);
    EXPECT_EQ(aicore_yolo_seg_det_count(nullptr), 0);
    EXPECT_EQ(aicore_yolo_seg_det_at(nullptr, 0).score, 0.0f);
    EXPECT_EQ(aicore_yolo_seg_det_class_name(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_yolo_seg_mask_at(nullptr, 0).data, nullptr);

    // Detect typed accessors: NULL ctx and model-less ctx are null-safe
    // (no engine -> no detections -> no names).
    EXPECT_EQ(aicore_yolo_detection_count(nullptr), -1);
    EXPECT_EQ(aicore_yolo_detection_count(ctx), 0);
    EXPECT_EQ(aicore_yolo_detection_at(nullptr, 0).score, 0.0f);
    EXPECT_EQ(aicore_yolo_detection_class_name(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_yolo_detection_class_name(ctx, 0), nullptr);

    // New-task inference entry points must reject a ctx with no loaded
    // model, and their accessors must be null-safe.
    EXPECT_EQ(aicore_yolo_pose_rgb(ctx, kRgb, 3, 3), nullptr);
    EXPECT_EQ(aicore_yolo_obb_rgb(ctx, kRgb, 3, 3), nullptr);
    EXPECT_EQ(aicore_yolo_semantic_rgb(ctx, kRgb, 3, 3), nullptr);
    EXPECT_EQ(aicore_yolo_classify_rgb(ctx, kRgb, 3, 3), nullptr);
    EXPECT_EQ(aicore_yolo_pose_image(ctx, &image), nullptr);
    EXPECT_EQ(aicore_yolo_obb_image(ctx, &image), nullptr);
    EXPECT_EQ(aicore_yolo_semantic_image(ctx, &image), nullptr);
    EXPECT_EQ(aicore_yolo_classify_image(ctx, &image), nullptr);
    EXPECT_EQ(aicore_yolo_pose_det_count(nullptr), 0);
    EXPECT_EQ(aicore_yolo_pose_det_at(nullptr, 0).score, 0.0f);
    EXPECT_EQ(aicore_yolo_pose_kpt_count(nullptr), 0);
    EXPECT_EQ(aicore_yolo_pose_kpt_at(nullptr, 0, 0).x, 0.0f);
    EXPECT_EQ(aicore_yolo_pose_det_class_name(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_yolo_obb_count(nullptr), 0);
    EXPECT_EQ(aicore_yolo_obb_at(nullptr, 0).score, 0.0f);
    EXPECT_EQ(aicore_yolo_obb_class_name(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_yolo_semantic_class_map(nullptr).data, nullptr);
    EXPECT_EQ(aicore_yolo_semantic_num_classes(nullptr), 0);
    EXPECT_EQ(aicore_yolo_semantic_class_name(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_yolo_classify_count(nullptr), 0);
    EXPECT_EQ(aicore_yolo_classify_prob_at(nullptr, 0), 0.0f);
    EXPECT_EQ(aicore_yolo_classify_class_name(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_yolo_context_has_text_input(nullptr), 0);
    EXPECT_EQ(aicore_yolo_context_has_text_input(ctx), 0);

    // Host-weight memory management: no engine loaded -> both must fail
    // cleanly (release/ensure need a live session).
    EXPECT_EQ(aicore_yolo_release_host_weights(nullptr), -1);
    EXPECT_EQ(aicore_yolo_ensure_host_weights(nullptr), -1);
    EXPECT_EQ(aicore_yolo_release_host_weights(ctx), -1);
    EXPECT_EQ(aicore_yolo_ensure_host_weights(ctx), -1);

    // set_detect_thresholds no-ops on NULL and accepts normal values.
    aicore_yolo_set_detect_thresholds(nullptr, 0.5f, 0.6f, 100);  // no-op
    aicore_yolo_set_detect_thresholds(ctx, 0.5f, 0.6f, 100);  // no ctx->engine

    // Loose-NMS recovery (TrackTrack track mode): NULL-safe no-op setter,
    // -1 count on null ctx, 0 rows on a context that never ran, and safe
    // zeroed rows for any index. Recovery rows only appear after a real
    // inference (model tier covers the numeric path).
    aicore_yolo_set_track_recovery(nullptr, 1);  // no-op
    EXPECT_EQ(aicore_yolo_recovery_count(nullptr), -1);
    aicore_yolo_set_track_recovery(ctx, 1);
    EXPECT_EQ(aicore_yolo_recovery_count(ctx), 0);
    const aicore_yolo_detection rec = aicore_yolo_recovery_at(ctx, 0);
    EXPECT_TRUE(rec.score == 0.0f && rec.class_id == 0);
    aicore_yolo_set_track_recovery(ctx, 0);
    EXPECT_EQ(aicore_yolo_recovery_count(ctx), 0);

    aicore_yolo_free(ctx);
    aicore_yolo_options_free(opts);
}

TEST(YoloCApiContract, NullIntrospection) {
    // Introspection on null ctx must not crash and return empty values.
    EXPECT_TRUE(aicore_yolo_context_task(nullptr) == nullptr ||
                std::strcmp(aicore_yolo_context_task(nullptr), "") == 0);
    EXPECT_NE(aicore_yolo_context_device(nullptr), nullptr);
    EXPECT_EQ(aicore_yolo_context_threads(nullptr), 0);
}

TEST(YoloCApiContract, MissingImageLoaderClearsOutputs) {
    // Image loader: nonexistent path must fail and zero the outputs.
    uint8_t* rgb = nullptr;
    int32_t iw = 0, ih = 0;
    EXPECT_EQ(aicore_yolo_load_path_rgb("/nonexistent/x.png", &rgb, &iw, &ih),
              -1);
    EXPECT_TRUE(rgb == nullptr && iw == 0 && ih == 0);
    EXPECT_EQ(aicore_yolo_load_path_rgb(nullptr, &rgb, &iw, &ih), -1);
}

TEST(YoloCApiContract, CatalogRolePartitionAndTextConditioning) {
    // Model catalog contract: task-tagged entries over the full published
    // release (230 = 3 quant x 77 variants minus the bridge's unpublished
    // f32; the 77 include the 5 native reid encoders published 2026-09-19;
    // counts must be stable).
    const int n_total = aicore_yolo_model_count(AICORE_YOLO_ROLE_ANY);
    EXPECT_GT(n_total, 0);
    EXPECT_EQ(aicore_yolo_model_at(-1, AICORE_YOLO_ROLE_ANY), nullptr);
    EXPECT_EQ(aicore_yolo_model_at(n_total, AICORE_YOLO_ROLE_ANY), nullptr);
    for (int i = 0; i < n_total; ++i) {
        const aicore_yolo_model_entry* e =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_ANY);
        EXPECT_TRUE(e != nullptr && e->filename != nullptr &&
                    e->download_url != nullptr && e->display_name != nullptr &&
                    e->license_note != nullptr && e->task != nullptr);
    }
    // Per-role counts: 77 variants x 3 quants, partitioned by role.
    struct RoleCount {
        enum aicore_yolo_model_role role;
        int expected;
    };
    const RoleCount kRoleCounts[] = {
            {AICORE_YOLO_ROLE_DETECTION, 30},  // 10 detect x 3
            {AICORE_YOLO_ROLE_DEPTH, 15},      // 5 depth x 3
            {AICORE_YOLO_ROLE_SEGMENT, 30},    // 10 closed-set seg x 3
            {AICORE_YOLO_ROLE_POSE, 15},       // 5 pose x 3
            {AICORE_YOLO_ROLE_OBB, 30},        // 10 obb (640 + 1024) x 3
            {AICORE_YOLO_ROLE_CLASSIFY, 30},   // 5 cls + 5 native reid x 3
            {AICORE_YOLO_ROLE_SEMANTIC, 30},   // 10 sem (640 + 1024) x 3
            {AICORE_YOLO_ROLE_WORLD, 12},      // 4 world x 3
            {AICORE_YOLO_ROLE_YOLOE, 30},      // 10 yoloe (incl. -pf) x 3
            {AICORE_YOLO_ROLE_TEXT, 8},  // 2 text towers x 3 + mclip f16+q8_0
    };
    int role_sum = 0;
    for (const RoleCount& rc : kRoleCounts) {
        const int n = aicore_yolo_model_count(rc.role);
        EXPECT_EQ(n, rc.expected);
        role_sum += n;
        EXPECT_NE(aicore_yolo_model_at(0, rc.role), nullptr);
        EXPECT_EQ(aicore_yolo_model_at(n, rc.role), nullptr);
    }
    // The roles partition the catalog.
    EXPECT_EQ(role_sum, n_total);
    // World entries are text-conditioned detectors; YOLOE entries are
    // text-conditioned segmenters; text towers are their own task.
    const aicore_yolo_model_entry* world =
            aicore_yolo_model_by_filename("yolov8s-world-f16.gguf");
    EXPECT_TRUE(world != nullptr && world->text_input == 1 &&
                std::strcmp(world->task, "detect") == 0);
    const aicore_yolo_model_entry* yoloe =
            aicore_yolo_model_by_filename("yoloe-26n-seg-f16.gguf");
    EXPECT_TRUE(yoloe != nullptr && yoloe->text_input == 1 &&
                std::strcmp(yoloe->task, "segment") == 0);
    const aicore_yolo_model_entry* clip =
            aicore_yolo_model_by_filename("clip-ViT-B-32-f16.gguf");
    EXPECT_TRUE(clip != nullptr && std::strcmp(clip->task, "text") == 0);
    const aicore_yolo_model_entry* mclip =
            aicore_yolo_model_by_filename("mobileclip2_b-q8_0.gguf");
    EXPECT_TRUE(mclip != nullptr && std::strcmp(mclip->task, "text") == 0);
    const aicore_yolo_model_entry* bridge =
            aicore_yolo_model_by_filename("mclip-labse-vitb32-f16.gguf");
    EXPECT_TRUE(bridge != nullptr && bridge->text_input == 1 &&
                std::strcmp(bridge->task, "text") == 0);
    // The multilingual bridge is published in F16 + Q8_0 (no F32), with
    // Q8_0 listed first (default quant).
    EXPECT_EQ(aicore_yolo_model_by_filename("mclip-labse-vitb32-f32.gguf"),
              nullptr);
    EXPECT_NE(aicore_yolo_model_by_filename("mclip-labse-vitb32-q8_0.gguf"),
              nullptr);
    {
        // The first mclip entry is the Q8_0 default.
        bool saw_q8_first = false;
        for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_TEXT);
             ++i) {
            const aicore_yolo_model_entry* e =
                    aicore_yolo_model_at(i, AICORE_YOLO_ROLE_TEXT);
            EXPECT_NE(e, nullptr);
            if (std::strncmp(e->filename, "mclip-labse", 11) == 0) {
                EXPECT_EQ(std::strcmp(e->filename,
                                      "mclip-labse-vitb32-q8_0.gguf"),
                          0);
                saw_q8_first = true;
                break;
            }
        }
        EXPECT_TRUE(saw_q8_first);
    }

    const aicore_yolo_model_entry* first =
            aicore_yolo_model_at(0, AICORE_YOLO_ROLE_ANY);
    EXPECT_EQ(aicore_yolo_model_by_filename(first->filename), first);
    EXPECT_EQ(aicore_yolo_model_by_filename("nope.gguf"), nullptr);
    EXPECT_TRUE(aicore_yolo_model_download_base() != nullptr &&
                std::strstr(aicore_yolo_model_download_base(),
                            "yolo_gguf_models") != nullptr);
}

TEST(YoloCApiContract,
     ModelVerificationDistinguishesUnknownNameAndMissingFile) {
    // verify_model contract (no model assets needed): NULL safety, unknown
    // basename and open failure layers.
    aicore_yolo_verify_report vr{};
    EXPECT_EQ(aicore_yolo_verify_model(nullptr, &vr), -1);
    EXPECT_EQ(aicore_yolo_verify_model("", &vr), -1);
    EXPECT_EQ(aicore_yolo_verify_model(nullptr, nullptr), -1);
    EXPECT_EQ(aicore_yolo_verify_model("/nonexistent/model.gguf", &vr), -1);
    EXPECT_EQ(vr.filename_ok, 0);  // basename not in the catalog

    // Catalog-named file that does not exist: basename matches, open fails
    // before the size layer runs.
    aicore_yolo_verify_report vr2{};
    EXPECT_EQ(
            aicore_yolo_verify_model("/nonexistent-dir/yolov8n-f16.gguf", &vr2),
            -1);
    EXPECT_EQ(vr2.filename_ok, 1);
    EXPECT_EQ(vr2.size_ok, 0);
    EXPECT_EQ(vr2.hash_ok, 0);
    EXPECT_EQ(vr2.magic_ok, 0);
    EXPECT_EQ(vr2.task_ok, 0);
}

TEST(YoloCApiContract, OwnedCacheAndEmptyContextInfoRelease) {
    // Cache dir / info json.
    char* dir = aicore_yolo_model_cache_dir();
    EXPECT_TRUE(dir != nullptr && std::strlen(dir) > 0);
    aicore_yolo_free_buffer(dir);

    // info_json on a ctx with no model must not crash (returns NULL because
    // there is no model metadata to serialize).
    aicore_yolo_ctx* empty =
            aicore_yolo_load_opts("/nonexistent/y.gguf", nullptr);
    EXPECT_NE(empty, nullptr);
    char* info = aicore_yolo_info_json(empty);
    aicore_yolo_free_buffer(info);  // NULL is a valid no-op input
    aicore_yolo_free(empty);
}

TEST(YoloCApiContract, VisualPromptsClearAndDropDegenerateBoxes) {
    // ---- YOLOE visual prompts (SAVPE) contract ----
    // Null guards first: every new entry point must tolerate NULL.
    aicore_yolo_options_set_visual_prompts(nullptr, nullptr, 0);
    EXPECT_EQ(aicore_yolo_options_get_visual_prompt_count(nullptr), 0);
    // A dedicated options struct: the shared one above was already freed.
    aicore_yolo_options* vopts = aicore_yolo_options_new();
    EXPECT_NE(vopts, nullptr);
    EXPECT_EQ(aicore_yolo_gguf_has_savpe(nullptr), 0);
    EXPECT_EQ(aicore_yolo_gguf_has_savpe(""), 0);
    EXPECT_EQ(aicore_yolo_gguf_has_savpe("/nonexistent/y.gguf"), 0);
    EXPECT_EQ(aicore_yolo_context_has_visual_prompts(nullptr), 0);
    // Clearing (NULL array / count <= 0) must reset the count to 0.
    const float kBoxes[] = {10.f,  20.f,  110.f, 220.f,
                            300.f, 400.f, 500.f, 600.f};
    aicore_yolo_options_set_visual_prompts(vopts, kBoxes, 2);
    EXPECT_EQ(aicore_yolo_options_get_visual_prompt_count(vopts), 2);
    aicore_yolo_options_set_visual_prompts(vopts, nullptr, 2);
    EXPECT_EQ(aicore_yolo_options_get_visual_prompt_count(vopts), 0);
    aicore_yolo_options_set_visual_prompts(vopts, kBoxes, 2);
    aicore_yolo_options_set_visual_prompts(vopts, kBoxes, -1);
    EXPECT_EQ(aicore_yolo_options_get_visual_prompt_count(vopts), 0);
    // Degenerate boxes (inverted / zero area / non-finite) are dropped; the
    // count only reflects usable prompts.
    const float kMixed[] = {10.f, 20.f, 110.f, 220.f,  // valid
                            50.f, 60.f, 40.f,  80.f,   // inverted x: dropped
                            5.f,  5.f,  5.f,   5.f,    // zero area: dropped
                            0.f,  0.f,  30.f,  40.f};  // valid
    aicore_yolo_options_set_visual_prompts(vopts, kMixed, 4);
    EXPECT_EQ(aicore_yolo_options_get_visual_prompt_count(vopts), 2);
    aicore_yolo_options_set_visual_prompts(vopts, nullptr, 0);
    EXPECT_EQ(aicore_yolo_options_get_visual_prompt_count(vopts), 0);

    aicore_yolo_options_free(vopts);
}

TEST(YoloCApiContract, CpuWarmupAndIdempotentShutdown) {
    // Warmup on cpu must succeed without any model loaded.
    EXPECT_EQ(aicore_yolo_warmup_backend("cpu"), 0);
    aicore_yolo_shutdown();
    aicore_yolo_shutdown();  // idempotent
}
