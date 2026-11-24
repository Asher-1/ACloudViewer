// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstring>

#include "aicore/backend_capi.h"
#include "aicore/depth_capi.h"

TEST(DepthCApiContract, BackendDiscoveryAndWarmup) {
    EXPECT_GE(aicore_backend_abi_version(), AICORE_BACKEND_ABI_VERSION);
    EXPECT_GE(aicore_device_count(), 2);
    EXPECT_NE(aicore_device_at(0), nullptr);
    EXPECT_EQ(std::strcmp(aicore_device_at(0)->id, "auto"), 0);
    EXPECT_EQ(aicore_device_available("cpu"), 1);
    EXPECT_EQ(aicore_device_available("blas"), 0);
    EXPECT_EQ(std::strstr(aicore_auto_device_order(), "blas"), nullptr);
    EXPECT_EQ(aicore_warmup_backend("cpu"), 0);
    EXPECT_NE(aicore_warmup_backend("blas"), 0);
    EXPECT_NE(aicore_warmup_backend("not-a-backend"), 0);
    EXPECT_NE(aicore_backend_last_error()[0], '\0');
}

TEST(DepthCApiContract, CpuSupportsFullGraphAndFp32) {
    aicore_model_device_info depthInfo{};
    depthInfo.struct_size = sizeof(depthInfo);
    EXPECT_EQ(aicore_model_device_info_query(AICORE_MODEL_DEPTH, "cpu",
                                             &depthInfo),
              0);
    EXPECT_NE((depthInfo.capabilities & AICORE_MODEL_CAP_FULL_GRAPH), 0);
    EXPECT_NE((depthInfo.precision & AICORE_MODEL_PRECISION_FP32), 0);
    EXPECT_GE(depthInfo.recommended_working_set_bytes,
              depthInfo.min_working_set_bytes);
}

TEST(DepthCApiContract, AbiAndNullRelease) {
    EXPECT_GE(aicore_depth_abi_version(), 8);

    aicore_depth_free(nullptr);
    aicore_depth_free_buffer(nullptr);
    aicore_depth_free_buffer(nullptr);
    aicore_depth_free_buffer(nullptr);
}

TEST(DepthCApiContract, OptionsAcceptValuesAndNullSetters) {
    // Options handle: null-safety and value semantics.
    aicore_depth_options_free(nullptr);
    aicore_depth_options_set_device(nullptr, "cpu");
    aicore_depth_options_set_threads(nullptr, 4);
    aicore_depth_options_set_fused_graph(nullptr, 0);
    aicore_depth_options_set_force_joint_multiview(nullptr, 1);
    aicore_depth_options_set_profile_logging(nullptr, 1);
    aicore_depth_options* opts = aicore_depth_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_depth_options_set_device(opts, nullptr);  // ignored, keeps "auto"
    aicore_depth_options_set_device(opts, "");
    aicore_depth_options_set_threads(opts, -1);  // clamped by backend later
    aicore_depth_options_set_fused_graph(opts, 0);
    aicore_depth_options_set_force_joint_multiview(opts, 1);
    aicore_depth_options_set_profile_logging(opts, 1);
    aicore_depth_options_set_fused_graph(opts, -1);  // invalid, ignored
    aicore_depth_options_set_keep_graph_buffers(nullptr, 1);  // no-op on NULL
    aicore_depth_options_set_keep_graph_buffers(opts, 1);
    aicore_depth_options_set_keep_graph_buffers(opts, 0);
    aicore_depth_options_set_keep_graph_buffers(opts, -1); /* ignored */

    aicore_depth_options_free(opts);
}

TEST(DepthCApiContract, NullLoadAndContextQueries) {
    EXPECT_EQ(aicore_depth_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_depth_is_ready(nullptr), 0);
    EXPECT_EQ(aicore_depth_load_nested_opts(nullptr, "m.gguf", nullptr),
              nullptr);
    EXPECT_EQ(aicore_depth_load_nested_opts("a.gguf", nullptr, nullptr),
              nullptr);
    EXPECT_EQ(aicore_depth_info_json(nullptr), nullptr);
    EXPECT_EQ(std::strcmp(aicore_depth_last_error(nullptr), ""), 0);
    EXPECT_EQ(std::strcmp(aicore_depth_device_name(nullptr), ""), 0);
}

TEST(DepthCApiContract, NullInferenceAndExportRejectArguments) {
    int h = 0, w = 0, n = 0, is_metric = 0;
    float ext[12] = {}, intr[9] = {};
    float* depth = nullptr;
    float* conf = nullptr;
    float* sky = nullptr;
    unsigned char* rgb = nullptr;

    EXPECT_EQ(aicore_depth_depth_path(nullptr, "x.png", &h, &w), nullptr);
    EXPECT_NE(aicore_depth_pose_path(nullptr, "x.png", ext, intr), 0);
    aicore_depth_multiview_result mv{};
    EXPECT_NE(aicore_depth_depth_pose_multi(nullptr, nullptr, 0, &mv), 0);
    aicore_depth_multiview_result_free(&mv);
    EXPECT_NE(aicore_depth_export_glb(nullptr, "x.png", "/tmp/x.glb"), 0);
    EXPECT_NE(aicore_depth_export_glb_image(nullptr, nullptr, "/tmp/x.glb"), 0);
    EXPECT_NE(aicore_depth_export_colmap_image(nullptr, nullptr, "x.png",
                                               "/tmp", 0),
              0);
    EXPECT_NE(aicore_depth_reconstruct_image(nullptr, nullptr, nullptr, nullptr,
                                             nullptr, nullptr, nullptr, nullptr,
                                             nullptr),
              0);
    EXPECT_NE(aicore_depth_export_colmap(nullptr, "x.png", "/tmp/x", 1), 0);
    EXPECT_NE(
            aicore_depth_export_colmap_multi(nullptr, nullptr, 0, "/tmp/x", 1),
            0);
    EXPECT_NE(aicore_depth_export_colmap_multi_named(nullptr, nullptr, nullptr,
                                                     0, "/tmp/x", 1),
              0);
    EXPECT_NE(aicore_depth_write_colmap_from_multiview(
                      nullptr, nullptr, nullptr, nullptr, "/tmp/x", 1),
              0);
    aicore_depth_dense_result dense{};
    EXPECT_NE(aicore_depth_depth_dense(nullptr, "x.png", &dense), 0);
    aicore_depth_dense_result_free(&dense);
    EXPECT_NE(aicore_depth_points(nullptr, "x.png", 0.5f, &n, &depth, &rgb), 0);
}

TEST(DepthCApiContract, NullWorkingMemoryAndResizeControls) {
    aicore_depth_set_img_resize_target(nullptr, 504);
    aicore_depth_release_gpu_working_memory(nullptr);
    EXPECT_EQ(aicore_depth_cap_img_resize_target(nullptr, 504), 504);
}

TEST(DepthCApiContract, OwnedModelCacheDirectory) {
    char* dir = aicore_depth_model_cache_dir();
    EXPECT_TRUE(dir != nullptr && dir[0] != '\0');
    EXPECT_TRUE(dir != nullptr && std::strstr(dir, "da3_models") != nullptr);
    aicore_depth_free_buffer(dir);
}
