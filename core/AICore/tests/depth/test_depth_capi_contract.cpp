// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>

#include "aicore/backend_capi.h"
#include "aicore/depth_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_backend_abi_version() >= AICORE_BACKEND_ABI_VERSION);
    AICORE_CHECK(aicore_device_count() >= 2);
    AICORE_CHECK(aicore_device_at(0) != nullptr);
    AICORE_CHECK(std::strcmp(aicore_device_at(0)->id, "auto") == 0);
    AICORE_CHECK(aicore_device_available("cpu") == 1);
    AICORE_CHECK(aicore_device_available("blas") == 0);
    AICORE_CHECK(std::strstr(aicore_auto_device_order(), "blas") == nullptr);
    AICORE_CHECK(aicore_warmup_backend("cpu") == 0);
    AICORE_CHECK(aicore_warmup_backend("blas") != 0);
    AICORE_CHECK(aicore_warmup_backend("not-a-backend") != 0);
    AICORE_CHECK(aicore_backend_last_error()[0] != '\0');

    aicore_model_device_info depthInfo{};
    depthInfo.struct_size = sizeof(depthInfo);
    AICORE_CHECK(aicore_model_device_info_query(AICORE_MODEL_DEPTH, "cpu",
                                                &depthInfo) == 0);
    AICORE_CHECK((depthInfo.capabilities & AICORE_MODEL_CAP_FULL_GRAPH) != 0);
    AICORE_CHECK((depthInfo.precision & AICORE_MODEL_PRECISION_FP32) != 0);
    AICORE_CHECK(depthInfo.recommended_working_set_bytes >=
                 depthInfo.min_working_set_bytes);

    AICORE_CHECK(aicore_depth_abi_version() >= 6);

    aicore_depth_free(nullptr);
    aicore_depth_free_buffer(nullptr);
    aicore_depth_free_buffer(nullptr);
    aicore_depth_free_buffer(nullptr);

    // Options handle: null-safety and value semantics.
    aicore_depth_options_free(nullptr);
    aicore_depth_options_set_device(nullptr, "cpu");
    aicore_depth_options_set_threads(nullptr, 4);
    aicore_depth_options_set_fused_graph(nullptr, 0);
    aicore_depth_options_set_force_joint_multiview(nullptr, 1);
    aicore_depth_options_set_profile_logging(nullptr, 1);
    aicore_depth_options* opts = aicore_depth_options_new();
    AICORE_CHECK(opts != nullptr);
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

    AICORE_CHECK(aicore_depth_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_depth_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_depth_load_nested_opts(nullptr, "m.gguf", nullptr) ==
                 nullptr);
    AICORE_CHECK(aicore_depth_load_nested_opts("a.gguf", nullptr, nullptr) ==
                 nullptr);
    AICORE_CHECK(aicore_depth_info_json(nullptr) == nullptr);
    AICORE_CHECK(std::strcmp(aicore_depth_last_error(nullptr), "") == 0);
    AICORE_CHECK(std::strcmp(aicore_depth_device_name(nullptr), "") == 0);

    int h = 0, w = 0, n = 0, is_metric = 0;
    float ext[12] = {}, intr[9] = {};
    float* depth = nullptr;
    float* conf = nullptr;
    float* sky = nullptr;
    unsigned char* rgb = nullptr;

    AICORE_CHECK(aicore_depth_depth_path(nullptr, "x.png", &h, &w) == nullptr);
    AICORE_CHECK(aicore_depth_pose_path(nullptr, "x.png", ext, intr) != 0);
    aicore_depth_multiview_result mv{};
    AICORE_CHECK(aicore_depth_depth_pose_multi(nullptr, nullptr, 0, &mv) != 0);
    aicore_depth_multiview_result_free(&mv);
    AICORE_CHECK(aicore_depth_export_glb(nullptr, "x.png", "/tmp/x.glb") != 0);
    AICORE_CHECK(aicore_depth_export_colmap(nullptr, "x.png", "/tmp/x", 1) !=
                 0);
    AICORE_CHECK(aicore_depth_export_colmap_multi(nullptr, nullptr, 0, "/tmp/x",
                                                  1) != 0);
    AICORE_CHECK(aicore_depth_export_colmap_multi_named(
                         nullptr, nullptr, nullptr, 0, "/tmp/x", 1) != 0);
    AICORE_CHECK(aicore_depth_write_colmap_from_multiview(
                         nullptr, nullptr, nullptr, nullptr, "/tmp/x", 1) != 0);
    aicore_depth_dense_result dense{};
    AICORE_CHECK(aicore_depth_depth_dense(nullptr, "x.png", &dense) != 0);
    aicore_depth_dense_result_free(&dense);
    AICORE_CHECK(
            aicore_depth_points(nullptr, "x.png", 0.5f, &n, &depth, &rgb) != 0);

    aicore_depth_set_img_resize_target(nullptr, 504);
    aicore_depth_release_gpu_working_memory(nullptr);
    AICORE_CHECK(aicore_depth_cap_img_resize_target(nullptr, 504) == 504);

    char* dir = aicore_depth_model_cache_dir();
    AICORE_CHECK(dir != nullptr && dir[0] != '\0');
    AICORE_CHECK(dir != nullptr && std::strstr(dir, "da3_models") != nullptr);
    aicore_depth_free_buffer(dir);

    std::fprintf(stderr, "depth_capi_contract ok (abi=%d)\n",
                 aicore_depth_abi_version());
    return failures == 0 ? 0 : 1;
}
