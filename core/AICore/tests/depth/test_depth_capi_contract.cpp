// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>

#include "aicore/backend_capi.h"
#include "aicore/depth_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_backend_abi_version() == 1);
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

    AICORE_CHECK(aicore_depth_abi_version() >= 5);

    aicore_depth_free(nullptr);
    aicore_depth_free_string(nullptr);
    aicore_depth_free_floats(nullptr);
    aicore_depth_free_bytes(nullptr);

    AICORE_CHECK(aicore_depth_load(nullptr, 1) == nullptr);
    AICORE_CHECK(aicore_depth_load_nested(nullptr, "m.gguf", 1) == nullptr);
    AICORE_CHECK(aicore_depth_load_nested("a.gguf", nullptr, 1) == nullptr);
    AICORE_CHECK(aicore_depth_info_json(nullptr) == nullptr);
    AICORE_CHECK(std::strcmp(aicore_depth_last_error(nullptr), "") == 0);

    int h = 0, w = 0, n = 0, is_metric = 0;
    float ext[12] = {}, intr[9] = {};
    float* depth = nullptr;
    float* conf = nullptr;
    float* sky = nullptr;
    unsigned char* rgb = nullptr;

    AICORE_CHECK(aicore_depth_depth_path(nullptr, "x.png", &h, &w) == nullptr);
    AICORE_CHECK(aicore_depth_pose_path(nullptr, "x.png", ext, intr) != 0);
    AICORE_CHECK(aicore_depth_depth_pose_multi(nullptr, nullptr, 0, &h, &w, &n,
                                               ext, intr) == nullptr);
    AICORE_CHECK(aicore_depth_export_glb(nullptr, "x.png", "/tmp/x.glb") != 0);
    AICORE_CHECK(aicore_depth_export_colmap(nullptr, "x.png", "/tmp/x", 1) !=
                 0);
    AICORE_CHECK(aicore_depth_export_colmap_multi(nullptr, nullptr, 0, "/tmp/x",
                                                  1) != 0);
    AICORE_CHECK(aicore_depth_export_colmap_multi_named(
                         nullptr, nullptr, nullptr, 0, "/tmp/x", 1) != 0);
    AICORE_CHECK(aicore_depth_write_colmap_from_multiview(
                         nullptr, nullptr, nullptr, 0, nullptr, nullptr,
                         nullptr, 0, 0, "/tmp/x", 1) != 0);
    AICORE_CHECK(aicore_depth_depth_dense(nullptr, "x.png", &h, &w, &depth,
                                          &conf, &sky, ext, intr,
                                          &is_metric) != 0);
    AICORE_CHECK(
            aicore_depth_points(nullptr, "x.png", 0.5f, &n, &depth, &rgb) != 0);

    aicore_depth_set_img_resize_target(nullptr, 504);
    aicore_depth_release_gpu_working_memory(nullptr);
    AICORE_CHECK(aicore_depth_cap_img_resize_target(nullptr, 504) == 504);

    char* dir = aicore_depth_model_cache_dir();
    AICORE_CHECK(dir != nullptr && dir[0] != '\0');
    AICORE_CHECK(dir != nullptr && std::strstr(dir, "da3_models") != nullptr);
    aicore_depth_free_string(dir);

    std::fprintf(stderr, "depth_capi_contract ok (abi=%d)\n",
                 aicore_depth_abi_version());
    return failures == 0 ? 0 : 1;
}
