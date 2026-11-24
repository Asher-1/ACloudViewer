// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include "aicore/gaussian_capi.h"

TEST(GaussianCApiContract, AbiAndNullRelease) {
    EXPECT_GE(aicore_gaussian_abi_version(), 1);

    aicore_gaussian_free(nullptr);
    aicore_gaussian_free_buffer(nullptr);
    aicore_gaussian_free_buffer(nullptr);
    aicore_gaussian_free_buffer(nullptr);
    aicore_gaussian_options_free(nullptr);
    aicore_gaussian_accumulator_free(nullptr);
}

TEST(GaussianCApiContract, NullLoadAndContextQueries) {
    EXPECT_EQ(aicore_gaussian_load(nullptr, 1), nullptr);
    EXPECT_EQ(aicore_gaussian_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_gaussian_is_ready(nullptr), 0);
    EXPECT_EQ(aicore_gaussian_info_json(nullptr), nullptr);
    aicore_gaussian_geometry geo{};
    EXPECT_NE(aicore_gaussian_geometry_of(nullptr, &geo), 0);
}

TEST(GaussianCApiContract, OptionsLifecycle) {
    aicore_gaussian_options* opts = aicore_gaussian_options_new();
    ASSERT_NE(opts, nullptr);
    aicore_gaussian_options_set_device(opts, "cpu");
    aicore_gaussian_options_set_threads(opts, 1);
    aicore_gaussian_options_set_dump_taps_dir(opts, "/tmp");
    aicore_gaussian_options_free(opts);
}

TEST(GaussianCApiContract, NullInferenceAndPlyExportsRejectArguments) {
    float* out = nullptr;
    size_t n_out = 0;
    EXPECT_NE(aicore_gaussian_run(nullptr, nullptr, 0, 0, 0, &out, &n_out), 0);
    EXPECT_NE(aicore_gaussian_run_paths(nullptr, nullptr, 0, &out, &n_out), 0);
    /* F-01 batch C: image_view entry is NULL-safe. */
    EXPECT_NE(
            aicore_gaussian_run_image_views(nullptr, nullptr, 0, &out, &n_out),
            0);
    EXPECT_NE(aicore_gaussian_estimate_poses(nullptr, nullptr, 0, 0.f, nullptr,
                                             nullptr),
              0);
    EXPECT_NE(
            aicore_gaussian_export_ply(nullptr, nullptr, 0, 0.f, "/tmp/x.ply"),
            0);

    unsigned char* bytes = nullptr;
    size_t byte_len = 0;
    EXPECT_NE(aicore_gaussian_export_ply_bytes(nullptr, nullptr, 0, 0.f, &bytes,
                                               &byte_len),
              0);
    EXPECT_NE(aicore_gaussian_run_and_export_ply(nullptr, nullptr, 0, 0.f,
                                                 "/tmp/x.ply"),
              0);
}

TEST(GaussianCApiContract, TensorSplatExportHasOne32ByteRecord) {
    const std::filesystem::path tensor_splat =
            std::filesystem::temp_directory_path() /
            "aicore_gaussian_contract_tensor.splat";
    float gaussian[23]{};
    gaussian[0] = 1.0f;
    gaussian[1] = 2.0f;
    gaussian[2] = 3.0f;
    gaussian[15] = 0.75f;
    gaussian[16] = gaussian[17] = gaussian[18] = 1.0f;
    gaussian[19] = 1.0f;
    EXPECT_EQ(aicore_gaussian_export_splat(gaussian, 1, 23, 0.1f, 0,
                                           tensor_splat.string().c_str()),
              0);
    std::error_code file_error;
    EXPECT_TRUE(std::filesystem::file_size(tensor_splat, file_error) == 32 &&
                !file_error);
    std::filesystem::remove(tensor_splat);
}

TEST(GaussianCApiContract, CloudSplatExportHasOne32ByteRecord) {
    std::error_code file_error;
    const std::filesystem::path cloud_splat =
            std::filesystem::temp_directory_path() /
            "aicore_gaussian_contract_cloud.splat";
    aicore_gaussian_point point{};
    point.opacity = 0.75f;
    point.sx = point.sy = point.sz = 1.0f;
    point.qw = 1.0f;
    EXPECT_EQ(aicore_gaussian_export_cloud_splat(&point, 1, 0, 1.0f,
                                                 cloud_splat.string().c_str()),
              0);
    file_error.clear();
    EXPECT_TRUE(std::filesystem::file_size(cloud_splat, file_error) == 32 &&
                !file_error);
    std::filesystem::remove(cloud_splat);
}

TEST(GaussianCApiContract, EmptyAccumulatorLifecycleAndFusion) {
    aicore_gaussian_geometry acc_geom{};
    acc_geom.image_height = 64;
    acc_geom.image_width = 64;
    aicore_gaussian_accumulator* acc =
            aicore_gaussian_accumulator_new(&acc_geom, 0.1f);
    ASSERT_NE(acc, nullptr);
    aicore_gaussian_accumulator_add_pair(acc, nullptr, 23);
    EXPECT_EQ(aicore_gaussian_accumulator_frame_count(acc), 0);
    aicore_gaussian_point* pts = nullptr;
    size_t n_pts = 0;
    aicore_gaussian_accumulator_cloud(acc, &pts, &n_pts);
    aicore_gaussian_accumulator_refine(acc, 0.01f, 1, 0.5f);
    EXPECT_NE(aicore_gaussian_accumulator_fuse(acc, 0.01f, 1, 0, &pts, &n_pts),
              0);
    aicore_gaussian_accumulator_free(acc);
}

TEST(GaussianCApiContract, InvalidParallaxAndCloudOperations) {
    aicore_gaussian_parallax px{};
    EXPECT_NE(aicore_gaussian_pair_parallax(nullptr, nullptr, 0, 0.f, &px), 0);
    aicore_gaussian_point* pts = nullptr;
    size_t n_pts = 0;
    EXPECT_NE(aicore_gaussian_tree_overlap(nullptr, 0, nullptr, 0.f, nullptr,
                                           &pts, &n_pts, nullptr),
              0);
    EXPECT_NE(aicore_gaussian_fuse_cloud(nullptr, 0, 0.01f, 1, 0, &pts, &n_pts),
              0);
    EXPECT_LT(aicore_gaussian_refine_cloud(nullptr, 0, 0.01f, 1, 0.5f), 0.0);
}

TEST(GaussianCApiContract, CpuWarmupAndOwnedCacheDirectory) {
    EXPECT_EQ(aicore_gaussian_warmup_backend("cpu"), 0);

    char* dir = aicore_gaussian_model_cache_dir();
    EXPECT_TRUE(dir != nullptr && dir[0] != '\0');
    EXPECT_TRUE(dir != nullptr &&
                std::strstr(dir, "freesplatter_models") != nullptr);
    aicore_gaussian_free_buffer(dir);
}
