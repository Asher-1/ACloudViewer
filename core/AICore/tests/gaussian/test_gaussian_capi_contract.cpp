// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>

#include "aicore/gaussian_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_gaussian_abi_version() >= 1);

    aicore_gaussian_free(nullptr);
    aicore_gaussian_free_floats(nullptr);
    aicore_gaussian_free_bytes(nullptr);
    aicore_gaussian_free_string(nullptr);
    aicore_gaussian_options_free(nullptr);
    aicore_gaussian_accumulator_free(nullptr);

    AICORE_CHECK(aicore_gaussian_load(nullptr, 1) == nullptr);
    AICORE_CHECK(aicore_gaussian_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_gaussian_info_json(nullptr) == nullptr);

    aicore_gaussian_options* opts = aicore_gaussian_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_gaussian_options_set_device(opts, "cpu");
    aicore_gaussian_options_set_threads(opts, 1);
    aicore_gaussian_options_set_dump_taps_dir(opts, "/tmp");
    aicore_gaussian_options_free(opts);

    aicore_gaussian_geometry geo{};
    AICORE_CHECK(aicore_gaussian_geometry_of(nullptr, &geo) != 0);

    float* out = nullptr;
    size_t n_out = 0;
    AICORE_CHECK(aicore_gaussian_run(nullptr, nullptr, 0, 0, 0, &out, &n_out) !=
                 0);
    AICORE_CHECK(aicore_gaussian_run_paths(nullptr, nullptr, 0, &out, &n_out) !=
                 0);
    AICORE_CHECK(aicore_gaussian_estimate_poses(nullptr, 0, 0, 0, 0, 0.f,
                                                nullptr, nullptr) != 0);
    AICORE_CHECK(aicore_gaussian_export_ply(nullptr, 0, 0, 0, 0, 0, 0.f,
                                            "/tmp/x.ply") != 0);

    unsigned char* bytes = nullptr;
    size_t byte_len = 0;
    AICORE_CHECK(aicore_gaussian_export_ply_bytes(nullptr, 0, 0, 0, 0, 0, 0.f,
                                                  &bytes, &byte_len) != 0);
    AICORE_CHECK(aicore_gaussian_run_and_export_ply(nullptr, nullptr, 0, 0.f,
                                                    "/tmp/x.ply") != 0);

    aicore_gaussian_parallax px{};
    AICORE_CHECK(aicore_gaussian_pair_parallax(nullptr, 0, 0, 0, 0, 0.f, &px) !=
                 0);

    aicore_gaussian_accumulator* acc =
            aicore_gaussian_accumulator_new(64, 64, 0.1f);
    AICORE_CHECK(acc != nullptr);
    aicore_gaussian_accumulator_add_pair(acc, nullptr, 23);
    AICORE_CHECK(aicore_gaussian_accumulator_frame_count(acc) == 0);
    aicore_gaussian_point* pts = nullptr;
    size_t n_pts = 0;
    aicore_gaussian_accumulator_cloud(acc, &pts, &n_pts);
    aicore_gaussian_accumulator_refine(acc, 0.01f, 1, 0.5f);
    AICORE_CHECK(aicore_gaussian_accumulator_fuse(acc, 0.01f, 1, 0, &pts,
                                                  &n_pts) != 0);
    aicore_gaussian_accumulator_free(acc);

    AICORE_CHECK(aicore_gaussian_tree_overlap(nullptr, 0, 0, 0, 0, 0.f, 0, 0, 0,
                                              0.f, 0, &pts, &n_pts,
                                              nullptr) != 0);
    AICORE_CHECK(aicore_gaussian_fuse_cloud(nullptr, 0, 0.01f, 1, 0, &pts,
                                            &n_pts) != 0);
    AICORE_CHECK(aicore_gaussian_refine_cloud(nullptr, 0, 0.01f, 1, 0.5f) <
                 0.0);

    AICORE_CHECK(aicore_gaussian_warmup_backend("cpu") == 0);

    char* dir = aicore_gaussian_model_cache_dir();
    AICORE_CHECK(dir != nullptr && dir[0] != '\0');
    AICORE_CHECK(dir != nullptr &&
                 std::strstr(dir, "freesplatter_models") != nullptr);
    aicore_gaussian_free_string(dir);

    std::fprintf(stderr, "gaussian_capi_contract ok (abi=%d)\n",
                 aicore_gaussian_abi_version());
    return failures == 0 ? 0 : 1;
}
