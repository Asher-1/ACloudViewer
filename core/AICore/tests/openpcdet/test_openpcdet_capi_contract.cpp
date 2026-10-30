// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// OpenPCDet C API contract test — fast, no GGUF assets required. Covers ABI,
// options lifecycle, catalog queries, geometry guard paths, null guards and
// the model-free introspection/error paths. Real-asset inference is exercised
// by bench_openpcdet_perf (validation runner) and the plugin workflow tests.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include "aicore/openpcdet_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_openpcdet_abi_version() >= 1);

    // Null-safe teardown / lifecycle.
    aicore_openpcdet_free(nullptr);
    aicore_openpcdet_options_free(nullptr);
    aicore_openpcdet_free_buffer(nullptr);
    aicore_openpcdet_shutdown();  // idempotent, no contexts alive

    AICORE_CHECK(aicore_openpcdet_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_openpcdet_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_openpcdet_last_error(nullptr) != nullptr);
    AICORE_CHECK(aicore_openpcdet_geometry_of(nullptr, nullptr) == -1);
    AICORE_CHECK(aicore_openpcdet_detection_count(nullptr) == -1);
    AICORE_CHECK(aicore_openpcdet_detection_at(nullptr, 0, nullptr) == -1);
    AICORE_CHECK(aicore_openpcdet_last_pipeline_timings(nullptr, nullptr) == -1);
    AICORE_CHECK(aicore_openpcdet_detect_points(nullptr, nullptr, 0, 0) == -1);
    AICORE_CHECK(aicore_openpcdet_info_json(nullptr) == nullptr);
    AICORE_CHECK(aicore_openpcdet_context_device(nullptr) != nullptr);
    AICORE_CHECK(aicore_openpcdet_context_threads(nullptr) == 0);

    // Options lifecycle: every setter must be a no-op on NULL (no crash).
    aicore_openpcdet_options_set_device(nullptr, "cpu");
    aicore_openpcdet_options_set_threads(nullptr, 4);
    aicore_openpcdet_options_set_score_thresh(nullptr, 0.5f);
    aicore_openpcdet_options_set_nms_thresh(nullptr, 0.5f);
    aicore_openpcdet_options_set_max_boxes(nullptr, 10);
    aicore_openpcdet_options_set_dump_taps_dir(nullptr, "/tmp");

    aicore_openpcdet_options* opts = aicore_openpcdet_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_openpcdet_options_set_device(opts, "cpu");
    aicore_openpcdet_options_set_threads(opts, 2);
    aicore_openpcdet_options_set_score_thresh(opts, 0.3f);
    aicore_openpcdet_options_set_nms_thresh(opts, 0.4f);
    aicore_openpcdet_options_set_max_boxes(opts, 33);
    aicore_openpcdet_options_set_dump_taps_dir(opts, nullptr);  // disable
    // NaN threshold overrides must be ignored (keep GGUF defaults).
    aicore_openpcdet_options_set_score_thresh(opts,
                                              std::numeric_limits<float>::quiet_NaN());
    aicore_openpcdet_options_set_nms_thresh(opts,
                                            std::numeric_limits<float>::quiet_NaN());
    aicore_openpcdet_options_set_max_boxes(opts, 0);
    aicore_openpcdet_options_free(opts);

    // Failed loads produce a non-ready context with an error string.
    aicore_openpcdet_options* bad = aicore_openpcdet_options_new();
    aicore_openpcdet_options_set_device(bad, "cpu");
    aicore_openpcdet_ctx* missing =
            aicore_openpcdet_load_opts("no_such_model.gguf", bad);
    AICORE_CHECK(missing != nullptr);  // non-NULL handle carrying the error
    AICORE_CHECK(aicore_openpcdet_is_ready(missing) == 0);
    AICORE_CHECK(aicore_openpcdet_last_error(missing) != nullptr &&
                 aicore_openpcdet_last_error(missing)[0] != '\0');
    AICORE_CHECK(aicore_openpcdet_detect_points(missing, nullptr, 0, 4) == -1);
    AICORE_CHECK(aicore_openpcdet_geometry_of(missing, nullptr) == -1);
    char* no_info = aicore_openpcdet_info_json(missing);
    AICORE_CHECK(no_info == nullptr);
    aicore_openpcdet_free(missing);
    aicore_openpcdet_options_free(bad);

    // Invalid detect arguments on a NULL context stay non-positive.
    float dummy[4] = {0.f, 0.f, 0.f, 0.f};
    AICORE_CHECK(aicore_openpcdet_detect_points(nullptr, dummy, 1, 4) == -1);

    // Catalog contract: base URL, count, entries, default index, lookups.
    AICORE_CHECK(aicore_openpcdet_model_count() == 54);
    AICORE_CHECK(aicore_openpcdet_model_default_index() >= 0 &&
                 aicore_openpcdet_model_default_index() <
                         aicore_openpcdet_model_count());
    const char* base = aicore_openpcdet_model_download_base();
    AICORE_CHECK(base != nullptr &&
                 std::strstr(base, "cloudViewer_downloads") != nullptr &&
                 std::strstr(base, "OpenPCDet-GGUF") != nullptr);
    for (int i = 0; i < aicore_openpcdet_model_count(); ++i) {
        const aicore_openpcdet_model_entry* e = aicore_openpcdet_model_at(i);
        AICORE_CHECK(e != nullptr);
        AICORE_CHECK(e->filename != nullptr && e->filename[0] != '\0');
        AICORE_CHECK(e->download_url != nullptr &&
                     std::strncmp(e->download_url, base, std::strlen(base)) == 0);
        AICORE_CHECK(std::strstr(e->download_url, e->filename) != nullptr);
        AICORE_CHECK(e->display_name != nullptr);
        AICORE_CHECK(e->quant_note != nullptr);
        AICORE_CHECK(e->family_note != nullptr);
        AICORE_CHECK(e->license_note != nullptr);
        const std::string name = e->filename;
        AICORE_CHECK(name.size() > 5 && name.substr(name.size() - 5) == ".gguf");
    }
    AICORE_CHECK(aicore_openpcdet_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_openpcdet_model_at(aicore_openpcdet_model_count()) ==
                 nullptr);
    AICORE_CHECK(aicore_openpcdet_model_by_filename(nullptr) == nullptr);
    AICORE_CHECK(aicore_openpcdet_model_by_filename("nope.gguf") == nullptr);
    const aicore_openpcdet_model_entry* known =
            aicore_openpcdet_model_by_filename("kitti_pointpillar.f16.gguf");
    AICORE_CHECK(known != nullptr &&
                 std::strstr(known->family_note, "PointPillar") != nullptr);
    known = aicore_openpcdet_model_by_filename("second_7862.f32.gguf");
    AICORE_CHECK(known != nullptr &&
                 std::strstr(known->family_note, "SECOND") != nullptr);

    // Scene ABI contract: version, null/invalid guards and error paths
    // (asset-free; multi-modal inference is covered by bench_openpcdet_perf).
    AICORE_CHECK(AICORE_OPENPCDET_SCENE_ABI_VERSION >= 1);
    AICORE_CHECK(aicore_openpcdet_detect_scene(nullptr, nullptr) == -1);
    aicore_openpcdet_scene scene{};
    scene.abi_version = AICORE_OPENPCDET_SCENE_ABI_VERSION;
    AICORE_CHECK(aicore_openpcdet_detect_scene(nullptr, &scene) == -1);
    aicore_openpcdet_free_scene(nullptr);  // NULL-safe
    aicore_openpcdet_free_scene(&scene);   // empty scene is safe to release
    char* scene_error = nullptr;
    AICORE_CHECK(aicore_openpcdet_load_scene_dir("no_such_scene_dir",
                                                 &scene, &scene_error) == -1);
    AICORE_CHECK(scene_error != nullptr && scene_error[0] != '\0');
    aicore_openpcdet_free_buffer(scene_error);
    // A stale ABI version must be rejected outright.
    aicore_openpcdet_options* stale_opts = aicore_openpcdet_options_new();
    aicore_openpcdet_options_set_device(stale_opts, "cpu");
    aicore_openpcdet_ctx* stale_ctx =
            aicore_openpcdet_load_opts("no_such_model.gguf", stale_opts);
    aicore_openpcdet_options_free(stale_opts);
    AICORE_CHECK(stale_ctx != nullptr);
    aicore_openpcdet_scene stale{};
    stale.abi_version = AICORE_OPENPCDET_SCENE_ABI_VERSION + 1;
    AICORE_CHECK(aicore_openpcdet_detect_scene(stale_ctx, &stale) == -1);
    aicore_openpcdet_free(stale_ctx);

    if (failures == 0) {
        std::printf("test_openpcdet_capi_contract: all checks passed\n");
        return 0;
    }
    std::printf("test_openpcdet_capi_contract: %d check(s) FAILED\n", failures);
    return 1;
}
