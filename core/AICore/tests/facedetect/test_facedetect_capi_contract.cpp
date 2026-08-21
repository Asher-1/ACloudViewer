// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// FaceDetect C API contract test — fast, no GGUF assets required. Covers ABI,
// model catalog queries, cache-dir / warmup / shutdown lifecycle and null
// guards. Model inference is exercised by the *_capi model tests.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/facedetect_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_facedetect_abi_version() >= 1);

    // Null-safe teardown.
    aicore_facedetect_free(nullptr);
    aicore_facedetect_options_free(nullptr);
    aicore_facedetect_free_buffer(nullptr);
    aicore_facedetect_free_buffer(nullptr);

    AICORE_CHECK(aicore_facedetect_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_facedetect_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_facedetect_last_error(nullptr) == nullptr);

    // Options lifecycle.
    aicore_facedetect_options* opts = aicore_facedetect_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_facedetect_options_set_device(opts, "cpu");
    aicore_facedetect_options_set_threads(opts, 1);
    aicore_facedetect_options_set_device(nullptr, "cpu");
    aicore_facedetect_options_set_threads(nullptr, 1);

    // Loading a nonexistent file must fail cleanly and report an error.
    aicore_facedetect_ctx* ctx =
            aicore_facedetect_load_opts("/nonexistent/fd.gguf", opts);
    AICORE_CHECK(ctx != nullptr);  // ctx is allocated; model inside is null
    AICORE_CHECK(aicore_facedetect_is_ready(ctx) == 0);
    AICORE_CHECK(aicore_facedetect_last_error(ctx) != nullptr);
    AICORE_CHECK(aicore_facedetect_info_json(ctx) != nullptr);
    aicore_facedetect_free(ctx);
    aicore_facedetect_options_free(opts);

    // Image loader: nonexistent path must fail and zero the outputs.
    uint8_t* rgb = nullptr;
    int32_t iw = 0, ih = 0;
    AICORE_CHECK(aicore_facedetect_load_path_rgb("/nonexistent/x.png", &rgb,
                                                 &iw, &ih) == -1);
    AICORE_CHECK(rgb == nullptr && iw == 0 && ih == 0);

    // Model catalog contract: entries must be well-formed and the
    // detector / landmark views must be consistent with the full list.
    const int n_total = aicore_facedetect_model_count();
    AICORE_CHECK(n_total > 0);
    AICORE_CHECK(aicore_facedetect_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_facedetect_model_at(n_total) == nullptr);
    for (int i = 0; i < n_total; ++i) {
        const aicore_facedetect_model_entry* e = aicore_facedetect_model_at(i);
        AICORE_CHECK(e != nullptr && e->filename != nullptr &&
                     e->download_url != nullptr && e->display_name != nullptr &&
                     e->license_note != nullptr);
    }
    const int n_det = aicore_facedetect_detector_model_count();
    const int n_land = aicore_facedetect_landmark_model_count();
    AICORE_CHECK(n_det > 0);
    AICORE_CHECK(n_land >= 0);
    AICORE_CHECK(n_det + n_land == n_total);
    AICORE_CHECK(aicore_facedetect_detector_model_at(0) != nullptr);
    AICORE_CHECK(aicore_facedetect_detector_model_at(n_det) == nullptr);
    if (n_land > 0) {
        AICORE_CHECK(aicore_facedetect_landmark_model_at(0) != nullptr);
        AICORE_CHECK(aicore_facedetect_landmark_model_at(n_land) == nullptr);
    }

    const aicore_facedetect_model_entry* first = aicore_facedetect_model_at(0);
    AICORE_CHECK(aicore_facedetect_model_by_filename(first->filename) == first);
    AICORE_CHECK(aicore_facedetect_model_by_filename("nope.gguf") == nullptr);
    AICORE_CHECK(aicore_facedetect_model_download_base() != nullptr &&
                 std::strstr(aicore_facedetect_model_download_base(),
                             "qFaceDetect") != nullptr);

    // Cache dir / warmup / shutdown (idempotent).
    char* dir = aicore_facedetect_model_cache_dir();
    AICORE_CHECK(dir != nullptr && std::strlen(dir) > 0);
    aicore_facedetect_free_buffer(dir);
    AICORE_CHECK(aicore_facedetect_warmup_backend("cpu") == 0);
    aicore_facedetect_shutdown();
    aicore_facedetect_shutdown();  // must be idempotent

    if (failures == 0) {
        std::printf("[facedetect] contract test passed\n");
    }
    return failures;
}
