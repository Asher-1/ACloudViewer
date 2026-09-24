// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SAM 3D C ABI contract test — fast, model-free. Covers ABI versioning,
// NULL-safe options/setters, error paths, the runtime model catalog shape,
// and NULL-context accessors. Real-asset accuracy/performance runs live in
// bench_sam3d_backend_acceptance (validation runner).

#include <cstdio>
#include <cstring>
#include <string>

#include "aicore/sam3d_capi.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool looks_like_sha256(const char* text) {
    if (text == nullptr || std::strlen(text) != 64) return false;
    for (const char* p = text; *p; ++p) {
        const char c = *p;
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

}  // namespace

int main() {
    check(aicore_sam3d_abi_version() >= 1, "abi version");
    check(std::strcmp(aicore_sam3d_backend(nullptr), "none") == 0,
          "backend(NULL) == none");
    check(aicore_sam3d_backend_note(nullptr) == nullptr,
          "backend_note(NULL) == NULL");

    // NULL-safe option setters and lifecycle.
    aicore_sam3d_options_set_models_dir(nullptr, "x");
    aicore_sam3d_options_set_dtype(nullptr, AICORE_SAM3D_DTYPE_Q8_0);
    aicore_sam3d_options_set_threads(nullptr, 4);
    aicore_sam3d_options_set_seed(nullptr, 1);
    aicore_sam3d_options_set_steps(nullptr, 2, 3);
    aicore_sam3d_options_set_strict_ss_attention(nullptr, 0);
    aicore_sam3d_options_set_gs_portable_attention(nullptr, 1);
    aicore_sam3d_options_set_philox_blocks(nullptr, 7);
    aicore_sam3d_options_set_disable_moge_cache(nullptr, 1);
    aicore_sam3d_options_set_moge_gguf(nullptr, "y");
    aicore_sam3d_options_set_noise_dir(nullptr, "noise");
    aicore_sam3d_options_set_conditions_out(nullptr, "conditions");
    aicore_sam3d_options_set_device(nullptr, "cpu");
    aicore_sam3d_options_free(nullptr);

    aicore_sam3d_options* options = aicore_sam3d_options_new();
    check(options != nullptr, "options_new");
    aicore_sam3d_options_free(options);

    // load_opts contract errors.
    char err[256] = {0};
    check(aicore_sam3d_load_opts(nullptr, err, sizeof(err)) == nullptr,
          "load_opts(NULL options) rejected");
    options = aicore_sam3d_options_new();
    aicore_sam3d_options_set_models_dir(options,
                                        "/nonexistent-sam3d-models-dir");
    aicore_sam3d_options_set_device(options, "cpu");
    aicore_sam3d_ctx* lazy_ctx =
            aicore_sam3d_load_opts(options, err, sizeof(err));
    check(lazy_ctx != nullptr,
          "load_opts with missing models still constructs (lazy weights)");
    aicore_sam3d_free(lazy_ctx);
    aicore_sam3d_options_free(options);

    // NULL-context accessors.
    check(aicore_sam3d_is_ready(nullptr) == 0, "is_ready(NULL) == 0");
    check(aicore_sam3d_caps(nullptr) == 0, "caps(NULL) == 0");
    check(aicore_sam3d_result_mesh_normals(nullptr) == nullptr,
          "mesh_normals(NULL) == NULL");
    check(aicore_sam3d_result_voxel_count(nullptr) == 0,
          "voxel_count(NULL) == 0");
    aicore_pipeline_timings timings{};
    check(aicore_sam3d_last_pipeline_timings(nullptr, &timings) == -1,
          "timings(NULL ctx) rejected");
    aicore_sam3d_free(nullptr);
    aicore_sam3d_result_free(nullptr);
    aicore_sam3d_free_buffer(nullptr);
    aicore_sam3d_shutdown();  // idempotent, must not destroy anything live

    // Runtime model catalog: published inventory with verifiable digests.
    check(aicore_sam3d_model_count() >= 21,
          "model count covers the raw matrix");
    check(aicore_sam3d_model_by_filename("does-not-exist.gguf") == nullptr,
          "unknown filename rejected");
    check(aicore_sam3d_model_by_filename(nullptr) == nullptr,
          "NULL filename rejected");
    for (int i = 0; i < aicore_sam3d_model_count(); ++i) {
        const aicore_sam3d_model_entry* e = aicore_sam3d_model_at(i);
        check(e != nullptr, "model_at valid index");
        if (e == nullptr) continue;
        check(e->filename != nullptr && *e->filename != '\0',
              "filename present");
        check(e->download_url != nullptr &&
                      std::strstr(e->download_url, "https://") ==
                              e->download_url,
              "download url present");
        check(looks_like_sha256(e->sha256), "sha256 present");
        check(e->size_bytes > 0, "size present");
        check(e->role != nullptr, "role present");
        const aicore_sam3d_model_entry* round_trip =
                aicore_sam3d_model_by_filename(e->filename);
        check(round_trip != nullptr &&
                      std::strcmp(round_trip->filename, e->filename) == 0,
              "by_filename round trip");
    }
    check(std::strcmp(aicore_sam3d_model_download_base(),
                      "https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF/"
                      "resolve/main/") == 0,
          "download base");
    check(aicore_sam3d_model_cache_dir() != nullptr, "cache dir");
    check(std::strstr(aicore_sam3d_info_json(), "\"task\":\"sam3d\"") !=
                  nullptr,
          "info json");

    if (failures == 0) {
        std::printf("sam3d capi contract: all checks passed\n");
        return 0;
    }
    std::printf("sam3d capi contract: %d failures\n", failures);
    return 1;
}
