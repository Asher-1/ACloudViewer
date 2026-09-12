// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// TRELLIS.2 C API contract test — fast, no GGUF assets required. Covers ABI,
// options lifecycle, error paths, image preprocessing (real decode path),
// model catalog and runtime plumbing.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aicore/trellis_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

// A tiny valid PNG (16x16 solid red, RGBA). Generated with a real zlib stream
// (python3 struct+zlib), so both stb-style and libpng decoders accept it.
// 16x16 (not 1x1) on purpose: the preprocess alpha-bbox crop requires a
// non-degenerate bounding box, so a 1x1 image fails the crop with
// "degenerate alpha bounding box".
static const unsigned char kPng1x1[] = {
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  // signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  // IHDR len + type
        0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,  // 16x16
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0xF3, 0xFF,  // 8-bit RGBA
        0x61, 0x00, 0x00, 0x00, 0x19, 0x49, 0x44, 0x41,  // IDAT len + type
        0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0,  // zlib stream (filter
        0x9F, 0x12, 0xCC, 0x30, 0x6A, 0xC0, 0xA8, 0x01,  // 0 rows + RGBA red)
        0xA3, 0x06, 0x0C, 0x17, 0x03, 0x00, 0x30, 0xC4, 0xFE, 0x10, 0x1C, 0x27,
        0xE4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,  // IDAT CRC +
                                                         // IEND
        0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82};

static void test_progress(void*, int, int, int) {}

// Tiny 3-vert / 1-tri mesh used by the GLB bake and export-prep contracts.
static const float kVerts[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
static const int kTris[] = {0, 1, 2};
static const float kPbr6[6] = {0.8f, 0.2f, 0.1f, 0.0f, 0.5f, 1.0f};

static int g_preview_blobs = 0;
static void test_preview(
        void*, int stage, int, int, const void* data, int len) {
    if (!data || len < 8) return;
    const char* magic = (const char*)data;
    if (std::strncmp(magic, "T2VOX01", 7) == 0 ||
        std::strncmp(magic, "T2MESH01", 8) == 0) {
        ++g_preview_blobs;
        (void)stage;
    }
}

int main() {
    AICORE_CHECK(aicore_trellis_abi_version() >= 3);

    // Null-safe teardown / lifecycle.
    aicore_trellis_free(nullptr);
    aicore_trellis_options_free(nullptr);
    aicore_trellis_free_buffer(nullptr);
    aicore_trellis_mesh_free(nullptr);

    AICORE_CHECK(aicore_trellis_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_trellis_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_last_error(nullptr) != nullptr);
    AICORE_CHECK(aicore_trellis_caps(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_backend(nullptr) != nullptr);
    AICORE_CHECK(std::strcmp(aicore_trellis_backend(nullptr), "none") == 0);
    AICORE_CHECK(aicore_trellis_backend_note(nullptr) != nullptr);
    AICORE_CHECK(aicore_trellis_backend_note(nullptr)[0] == '\0');
    // RMBG-result accessors are null-safe and empty on a null mesh.
    AICORE_CHECK(aicore_trellis_mesh_has_rmbg(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_mesh_rmbg_rgba(nullptr) == nullptr);
    AICORE_CHECK(aicore_trellis_mesh_rmbg_w(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_mesh_rmbg_h(nullptr) == 0);

    // Options lifecycle + NULL no-ops.
    aicore_trellis_options* opts = aicore_trellis_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_trellis_options_set_device(opts, "cpu");
    aicore_trellis_options_set_threads(opts, 1);
    aicore_trellis_options_set_rmbg_gguf(opts, "/nonexistent/rmbg_f16.gguf");
    aicore_trellis_options_set_shape_dec_placement(opts, "cpu");
    aicore_trellis_options_set_sdpa_exact(opts, 1);
    aicore_trellis_options_set_sdpa_flash(opts, 0);
    aicore_trellis_options_set_timing(opts, 0);
    aicore_trellis_options_set_device(nullptr, "cpu");
    aicore_trellis_options_set_threads(nullptr, 1);
    aicore_trellis_options_set_rmbg_gguf(nullptr, "x");
    aicore_trellis_options_set_shape_dec_placement(nullptr, "cpu");
    aicore_trellis_options_set_sdpa_exact(nullptr, 1);
    aicore_trellis_options_set_sdpa_flash(nullptr, 0);
    aicore_trellis_options_set_timing(nullptr, 0);

    // Loading nonexistent model files must fail cleanly with a null ctx and
    // no crash; the error is visible through last_error only when a ctx
    // exists, so the contract here is just the null return.
    aicore_trellis_model_paths paths{};
    paths.dino_gguf = "/nonexistent/dino_f16.gguf";
    paths.ss_flow_gguf = "/nonexistent/ss_flow_q8.gguf";
    paths.ss_dec_gguf = "/nonexistent/ss_dec_f16.gguf";
    aicore_trellis_ctx* ctx = aicore_trellis_load_opts(&paths, opts);
    AICORE_CHECK(ctx == nullptr);

    // Missing required paths -> null.
    aicore_trellis_model_paths empty{};
    AICORE_CHECK(aicore_trellis_load_opts(&empty, nullptr) == nullptr);

    // Inference guards: null context / invalid image must not crash.
    char err[256] = {0};
    AICORE_CHECK(aicore_trellis_generate(nullptr, kPng1x1, (int)sizeof(kPng1x1),
                                         nullptr, test_progress, nullptr, err,
                                         sizeof(err)) == nullptr);
    AICORE_CHECK(aicore_trellis_generate(ctx /* null */, nullptr, 0, nullptr,
                                         nullptr, nullptr, err,
                                         sizeof(err)) == nullptr);
    // generate_ex: null-context guard + null preview == generate.
    AICORE_CHECK(aicore_trellis_generate_ex(
                         nullptr, kPng1x1, (int)sizeof(kPng1x1), nullptr,
                         test_progress, nullptr, test_preview, nullptr, err,
                         sizeof(err)) == nullptr);

    // Standalone texturing / export-prep contracts: guards must not crash.
    AICORE_CHECK(aicore_trellis_texture_mesh(
                         nullptr, kVerts, 3, kTris, 1, nullptr, 0, nullptr, 0,
                         AICORE_TRELLIS_PIPE_512, kPng1x1, (int)sizeof(kPng1x1),
                         AICORE_TRELLIS_BG_AUTO, 0, 0, test_progress, nullptr,
                         err, sizeof(err)) == nullptr);
    AICORE_CHECK(aicore_trellis_prepare_mesh(nullptr, 0, nullptr, 0, nullptr, 0,
                                             err, sizeof(err)) == nullptr);
    AICORE_CHECK(aicore_trellis_prepare_mesh(kVerts, 3, kTris, 1, nullptr, 3,
                                             err, sizeof(err)) ==
                 nullptr);  // bad filter
    AICORE_CHECK(aicore_trellis_prepare_mesh(kVerts, 3, kTris, 1, nullptr, 2,
                                             err, sizeof(err)) != nullptr);
    // Print remesh availability is a build-time constant; the call must be
    // safe either way and the prepare guard must hold.
    const int printable = aicore_trellis_print_remesh_available();
    AICORE_CHECK(printable == 0 || printable == 1);
    AICORE_CHECK(aicore_trellis_prepare_print_mesh(kVerts, 3, kTris, 1, nullptr,
                                                   0, 0.01f, 0.01f, err,
                                                   sizeof(err)) ==
                 nullptr);  // degenerate single triangle
    // Projected GLB bake: guards + (unavailable CGAL -> null, not crash).
    int out_len = 0;
    AICORE_CHECK(aicore_trellis_bake_projected_glb(
                         nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
                         nullptr, 0, 0, &out_len, err, sizeof(err)) == nullptr);
    AICORE_CHECK(aicore_trellis_bake_projected_glb(
                         kVerts, 3, kTris, 1, kVerts, 3, kTris, 1, kPbr6, 0, 0,
                         &out_len, err, sizeof(err)) ==
                 nullptr);  // CGAL unavailable in-tree, wrap target degenerate

    // prepare_mesh returns the same topology for KeepAll and fills normals.
    aicore_trellis_mesh* prepared = aicore_trellis_prepare_mesh(
            kVerts, 3, kTris, 1, nullptr, 2, err, sizeof(err));
    AICORE_CHECK(prepared != nullptr);
    AICORE_CHECK(aicore_trellis_mesh_n_verts(prepared) == 3);
    AICORE_CHECK(aicore_trellis_mesh_n_tris(prepared) == 1);
    AICORE_CHECK(aicore_trellis_mesh_normals(prepared) != nullptr);
    aicore_trellis_mesh_free(prepared);

    // Preprocess: real decode path on the tiny PNG; 16x16 -> 512x512 RGB.
    std::vector<unsigned char> rgb((size_t)512 * 512 * 3, 0x7F);
    AICORE_CHECK(aicore_trellis_preprocess_image_bytes(
                         kPng1x1, (int)sizeof(kPng1x1), 512, rgb.data(),
                         AICORE_TRELLIS_BG_AUTO, err, sizeof(err)) == 0);
    // Invalid arguments are rejected.
    AICORE_CHECK(aicore_trellis_preprocess_image_bytes(
                         nullptr, 0, 512, rgb.data(), AICORE_TRELLIS_BG_AUTO,
                         err, sizeof(err)) != 0);
    AICORE_CHECK(aicore_trellis_preprocess_image_bytes(
                         kPng1x1, (int)sizeof(kPng1x1), 512, nullptr,
                         AICORE_TRELLIS_BG_AUTO, err, sizeof(err)) != 0);
    AICORE_CHECK(aicore_trellis_preprocess_image_bytes(
                         kPng1x1, (int)sizeof(kPng1x1), 0, rgb.data(),
                         AICORE_TRELLIS_BG_AUTO, err, sizeof(err)) != 0);
    // Trash bytes fail decode cleanly.
    static const unsigned char kTrash[64] = {0};
    AICORE_CHECK(aicore_trellis_preprocess_image_bytes(
                         kTrash, (int)sizeof(kTrash), 512, rgb.data(),
                         AICORE_TRELLIS_BG_AUTO, err, sizeof(err)) != 0);

    // GLB bake contract: NULL guards and invalid component filter.
    AICORE_CHECK(aicore_trellis_bake_glb(nullptr, 0, nullptr, 0, nullptr, 0, 0,
                                         &out_len, err,
                                         sizeof(err)) == nullptr);
    AICORE_CHECK(aicore_trellis_bake_glb(kVerts, 3, kTris, 1, nullptr, 0, 3,
                                         &out_len, err, sizeof(err)) ==
                 nullptr);  // bad filter
    AICORE_CHECK(aicore_trellis_bake_glb(kVerts, 3, kTris, 1, nullptr, 0, 0,
                                         &out_len, err,
                                         sizeof(err)) != nullptr);
    AICORE_CHECK(out_len > 0);

    // Mesh accessor guards on a null handle.
    AICORE_CHECK(aicore_trellis_mesh_n_verts(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_mesh_n_tris(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_mesh_verts(nullptr) == nullptr);
    AICORE_CHECK(aicore_trellis_mesh_normals(nullptr) == nullptr);
    AICORE_CHECK(aicore_trellis_mesh_tris(nullptr) == nullptr);
    AICORE_CHECK(aicore_trellis_mesh_has_pbr(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_mesh_pbr(nullptr) == nullptr);
    AICORE_CHECK(aicore_trellis_mesh_grid_res(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_mesh_grid_nvox(nullptr) == 0);
    AICORE_CHECK(aicore_trellis_mesh_grid_feats(nullptr) == nullptr);
    AICORE_CHECK(aicore_trellis_mesh_grid_coords(nullptr) == nullptr);

    // Model catalog contract (must match the HF TRELLIS.2 published assets).
    AICORE_CHECK(aicore_trellis_model_count() == 26);
    static const char* kExpected[] = {"dino_f16.gguf",
                                      "dino_q8.gguf",
                                      "dino_f32.gguf",
                                      "ss_flow_f16.gguf",
                                      "ss_flow_q8.gguf",
                                      "ss_flow_f32.gguf",
                                      "ss_dec_f16.gguf",
                                      "ss_dec_q8.gguf",
                                      "ss_dec_f32.gguf",
                                      "slat_flow_f16.gguf",
                                      "slat_flow_q8.gguf",
                                      "slat_flow_f32.gguf",
                                      "slat_flow_1024_f16.gguf",
                                      "slat_flow_1024_q8.gguf",
                                      "slat_flow_1024_f32.gguf",
                                      "shape_dec_f16.gguf",
                                      "shape_dec_f32.gguf",
                                      "shape_enc_f16.gguf",
                                      "tex_dec_f16.gguf",
                                      "tex_slat_flow_512_f16.gguf",
                                      "tex_slat_flow_512_q8.gguf",
                                      "tex_slat_flow_1024_f16.gguf",
                                      "tex_slat_flow_1024_q8.gguf",
                                      "rmbg_f32.gguf",
                                      "rmbg_f16.gguf",
                                      "rmbg_q8.gguf"};
    for (int i = 0; i < 26; ++i) {
        const aicore_trellis_model_entry* e = aicore_trellis_model_at(i);
        AICORE_CHECK(e != nullptr && e->filename != nullptr &&
                     std::strcmp(e->filename, kExpected[i]) == 0 &&
                     e->download_url != nullptr && e->display_name != nullptr &&
                     e->quant_note != nullptr && e->license_note != nullptr &&
                     e->role != nullptr && e->size_bytes > 0 &&
                     e->sha256 != nullptr && std::strlen(e->sha256) == 64 &&
                     std::strstr(e->download_url, "huggingface.co/") !=
                             nullptr);
    }
    AICORE_CHECK(aicore_trellis_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_trellis_model_at(26) == nullptr);
    AICORE_CHECK(aicore_trellis_model_by_filename("dino_q8.gguf") != nullptr);
    AICORE_CHECK(aicore_trellis_model_by_filename("shape_dec_f16.gguf") !=
                 nullptr);
    AICORE_CHECK(aicore_trellis_model_by_filename("ss_flow_f32.gguf") !=
                 nullptr);
    AICORE_CHECK(aicore_trellis_model_by_filename("nope.gguf") == nullptr);
    AICORE_CHECK(aicore_trellis_model_by_filename(nullptr) == nullptr);
    AICORE_CHECK(aicore_trellis_model_download_base() != nullptr &&
                 std::strstr(aicore_trellis_model_download_base(),
                             "huggingface.co/Asher-1/Trellis2-models") !=
                         nullptr);

    // Device enumeration / warmup.
    AICORE_CHECK(aicore_trellis_warmup_backend("cpu") == 0);

    char* dir = aicore_trellis_model_cache_dir();
    AICORE_CHECK(dir != nullptr && std::strlen(dir) > 0);
    aicore_trellis_free_buffer(dir);

    // info_json contract (null-safe, parseable shell).
    char* info = aicore_trellis_info_json(nullptr);
    AICORE_CHECK(info != nullptr && std::strstr(info, "error") != nullptr);
    aicore_trellis_free_buffer(info);

    // Shutdown is idempotent and must not disturb later warmups.
    aicore_trellis_shutdown();
    aicore_trellis_shutdown();

    aicore_trellis_options_free(opts);

    if (failures == 0) {
        std::printf("[trellis] contract test passed\n");
    }
    return failures;
}
