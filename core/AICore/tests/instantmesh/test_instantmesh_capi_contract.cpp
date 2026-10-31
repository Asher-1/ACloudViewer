// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// InstantMesh C API contract test — fast, no GGUF assets required. Covers
// ABI, options lifecycle, error paths, image preprocessing (real decode
// path), model catalog and runtime plumbing.

#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "aicore/asset_digests.h"
#include "aicore/instantmesh_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

// A tiny valid PNG (16x16 solid red, RGBA) — same bytes as the trellis
// contract fixture (real zlib stream, decoded by the Qt image path).
static const unsigned char kPng1x1[] = {
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  // signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  // IHDR len + type
        0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,  // 16x16
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0xF3, 0xFF,  // 8-bit RGBA
        0x61, 0x00, 0x00, 0x00, 0x19, 0x49, 0x44, 0x41,  // IDAT len + type
        0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0,  // zlib stream
        0x9F, 0x12, 0xCC, 0x30, 0x6A, 0xC0, 0xA8, 0x01, 0xA3, 0x06, 0x0C, 0x17,
        0x03, 0x00, 0x30, 0xC4, 0xFE, 0x10, 0x1C, 0x27, 0xE4, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82};

static void test_progress(void*, int, int, int) {}

// Tiny 3-vert / 1-tri mesh used by the GLB bake contract.
static const float kVerts[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
static const int kTris[] = {0, 1, 2};

static void test_versioned_load_errors() {
#if defined(AICORE_INSTANTMESH_ABI_VERSION)
    char err[256] = {};
    AICORE_CHECK(aicore_instantmesh_load_opts_ex(nullptr, nullptr, err,
                                                 sizeof(err)) == nullptr);
    AICORE_CHECK(std::strstr(err, "paths") != nullptr);
    aicore_instantmesh_model_paths paths = AICORE_INSTANTMESH_MODEL_PATHS_INIT;
    paths.struct_size = sizeof(paths) - 1;
    AICORE_CHECK(aicore_instantmesh_load_opts_ex(&paths, nullptr, err,
                                                 sizeof(err)) == nullptr);
    AICORE_CHECK(std::strstr(err, "struct_size") != nullptr);
    paths.struct_size = sizeof(paths);
    paths.abi_version = AICORE_INSTANTMESH_ABI_VERSION + 1;
    AICORE_CHECK(aicore_instantmesh_load_opts_ex(&paths, nullptr, err,
                                                 sizeof(err)) == nullptr);
    AICORE_CHECK(std::strstr(err, "abi_version") != nullptr);
    paths.abi_version = AICORE_INSTANTMESH_ABI_VERSION;
    paths.dino_gguf = "missing-dino.gguf";
    paths.transformer_gguf = "missing-transformer.gguf";
    paths.synthesizer_gguf = "missing-synthesizer.gguf";
    paths.cond_gguf = "missing-cond.gguf";
    AICORE_CHECK(aicore_instantmesh_load_opts_ex(&paths, nullptr, err,
                                                 sizeof(err)) == nullptr);
    AICORE_CHECK(std::strstr(err, "Zero123++") != nullptr);
    paths.cond_gguf = nullptr;
    aicore_instantmesh_options* opts = aicore_instantmesh_options_new();
    aicore_instantmesh_options_set_device(opts, "cpu");
    AICORE_CHECK(aicore_instantmesh_load_opts_ex(&paths, opts, err,
                                                 sizeof(err)) == nullptr);
    AICORE_CHECK(std::strstr(err, "load") != nullptr);
    char tiny[1] = {'x'};
    aicore_instantmesh_load_opts_ex(nullptr, opts, tiny, sizeof(tiny));
    AICORE_CHECK(tiny[0] == '\0');
    aicore_instantmesh_options_set_device(opts, "cuda:9999");
    aicore_instantmesh_load_opts_ex(&paths, opts, err, sizeof(err));
    AICORE_CHECK(std::strstr(err, "backend unavailable") != nullptr);
    aicore_instantmesh_options_set_device(opts, "cpu");
    aicore_instantmesh_options_set_steps(opts, 1);
    aicore_instantmesh_load_opts_ex(&paths, opts, err, sizeof(err));
    AICORE_CHECK(std::strstr(err, "steps") != nullptr);
    aicore_instantmesh_options_free(opts);
#else
    std::fprintf(stderr,
                 "Missing versioned structs and load entry point with returned "
                 "errors\n");
    AICORE_CHECK(false);
#endif
}

static void test_generate_parameter_versions() {
#if defined(AICORE_INSTANTMESH_ABI_VERSION)
    aicore_instantmesh_generate_params params =
            AICORE_INSTANTMESH_GENERATE_PARAMS_INIT;
    char err[256] = {};
    params.struct_size = sizeof(params) - 1;
    AICORE_CHECK(aicore_instantmesh_generate(nullptr, kPng1x1, sizeof(kPng1x1),
                                             &params, nullptr, nullptr, err,
                                             sizeof(err)) == nullptr);
    AICORE_CHECK(std::strstr(err, "struct_size") != nullptr);
    params.struct_size = sizeof(params);
    params.abi_version++;
    AICORE_CHECK(aicore_instantmesh_generate(nullptr, kPng1x1, sizeof(kPng1x1),
                                             &params, nullptr, nullptr, err,
                                             sizeof(err)) == nullptr);
    AICORE_CHECK(std::strstr(err, "abi_version") != nullptr);
#endif
}

static void test_invalid_numeric_parameters() {
    aicore_instantmesh_generate_params params =
            AICORE_INSTANTMESH_GENERATE_PARAMS_INIT;
    char err[256] = {};
    const auto rejected = [&](const char* field) {
        aicore_instantmesh_generate(nullptr, kPng1x1, sizeof(kPng1x1), &params,
                                    nullptr, nullptr, err, sizeof(err));
        AICORE_CHECK(std::strstr(err, field) != nullptr);
    };
    params.steps = 1;
    rejected("steps");
    params.steps = 1001;
    rejected("steps");
    params.steps = 0;
    params.guidance = std::numeric_limits<float>::quiet_NaN();
    rejected("guidance");
    params.guidance = std::numeric_limits<float>::infinity();
    rejected("guidance");
    params.guidance = 0;
    params.grid_res = 257;
    rejected("grid_res");
    params.grid_res = 1;
    rejected("grid_res");
    params.grid_res = 0;
    params.grid_scale = std::numeric_limits<float>::infinity();
    rejected("grid_scale");
    params.grid_scale = 0;
    params.texture_res = 4097;
    rejected("texture_res");
    params.texture_res = 0;
    params.bake_texture = 2;
    rejected("bake_texture");
}

static void test_decoded_formats_and_stride() {
#if defined(AICORE_INSTANTMESH_ABI_VERSION)
    // Two real padded pixel rows; retain alpha until the white-compositing
    // boundary.
    const uint8_t rgb[] = {255, 0, 0, 17, 255, 0, 0, 23};
    const uint8_t bgr[] = {0, 0, 255, 17, 0, 0, 255, 23};
    const uint8_t rgba[] = {255, 0, 0, 128, 17, 255, 0, 0, 128, 23};
    const uint8_t bgra[] = {0, 0, 255, 128, 17, 0, 0, 255, 128, 23};
    const uint8_t gray[] = {91, 17, 91, 23};
    const uint8_t* buffers[] = {rgb, rgba, gray, bgr, bgra};
    const size_t strides[] = {4, 5, 2, 4, 5};
    char err[256] = {};
    uint8_t out[3] = {};
    for (int f = AICORE_IMAGE_RGB8; f <= AICORE_IMAGE_BGRA8; ++f) {
        aicore_image_view view{buffers[f - 1], 1, 2, strides[f - 1],
                               static_cast<aicore_image_format>(f)};
        AICORE_CHECK(aicore_instantmesh_preprocess_image_view(
                             &view, 1, out, err, sizeof(err)) == 0);
        if (f == AICORE_IMAGE_GRAY8) {
            AICORE_CHECK(out[0] == 91 && out[1] == 91 && out[2] == 91);
        } else {
            AICORE_CHECK(out[0] == 255);
            const int expected =
                    (f == AICORE_IMAGE_RGBA8 || f == AICORE_IMAGE_BGRA8) ? 126
                                                                         : 0;
            AICORE_CHECK(std::abs(int(out[1]) - expected) <= 1);
            AICORE_CHECK(out[1] == out[2]);
        }
        view.row_stride_bytes = 0;
        AICORE_CHECK(aicore_instantmesh_preprocess_image_view(
                             &view, 1, out, err, sizeof(err)) != 0);
        AICORE_CHECK(std::strstr(err, "stride") != nullptr);
    }
    aicore_image_view bad{rgb, 1, 2, std::numeric_limits<size_t>::max(),
                          AICORE_IMAGE_RGB8};
    AICORE_CHECK(aicore_instantmesh_preprocess_image_view(&bad, 1, out, err,
                                                          sizeof(err)) != 0);
    bad.row_stride_bytes = 4;
    bad.width = std::numeric_limits<int32_t>::max();
    AICORE_CHECK(aicore_instantmesh_preprocess_image_view(&bad, 1, out, err,
                                                          sizeof(err)) != 0);
    bad.width = -1;
    AICORE_CHECK(aicore_instantmesh_preprocess_image_view(&bad, 1, out, err,
                                                          sizeof(err)) != 0);
    AICORE_CHECK(aicore_instantmesh_preprocess_image_view(nullptr, 1, out, err,
                                                          sizeof(err)) != 0);
    bad = {rgb, 1, 2, 4, static_cast<aicore_image_format>(99)};
    AICORE_CHECK(aicore_instantmesh_preprocess_image_view(&bad, 1, out, err,
                                                          sizeof(err)) != 0);
#else
    std::fprintf(
            stderr,
            "Missing unified decoded image-view preprocessing entry point\n");
    AICORE_CHECK(false);
#endif
}

static uint32_t read_u32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 |
           uint32_t(p[3]) << 24;
}

static void test_glb_texture_layout() {
    // Two faces share vertices but seam corners have distinct UVs; expectations
    // are independently hand-specified.
    const float vertices[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    const int faces[] = {0, 1, 2, 0, 2, 3};
    const float uv[] = {0, 0, 1, 0, 1, 1, 0.25f, 0.75f, 0.75f, 1, 0, 1};
    const uint8_t rgba[] = {255, 0, 0,   255, 0,   255, 0,   128,
                            0,   0, 255, 64,  255, 255, 255, 255};
    char err[256] = {};
    int len = 0;
    uint8_t* glb = aicore_instantmesh_bake_glb(vertices, 4, faces, 2, uv, rgba,
                                               2, 2, &len, err, sizeof(err));
    AICORE_CHECK(glb && len > 20);
    if (!glb) return;
    AICORE_CHECK(read_u32(glb + 8) == uint32_t(len));
    const uint32_t json_len = read_u32(glb + 12);
    AICORE_CHECK(json_len % 4 == 0 && json_len + 28 <= uint32_t(len));
    const auto json =
            QJsonDocument::fromJson(
                    QByteArray(reinterpret_cast<char*>(glb + 20), json_len))
                    .object();
    const auto acc = json["accessors"].toArray();
    const auto views = json["bufferViews"].toArray();
    AICORE_CHECK(acc.size() == 3 && views.size() == 4);
    if (acc.size() == 3 && views.size() == 4) {
        AICORE_CHECK(acc[0].toObject()["count"].toInt() == 6);
        AICORE_CHECK(acc[2].toObject()["count"].toInt() == 6);
        const uint8_t* bin = glb + 28 + json_len;
        const auto component = [&](int view, int element) {
            float value;
            std::memcpy(&value,
                        bin + views[view].toObject()["byteOffset"].toInt() +
                                element * 4,
                        4);
            return value;
        };
        const uint8_t* indices =
                bin + views[1].toObject()["byteOffset"].toInt();
        for (int i = 0; i < 6; ++i)
            AICORE_CHECK(read_u32(indices + i * 4) == uint32_t(i));
        AICORE_CHECK(component(0, 9) == 0 && component(0, 10) == 0 &&
                     component(0, 11) == 0);
        AICORE_CHECK(component(2, 0) == 0 && component(2, 1) == 1);
        AICORE_CHECK(component(2, 6) == 0.25f && component(2, 7) == 0.25f);
        const auto image_view = views[3].toObject();
        const QImage png =
                QImage::fromData(bin + image_view["byteOffset"].toInt(),
                                 image_view["byteLength"].toInt(), "PNG");
        AICORE_CHECK(png.size() == QSize(2, 2));
        if (!png.isNull()) {
            AICORE_CHECK(png.pixel(0, 0) == qRgba(255, 0, 0, 255));
            AICORE_CHECK(png.pixel(1, 0) == qRgba(0, 255, 0, 128));
            AICORE_CHECK(png.pixel(0, 1) == qRgba(0, 0, 255, 64));
        }
    }
    aicore_instantmesh_free_buffer(glb);
    glb = aicore_instantmesh_bake_glb(vertices, 4, faces, 2, nullptr, nullptr,
                                      0, 0, &len, err, sizeof(err));
    if (glb) {
        const auto untextured =
                QJsonDocument::fromJson(
                        QByteArray(reinterpret_cast<char*>(glb + 20),
                                   read_u32(glb + 12)))
                        .object();
        const auto pbr = untextured["materials"]
                                 .toArray()[0]
                                 .toObject()["pbrMetallicRoughness"]
                                 .toObject();
        AICORE_CHECK(!pbr.contains("baseColorTexture") &&
                     !untextured.contains("textures"));
        aicore_instantmesh_free_buffer(glb);
    } else
        AICORE_CHECK(false);
}

static void test_glb_invalid_mesh() {
    char err[256] = {};
    int len = 123;
    int faces[] = {0, 1, 3};
    float vertices[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    const auto reject = [&](const float* uv, const uint8_t* tex, int w, int h,
                            const char* field) {
        len = 123;
        auto* glb = aicore_instantmesh_bake_glb(vertices, 3, faces, 1, uv, tex,
                                                w, h, &len, err, sizeof(err));
        AICORE_CHECK(glb == nullptr && len == 0 && std::strstr(err, field));
        aicore_instantmesh_free_buffer(glb);
    };
    reject(nullptr, nullptr, 0, 0, "index");
    faces[2] = -1;
    reject(nullptr, nullptr, 0, 0, "index");
    faces[2] = 2;
    vertices[0] = std::numeric_limits<float>::infinity();
    reject(nullptr, nullptr, 0, 0, "finite");
    vertices[0] = 0;
    float uv[] = {0, 0, 1, 0, 0, 1};
    const std::vector<uint8_t> tex(4097 * 4, 255);
    reject(uv, nullptr, 0, 0, "texture");
    reject(nullptr, tex.data(), 1, 1, "texture");
    reject(uv, tex.data(), 0, 1, "texture");
    reject(uv, tex.data(), 4097, 1, "texture");
    uv[0] = std::numeric_limits<float>::quiet_NaN();
    reject(uv, tex.data(), 1, 1, "finite");
}

static void test_catalog_digest_scope(const char* catalog_binary) {
    // Same-named DINO files come from different conversions; Trellis basename
    // digests must not replace InstantMesh's.
    for (int i = 0; i < aicore_instantmesh_model_count(); ++i) {
        const auto* entry = aicore_instantmesh_model_at(i);
        const auto key = std::string("instantmesh_models/") + entry->filename;
        const char* digest = aicore::AssetDigestForFile(key.c_str());
        AICORE_CHECK(digest && std::strcmp(digest, entry->sha256) == 0);
    }
    AICORE_CHECK(std::strcmp(aicore::AssetDigestForFile("dino_f16.gguf"),
                             "385d8186a38a2328ec740fb2ac1f33f9194d8774efc7ccafd"
                             "4aa2e51cf5f6450") == 0);
    if (!catalog_binary) return;
    QProcess process;
    process.start(QString::fromUtf8(catalog_binary),
                  QStringList{QStringLiteral("--json")});
    AICORE_CHECK(process.waitForFinished(30000) && process.exitCode() == 0);
    int count = 0;
    for (const auto& line : process.readAllStandardOutput().split('\n')) {
        const auto row = QJsonDocument::fromJson(line).object();
        if (row["task"].toString() != QStringLiteral("instantmesh")) continue;
        ++count;
        const QByteArray name =
                row["relative_path"].toString().section('/', -1).toUtf8();
        const auto* entry =
                aicore_instantmesh_model_by_filename(name.constData());
        AICORE_CHECK(entry != nullptr);
        if (entry) {
            AICORE_CHECK(row["sha256"].toString() ==
                         QString::fromUtf8(entry->sha256));
            AICORE_CHECK(row["size_bytes"].toDouble() ==
                         double(entry->size_bytes));
        }
    }
    AICORE_CHECK(count == 14);
}

int main(int argc, char** argv) {
    test_catalog_digest_scope(argc > 1 ? argv[1] : nullptr);
    test_glb_texture_layout();
    test_glb_invalid_mesh();
    test_versioned_load_errors();
    test_generate_parameter_versions();
    test_invalid_numeric_parameters();
    test_decoded_formats_and_stride();
    AICORE_CHECK(aicore_instantmesh_abi_version() >= 1);

    // Null-safe teardown / lifecycle.
    aicore_instantmesh_free(nullptr);
    aicore_instantmesh_options_free(nullptr);
    aicore_instantmesh_free_buffer(nullptr);
    aicore_instantmesh_mesh_free(nullptr);

    AICORE_CHECK(aicore_instantmesh_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_instantmesh_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_instantmesh_last_error(nullptr) != nullptr);
    AICORE_CHECK(aicore_instantmesh_has_multiview(nullptr) == 0);
    AICORE_CHECK(aicore_instantmesh_backend(nullptr) != nullptr);

    // Mesh accessors are null-safe.
    AICORE_CHECK(aicore_instantmesh_mesh_n_verts(nullptr) == 0);
    AICORE_CHECK(aicore_instantmesh_mesh_n_tris(nullptr) == 0);
    AICORE_CHECK(aicore_instantmesh_mesh_verts(nullptr) == nullptr);
    AICORE_CHECK(aicore_instantmesh_mesh_tris(nullptr) == nullptr);
    AICORE_CHECK(aicore_instantmesh_mesh_vcolors(nullptr) == nullptr);
    AICORE_CHECK(aicore_instantmesh_mesh_uvs(nullptr) == nullptr);
    AICORE_CHECK(aicore_instantmesh_mesh_texture(nullptr) == nullptr);
    AICORE_CHECK(aicore_instantmesh_mesh_texture_w(nullptr) == 0);
    AICORE_CHECK(aicore_instantmesh_mesh_texture_h(nullptr) == 0);

    // Options lifecycle: full setter sweep must not crash.
    {
        aicore_instantmesh_options* opts = aicore_instantmesh_options_new();
        AICORE_CHECK(opts != nullptr);
        aicore_instantmesh_options_set_device(opts, "cpu");
        aicore_instantmesh_options_set_threads(opts, 2);
        aicore_instantmesh_options_set_rmbg_gguf(opts, "");
        aicore_instantmesh_options_set_steps(opts, 8);
        aicore_instantmesh_options_set_seed(opts, 7);
        aicore_instantmesh_options_set_guidance(opts, 3.0f);
        aicore_instantmesh_options_set_grid_res(opts, 32);
        aicore_instantmesh_options_set_grid_scale(opts, 2.0f);
        aicore_instantmesh_options_set_texture_res(opts, 256);
        aicore_instantmesh_options_set_bake_texture(opts, 0);
        aicore_instantmesh_options_free(opts);
    }

    // Generate error paths (no models loaded).
    {
        char err[512] = {0};
        AICORE_CHECK(aicore_instantmesh_generate(nullptr, nullptr, 0, nullptr,
                                                 nullptr, nullptr, err,
                                                 (int)sizeof(err)) == nullptr);
        aicore_instantmesh_ctx* ctx = nullptr;
        AICORE_CHECK(aicore_instantmesh_generate(ctx, kPng1x1,
                                                 (int)sizeof(kPng1x1), nullptr,
                                                 nullptr, nullptr, err,
                                                 (int)sizeof(err)) == nullptr);
        // Missing-model load fails cleanly with a non-null last error.
        aicore_instantmesh_options* opts = aicore_instantmesh_options_new();
        AICORE_CHECK(opts != nullptr);
        aicore_instantmesh_options_set_device(opts, "cpu");
#if defined(AICORE_INSTANTMESH_ABI_VERSION)
        aicore_instantmesh_model_paths paths =
                AICORE_INSTANTMESH_MODEL_PATHS_INIT;
#else
        aicore_instantmesh_model_paths paths{};
#endif
        paths.dino_gguf = "/nonexistent/dino_f16.gguf";
        paths.transformer_gguf = "/nonexistent/lrm_f16.gguf";
        paths.synthesizer_gguf = "/nonexistent/synth_f16.gguf";
        aicore_instantmesh_ctx* bad =
                aicore_instantmesh_load_opts(&paths, opts);
        aicore_instantmesh_options_free(opts);
        AICORE_CHECK(bad == nullptr);
    }

    // Image preprocessing only (no models): decode + alpha-over-white +
    // resize into a solid 64 target.
    {
        unsigned char out[64 * 64 * 3];
        std::memset(out, 0, sizeof(out));
        char err[512] = {0};
        const int rc = aicore_instantmesh_preprocess_image_bytes(
                kPng1x1, (int)sizeof(kPng1x1), 64, out, err, (int)sizeof(err));
        AICORE_CHECK(rc == 0);
        bool nonwhite = false;
        for (size_t i = 0; i < sizeof(out); i += 3) {
            if (out[i] != 255 || out[i + 1] != 255) nonwhite = true;
        }
        AICORE_CHECK(nonwhite);  // the red square must survive
    }

    // GLB bake contract: a minimal triangle bakes to a valid GLB container.
    {
        char err[512] = {0};
        int glb_len = 0;
        uint8_t* glb = aicore_instantmesh_bake_glb(kVerts, 3, kTris, 1, nullptr,
                                                   nullptr, 0, 0, &glb_len, err,
                                                   (int)sizeof(err));
        AICORE_CHECK(glb != nullptr && glb_len > 12);
        if (glb) {
            AICORE_CHECK(std::memcmp(glb, "glTF", 4) == 0);
            aicore_instantmesh_free_buffer(glb);
        }
    }

    // Model catalog: 14 published entries, all roles present, stable lookup,
    // every URL on the pinned mirror, RMBG absent (single-owner rule).
    {
        const int n = aicore_instantmesh_model_count();
        AICORE_CHECK(n == 14);
        int roles_dino = 0, roles_transformer = 0, roles_synth = 0,
            roles_cond = 0, roles_unet = 0, roles_vae = 0;
        for (int i = 0; i < n; ++i) {
            const aicore_instantmesh_model_entry* e =
                    aicore_instantmesh_model_at(i);
            AICORE_CHECK(e != nullptr);
            AICORE_CHECK(e->filename && e->filename[0]);
            AICORE_CHECK(e->download_url &&
                         std::strstr(e->download_url,
                                     "huggingface.co/Asher-1/"
                                     "InstantMesh-GUFF") != nullptr);
            AICORE_CHECK(e->sha256 && std::strlen(e->sha256) == 64);
            AICORE_CHECK(e->size_bytes > 0);
            AICORE_CHECK(e->role && e->role[0]);
            const char* r = e->role;
            if (!std::strcmp(r, "dino"))
                ++roles_dino;
            else if (!std::strcmp(r, "transformer"))
                ++roles_transformer;
            else if (!std::strcmp(r, "synthesizer"))
                ++roles_synth;
            else if (!std::strcmp(r, "cond"))
                ++roles_cond;
            else if (!std::strcmp(r, "unet"))
                ++roles_unet;
            else if (!std::strcmp(r, "vae"))
                ++roles_vae;
            else
                AICORE_CHECK(false);
            // by_filename round-trips.
            AICORE_CHECK(aicore_instantmesh_model_by_filename(e->filename) ==
                         e);
        }
        AICORE_CHECK(roles_dino == 3 && roles_transformer == 3 &&
                     roles_synth == 3);
        AICORE_CHECK(roles_cond == 1 && roles_unet == 2 && roles_vae == 2);
        AICORE_CHECK(aicore_instantmesh_model_at(-1) == nullptr);
        AICORE_CHECK(aicore_instantmesh_model_at(n) == nullptr);
        AICORE_CHECK(aicore_instantmesh_model_by_filename("nope.gguf") ==
                     nullptr);
        AICORE_CHECK(aicore_instantmesh_model_by_filename("rmbg_q8.gguf") ==
                     nullptr);
        const char* base = aicore_instantmesh_model_download_base();
        AICORE_CHECK(base && std::strstr(base, "InstantMesh-GUFF") != nullptr);
    }

    // Cache dir + info JSON plumbing.
    {
        char* dir = aicore_instantmesh_model_cache_dir();
        AICORE_CHECK(dir != nullptr && dir[0]);
        aicore_instantmesh_free_buffer(dir);
        char* json = aicore_instantmesh_info_json(nullptr);
        AICORE_CHECK(json != nullptr);
        aicore_instantmesh_free_buffer(json);
    }

    // Timings: no generate yet -> -1.
    {
        aicore_pipeline_timings t{};
        AICORE_CHECK(aicore_instantmesh_last_pipeline_timings(nullptr, &t) ==
                     -1);
    }

    // Progress callback plumbing is accepted (never invoked without models).
    {
        char err[512] = {0};
        aicore_instantmesh_generate_params params{};
        AICORE_CHECK(aicore_instantmesh_generate(nullptr, kPng1x1,
                                                 (int)sizeof(kPng1x1), &params,
                                                 test_progress, nullptr, err,
                                                 (int)sizeof(err)) == nullptr);
    }

    aicore_instantmesh_shutdown();
    aicore_instantmesh_shutdown();

    if (failures == 0) {
        std::printf("test_instantmesh_capi_contract: ALL PASS\n");
        return 0;
    }
    std::printf("test_instantmesh_capi_contract: %d FAILURES\n", failures);
    return 1;
}
