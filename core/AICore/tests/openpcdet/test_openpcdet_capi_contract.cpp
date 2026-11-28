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

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "aicore/openpcdet_capi.h"

namespace {
std::string g_model_path;
std::string g_catalog_dir;
}  // namespace

// Minimal GGUF v3 writer for the malformed-metadata probes: file header plus
// one string KV ("general.architecture") and zero tensors. Returns true when
// the file was fully written.
static bool write_minimal_gguf(const std::string& path, const char* arch) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const char magic[4] = {'G', 'G', 'U', 'F'};
    const uint32_t version = 3;
    const uint64_t tensor_count = 0;
    const uint64_t kv_count = 1;
    bool ok = std::fwrite(magic, 1, sizeof(magic), f) == sizeof(magic) &&
              std::fwrite(&version, sizeof(version), 1, f) == 1 &&
              std::fwrite(&tensor_count, sizeof(tensor_count), 1, f) == 1 &&
              std::fwrite(&kv_count, sizeof(kv_count), 1, f) == 1;
    if (ok) {
        const char key[] = "general.architecture";
        const uint64_t key_len = sizeof(key) - 1;
        const uint32_t string_type = 8;  // GGUF_TYPE_STRING
        const uint64_t val_len = std::strlen(arch);
        ok = std::fwrite(&key_len, sizeof(key_len), 1, f) == 1 &&
             std::fwrite(key, 1, key_len, f) == key_len &&
             std::fwrite(&string_type, sizeof(string_type), 1, f) == 1 &&
             std::fwrite(&val_len, sizeof(val_len), 1, f) == 1 &&
             std::fwrite(arch, 1, val_len, f) == val_len;
    }
    std::fclose(f);
    return ok;
}

// Fixed-content garbage probe (not a GGUF at all).
static bool write_garbage_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const char payload[64] = {0};
    const bool ok =
            std::fwrite(payload, 1, sizeof(payload), f) == sizeof(payload);
    std::fclose(f);
    return ok;
}

// Initializes the caller-side model_info envelope the way every consumer
// must before calling aicore_openpcdet_read_model_info.
static aicore_openpcdet_model_info make_model_info() {
    aicore_openpcdet_model_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = AICORE_OPENPCDET_MODEL_INFO_ABI_VERSION;
    return info;
}

TEST(OpenPcdetCApiContract, ExplicitModelPathReportsBackendErrors) {
    if (g_model_path.empty()) GTEST_SKIP() << "No model path argument supplied";

    auto* unavailable = aicore_openpcdet_options_new();
    ASSERT_NE(unavailable, nullptr);
    aicore_openpcdet_options_set_device(unavailable, "cuda:999999");
    auto* ctx = aicore_openpcdet_load_opts(g_model_path.c_str(), unavailable);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(aicore_openpcdet_is_ready(ctx), 0);
    const char* error = aicore_openpcdet_last_error(ctx);
    EXPECT_TRUE(error && std::strstr(error, "cuda:999999"));
    aicore_openpcdet_free(ctx);
    aicore_openpcdet_options_free(unavailable);

    // An explicit unknown device must fail with a non-empty error naming the
    // request, reproducibly even on CPU-only hosts.
    auto* unknown_dev = aicore_openpcdet_options_new();
    ASSERT_NE(unknown_dev, nullptr);
    aicore_openpcdet_options_set_device(unknown_dev, "nosuchdevice");
    auto* unknown_ctx =
            aicore_openpcdet_load_opts(g_model_path.c_str(), unknown_dev);
    ASSERT_NE(unknown_ctx, nullptr);
    EXPECT_EQ(aicore_openpcdet_is_ready(unknown_ctx), 0);
    const char* unknown_error = aicore_openpcdet_last_error(unknown_ctx);
    EXPECT_TRUE(unknown_error && unknown_error[0] != '\0');
    EXPECT_NE(std::strstr(unknown_error, "nosuchdevice"), nullptr);
    aicore_openpcdet_free(unknown_ctx);
    aicore_openpcdet_options_free(unknown_dev);
}

TEST(OpenPcdetCApiContract, CachedCatalogMetadataMatchesCatalogAndDefaults) {
    if (g_catalog_dir.empty())
        GTEST_SKIP() << "No --catalog cache directory supplied";
    // Explicit full-coverage mode: read the metadata of EVERY published
    // catalog GGUF found under g_catalog_dir.c_str() through both metadata
    // APIs and cross-check them. Missing assets are failures here, not
    // skips. Nothing in this mode loads tensor data or initializes a
    // backend.
    const std::string cache_dir = g_catalog_dir.c_str();
    int present = 0;
    int scene_models = 0;
    for (int i = 0; i < aicore_openpcdet_model_count(); ++i) {
        const aicore_openpcdet_model_entry* e = aicore_openpcdet_model_at(i);
        ASSERT_TRUE(e != nullptr && e->filename != nullptr);
        const std::string path = cache_dir + "/" + e->filename;
        std::FILE* probe = std::fopen(path.c_str(), "rb");
        if (!probe) {
            std::fprintf(stderr, "FAIL catalog asset missing: %s\n",
                         e->filename);
            ADD_FAILURE();
            continue;
        }
        std::fclose(probe);
        ++present;
        aicore_openpcdet_model_info info = make_model_info();
        char* err = nullptr;
        if (aicore_openpcdet_read_model_info(path.c_str(), &info, &err) != 0) {
            std::fprintf(stderr, "FAIL model_info(%s): %s\n", e->filename,
                         err ? err : "no error text");
            aicore_openpcdet_free_buffer(err);
            ADD_FAILURE();
            continue;
        }
        EXPECT_EQ(info.struct_size, sizeof(info));
        EXPECT_GE(info.num_point_features, 3);
        EXPECT_GT(info.num_class, 0);
        EXPECT_GT(info.max_boxes, 0);
        EXPECT_NE(info.arch[0], '\0');
        // The scene expectation follows the catalog pairing rule — the
        // same filenames the plugin's inputKindForFilename treats as
        // multi-modal (CaDDN / BEVFusion / focal multimodal VoxelRCNN,
        // the last declared via focal.multimodal in the GGUF rather
        // than a dedicated architecture string).
        const std::string fname = e->filename;
        const bool scene_expected =
                fname.rfind("kitti_caddn.", 0) == 0 ||
                fname.rfind("nuscenes_bevfusion.", 0) == 0 ||
                fname.rfind("kitti_voxelrcnn_focal_multimodal.", 0) == 0;
        EXPECT_EQ(info.needs_scene, (scene_expected ? 1 : 0));
        if (info.needs_scene) ++scene_models;
        float score = -1.0f;
        float nms = -1.0f;
        int32_t boxes = -1;
        EXPECT_EQ(aicore_openpcdet_model_defaults(path.c_str(), &score, &nms,
                                                  &boxes),
                  0);
        // model_defaults returns the per-checkpoint VISUALIZATION
        // defaults (score + NMS), not the GGUF evaluation thresholds
        // (read_model_info above keeps those). max boxes stays GGUF.
        struct VisExpect {
            const char* stem;
            float score;
            float nms;
        };
        static const VisExpect kVisExpect[] = {
                {"kitti_pointpillar", 0.10f, 0.01f},
                {"second_7862", 0.10f, 0.01f},
                {"kitti_caddn", 0.10f, 0.01f},
                {"kitti_second_iou", 0.10f, 0.01f},
                {"kitti_pvrcnn", 0.10f, 0.10f},
                {"kitti_parta2_anchor", 0.10f, 0.10f},
                {"kitti_pointrcnn", 0.10f, 0.10f},
                {"kitti_pointrcnn_iou", 0.10f, 0.10f},
                {"kitti_voxelrcnn_car", 0.30f, 0.10f},
                {"kitti_voxelrcnn_focal_multimodal", 0.30f, 0.10f},
                {"nuscenes_pointpillar_multihead", 0.20f, 0.15f},
                {"nuscenes_second_multihead", 0.20f, 0.15f},
                {"nuscenes_dynpillar_centerpoint", 0.10f, 0.10f},
                {"nuscenes_centerpoint_voxel0075", 0.10f, 0.10f},
                {"nuscenes_voxelnext", 0.20f, 0.15f},
                {"nuscenes_voxelnext_doubleflip", 0.20f, 0.15f},
                {"nuscenes_transfusion_lidar", 0.20f, 0.10f},
                {"nuscenes_bevfusion", 0.20f, 0.10f},
        };
        const std::string stem = fname.substr(0, fname.find('.'));
        float exp_score = 0.3f;
        float exp_nms = info.nms_thresh;
        for (const auto& v : kVisExpect) {
            if (stem == v.stem) {
                exp_score = v.score;
                exp_nms = v.nms;
                break;
            }
        }
        EXPECT_EQ(score, exp_score);
        EXPECT_EQ(nms, exp_nms);
        EXPECT_EQ(boxes, info.max_boxes);
    }
    EXPECT_EQ(present, aicore_openpcdet_model_count());
    EXPECT_GT(scene_models, 0);
}

TEST(OpenPcdetCApiContract, ModelFreeAbiCatalogAndNullContracts) {
    EXPECT_GE(aicore_openpcdet_abi_version(), 1);
    // Null-safe teardown / lifecycle.
    aicore_openpcdet_free(nullptr);
    aicore_openpcdet_options_free(nullptr);
    aicore_openpcdet_free_buffer(nullptr);
    aicore_openpcdet_shutdown();  // idempotent, no contexts alive

    EXPECT_EQ(aicore_openpcdet_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_openpcdet_is_ready(nullptr), 0);
    EXPECT_NE(aicore_openpcdet_last_error(nullptr), nullptr);
    EXPECT_EQ(aicore_openpcdet_geometry_of(nullptr, nullptr), -1);
    EXPECT_EQ(aicore_openpcdet_detection_count(nullptr), -1);
    EXPECT_EQ(aicore_openpcdet_detection_at(nullptr, 0, nullptr), -1);
    EXPECT_EQ(aicore_openpcdet_last_pipeline_timings(nullptr, nullptr), -1);
    EXPECT_EQ(aicore_openpcdet_detect_points(nullptr, nullptr, 0, 0), -1);
    EXPECT_EQ(aicore_openpcdet_info_json(nullptr), nullptr);
    EXPECT_NE(aicore_openpcdet_context_device(nullptr), nullptr);
    EXPECT_EQ(aicore_openpcdet_context_threads(nullptr), 0);

    // Options lifecycle: every setter must be a no-op on NULL (no crash).
    aicore_openpcdet_options_set_device(nullptr, "cpu");
    aicore_openpcdet_options_set_threads(nullptr, 4);
    aicore_openpcdet_options_set_score_thresh(nullptr, 0.5f);
    aicore_openpcdet_options_set_nms_thresh(nullptr, 0.5f);
    aicore_openpcdet_options_set_max_boxes(nullptr, 10);
    aicore_openpcdet_options_set_dump_taps_dir(nullptr, "/tmp");

    aicore_openpcdet_options* opts = aicore_openpcdet_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_openpcdet_options_set_device(opts, "cpu");
    aicore_openpcdet_options_set_threads(opts, 2);
    aicore_openpcdet_options_set_score_thresh(opts, 0.3f);
    aicore_openpcdet_options_set_nms_thresh(opts, 0.4f);
    aicore_openpcdet_options_set_max_boxes(opts, 33);
    aicore_openpcdet_options_set_dump_taps_dir(opts, nullptr);  // disable
    // NaN threshold overrides must be ignored (keep GGUF defaults).
    aicore_openpcdet_options_set_score_thresh(
            opts, std::numeric_limits<float>::quiet_NaN());
    aicore_openpcdet_options_set_nms_thresh(
            opts, std::numeric_limits<float>::quiet_NaN());
    aicore_openpcdet_options_set_max_boxes(opts, 0);
    aicore_openpcdet_options_free(opts);

    // Failed loads produce a non-ready context with an error string.
    aicore_openpcdet_options* bad = aicore_openpcdet_options_new();
    aicore_openpcdet_options_set_device(bad, "cpu");
    aicore_openpcdet_ctx* missing =
            aicore_openpcdet_load_opts("no_such_model.gguf", bad);
    ASSERT_NE(missing, nullptr);  // non-NULL handle carrying the error
    EXPECT_EQ(aicore_openpcdet_is_ready(missing), 0);
    EXPECT_TRUE(aicore_openpcdet_last_error(missing) != nullptr &&
                aicore_openpcdet_last_error(missing)[0] != '\0');
    EXPECT_EQ(aicore_openpcdet_detect_points(missing, nullptr, 0, 4), -1);
    EXPECT_EQ(aicore_openpcdet_geometry_of(missing, nullptr), -1);
    char* no_info = aicore_openpcdet_info_json(missing);
    EXPECT_EQ(no_info, nullptr);
    aicore_openpcdet_free(missing);
    aicore_openpcdet_options_free(bad);

    // Invalid detect arguments on a NULL context stay non-positive.
    float dummy[4] = {0.f, 0.f, 0.f, 0.f};
    EXPECT_EQ(aicore_openpcdet_detect_points(nullptr, dummy, 1, 4), -1);

    // Catalog contract: base URL, count, entries, default index, lookups.
    EXPECT_EQ(aicore_openpcdet_model_count(), 54);
    EXPECT_TRUE(aicore_openpcdet_model_default_index() >= 0 &&
                aicore_openpcdet_model_default_index() <
                        aicore_openpcdet_model_count());
    const char* base = aicore_openpcdet_model_download_base();
    EXPECT_TRUE(base != nullptr &&
                std::strstr(base, "cloudViewer_downloads") != nullptr &&
                std::strstr(base, "OpenPCDet-GGUF") != nullptr);
    for (int i = 0; i < aicore_openpcdet_model_count(); ++i) {
        const aicore_openpcdet_model_entry* e = aicore_openpcdet_model_at(i);
        ASSERT_NE(e, nullptr);
        EXPECT_TRUE(e->filename != nullptr && e->filename[0] != '\0');
        EXPECT_TRUE(e->download_url != nullptr &&
                    std::strncmp(e->download_url, base, std::strlen(base)) ==
                            0);
        EXPECT_NE(std::strstr(e->download_url, e->filename), nullptr);
        EXPECT_NE(e->display_name, nullptr);
        EXPECT_NE(e->quant_note, nullptr);
        EXPECT_NE(e->family_note, nullptr);
        EXPECT_NE(e->license_note, nullptr);
        const std::string name = e->filename;
        EXPECT_TRUE(name.size() > 5 && name.substr(name.size() - 5) == ".gguf");
    }
    EXPECT_EQ(aicore_openpcdet_model_at(-1), nullptr);
    EXPECT_EQ(aicore_openpcdet_model_at(aicore_openpcdet_model_count()),
              nullptr);
    EXPECT_EQ(aicore_openpcdet_model_by_filename(nullptr), nullptr);
    EXPECT_EQ(aicore_openpcdet_model_by_filename("nope.gguf"), nullptr);
    const aicore_openpcdet_model_entry* known =
            aicore_openpcdet_model_by_filename("kitti_pointpillar.f16.gguf");
    EXPECT_TRUE(known != nullptr &&
                std::strstr(known->family_note, "PointPillar") != nullptr);
    known = aicore_openpcdet_model_by_filename("second_7862.f32.gguf");
    EXPECT_TRUE(known != nullptr &&
                std::strstr(known->family_note, "SECOND") != nullptr);

    // Declared post-processing defaults: null/missing paths must fail
    // cleanly (the positive path runs against the real downloaded GGUF in
    // bench_openpcdet_perf / the validation matrix, which own the assets).
    EXPECT_EQ(
            aicore_openpcdet_model_defaults(nullptr, nullptr, nullptr, nullptr),
            -1);
    EXPECT_EQ(aicore_openpcdet_model_defaults("", nullptr, nullptr, nullptr),
              -1);
    EXPECT_EQ(aicore_openpcdet_model_defaults("no_such_model.gguf", nullptr,
                                              nullptr, nullptr),
              -1);

    // Versioned metadata reader guards (asset-free). A failed read must
    // leave the caller's envelope untouched and produce a freeable
    // diagnostic; declared zero thresholds flow through unchanged (the
    // positive path is the catalog mode above, which compares both APIs).
    {
        aicore_openpcdet_model_info info = make_model_info();
        char* err = nullptr;
        EXPECT_EQ(aicore_openpcdet_read_model_info(nullptr, &info, &err), -1);
        EXPECT_TRUE(err != nullptr && err[0] != '\0');
        aicore_openpcdet_free_buffer(err);
        EXPECT_EQ(aicore_openpcdet_read_model_info("", &info, &err), -1);
        aicore_openpcdet_free_buffer(err);
        EXPECT_EQ(aicore_openpcdet_read_model_info("x.gguf", nullptr, &err),
                  -1);
        EXPECT_NE(err, nullptr);
        aicore_openpcdet_free_buffer(err);
        // The diagnostic output is optional.
        EXPECT_EQ(aicore_openpcdet_read_model_info("", &info, nullptr), -1);
        // A failed read must not touch the caller's envelope.
        aicore_openpcdet_model_info sentinel = make_model_info();
        sentinel.score_thresh = 42.0f;
        const aicore_openpcdet_model_info before = sentinel;
        EXPECT_EQ(aicore_openpcdet_read_model_info("no_such_model.gguf",
                                                   &sentinel, &err),
                  -1);
        aicore_openpcdet_free_buffer(err);
        EXPECT_EQ(std::memcmp(&sentinel, &before, sizeof(sentinel)), 0);
        // Wrong struct_size / abi_version envelopes are rejected outright.
        aicore_openpcdet_model_info bad = make_model_info();
        bad.struct_size = sizeof(bad) + 1;
        EXPECT_EQ(aicore_openpcdet_read_model_info("x.gguf", &bad, &err), -1);
        aicore_openpcdet_free_buffer(err);
        bad = make_model_info();
        bad.abi_version = AICORE_OPENPCDET_MODEL_INFO_ABI_VERSION + 1;
        EXPECT_EQ(aicore_openpcdet_read_model_info("x.gguf", &bad, &err), -1);
        aicore_openpcdet_free_buffer(err);
    }
    // Malformed files must fail with an error, never a crash or gguf
    // assert: a garbage file, and a well-formed GGUF declaring a foreign
    // arch.
    {
        const std::string garbage = "test_openpcdet_model_info_garbage.bin";
        const std::string foreign = "test_openpcdet_model_info_foreign.gguf";
        EXPECT_TRUE(write_garbage_file(garbage));
        EXPECT_TRUE(write_minimal_gguf(foreign, "llama"));
        aicore_openpcdet_model_info info = make_model_info();
        char* err = nullptr;
        EXPECT_EQ(
                aicore_openpcdet_read_model_info(garbage.c_str(), &info, &err),
                -1);
        EXPECT_TRUE(err != nullptr && err[0] != '\0');
        aicore_openpcdet_free_buffer(err);
        EXPECT_EQ(
                aicore_openpcdet_read_model_info(foreign.c_str(), &info, &err),
                -1);
        EXPECT_TRUE(err != nullptr && std::strstr(err, "llama") != nullptr);
        aicore_openpcdet_free_buffer(err);
        std::remove(garbage.c_str());
        std::remove(foreign.c_str());
    }

    // Scene ABI contract: version, null/invalid guards and error paths
    // (asset-free; multi-modal inference is covered by
    // bench_openpcdet_perf).
    EXPECT_GE(AICORE_OPENPCDET_SCENE_ABI_VERSION, 1);
    EXPECT_EQ(aicore_openpcdet_detect_scene(nullptr, nullptr), -1);
    aicore_openpcdet_scene scene{};
    scene.abi_version = AICORE_OPENPCDET_SCENE_ABI_VERSION;
    EXPECT_EQ(aicore_openpcdet_detect_scene(nullptr, &scene), -1);
    aicore_openpcdet_free_scene(nullptr);  // NULL-safe
    aicore_openpcdet_free_scene(&scene);   // empty scene is safe to release
    char* scene_error = nullptr;
    EXPECT_EQ(aicore_openpcdet_load_scene_dir("no_such_scene_dir", &scene,
                                              &scene_error),
              -1);
    EXPECT_TRUE(scene_error != nullptr && scene_error[0] != '\0');
    aicore_openpcdet_free_buffer(scene_error);
    // A stale ABI version must be rejected outright.
    aicore_openpcdet_options* stale_opts = aicore_openpcdet_options_new();
    aicore_openpcdet_options_set_device(stale_opts, "cpu");
    aicore_openpcdet_ctx* stale_ctx =
            aicore_openpcdet_load_opts("no_such_model.gguf", stale_opts);
    aicore_openpcdet_options_free(stale_opts);
    EXPECT_NE(stale_ctx, nullptr);
    aicore_openpcdet_scene stale{};
    stale.abi_version = AICORE_OPENPCDET_SCENE_ABI_VERSION + 1;
    EXPECT_EQ(aicore_openpcdet_detect_scene(stale_ctx, &stale), -1);
    aicore_openpcdet_free(stale_ctx);
}

int main(int argc, char** argv) {
    std::vector<char*> gtest_args;
    gtest_args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--catalog") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--catalog requires a cache directory\n");
                return 2;
            }
            g_catalog_dir = argv[++i];
        } else if (argv[i][0] != '-') {
            if (!g_model_path.empty()) {
                std::fprintf(stderr, "only one model path may be supplied\n");
                return 2;
            }
            g_model_path = argv[i];
        } else {
            gtest_args.push_back(argv[i]);
        }
    }
    int gtest_argc = static_cast<int>(gtest_args.size());
    gtest_args.push_back(nullptr);
    ::testing::InitGoogleTest(&gtest_argc, gtest_args.data());
    return RUN_ALL_TESTS();
}
