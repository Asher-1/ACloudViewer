// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

#include "aicore/depth_capi.h"

namespace {
using Context = std::unique_ptr<aicore_depth_ctx, decltype(&aicore_depth_free)>;

const char* Asset(const char* variable) {
    const char* value = std::getenv(variable);
    return value && *value ? value : nullptr;
}

Context Load(const char* model, const char* metric = nullptr) {
    std::unique_ptr<aicore_depth_options, decltype(&aicore_depth_options_free)>
            options(aicore_depth_options_new(), aicore_depth_options_free);
    if (!options) return Context(nullptr, aicore_depth_free);
    aicore_depth_options_set_threads(options.get(), 1);
    aicore_depth_options_set_device(options.get(), "cpu");
    return Context(
            metric ? aicore_depth_load_nested_opts(model, metric, options.get())
                   : aicore_depth_load_opts(model, options.get()),
            aicore_depth_free);
}

void ExpectFinite(const float* values, size_t count) {
    ASSERT_NE(values, nullptr);
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(std::isfinite(values[i])) << "index=" << i;
    }
}

void ExpectDense(aicore_depth_ctx* ctx,
                 const char* image,
                 bool metric,
                 bool confidence,
                 bool sky,
                 bool pose) {
    aicore_depth_dense_result dense{};
    // Keep ownership valid on every fatal GoogleTest assertion path.
    std::unique_ptr<aicore_depth_dense_result,
                    decltype(&aicore_depth_dense_result_free)>
            cleanup(&dense, aicore_depth_dense_result_free);
    ASSERT_EQ(aicore_depth_depth_dense(ctx, image, &dense), 0)
            << aicore_depth_last_error(ctx);
    ASSERT_GT(dense.height, 0);
    ASSERT_GT(dense.width, 0);
    const size_t pixels = static_cast<size_t>(dense.height) * dense.width;
    ExpectFinite(dense.depth, pixels);
    EXPECT_EQ(dense.is_metric, metric ? 1 : 0);
    EXPECT_EQ(dense.conf != nullptr, confidence);
    EXPECT_EQ(dense.sky != nullptr, sky);
    if (confidence) ExpectFinite(dense.conf, pixels);
    if (sky) ExpectFinite(dense.sky, pixels);
    if (pose) {
        EXPECT_GT(dense.intr[0], 0.f);
        EXPECT_GT(dense.intr[4], 0.f);
        ExpectFinite(dense.intr, 9);
        ExpectFinite(dense.ext, 12);
    }
}

void ExpectPoints(aicore_depth_ctx* ctx, const char* image, bool supported) {
    int count = 0;
    float* xyz = nullptr;
    unsigned char* rgb = nullptr;
    const int status = aicore_depth_points(ctx, image, 0.f, &count, &xyz, &rgb);
    std::unique_ptr<float, decltype(&aicore_depth_free_buffer)> points(
            xyz, aicore_depth_free_buffer);
    std::unique_ptr<unsigned char, decltype(&aicore_depth_free_buffer)> colors(
            rgb, aicore_depth_free_buffer);
    if (supported) {
        ASSERT_EQ(status, 0) << aicore_depth_last_error(ctx);
        ASSERT_GT(count, 0);
        EXPECT_NE(rgb, nullptr);
        ExpectFinite(xyz, static_cast<size_t>(count) * 3);
    } else {
        EXPECT_EQ(status, -1);
        EXPECT_EQ(xyz, nullptr);
        EXPECT_EQ(rgb, nullptr);
    }
}
}  // namespace

TEST(DepthModelContract, DualDptDenseAndPoints) {
    const char* model = Asset("AICORE_TEST_DEPTH_GGUF");
    const char* image = Asset("AICORE_TEST_DEPTH_IMAGE");
    if (!model || !image) GTEST_SKIP() << "DualDPT model/image not supplied";
    auto ctx = Load(model);
    ASSERT_NE(ctx, nullptr);
    ExpectDense(ctx.get(), image, false, true, false, true);
    ExpectPoints(ctx.get(), image, true);
}

TEST(DepthModelContract, MonoDenseAndUnsupportedPoints) {
    const char* model = Asset("AICORE_TEST_DEPTH_GGUF_MONO");
    const char* image = Asset("AICORE_TEST_DEPTH_IMAGE");
    if (!model || !image) GTEST_SKIP() << "Mono model/image not supplied";
    auto ctx = Load(model);
    ASSERT_NE(ctx, nullptr);
    ExpectDense(ctx.get(), image, true, false, true, false);
    ExpectPoints(ctx.get(), image, false);
}

TEST(DepthModelContract, NestedMetricDenseAndPose) {
    const char* model = Asset("AICORE_TEST_DEPTH_GGUF");
    const char* metric = Asset("AICORE_TEST_DEPTH_GGUF_METRIC");
    const char* image = Asset("AICORE_TEST_DEPTH_IMAGE");
    if (!model || !metric || !image)
        GTEST_SKIP() << "Nested model/image not supplied";
    auto ctx = Load(model, metric);
    ASSERT_NE(ctx, nullptr);
    ExpectDense(ctx.get(), image, true, false, false, true);
}

TEST(DepthModelContract, Da2RelativeDepthAndUnsupportedPose) {
    const char* model = Asset("AICORE_TEST_DEPTH_GGUF_DA2");
    const char* image = Asset("AICORE_TEST_DEPTH_IMAGE");
    if (!model || !image) GTEST_SKIP() << "DA2 model/image not supplied";
    auto ctx = Load(model);
    ASSERT_NE(ctx, nullptr);
    ExpectDense(ctx.get(), image, false, false, false, false);
    float ext[12]{}, intr[9]{};
    EXPECT_EQ(aicore_depth_pose_path(ctx.get(), image, ext, intr), -1);
}

TEST(DepthModelContract, GlbAndColmapExport) {
    const char* model = Asset("AICORE_TEST_DEPTH_GGUF");
    const char* image = Asset("AICORE_TEST_DEPTH_IMAGE");
    if (!model || !image)
        GTEST_SKIP() << "Pose-capable model/image not supplied";
    auto ctx = Load(model);
    ASSERT_NE(ctx, nullptr);
    // A fresh directory makes stale exports unable to satisfy these checks.
    const auto run_id =
            std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output = std::filesystem::current_path() /
                        ("depth-model-export-" + std::to_string(run_id));
    ASSERT_TRUE(std::filesystem::create_directory(output));
    const auto glb = output / "depth.glb";
    const auto colmap = output / "colmap";
    ASSERT_EQ(aicore_depth_export_glb(ctx.get(), image, glb.string().c_str()),
              0);
    EXPECT_GT(std::filesystem::file_size(glb), 0u);
    ASSERT_EQ(aicore_depth_export_colmap(ctx.get(), image,
                                         colmap.string().c_str(), 1),
              0);
    ASSERT_TRUE(std::filesystem::is_directory(colmap));
    for (const char* name : {"cameras.bin", "images.bin", "points3D.bin"}) {
        SCOPED_TRACE(name);
        const auto file = colmap / name;
        ASSERT_TRUE(std::filesystem::is_regular_file(file));
        ASSERT_GT(std::filesystem::file_size(file), 8u);
        std::ifstream input(file, std::ios::binary);
        unsigned char bytes[8]{};
        ASSERT_TRUE(input.read(reinterpret_cast<char*>(bytes), sizeof(bytes)));
        uint64_t count = 0;
        for (unsigned i = 0; i < 8; ++i) count |= uint64_t(bytes[i]) << (8 * i);
        if (std::string(name) == "points3D.bin") {
            EXPECT_GT(count, 0u);
            // One 59-byte point record, including its single observation track.
            EXPECT_EQ(std::filesystem::file_size(file), 8u + 59u * count);
        } else {
            EXPECT_EQ(count, 1u);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int status = RUN_ALL_TESTS();
    const auto* tests = ::testing::UnitTest::GetInstance();
    return status == 0 && tests->successful_test_count() == 0 &&
                           tests->skipped_test_count() > 0
                   ? 77
                   : status;
}
