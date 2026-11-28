// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Asset-free tests for the untrusted image-input validator.
#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "tasks/gaussian/gguf_loader.h"
#include "tasks/gaussian/image.h"

class GaussianImageInput : public ::testing::Test {
protected:
    void SetUp() override {
        hp.image_size = H;
        hp.in_channels = C;
        hp.patch_size = 8;
    }

    static constexpr int H = 512, W = 512, C = 3;
    static constexpr size_t per_view = static_cast<size_t>(C) * H * W;
    aicore::gaussian::hparams hp;
    std::vector<float> out;
    std::string error;
};

TEST_F(GaussianImageInput, AcceptsTwoMidGreyViews) {
    std::vector<float> image(2 * per_view, 0.5f);
    ASSERT_TRUE(aicore::gaussian::ingest_images(hp, image.data(), 2, H, W, out,
                                                error))
            << error;
    EXPECT_TRUE(error.empty()) << error;
    EXPECT_EQ(out.size(), 2 * per_view);
}

TEST_F(GaussianImageInput, RejectsWrongResolution) {
    std::vector<float> image(static_cast<size_t>(C) * 256 * 256, 0.0f);
    EXPECT_FALSE(aicore::gaussian::ingest_images(hp, image.data(), 1, 256, 256,
                                                 out, error));
    EXPECT_FALSE(error.empty());
}

TEST_F(GaussianImageInput, RejectsNaN) {
    std::vector<float> image(per_view, 0.0f);
    image[123] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(aicore::gaussian::ingest_images(hp, image.data(), 1, H, W, out,
                                                 error));
}

TEST_F(GaussianImageInput, RejectsInfinity) {
    std::vector<float> image(per_view, 0.0f);
    image[per_view - 1] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(aicore::gaussian::ingest_images(hp, image.data(), 1, H, W, out,
                                                 error));
}

TEST_F(GaussianImageInput, RejectsZeroViews) {
    std::vector<float> image(per_view, 0.0f);
    EXPECT_FALSE(aicore::gaussian::ingest_images(hp, image.data(), 0, H, W, out,
                                                 error));
}

TEST_F(GaussianImageInput, RejectsNullInput) {
    EXPECT_FALSE(
            aicore::gaussian::ingest_images(hp, nullptr, 1, H, W, out, error));
}
