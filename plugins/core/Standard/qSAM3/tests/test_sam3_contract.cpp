// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Contract test for the SAM3 C-API.
//
// Covers:
//   1. ABI version sanity
//   2. Model catalog integrity (count, entries, download base)
//   3. Options lifecycle
//   4. Precision contract (when model + image are available)
//
// The precision-contract test is skipped (exit 77) when the model file or
// reference image is not found (download via the catalog URL first).

#include <aicore/sam3_capi.h>
#include <gtest/gtest.h>

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QStringList>

// ---------------------------------------------------------------------------
// ABI version
// ---------------------------------------------------------------------------

TEST(SAM3Contract, AbiVersion) { EXPECT_GE(aicore_sam3_abi_version(), 1); }

// ---------------------------------------------------------------------------
// Model catalog
// ---------------------------------------------------------------------------

TEST(SAM3Contract, ModelCount) {
    const int n = aicore_sam3_model_count();
    EXPECT_GT(n, 0);
}

TEST(SAM3Contract, ModelEntries) {
    const int n = aicore_sam3_model_count();
    EXPECT_GT(n, 0);

    bool foundSam3 = false;
    bool foundSam21 = false;
    bool foundSam2 = false;

    for (int i = 0; i < n; ++i) {
        const auto* e = aicore_sam3_model_at(i);
        ASSERT_NE(e, nullptr);
        ASSERT_NE(e->filename, nullptr);
        ASSERT_NE(e->download_url, nullptr);
        ASSERT_NE(e->display_name, nullptr);
        ASSERT_NE(e->model_family, nullptr);

        EXPECT_GT(e->size_bytes, 0);
        EXPECT_TRUE(e->size_bytes < 10000000000LL) << e->filename;

        // No f32 in catalog (too large)
        EXPECT_EQ(nullptr, strstr(e->filename, "-f32")) << e->filename;

        // Verify download URLs
        std::string url(e->download_url);
        EXPECT_TRUE(url.find("cloudViewer_downloads") != std::string::npos ||
                    url.find("github.com") != std::string::npos);

        if (strcmp(e->model_family, "sam3") == 0) foundSam3 = true;
        if (strstr(e->model_family, "sam2.1")) foundSam21 = true;
        if (strcmp(e->model_family, "sam2") == 0) foundSam2 = true;
    }

    EXPECT_TRUE(foundSam3) << "No sam3 entries in catalog";
    EXPECT_TRUE(foundSam21) << "No sam2.1 entries in catalog";
    EXPECT_TRUE(foundSam2) << "No sam2 entries in catalog";
}

TEST(SAM3Contract, ModelByFilename) {
    const auto* e = aicore_sam3_model_by_filename("sam3-q4_0.gguf");
    ASSERT_NE(e, nullptr);
    EXPECT_STREQ(e->filename, "sam3-q4_0.gguf");
    EXPECT_EQ(e->visual_only, 0);  // sam3 has text encoder

    const auto* notFound = aicore_sam3_model_by_filename("nonexistent.gguf");
    EXPECT_EQ(notFound, nullptr);
}

TEST(SAM3Contract, ModelDownloadBase) {
    const char* base = aicore_sam3_model_download_base();
    ASSERT_NE(base, nullptr);
    EXPECT_TRUE(strlen(base) > 0);
}

// ---------------------------------------------------------------------------
// Options lifecycle
// ---------------------------------------------------------------------------

TEST(SAM3Contract, OptionsCreateFree) {
    auto* opts = aicore_sam3_options_new();
    ASSERT_NE(opts, nullptr);

    // Setters should not crash on valid opts
    aicore_sam3_options_set_device(opts, "cpu");
    aicore_sam3_options_set_threads(opts, 2);
    aicore_sam3_options_set_encode_img_size(opts, 512);
    aicore_sam3_options_set_score_threshold(opts, 0.3f);
    aicore_sam3_options_set_nms_threshold(opts, 0.2f);
    aicore_sam3_options_set_assoc_iou_threshold(opts, 0.15f);
    aicore_sam3_options_set_hotstart_delay(opts, 10);
    aicore_sam3_options_set_max_keep_alive(opts, 20);
    aicore_sam3_options_set_recondition_every(opts, 8);
    aicore_sam3_options_set_fill_hole_area(opts, 32);

    // Setters should not crash on null (documented no-op)
    aicore_sam3_options_set_device(nullptr, "cpu");
    aicore_sam3_options_set_threads(nullptr, 0);

    aicore_sam3_options_free(opts);
}

// ---------------------------------------------------------------------------
// Context lifecycle (without model — just null-safety tests)
// ---------------------------------------------------------------------------

TEST(SAM3Contract, ContextNullSafety) {
    // Calling introspection on null should not crash
    EXPECT_EQ(aicore_sam3_context_model_type(nullptr), -1);
    EXPECT_EQ(aicore_sam3_context_visual_only(nullptr), 0);
    EXPECT_EQ(aicore_sam3_context_backend_name(nullptr), std::string("none"));
    EXPECT_EQ(aicore_sam3_context_threads(nullptr), 0);
    EXPECT_EQ(aicore_sam3_is_ready(nullptr), 0);
    EXPECT_EQ(aicore_sam3_last_error(nullptr), std::string(""));
    EXPECT_EQ(aicore_sam3_has_encoded_image(nullptr), 0);

    // Free on null is safe
    aicore_sam3_free(nullptr);
    aicore_sam3_free_buffer(nullptr);
    aicore_sam3_seg_result_free(nullptr);
    aicore_sam3_tracker_free(nullptr);
}

// ---------------------------------------------------------------------------
// Warm-up / shutdown
// ---------------------------------------------------------------------------

TEST(SAM3Contract, WarmupShutdown) {
    // Warmup should succeed (returns 0) even for "nonexistent" backend
    // because the function delegates to the backend registry.
    const int ret = aicore_sam3_warmup_backend("cpu");
    EXPECT_EQ(ret, 0);

    // Shutdown is idempotent
    aicore_sam3_shutdown();
    aicore_sam3_shutdown();
}

// ---------------------------------------------------------------------------
// Timings (null-safe)
// ---------------------------------------------------------------------------

TEST(SAM3Contract, TimingsNull) {
    aicore_sam3_timings t{};
    EXPECT_EQ(aicore_sam3_last_timings(nullptr, &t), -1);
}

// ---------------------------------------------------------------------------
// Precision contract (requires model + image, skipped when unavailable)
// ---------------------------------------------------------------------------

class SAM3PrecisionContract : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Look for a test model in standard locations
        const QStringList candidates = {
                "/home/ludahai/develop/code/github/dl/sam3-ggml/models/"
                "sam2.1_hiera_tiny_f16.gguf",
                QDir::homePath() +
                        "/.cache/cloudViewer/models/sam/"
                        "sam2.1_hiera_tiny_f16.gguf",
        };
        s_modelPath.clear();
        for (const auto& p : candidates) {
            if (QFileInfo::exists(p)) {
                s_modelPath = p;
                break;
            }
        }

        s_imagePath =
                "/home/ludahai/develop/code/github/dl/sam3-ggml/data/"
                "test_image.jpg";
    }

    static QString s_modelPath;
    static QString s_imagePath;
};

QString SAM3PrecisionContract::s_modelPath;
QString SAM3PrecisionContract::s_imagePath;

TEST_F(SAM3PrecisionContract, EncodeSegmentPVS) {
    if (s_modelPath.isEmpty() || !QFileInfo::exists(s_modelPath)) {
        GTEST_SKIP() << "Model file not found: " << s_modelPath.toStdString();
    }

    QImage img(s_imagePath);
    if (img.isNull()) {
        GTEST_SKIP() << "Test image not found: " << s_imagePath.toStdString();
    }

    img = img.convertToFormat(QImage::Format_RGB888);

    // Load model
    auto* opts = aicore_sam3_options_new();
    aicore_sam3_options_set_device(opts, "cpu");
    aicore_sam3_options_set_threads(opts, 4);

    aicore_sam3_ctx* ctx =
            aicore_sam3_load_opts(s_modelPath.toUtf8().constData(), opts);
    aicore_sam3_options_free(opts);

    ASSERT_NE(ctx, nullptr) << "Model load failed";
    ASSERT_TRUE(aicore_sam3_is_ready(ctx));

    // Encode
    EXPECT_EQ(aicore_sam3_encode_rgb(
                      ctx, img.constBits(), img.width(), img.height(),
                      static_cast<size_t>(img.bytesPerLine()), 1),
              0);
    EXPECT_TRUE(aicore_sam3_has_encoded_image(ctx));

    // Segment with point prompt
    aicore_sam3_pvs_prompt prompt{};
    aicore_sam3_point pos = {315.0f, 250.0f};
    prompt.pos_points = &pos;
    prompt.n_pos_points = 1;
    prompt.multimask = 0;

    aicore_sam3_seg_result* res = aicore_sam3_segment_pvs_rgb(
            ctx, &prompt, img.constBits(), img.width(), img.height(),
            static_cast<size_t>(img.bytesPerLine()));
    ASSERT_NE(res, nullptr);

    // Verify detection structure
    const int n = aicore_sam3_seg_det_count(res);
    EXPECT_GE(n, 1) << "Expected at least 1 detection";

    if (n > 0) {
        const aicore_sam3_box box = aicore_sam3_seg_det_box_at(res, 0);
        const float score = aicore_sam3_seg_det_score_at(res, 0);
        const float iou = aicore_sam3_seg_det_iou_at(res, 0);
        const aicore_sam3_plane_view mask = aicore_sam3_seg_mask_at(res, 0);

        EXPECT_GE(score, 0.0f);
        EXPECT_LE(score, 1.0f);
        EXPECT_GE(iou, 0.0f);

        // Box should be within image bounds
        EXPECT_GE(box.x0, 0.0f);
        EXPECT_GE(box.y0, 0.0f);
        EXPECT_LE(box.x1, static_cast<float>(img.width()));
        EXPECT_LE(box.y1, static_cast<float>(img.height()));

        // Mask should match image dimensions
        EXPECT_EQ(mask.width, img.width());
        EXPECT_EQ(mask.height, img.height());

        // Verify timings are populated
        aicore_sam3_timings t{};
        EXPECT_EQ(aicore_sam3_last_timings(ctx, &t), 0);
        EXPECT_GT(t.e2e_ms, 0.0);
    }

    aicore_sam3_seg_result_free(res);
    aicore_sam3_free(ctx);

    // Shutdown backends to clean up GPU resources
    aicore_sam3_shutdown();
}