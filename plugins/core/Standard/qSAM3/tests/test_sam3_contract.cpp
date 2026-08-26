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
#include <QDirIterator>
#include <QFileInfo>
#include <QImage>
#include <QStringList>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

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
    const std::string base = aicore_sam3_model_download_base();
    std::vector<std::string> urls;
    urls.reserve(static_cast<size_t>(n));

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
        EXPECT_EQ(url, base + e->filename);
        urls.push_back(url);
        EXPECT_TRUE(url.find("cloudViewer_downloads") != std::string::npos ||
                    url.find("github.com") != std::string::npos);

        if (strcmp(e->model_family, "sam3") == 0) foundSam3 = true;
        if (strstr(e->model_family, "sam2.1")) foundSam21 = true;
        if (strcmp(e->model_family, "sam2") == 0) foundSam2 = true;
    }

    EXPECT_TRUE(foundSam3) << "No sam3 entries in catalog";
    EXPECT_TRUE(foundSam21) << "No sam2.1 entries in catalog";
    EXPECT_TRUE(foundSam2) << "No sam2 entries in catalog";

    // Every returned pointer is documented as process-lifetime stable.
    // Revisit after the catalog has fully initialized to catch URL storage
    // reallocation bugs.
    for (int i = 0; i < n; ++i) {
        const auto* e = aicore_sam3_model_at(i);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(urls[static_cast<size_t>(i)], e->download_url);
    }
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

TEST(SAM3Contract, CachedModelSizesMatchCatalog) {
    char* cacheDirRaw = aicore_sam3_model_cache_dir();
    ASSERT_NE(cacheDirRaw, nullptr);
    const QDir cacheDir(QString::fromUtf8(cacheDirRaw));
    aicore_sam3_free_buffer(cacheDirRaw);

    for (int i = 0; i < aicore_sam3_model_count(); ++i) {
        const auto* entry = aicore_sam3_model_at(i);
        ASSERT_NE(entry, nullptr);
        const QFileInfo file(cacheDir.filePath(QString::fromUtf8(entry->filename)));
        if (!file.exists()) continue;
        EXPECT_EQ(file.size(), entry->size_bytes) << entry->filename;
    }
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
    EXPECT_EQ(aicore_sam3_set_score_threshold(nullptr, 0.5f), -1);
    EXPECT_EQ(aicore_sam3_has_encoded_image(nullptr), 0);

    // Free on null is safe
    aicore_sam3_free(nullptr);
    aicore_sam3_free_buffer(nullptr);
    aicore_sam3_seg_result_free(nullptr);
    aicore_sam3_tracker_free(nullptr);
}

TEST(SAM3Contract, LoadErrorSurvivesNullContext) {
    EXPECT_EQ(aicore_sam3_load_opts(nullptr, nullptr), nullptr);
    EXPECT_STREQ(aicore_sam3_last_load_error(), "empty model path");
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
// Model cache directory
// ---------------------------------------------------------------------------

TEST(SAM3Contract, ModelCacheDir) {
    char* dir = aicore_sam3_model_cache_dir();
    ASSERT_NE(dir, nullptr);
    EXPECT_GT(strlen(dir), 0u);
    // Must point inside the shared AICore data root (sam3_models).
    EXPECT_NE(strstr(dir, "sam3_models"), nullptr);
    aicore_sam3_free_buffer(dir);
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
        // Look for a test model in the standard AICore model cache
        // ({CLOUDVIEWER_DATA_ROOT|~}/cloudViewer_data/extract/sam3_models).
        char* cacheDir = aicore_sam3_model_cache_dir();
        const QString modelDir = QString::fromUtf8(cacheDir);
        aicore_sam3_free_buffer(cacheDir);
        QStringList candidates;
        const QString requested =
                QString::fromLocal8Bit(qgetenv("AICORE_SAM3_TEST_MODEL"));
        if (!requested.isEmpty()) candidates.append(requested);
        candidates.append({QStringLiteral("sam3-f16.gguf"),
                           QStringLiteral("sam3-visual-q4_0.gguf"),
                           QStringLiteral("sam2.1_hiera_tiny_q4_0.gguf"),
                           QStringLiteral("sam2_hiera_tiny_q4_0.gguf")});
        for (const QString& filename : candidates) {
            const QString path = QFileInfo(filename).isAbsolute()
                                         ? filename
                                         : modelDir + QLatin1Char('/') + filename;
            if (QFileInfo::exists(path)) {
                s_modelPath = path;
                break;
            }
        }

        // The reference image comes from the shared SAM3 test-data bundle
        // (extract/sam_test_data/images under the data root).
        QString dataRoot =
                QString::fromLocal8Bit(qgetenv("CLOUDVIEWER_DATA_ROOT"));
        if (dataRoot.isEmpty()) {
            dataRoot = QDir::homePath() + QLatin1String("/cloudViewer_data");
        }
        const QString imageDir =
                dataRoot + QLatin1String("/extract/sam_test_data/images");
        const QString reference =
                imageDir + QLatin1String("/test_image_market_03.jpg");
        const QString requestedImage = QString::fromLocal8Bit(
                qgetenv("AICORE_SAM3_TEST_IMAGE"));
        if (!requestedImage.isEmpty() && QFileInfo::exists(requestedImage)) {
            s_imagePath = requestedImage;
        } else if (QFileInfo::exists(reference)) {
            s_imagePath = reference;
        }
        const QString requestedSecondImage = QString::fromLocal8Bit(
                qgetenv("AICORE_SAM3_TEST_SECOND_IMAGE"));
        if (!requestedSecondImage.isEmpty() &&
            QFileInfo::exists(requestedSecondImage)) {
            s_secondImagePath = requestedSecondImage;
        }
        s_textPrompt = QString::fromLocal8Bit(
                               qgetenv("AICORE_SAM3_TEST_TEXT_PROMPT"))
                               .trimmed();
        if (s_textPrompt.isEmpty()) s_textPrompt = QStringLiteral("pepper");
    }

    static QString s_modelPath;
    static QString s_imagePath;
    static QString s_secondImagePath;
    static QString s_textPrompt;
};

QString SAM3PrecisionContract::s_modelPath;
QString SAM3PrecisionContract::s_imagePath;
QString SAM3PrecisionContract::s_secondImagePath;
QString SAM3PrecisionContract::s_textPrompt;

TEST_F(SAM3PrecisionContract, RepeatedContextRelease) {
    if (s_modelPath.isEmpty() || !QFileInfo::exists(s_modelPath)) {
        GTEST_SKIP() << "Model file not found: " << s_modelPath.toStdString();
    }

    QByteArray device = qgetenv("AICORE_SAM3_TEST_DEVICE");
    if (device.isEmpty()) device = "cpu";

    // Exercise several complete context lifetimes in one process. This is the
    // sequence produced by switching qSAM3 tabs between different model
    // families; model buffers must be released by aicore_sam3_free(), not only
    // when the process exits.
    constexpr int kReloadCount = 5;
    for (int pass = 0; pass < kReloadCount; ++pass) {
        auto* opts = aicore_sam3_options_new();
        ASSERT_NE(opts, nullptr);
        aicore_sam3_options_set_device(opts, device.constData());
        aicore_sam3_options_set_threads(opts, 4);

        aicore_sam3_ctx* ctx =
                aicore_sam3_load_opts(s_modelPath.toUtf8().constData(), opts);
        aicore_sam3_options_free(opts);
        ASSERT_NE(ctx, nullptr)
                << "model reload " << pass << " failed: "
                << aicore_sam3_last_load_error();
        ASSERT_TRUE(aicore_sam3_is_ready(ctx));

        aicore_sam3_free(ctx);
        aicore_sam3_shutdown();
    }
}

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
    QByteArray device = qgetenv("AICORE_SAM3_TEST_DEVICE");
    if (device.isEmpty()) device = "cpu";
    aicore_sam3_options_set_device(opts, device.constData());
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

    std::vector<uint8_t> referenceMask;
    aicore_sam3_box referenceBox{};
    float referenceScore = 0.0f;
    for (int pass = 0; pass < 2; ++pass) {
        aicore_sam3_seg_result* res = aicore_sam3_segment_pvs_rgb(
                ctx, &prompt, img.constBits(), img.width(), img.height(),
                static_cast<size_t>(img.bytesPerLine()));
        ASSERT_NE(res, nullptr) << "pass " << pass;

        // Two forwards are required: one successful CUDA graph does not
        // prove that persistent buffers and destination strides are stable.
        const int n = aicore_sam3_seg_det_count(res);
        ASSERT_GE(n, 1) << "Expected at least 1 detection on pass " << pass;

        const aicore_sam3_box box = aicore_sam3_seg_det_box_at(res, 0);
        const float score = aicore_sam3_seg_det_score_at(res, 0);
        const float iou = aicore_sam3_seg_det_iou_at(res, 0);
        const aicore_sam3_plane_view mask = aicore_sam3_seg_mask_at(res, 0);

        EXPECT_TRUE(std::isfinite(score));
        EXPECT_TRUE(std::isfinite(iou));
        EXPECT_GE(score, 0.0f);
        EXPECT_LE(score, 1.0f);
        EXPECT_GE(iou, 0.0f);

        // Box should be within image bounds
        EXPECT_GE(box.x0, 0.0f);
        EXPECT_GE(box.y0, 0.0f);
        EXPECT_LE(box.x1, static_cast<float>(img.width()));
        EXPECT_LE(box.y1, static_cast<float>(img.height()));
        EXPECT_GT(box.x1, box.x0);
        EXPECT_GT(box.y1, box.y0);

        // Mask should match image dimensions
        EXPECT_EQ(mask.width, img.width());
        EXPECT_EQ(mask.height, img.height());
        ASSERT_NE(mask.data, nullptr);

        std::vector<uint8_t> packed(static_cast<size_t>(mask.width) *
                                    mask.height);
        size_t foreground = 0;
        const auto* src = static_cast<const uint8_t*>(mask.data);
        for (int y = 0; y < mask.height; ++y) {
            const auto* row = src + static_cast<size_t>(y) *
                                      mask.row_stride_bytes;
            std::copy(row, row + mask.width,
                      packed.begin() + static_cast<size_t>(y) * mask.width);
            foreground += static_cast<size_t>(
                    std::count_if(row, row + mask.width,
                                  [](uint8_t v) { return v > 127; }));
        }
        EXPECT_GT(foreground, 0u) << "empty mask on pass " << pass;

        if (pass == 0) {
            referenceMask = packed;
            referenceBox = box;
            referenceScore = score;
        } else {
            EXPECT_EQ(packed, referenceMask);
            EXPECT_FLOAT_EQ(box.x0, referenceBox.x0);
            EXPECT_FLOAT_EQ(box.y0, referenceBox.y0);
            EXPECT_FLOAT_EQ(box.x1, referenceBox.x1);
            EXPECT_FLOAT_EQ(box.y1, referenceBox.y1);
            EXPECT_FLOAT_EQ(score, referenceScore);
        }

        // Verify timings are populated
        aicore_sam3_timings t{};
        EXPECT_EQ(aicore_sam3_last_timings(ctx, &t), 0);
        EXPECT_GT(t.e2e_ms, 0.0);
        aicore_sam3_seg_result_free(res);
    }

    // Box prompting is a separate decoder path from point prompting. Keep the
    // range identical to upstream test_visual_only_compare.cpp and run it
    // twice to exercise persistent graph/buffer reuse.
    aicore_sam3_pvs_prompt boxPrompt{};
    boxPrompt.box = {img.width() * 0.1f, img.height() * 0.1f,
                     img.width() * 0.9f, img.height() * 0.9f};
    boxPrompt.use_box = 1;
    std::vector<uint8_t> referenceBoxMask;
    for (int pass = 0; pass < 2; ++pass) {
        aicore_sam3_seg_result* res = aicore_sam3_segment_pvs_rgb(
                ctx, &boxPrompt, img.constBits(), img.width(), img.height(),
                static_cast<size_t>(img.bytesPerLine()));
        ASSERT_NE(res, nullptr)
                << "box pass " << pass << ": " << aicore_sam3_last_error(ctx);
        ASSERT_GE(aicore_sam3_seg_det_count(res), 1)
                << "Expected a box-prompt detection on pass " << pass;

        const aicore_sam3_box box = aicore_sam3_seg_det_box_at(res, 0);
        const float score = aicore_sam3_seg_det_score_at(res, 0);
        const float iou = aicore_sam3_seg_det_iou_at(res, 0);
        const aicore_sam3_plane_view mask = aicore_sam3_seg_mask_at(res, 0);
        EXPECT_TRUE(std::isfinite(box.x0));
        EXPECT_TRUE(std::isfinite(box.y0));
        EXPECT_TRUE(std::isfinite(box.x1));
        EXPECT_TRUE(std::isfinite(box.y1));
        EXPECT_TRUE(std::isfinite(score));
        EXPECT_TRUE(std::isfinite(iou));
        ASSERT_EQ(mask.width, img.width());
        ASSERT_EQ(mask.height, img.height());
        ASSERT_NE(mask.data, nullptr);

        std::vector<uint8_t> packed(static_cast<size_t>(mask.width) *
                                    mask.height);
        size_t foreground = 0;
        const auto* src = static_cast<const uint8_t*>(mask.data);
        for (int y = 0; y < mask.height; ++y) {
            const auto* row =
                    src + static_cast<size_t>(y) * mask.row_stride_bytes;
            std::copy(row, row + mask.width,
                      packed.begin() + static_cast<size_t>(y) * mask.width);
            foreground += static_cast<size_t>(std::count_if(
                    row, row + mask.width, [](uint8_t v) { return v > 127; }));
        }
        EXPECT_GT(foreground, 0u) << "empty box mask on pass " << pass;
        if (pass == 0) {
            referenceBoxMask = packed;
        } else {
            EXPECT_EQ(packed, referenceBoxMask);
        }
        aicore_sam3_seg_result_free(res);
    }

    // Video tracking owns independent temporal state. Process one frame to
    // establish encoded features, initialize an instance from the same box,
    // then require it to survive the next frame.
    aicore_sam3_tracker_ctx* tracker = aicore_sam3_tracker_create(ctx);
    ASSERT_NE(tracker, nullptr) << aicore_sam3_last_error(ctx);
    const bool visualOnly = aicore_sam3_context_visual_only(ctx) != 0;
    aicore_sam3_seg_result* firstFrame =
            visualOnly
                    ? aicore_sam3_propagate_frame(
                              tracker, img.constBits(), img.width(), img.height(),
                              static_cast<size_t>(img.bytesPerLine()))
                    : aicore_sam3_track_frame(
                              tracker, img.constBits(), img.width(), img.height(),
                              static_cast<size_t>(img.bytesPerLine()));
    ASSERT_NE(firstFrame, nullptr) << aicore_sam3_tracker_last_error(tracker);
    aicore_sam3_seg_result_free(firstFrame);

    const int instanceId =
            aicore_sam3_tracker_add_instance(tracker, &boxPrompt);
    ASSERT_GE(instanceId, 0) << aicore_sam3_tracker_last_error(tracker);
    aicore_sam3_seg_result* currentMask =
            aicore_sam3_tracker_segment_pvs(tracker, &boxPrompt);
    ASSERT_NE(currentMask, nullptr) << aicore_sam3_tracker_last_error(tracker);
    ASSERT_GE(aicore_sam3_seg_det_count(currentMask), 1);
    aicore_sam3_seg_result_free(currentMask);

    aicore_sam3_seg_result* secondFrame =
            visualOnly
                    ? aicore_sam3_propagate_frame(
                              tracker, img.constBits(), img.width(), img.height(),
                              static_cast<size_t>(img.bytesPerLine()))
                    : aicore_sam3_track_frame(
                              tracker, img.constBits(), img.width(), img.height(),
                              static_cast<size_t>(img.bytesPerLine()));
    ASSERT_NE(secondFrame, nullptr) << aicore_sam3_tracker_last_error(tracker);
    const int secondFrameCount = aicore_sam3_seg_det_count(secondFrame);
    ASSERT_GE(secondFrameCount, 1);
    bool retainedInstanceId = false;
    for (int i = 0; i < secondFrameCount; ++i) {
        retainedInstanceId =
                retainedInstanceId ||
                aicore_sam3_seg_det_instance_id_at(secondFrame, i) == instanceId;
    }
    EXPECT_TRUE(retainedInstanceId)
            << "tracker replaced the initialized object on the next frame";
    aicore_sam3_seg_result_free(secondFrame);
    aicore_sam3_timings trackerTimings{};
    EXPECT_EQ(aicore_sam3_tracker_last_timings(tracker, &trackerTimings), 0);
    EXPECT_GT(trackerTimings.e2e_ms, 0.0);
    aicore_sam3_tracker_free(tracker);

    // Full SAM3 text tracking starts detections in the pending/hotstart pool.
    // The next frame must associate at least one prompted object with its
    // existing ID instead of allocating a fresh identity every frame.
    if (!visualOnly) {
        aicore_sam3_tracker_ctx* textTracker =
                aicore_sam3_tracker_create(ctx);
        ASSERT_NE(textTracker, nullptr) << aicore_sam3_last_error(ctx);
        aicore_sam3_tracker_set_text_prompt(
                textTracker, s_textPrompt.toUtf8().constData());
        aicore_sam3_seg_result* textFirst = aicore_sam3_track_frame(
                textTracker, img.constBits(), img.width(), img.height(),
                static_cast<size_t>(img.bytesPerLine()));
        ASSERT_NE(textFirst, nullptr)
                << aicore_sam3_tracker_last_error(textTracker);
        const int textFirstCount = aicore_sam3_seg_det_count(textFirst);
        ASSERT_GE(textFirstCount, 1)
                << "expected detections for " << s_textPrompt.toStdString();
        std::vector<int> textFirstIds;
        for (int i = 0; i < textFirstCount; ++i) {
            textFirstIds.push_back(
                    aicore_sam3_seg_det_instance_id_at(textFirst, i));
        }
        aicore_sam3_seg_result_free(textFirst);

        QImage textSecondImage = s_secondImagePath.isEmpty()
                                         ? img
                                         : QImage(s_secondImagePath);
        ASSERT_FALSE(textSecondImage.isNull())
                << "second tracking image could not be loaded";
        textSecondImage =
                textSecondImage.convertToFormat(QImage::Format_RGB888);
        aicore_sam3_seg_result* textSecond = aicore_sam3_track_frame(
                textTracker, textSecondImage.constBits(),
                textSecondImage.width(), textSecondImage.height(),
                static_cast<size_t>(textSecondImage.bytesPerLine()));
        ASSERT_NE(textSecond, nullptr)
                << aicore_sam3_tracker_last_error(textTracker);
        bool retainedTextId = false;
        for (int i = 0; i < aicore_sam3_seg_det_count(textSecond); ++i) {
            const int id =
                    aicore_sam3_seg_det_instance_id_at(textSecond, i);
            retainedTextId =
                    retainedTextId ||
                    std::find(textFirstIds.begin(), textFirstIds.end(), id) !=
                            textFirstIds.end();
        }
        EXPECT_TRUE(retainedTextId)
                << "text tracker replaced all pending IDs on the next frame";
        aicore_sam3_seg_result_free(textSecond);
        aicore_sam3_tracker_free(textTracker);
    }

    aicore_sam3_free(ctx);

    // Shutdown backends to clean up GPU resources
    aicore_sam3_shutdown();
}
