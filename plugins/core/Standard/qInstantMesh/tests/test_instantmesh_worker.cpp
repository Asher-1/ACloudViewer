// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QTemporaryDir>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <limits>
#include <thread>

#include "InstantMeshWorker.h"
#include "aicore/runtime_capi.h"

namespace {
QString modelRoot;
QString modelDevice;

class InstantMeshWorkerTest : public ::testing::Test {
protected:
    QTemporaryDir tmp;
    QString path;
    InstantMeshWorker::Settings settings;
    void SetUp() override {
        ASSERT_TRUE(tmp.isValid());
        qputenv("CLOUDVIEWER_DATA_ROOT", tmp.path().toUtf8());
        QImage image(8, 8, QImage::Format_RGBA8888);
        image.fill(Qt::red);
        path = tmp.filePath(QStringLiteral("input.png"));
        ASSERT_TRUE(image.save(path));
        settings.inputPath = path;
        settings.device = QStringLiteral("cpu");
        settings.modelPaths =
                QStringList{QString(),
                            QString(),
                            QString(),
                            QStringLiteral("missing-dino.gguf"),
                            QStringLiteral("missing-transformer.gguf"),
                            QStringLiteral("missing-synthesizer.gguf")};
    }
};

class InstantMeshPreviewTest : public ::testing::Test {
protected:
    InstantMeshRunResult previewResult;
    void SetUp() override {
        previewResult.verts = {0, 0, 0, 1, 0, 0, 0, 1, 0};
        previewResult.tris = {0, 1, 2};
        previewResult.vcolors = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        previewResult.uvs = {0, 0, 1, 0, 0, 1};
        previewResult.texRgba = {255, 0, 0, 128};
        previewResult.texW = previewResult.texH = 1;
    }
};
}  // namespace

TEST(InstantMeshNormals, CounterClockwiseTriangleHasUnitFaceNormals) {
    const QVector<float> verts = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    const QVector<int> tris = {0, 1, 2};
    const auto normals = InstantMeshWorker::vertexNormals(verts, tris);
    ASSERT_EQ(normals.size(), 9);
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(normals[3 * i], 0.0f);
        EXPECT_FLOAT_EQ(normals[3 * i + 1], 0.0f);
        EXPECT_GT(normals[3 * i + 2], 0.99f);
    }
}

TEST(InstantMeshNormals, RejectsOutOfRangeIndicesAndKeepsZeroAreaNormals) {
    const QVector<float> verts = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    EXPECT_TRUE(InstantMeshWorker::vertexNormals(verts, {0, 1, 9}).isEmpty());
    const auto flat = InstantMeshWorker::vertexNormals(QVector<float>(9, 0.0f),
                                                       {0, 1, 2});
    ASSERT_EQ(flat.size(), 9);
    EXPECT_FLOAT_EQ(flat[0], 0.0f);
    EXPECT_FLOAT_EQ(flat[8], 0.0f);
}

TEST_F(InstantMeshWorkerTest,
       CancellationInterruptsDeviceQueueAndCompletesOnce) {
    std::atomic<int> completed{0}, results{0};
    std::promise<int> acquired;
    std::promise<void> release;
    auto release_future = release.get_future();
    std::thread holder([&] {
        const int rc = aicore_device_task_lock("cpu");
        acquired.set_value(rc);
        release_future.wait();
        if (rc == 0) aicore_device_task_unlock();
    });
    const int acquiredResult = acquired.get_future().get();
    EXPECT_EQ(acquiredResult, 0) << "Acquire the shared CPU device queue";
    {
        InstantMeshWorker worker(settings);
        QObject::connect(
                &worker, &InstantMeshWorker::taskFinished, &worker,
                [&](bool ok) {
                    ++completed;
                    if (ok) ++results;
                },
                Qt::DirectConnection);
        QObject::connect(
                &worker, &InstantMeshWorker::resultReady, &worker,
                [&](const InstantMeshRunResult&) { ++results; },
                Qt::DirectConnection);
        worker.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const bool invoked = QMetaObject::invokeMethod(&worker, "cancel",
                                                       Qt::DirectConnection);
        EXPECT_TRUE(invoked) << "Worker exposes cancellation";
        const bool stopped = invoked && worker.wait(1000);
        EXPECT_TRUE(!invoked || stopped)
                << "Cancellation must interrupt the shared device-lock wait";
        release.set_value();
        holder.join();
        worker.wait();
    }
    EXPECT_EQ(completed.load(), 1);
    EXPECT_EQ(results.load(), 0);
}

TEST_F(InstantMeshWorkerTest,
       MissingSelectedRmbgDoesNotSubstituteAlternateWeights) {
    const QString rmbgDirectory =
            tmp.filePath(QStringLiteral("extract/rmbg_models"));
    QDir().mkpath(rmbgDirectory);
    QFile other(rmbgDirectory + QStringLiteral("/rmbg_q8.gguf"));
    ASSERT_FALSE((!other.open(QIODevice::WriteOnly)));
    other.write("GGUF invalid alternate model");
    other.close();
    settings.useRmbg = true;
    settings.rmbgModelPath = rmbgDirectory + QStringLiteral("/rmbg_f32.gguf");
    QString logs;
    {
        InstantMeshWorker worker(settings);
        QObject::connect(
                &worker, &InstantMeshWorker::logMessage, &worker,
                [&](const QString& text) { logs += text; },
                Qt::DirectConnection);
        worker.start();
        worker.wait();
    }
    EXPECT_TRUE(logs.contains(QStringLiteral("RMBG")));
    EXPECT_TRUE(logs.contains(QStringLiteral("rmbg_f32.gguf")))
            << "The selected missing RMBG model must fail explicitly";
}

TEST_F(InstantMeshPreviewTest, RasterizesTriangleSurfaceUsingRgbaTexture) {
    const auto preview =
            InstantMeshWorker::renderResultPreview(previewResult, 64);
    int surfacePixels = 0, redPixels = 0;
    for (int y = 0; y < preview.height(); ++y)
        for (int x = 0; x < preview.width(); ++x) {
            const QRgb pixel = preview.pixel(x, y);
            if (qAlpha(pixel)) {
                ++surfacePixels;
                if (qRed(pixel) > 0 && qGreen(pixel) == 0 &&
                    qBlue(pixel) == 0 && qAlpha(pixel) == 128)
                    ++redPixels;
            }
        }
    EXPECT_GT(surfacePixels, 100);
    EXPECT_EQ(redPixels, surfacePixels);
}

TEST_F(InstantMeshPreviewTest,
       PerCornerUvSeamsSampleSharedVerticesIndependently) {
    InstantMeshRunResult seam = previewResult;
    seam.verts = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    seam.tris = {0, 1, 2, 0, 2, 3};
    seam.vcolors.fill(1, 12);
    seam.uvs = {0.25f, 0.75f, 0.25f, 0.75f, 0.25f, 0.75f,
                0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f};
    seam.texW = seam.texH = 2;
    seam.texRgba = {255, 0, 0,   255, 255, 0, 0,   255,
                    0,   0, 255, 255, 0,   0, 255, 255};
    const QImage seamImage = InstantMeshWorker::renderResultPreview(seam, 64);
    int topRed = 0, bottomBlue = 0;
    for (int y = 0; y < seamImage.height(); ++y)
        for (int x = 0; x < seamImage.width(); ++x) {
            const QRgb pixel = seamImage.pixel(x, y);
            if (qAlpha(pixel) && qRed(pixel) > 0 && qBlue(pixel) == 0) ++topRed;
            if (qAlpha(pixel) && qBlue(pixel) > 0 && qRed(pixel) == 0)
                ++bottomBlue;
        }
    EXPECT_GE(topRed, 100);
    EXPECT_GE(bottomBlue, 100);
}

TEST_F(InstantMeshPreviewTest, RejectsInvalidIndicesNonfiniteUvsAndColors) {
    InstantMeshRunResult seam = previewResult;
    seam.tris[0] = -1;
    EXPECT_TRUE(InstantMeshWorker::renderResultPreview(seam, 64).isNull());
    seam = previewResult;
    seam.uvs[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(InstantMeshWorker::renderResultPreview(seam, 64).isNull());
    seam = previewResult;
    seam.texRgba.clear();
    seam.uvs.clear();
    seam.texW = seam.texH = 0;
    seam.vcolors[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(InstantMeshWorker::renderResultPreview(seam, 64).isNull())
            << "Nonfinite colors must not reach integer pixel conversion";
}

TEST_F(InstantMeshWorkerTest,
       RealInferencePropagatesSourceGenerationAndCommonTiming) {
    if (modelRoot.isEmpty())
        GTEST_SKIP()
                << "Pass model root and optional device for real inference";
    const QString models = modelRoot + "/instantmesh_models/";
    settings.modelPaths = QStringList{QString(),
                                      QString(),
                                      QString(),
                                      models + "dino_f16.gguf",
                                      models + "lrm_transformer_f16.gguf",
                                      models + "synthesizer_f16.gguf"};
    settings.useRmbg = false;
    settings.gridRes = 8;
    settings.bakeTexture = false;
    settings.generation = 17;
    settings.device =
            modelDevice.isEmpty() ? QStringLiteral("cpu") : modelDevice;
    InstantMeshRunResult result;
    bool success = false;
    {
        InstantMeshWorker worker(settings);
        QObject::connect(
                &worker, &InstantMeshWorker::resultReady, &worker,
                [&](const InstantMeshRunResult& value) { result = value; },
                Qt::DirectConnection);
        QObject::connect(
                &worker, &InstantMeshWorker::taskFinished, &worker,
                [&](bool value) { success = value; }, Qt::DirectConnection);
        worker.start();
        worker.wait();
    }
    EXPECT_TRUE(success);
    EXPECT_EQ(result.generation, 17);
    EXPECT_EQ(result.sourceImage, path);
    EXPECT_NE(result.timings.valid_fields, 0);
    EXPECT_GT(result.timings.e2e_ms, 0);
    EXPECT_EQ(result.totalRuntimeMs, result.timings.e2e_ms);
    EXPECT_GT(result.loadRuntimeMs, 0);
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    if (argc > 1) modelRoot = QString::fromLocal8Bit(argv[1]);
    if (argc > 2) modelDevice = QString::fromLocal8Bit(argv[2]);
    return RUN_ALL_TESTS();
}
