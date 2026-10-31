// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir tmp;
    qputenv("CLOUDVIEWER_DATA_ROOT", tmp.path().toUtf8());
    QImage image(8, 8, QImage::Format_RGBA8888);
    image.fill(Qt::red);
    const QString path = tmp.filePath(QStringLiteral("input.png"));
    if (!image.save(path)) return 1;
    InstantMeshWorker::Settings settings;
    settings.inputPath = path;
    settings.device = QStringLiteral("cpu");
    settings.modelPaths =
            QStringList{QString(),
                        QString(),
                        QString(),
                        QStringLiteral("missing-dino.gguf"),
                        QStringLiteral("missing-transformer.gguf"),
                        QStringLiteral("missing-synthesizer.gguf")};
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
    int failures = acquired.get_future().get() != 0;
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
        if (!invoked) {
            std::fprintf(
                    stderr,
                    "FAIL: worker has no callable cancellation entry point\n");
            ++failures;
        }
        const bool stopped = invoked && worker.wait(1000);
        if (invoked && !stopped) {
            std::fprintf(stderr,
                         "FAIL: cancellation did not interrupt the shared "
                         "device-lock wait\n");
            ++failures;
        }
        release.set_value();
        holder.join();
        worker.wait();
    }
    if (completed != 1 || results != 0) {
        std::fprintf(stderr, "FAIL: completion=%d results=%d\n",
                     completed.load(), results.load());
        ++failures;
    }
    const QString rmbgDirectory =
            tmp.filePath(QStringLiteral("extract/rmbg_models"));
    QDir().mkpath(rmbgDirectory);
    QFile other(rmbgDirectory + QStringLiteral("/rmbg_q8.gguf"));
    if (!other.open(QIODevice::WriteOnly)) return 1;
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
    if (!logs.contains(QStringLiteral("RMBG")) ||
        !logs.contains(QStringLiteral("rmbg_f32.gguf"))) {
        std::fprintf(stderr,
                     "FAIL: worker did not explicitly fail for the selected "
                     "RMBG model\n");
        ++failures;
    }
    // Hand-built triangles and constant RGBA: detect discarded textures or
    // vertex-only projection.
    InstantMeshRunResult previewResult;
    previewResult.verts = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    previewResult.tris = {0, 1, 2};
    previewResult.vcolors = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    previewResult.uvs = {0, 0, 1, 0, 0, 1};
    previewResult.texRgba = {255, 0, 0, 128};
    previewResult.texW = previewResult.texH = 1;
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
    if (surfacePixels <= 100 || redPixels != surfacePixels) {
        std::fprintf(stderr,
                     "FAIL: preview must rasterize triangles and consume RGBA, "
                     "surface=%d red=%d\n",
                     surfacePixels, redPixels);
        ++failures;
    }
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
    if (topRed < 100 || bottomBlue < 100) {
        std::fprintf(stderr,
                     "FAIL: per-corner UV seams on shared vertices must sample "
                     "top red/bottom blue separately\n");
        ++failures;
    }
    seam.tris[0] = -1;
    if (!InstantMeshWorker::renderResultPreview(seam, 64).isNull()) ++failures;
    seam = previewResult;
    seam.uvs[0] = std::numeric_limits<float>::quiet_NaN();
    if (!InstantMeshWorker::renderResultPreview(seam, 64).isNull()) ++failures;
    seam = previewResult;
    seam.texRgba.clear();
    seam.uvs.clear();
    seam.texW = seam.texH = 0;
    seam.vcolors[0] = std::numeric_limits<float>::quiet_NaN();
    if (!InstantMeshWorker::renderResultPreview(seam, 64).isNull()) {
        std::fprintf(stderr,
                     "FAIL: nonfinite vertex colors must not reach integer "
                     "pixel conversion\n");
        ++failures;
    }
    if (argc > 1) {
        const QString models =
                QString::fromLocal8Bit(argv[1]) + "/instantmesh_models/";
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
        settings.device = argc > 2 ? QString::fromLocal8Bit(argv[2])
                                   : QStringLiteral("cpu");
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
        if (!success || result.generation != 17 || result.sourceImage != path ||
            result.timings.valid_fields == 0 || result.timings.e2e_ms <= 0 ||
            result.totalRuntimeMs != result.timings.e2e_ms ||
            result.loadRuntimeMs <= 0) {
            std::fprintf(stderr,
                         "FAIL: real worker did not propagate common "
                         "timing/source/generation\n");
            ++failures;
        }
    }
    return failures ? 1 : 0;
}
