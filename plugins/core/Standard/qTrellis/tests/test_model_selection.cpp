// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <FileSystem.h>

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QSettings>
#include <QTemporaryDir>
#include <cstdio>

#include "TrellisDialog.h"
#include "TrellisWorker.h"
#include "ecvAssetIntegrity.h"

#ifdef AICore_ENABLED
#include <aicore/trellis_capi.h>
#endif

#define REQUIRE(condition)                                               \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QTemporaryDir temporary;
    REQUIRE(temporary.isValid());
    REQUIRE(cloudViewer::utility::filesystem::SetEnvVar(
            "CLOUDVIEWER_DATA_ROOT", temporary.path().toUtf8().toStdString()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       temporary.path());
    app.setOrganizationName("ACloudViewerCacheTest");
    app.setApplicationName("TrellisSelection");
    TrellisDialog dialog;
    auto* preset = dialog.findChild<QComboBox*>("pipelinePreset");
    auto* quant = dialog.findChild<QComboBox*>("quantization");
    auto* texture = dialog.findChild<QCheckBox*>("textureEnabled");
    auto* rmbg = dialog.findChild<QCheckBox*>("removeBackground");
    auto* image = dialog.findChild<QLineEdit*>("sourceImage");
    REQUIRE(preset && quant && texture && rmbg && image);
    preset->setCurrentIndex(1);
    quant->setCurrentIndex(quant->findData("q8"));
    texture->setChecked(false);
    rmbg->setChecked(false);
    image->setText("ui-fixture.png");
    // Inject UI-only ingestion evidence for geometry, leaving every texture
    // weight and alternate quantization absent. No inference is performed.
    const auto paths = dialog.getSettings().modelPaths;
    for (const QString& path : paths.mid(0, 6)) {
        if (path.isEmpty()) continue;
        REQUIRE(QDir().mkpath(QFileInfo(path).absolutePath()));
        QFile file(path);
        REQUIRE(file.open(QIODevice::WriteOnly));
        REQUIRE(file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x')) >
                0);
        file.close();
        ecvAssetIntegrity::markVerified(
                path,
                {QCryptographicHash::Sha256,
                 ecvAssetIntegrity::PinnedDigest(QFileInfo(path).fileName())});
    }
    int runs = 0;
    TrellisRunMode emitted = TrellisRunMode::GeometryOnly;
    QObject::connect(&dialog, &TrellisDialog::runRequested, &app,
                     [&](const TrellisDialog::Settings& settings) {
                         ++runs;
                         emitted = settings.runMode;
                     });
    dialog.runWithMode(TrellisRunMode::Full);
    REQUIRE(runs == 1 && emitted == TrellisRunMode::Full);
    REQUIRE(!dialog.findChild<ecvModelDownloader*>()->isBusy());
    dialog.runWithMode(TrellisRunMode::GeometryOnly);
    REQUIRE(runs == 2 && emitted == TrellisRunMode::GeometryOnly);
    REQUIRE(QMetaObject::invokeMethod(&dialog, "onDownloadModels"));
    REQUIRE(!dialog.findChild<ecvModelDownloader*>()->isBusy());
    for (const QString& path : paths.mid(6)) {
        if (!path.isEmpty()) REQUIRE(!QFile::exists(path));
    }
    // Missing one selected file: Full must remain Full after the download.
    const QString selectedPath = paths.first();
    dialog.findChild<ecvModelDownloader*>()
            ->findChild<QNetworkAccessManager*>()
            ->setProxy(QNetworkProxy(QNetworkProxy::HttpProxy,
                                     QStringLiteral("127.0.0.1"), 9));
    REQUIRE(QFile::remove(selectedPath));
    dialog.runWithMode(TrellisRunMode::Full);
    auto* downloader = dialog.findChild<ecvModelDownloader*>();
    REQUIRE(downloader->isBusy() && runs == 2);
    auto* cancel = dialog.findChild<QPushButton*>("cancelTask");
    REQUIRE(cancel && cancel->isEnabled() && !quant->isEnabled());
    cancel->click();
    REQUIRE(!downloader->isBusy() && runs == 2);
    REQUIRE(quant->isEnabled() && !cancel->isEnabled());
    dialog.runWithMode(TrellisRunMode::Full);
    REQUIRE(downloader->isBusy());
    downloader->blockSignals(true);
    downloader->cancel();
    downloader->blockSignals(false);
    {
        QFile file(selectedPath);
        REQUIRE(file.open(QIODevice::WriteOnly));
        file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x'));
        file.close();
        REQUIRE(ecvAssetIntegrity::markVerified(
                selectedPath, {QCryptographicHash::Sha256,
                               ecvAssetIntegrity::PinnedDigest(
                                       QFileInfo(selectedPath).fileName())}));
    }
    downloader->finished(true, selectedPath);
    REQUIRE(runs == 3 && emitted == TrellisRunMode::Full);
    REQUIRE(!downloader->isBusy());
    REQUIRE(QFile::remove(selectedPath));
    dialog.runWithMode(TrellisRunMode::Full);
    REQUIRE(downloader->isBusy());
    dialog.reject();
    REQUIRE(!downloader->isBusy() && quant->isEnabled());
    {
        QFile file(selectedPath);
        REQUIRE(file.open(QIODevice::WriteOnly));
        file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x'));
        file.close();
        REQUIRE(ecvAssetIntegrity::markVerified(
                selectedPath, {QCryptographicHash::Sha256,
                               ecvAssetIntegrity::PinnedDigest(
                                       QFileInfo(selectedPath).fileName())}));
    }
    downloader->finished(true, selectedPath);
    REQUIRE(runs == 3);
    dialog.runWithMode(TrellisRunMode::GeometryOnly);
    REQUIRE(runs == 4 && emitted == TrellisRunMode::GeometryOnly);
    int cancellations = 0;
    QObject::connect(&dialog, &TrellisDialog::cancelRequested, &app,
                     [&]() { ++cancellations; });
    dialog.setRunning(true);
    REQUIRE(cancel->isEnabled() && !quant->isEnabled());
    cancel->click();
    REQUIRE(cancellations == 1);
    dialog.setRunning(false);
    REQUIRE(!cancel->isEnabled() && quant->isEnabled());

    // ── T2IMG01 preprocess preview: decode round-trip ──────────────────────
    // The AICore preprocess stage streams the RMBG-matted RGBA image as a
    // T2IMG01 blob; the worker must decode it to a full-resolution QImage
    // with intact alpha (the chip's checker backdrop needs it) and reject
    // damaged payload lengths.
    {
        const int w = 4, h = 3;
        QByteArray payload;
        payload.append("T2IMG01", 8);
        const quint32 dims[2] = {(quint32)w, (quint32)h};
        payload.append((const char*)dims, 8);
        QByteArray px(w * h * 4, 0);
        for (int i = 0; i < w * h; ++i) {
            px[i * 4 + 0] = (quint8)(10 + i);
            px[i * 4 + 1] = 20;
            px[i * 4 + 2] = 30;
            px[i * 4 + 3] = (i % 2) ? 255 : 0;  // mixed alpha on purpose
        }
        payload.append(px);

        TrellisWorker worker(TrellisWorker::Settings{});
        const QImage img = worker.renderPreviewBlob(payload.constData(),
                                                    payload.size(), 256);
        REQUIRE(img.width() == w && img.height() == h);
        REQUIRE(img.format() == QImage::Format_RGBA8888);
        REQUIRE(img.pixelColor(0, 0).alpha() == 0);
        REQUIRE(img.pixelColor(1, 0).alpha() == 255);
        REQUIRE(img.pixelColor(1, 0).red() == 11);
        // A payload length inconsistent with the header must be rejected.
        REQUIRE(worker.renderPreviewBlob(payload.constData(),
                                         payload.size() - 1, 256)
                        .isNull());
        REQUIRE(worker.renderPreviewBlob(payload.constData(),
                                         payload.size() + 1, 256)
                        .isNull());
    }

    // The live PREPROCESS stage maps onto the Preprocess chip (slot 1) and
    // must not disturb the voxel/mesh slot mapping.
    {
        QImage chip(6, 5, QImage::Format_RGBA8888);
        chip.fill(0);
        chip.setPixel(0, 0, qRgba(1, 2, 3, 0));
        TrellisStagePreview preview;
        preview.stage = AICORE_TRELLIS_STAGE_PREPROCESS;
        preview.image = chip;
        preview.label = QStringLiteral("preprocess");
        dialog.resetStageStrip(QImage(8, 8, QImage::Format_RGB32));
        dialog.setStagePreview(preview);
        std::puts("TRELLIS T2IMG01 preprocess preview round-trip passed");
    }
    return 0;
}
