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
#include "ecvAssetIntegrity.h"

#define REQUIRE(condition)                                               \
    do {                                                                 \
        std::fprintf(stderr, "checkpoint line %d\n", __LINE__); \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
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
    std::puts(
            "TRELLIS geometry-only dependency and cached Run workflow passed "
            "(UI fixtures only)");
    std::fprintf(stderr, "body done, returning from main\n");
    return 0;
}
