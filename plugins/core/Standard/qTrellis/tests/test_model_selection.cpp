// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <FileSystem.h>
#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QPushButton>
#include <QSettings>
#include <QTemporaryDir>

#include "TrellisDialog.h"
#include "TrellisWorker.h"
#include "ecvAssetIntegrity.h"

#ifdef AICore_ENABLED
#include <aicore/trellis_capi.h>
#endif

class TrellisModelSelectionTest : public ::testing::Test {
protected:
    QTemporaryDir temporary;
    const QByteArray previousDataRoot = qgetenv("CLOUDVIEWER_DATA_ROOT");
    const bool hadDataRoot = qEnvironmentVariableIsSet("CLOUDVIEWER_DATA_ROOT");

    void CacheRequiredModels(const QStringList& paths) {
        for (const QString& path : paths.mid(0, 6)) {
            if (path.isEmpty()) continue;
            ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
            QFile file(path);
            ASSERT_TRUE(file.open(QIODevice::WriteOnly));
            ASSERT_GT(
                    file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x')),
                    0);
            file.close();
            ASSERT_TRUE(ecvAssetIntegrity::markVerified(
                    path, {QCryptographicHash::Sha256,
                           ecvAssetIntegrity::PinnedDigest(
                                   QFileInfo(path).fileName())}));
        }
    }

    void TearDown() override {
        if (hadDataRoot) {
            qputenv("CLOUDVIEWER_DATA_ROOT", previousDataRoot);
        } else {
            qunsetenv("CLOUDVIEWER_DATA_ROOT");
        }
    }

    void SetUp() override {
        ASSERT_TRUE(temporary.isValid());
        ASSERT_TRUE(cloudViewer::utility::filesystem::SetEnvVar(
                "CLOUDVIEWER_DATA_ROOT",
                temporary.path().toUtf8().toStdString()));
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                           temporary.path());
        QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope,
                           temporary.path());
        QCoreApplication::setOrganizationName("ACloudViewerCacheTest");
        QCoreApplication::setApplicationName("TrellisSelection");
    }
};

TEST_F(TrellisModelSelectionTest,
       CachedGeometryRunsAndMissingModelsDoNotStart) {
    TrellisDialog dialog;
    auto* preset = dialog.findChild<QComboBox*>("pipelinePreset");
    auto* quant = dialog.findChild<QComboBox*>("quantization");
    auto* texture = dialog.findChild<QCheckBox*>("textureEnabled");
    auto* rmbg = dialog.findChild<QCheckBox*>("removeBackground");
    auto* image = dialog.findChild<QLineEdit*>("sourceImage");
    ASSERT_TRUE(preset && quant && texture && rmbg && image);
    preset->setCurrentIndex(1);
    quant->setCurrentIndex(quant->findData("q8"));
    texture->setChecked(false);
    rmbg->setChecked(false);
    image->setText("ui-fixture.png");
    const auto paths = dialog.getSettings().modelPaths;
    ASSERT_NO_FATAL_FAILURE(CacheRequiredModels(paths));
    int runs = 0;
    TrellisRunMode emitted = TrellisRunMode::GeometryOnly;
    QObject::connect(&dialog, &TrellisDialog::runRequested, &dialog,
                     [&](const TrellisDialog::Settings& settings) {
                         ++runs;
                         emitted = settings.runMode;
                     });
    dialog.runWithMode(TrellisRunMode::Full);
    EXPECT_EQ(runs, 1);
    EXPECT_EQ(emitted, TrellisRunMode::Full);
    EXPECT_FALSE(dialog.findChild<ecvModelDownloader*>()->isBusy());
    dialog.runWithMode(TrellisRunMode::GeometryOnly);
    EXPECT_EQ(runs, 2);
    EXPECT_EQ(emitted, TrellisRunMode::GeometryOnly);
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "onDownloadModels"));
    EXPECT_FALSE(dialog.findChild<ecvModelDownloader*>()->isBusy());
    for (const QString& path : paths.mid(6)) {
        if (!path.isEmpty()) EXPECT_FALSE(QFile::exists(path));
    }
}

TEST_F(TrellisModelSelectionTest, DownloadCancellationRetryAndLateCompletion) {
    TrellisDialog dialog;
    auto* preset = dialog.findChild<QComboBox*>("pipelinePreset");
    auto* quant = dialog.findChild<QComboBox*>("quantization");
    auto* texture = dialog.findChild<QCheckBox*>("textureEnabled");
    auto* rmbg = dialog.findChild<QCheckBox*>("removeBackground");
    auto* image = dialog.findChild<QLineEdit*>("sourceImage");
    ASSERT_TRUE(preset && quant && texture && rmbg && image);
    image->setText("ui-fixture.png");
    preset->setCurrentIndex(1);
    quant->setCurrentIndex(quant->findData("q8"));
    texture->setChecked(false);
    rmbg->setChecked(false);
    auto* downloader = dialog.findChild<ecvModelDownloader*>();
    ASSERT_NE(downloader, nullptr);
    const auto paths = dialog.getSettings().modelPaths;
    ASSERT_FALSE(paths.isEmpty());
    ASSERT_NO_FATAL_FAILURE(CacheRequiredModels(paths));
    const QString selectedPath = paths.first();
    downloader->findChild<QNetworkAccessManager*>()->setProxy(QNetworkProxy(
            QNetworkProxy::HttpProxy, QStringLiteral("127.0.0.1"), 9));
    int runs = 0;
    TrellisRunMode emitted = TrellisRunMode::GeometryOnly;
    QObject::connect(&dialog, &TrellisDialog::runRequested, &dialog,
                     [&](const TrellisDialog::Settings& settings) {
                         ++runs;
                         emitted = settings.runMode;
                     });
    ASSERT_TRUE(QFile::remove(selectedPath));
    dialog.runWithMode(TrellisRunMode::Full);
    ASSERT_TRUE(downloader->isBusy());
    EXPECT_EQ(runs, 0);
    auto* cancel = dialog.findChild<QPushButton*>("cancelTask");
    ASSERT_TRUE(cancel && cancel->isEnabled());
    EXPECT_FALSE(quant->isEnabled());
    cancel->click();
    EXPECT_FALSE(downloader->isBusy());
    EXPECT_EQ(runs, 0);
    EXPECT_TRUE(quant->isEnabled());
    EXPECT_FALSE(cancel->isEnabled());

    dialog.runWithMode(TrellisRunMode::Full);
    ASSERT_TRUE(downloader->isBusy());
    downloader->blockSignals(true);
    downloader->cancel();
    downloader->blockSignals(false);
    {
        QFile file(selectedPath);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_GT(file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x')),
                  0);
        file.close();
        ASSERT_TRUE(ecvAssetIntegrity::markVerified(
                selectedPath, {QCryptographicHash::Sha256,
                               ecvAssetIntegrity::PinnedDigest(
                                       QFileInfo(selectedPath).fileName())}));
    }
    downloader->finished(true, selectedPath);
    EXPECT_EQ(runs, 1);
    EXPECT_EQ(emitted, TrellisRunMode::Full);
    EXPECT_FALSE(downloader->isBusy());

    ASSERT_TRUE(QFile::remove(selectedPath));
    dialog.runWithMode(TrellisRunMode::Full);
    ASSERT_TRUE(downloader->isBusy());
    dialog.reject();
    EXPECT_FALSE(downloader->isBusy());
    EXPECT_TRUE(quant->isEnabled());
    {
        QFile file(selectedPath);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_GT(file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x')),
                  0);
        file.close();
        ASSERT_TRUE(ecvAssetIntegrity::markVerified(
                selectedPath, {QCryptographicHash::Sha256,
                               ecvAssetIntegrity::PinnedDigest(
                                       QFileInfo(selectedPath).fileName())}));
    }
    downloader->finished(true, selectedPath);
    EXPECT_EQ(runs, 1);

    dialog.runWithMode(TrellisRunMode::GeometryOnly);
    EXPECT_EQ(runs, 2);
    EXPECT_EQ(emitted, TrellisRunMode::GeometryOnly);

    int cancellations = 0;
    QObject::connect(&dialog, &TrellisDialog::cancelRequested, &dialog,
                     [&]() { ++cancellations; });
    dialog.setRunning(true);
    EXPECT_TRUE(cancel->isEnabled());
    EXPECT_FALSE(quant->isEnabled());
    cancel->click();
    EXPECT_EQ(cancellations, 1);
    dialog.setRunning(false);
    EXPECT_FALSE(cancel->isEnabled());
    EXPECT_TRUE(quant->isEnabled());
}

TEST(TrellisWorkerPreview, DecodesT2img01AndRejectsInvalidLength) {
    const int width = 4, height = 3;
    QByteArray payload;
    payload.append("T2IMG01", 8);
    const quint32 dimensions[2] = {static_cast<quint32>(width),
                                   static_cast<quint32>(height)};
    payload.append(reinterpret_cast<const char*>(dimensions), 8);
    QByteArray pixels(width * height * 4, 0);
    for (int i = 0; i < width * height; ++i) {
        pixels[i * 4] = static_cast<quint8>(10 + i);
        pixels[i * 4 + 1] = 20;
        pixels[i * 4 + 2] = 30;
        pixels[i * 4 + 3] = (i % 2) ? 255 : 0;
    }
    payload.append(pixels);
    TrellisWorker worker(TrellisWorker::Settings{});
    const QImage image =
            worker.renderPreviewBlob(payload.constData(), payload.size(), 256);
    ASSERT_EQ(image.size(), QSize(width, height));
    EXPECT_EQ(image.format(), QImage::Format_RGBA8888);
    EXPECT_EQ(image.pixelColor(0, 0).alpha(), 0);
    EXPECT_EQ(image.pixelColor(1, 0).alpha(), 255);
    EXPECT_EQ(image.pixelColor(1, 0).red(), 11);
    EXPECT_TRUE(worker.renderPreviewBlob(payload.constData(),
                                         payload.size() - 1, 256)
                        .isNull());
    EXPECT_TRUE(worker.renderPreviewBlob(payload.constData(),
                                         payload.size() + 1, 256)
                        .isNull());
}

TEST_F(TrellisModelSelectionTest, PreprocessStagePreviewCanBeApplied) {
    TrellisDialog dialog;
    QImage chip(6, 5, QImage::Format_RGBA8888);
    chip.fill(0);
    chip.setPixel(0, 0, qRgba(1, 2, 3, 0));
    TrellisStagePreview preview;
    preview.stage = AICORE_TRELLIS_STAGE_PREPROCESS;
    preview.image = chip;
    preview.label = QStringLiteral("preprocess");
    dialog.resetStageStrip(QImage(8, 8, QImage::Format_RGB32));
    dialog.setStagePreview(preview);
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
