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
#include <QImage>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QPushButton>
#include <QSettings>
#include <QTemporaryDir>

#include "Sam3dDialog.h"
#include "aicore/sam3d_capi.h"
#include "ecvAssetIntegrity.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

class Sam3dModelSelectionTest : public ::testing::Test {
protected:
    QTemporaryDir temporary;

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
        QCoreApplication::setApplicationName("Sam3dSelection");
    }
};

TEST_F(Sam3dModelSelectionTest, SourcePreviewTracksTheSelectedImage) {
    Sam3dDialog dialog;
    auto* image = dialog.findChild<QLineEdit*>("sourceImage");
    ASSERT_NE(image, nullptr);
    auto* preview = dialog.findChild<ecvClickableImageLabel*>();
    ASSERT_NE(preview, nullptr);
    EXPECT_TRUE(preview->fullImage().isNull());
    const QString previewPath = QDir(temporary.path()).filePath("preview.png");
    QImage fixture(64, 48, QImage::Format_RGB32);
    fixture.fill(Qt::red);
    ASSERT_TRUE(fixture.save(previewPath));
    image->setText(previewPath);
    ASSERT_FALSE(preview->fullImage().isNull());
    EXPECT_EQ(preview->fullImage().size(), QSize(64, 48));
    image->clear();
    EXPECT_TRUE(preview->fullImage().isNull());
}

TEST_F(Sam3dModelSelectionTest, RequiredFilesFollowQuantizationAndMode) {
    Sam3dDialog dialog;
    auto* quant = dialog.findChild<QComboBox*>("quantization");
    auto* rmbgQuant = dialog.findChild<QComboBox*>("rmbgQuantization");
    auto* rmbg = dialog.findChild<QCheckBox*>("removeBackground");
    auto* mesh = dialog.findChild<QCheckBox*>("texturedMesh");
    auto* scene = dialog.findChild<QCheckBox*>("sceneMode");
    auto* prompts = dialog.findChild<QLineEdit*>("conceptPrompts");
    auto* masks = dialog.findChild<QLineEdit*>("masksDirectory");
    ASSERT_TRUE(quant && rmbgQuant && rmbg && mesh && scene && prompts &&
                masks);
    scene->setChecked(false);
    mesh->setChecked(false);
    rmbg->setChecked(false);
    const QStringList suffixes{"f16", "q8_0", "q4_k"};
    for (int q = 0; q < 3; ++q) {
        quant->setCurrentIndex(q);
        const auto files = dialog.requiredModelFiles();
        ASSERT_EQ(files.size(), 5);
        EXPECT_TRUE(files.contains("moge_vitl-f16.gguf"));
        EXPECT_TRUE(files.contains(
                QString("ss_generator-%1.gguf").arg(suffixes[q])));
        EXPECT_FALSE(files.contains("sam3-q8_0.gguf"));
        EXPECT_TRUE(dialog.requiredModelFiles(true).isEmpty());
    }
    mesh->setChecked(true);
    EXPECT_EQ(dialog.requiredModelFiles().size(), 6);
    mesh->setChecked(false);
    rmbg->setChecked(true);
    EXPECT_TRUE(dialog.requiredModelFiles().contains("rmbg_q8.gguf"));
    rmbgQuant->setCurrentIndex(1);
    EXPECT_TRUE(dialog.requiredModelFiles().contains("rmbg_f16.gguf"));
    EXPECT_FALSE(dialog.requiredModelFiles().contains("rmbg_q8.gguf"));
    scene->setChecked(true);
    prompts->setText("chair");
    masks->clear();
    const QString selectedSam = dialog.settings().sam3ModelFile;
    EXPECT_EQ(dialog.requiredModelFiles(true), QStringList{selectedSam});
    EXPECT_FALSE(dialog.requiredModelFiles().contains("rmbg_f16.gguf"));
    prompts->clear();
    masks->setText(temporary.path());
    EXPECT_TRUE(dialog.requiredModelFiles(true).isEmpty());
    EXPECT_FALSE(dialog.requiredModelFiles().contains(selectedSam));
}

TEST_F(Sam3dModelSelectionTest, CachedRunAndDownloadContinuation) {
    Sam3dDialog dialog;
    auto* quant = dialog.findChild<QComboBox*>("quantization");
    auto* rmbg = dialog.findChild<QCheckBox*>("removeBackground");
    auto* mesh = dialog.findChild<QCheckBox*>("texturedMesh");
    auto* scene = dialog.findChild<QCheckBox*>("sceneMode");
    auto* image = dialog.findChild<QLineEdit*>("sourceImage");
    ASSERT_TRUE(quant && rmbg && mesh && scene && image);
    scene->setChecked(false);
    mesh->setChecked(false);
    rmbg->setChecked(false);
    quant->setCurrentIndex(2);
    const QString cache = QString::fromUtf8(aicore_sam3d_model_cache_dir());
    ASSERT_TRUE(QDir().mkpath(cache));
    for (const QString& name : dialog.requiredModelFiles()) {
        const QString path = QDir(cache).filePath(name);
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_GT(file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x')),
                  0);
        file.close();
        const auto* entry =
                aicore_sam3d_model_by_filename(name.toUtf8().constData());
        ASSERT_NE(entry, nullptr);
        ASSERT_TRUE(ecvAssetIntegrity::markVerified(
                path, {QCryptographicHash::Sha256, QByteArray(entry->sha256)}));
    }
    image->setText("ui-fixture.png");
    int runs = 0;
    QObject::connect(&dialog, &Sam3dDialog::runRequested, &dialog,
                     [&](const Sam3dDialog::Settings&) { ++runs; });
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    EXPECT_EQ(runs, 2);
    auto* downloader = dialog.findChild<ecvModelDownloader*>();
    ASSERT_NE(downloader, nullptr);
    EXPECT_FALSE(downloader->isBusy());
    EXPECT_FALSE(QFile::exists(QDir(cache).filePath("ss_generator-f16.gguf")));
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "onDownloadModels"));
    EXPECT_TRUE(dialog.findChild<QPushButton*>("downloadModels")->isEnabled());

    const QString selected = dialog.requiredModelFiles().first();
    const QString selectedPath = QDir(cache).filePath(selected);
    auto restoreSelected = [&]() {
        QFile file(selectedPath);
        if (!file.open(QIODevice::WriteOnly)) return false;
        file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x'));
        file.close();
        const auto* entry =
                aicore_sam3d_model_by_filename(selected.toUtf8().constData());
        return entry && ecvAssetIntegrity::markVerified(
                                selectedPath, {QCryptographicHash::Sha256,
                                               QByteArray(entry->sha256)});
    };
    downloader->findChild<QNetworkAccessManager*>()->setProxy(QNetworkProxy(
            QNetworkProxy::HttpProxy, QStringLiteral("127.0.0.1"), 9));
    ASSERT_TRUE(QFile::remove(selectedPath));
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    EXPECT_TRUE(downloader->isBusy());
    EXPECT_FALSE(quant->isEnabled());
    auto* cancel = dialog.findChild<QPushButton*>("cancelTask");
    ASSERT_TRUE(cancel && cancel->isEnabled());
    cancel->click();
    EXPECT_FALSE(downloader->isBusy());
    EXPECT_TRUE(quant->isEnabled());
    EXPECT_TRUE(dialog.findChild<QPushButton*>("downloadModels")->isEnabled());
    EXPECT_EQ(runs, 2);
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    EXPECT_TRUE(downloader->isBusy());
    downloader->blockSignals(true);
    downloader->cancel();
    downloader->blockSignals(false);
    ASSERT_TRUE(restoreSelected());
    downloader->finished(true, selectedPath);
    EXPECT_EQ(runs, 3);
    EXPECT_TRUE(quant->isEnabled());
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    EXPECT_EQ(runs, 4);
    EXPECT_FALSE(downloader->isBusy());

    ASSERT_TRUE(QFile::remove(selectedPath));
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    EXPECT_TRUE(downloader->isBusy());
    dialog.reject();
    EXPECT_FALSE(downloader->isBusy());
    EXPECT_TRUE(quant->isEnabled());
    ASSERT_TRUE(restoreSelected());
    downloader->finished(true, selectedPath);
    EXPECT_EQ(runs, 4);
    ASSERT_TRUE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    EXPECT_EQ(runs, 5);
    EXPECT_FALSE(downloader->isBusy());
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
