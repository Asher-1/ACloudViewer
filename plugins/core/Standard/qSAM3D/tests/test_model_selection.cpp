// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <FileSystem.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QPushButton>
#include <QTemporaryDir>
#include <cstdio>

#include "Sam3dDialog.h"
#include "aicore/sam3d_capi.h"
#include "ecvAssetIntegrity.h"
#include "ecvModelDownloader.h"

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
    Sam3dDialog dialog;
    auto* quant = dialog.findChild<QComboBox*>("quantization");
    auto* rmbgQuant = dialog.findChild<QComboBox*>("rmbgQuantization");
    auto* rmbg = dialog.findChild<QCheckBox*>("removeBackground");
    auto* mesh = dialog.findChild<QCheckBox*>("texturedMesh");
    auto* scene = dialog.findChild<QCheckBox*>("sceneMode");
    auto* prompts = dialog.findChild<QLineEdit*>("conceptPrompts");
    auto* masks = dialog.findChild<QLineEdit*>("masksDirectory");
    auto* image = dialog.findChild<QLineEdit*>("sourceImage");
    REQUIRE(quant && rmbgQuant && rmbg && mesh && scene && prompts && masks &&
            image);
    scene->setChecked(false);
    mesh->setChecked(false);
    rmbg->setChecked(false);
    const QStringList suffixes{"f16", "q8_0", "q4_k"};
    for (int q = 0; q < 3; ++q) {
        quant->setCurrentIndex(q);
        const auto files = dialog.requiredModelFiles();
        REQUIRE(files.size() == 5);
        REQUIRE(files.contains("moge_vitl-f16.gguf"));
        REQUIRE(files.contains(
                QString("ss_generator-%1.gguf").arg(suffixes[q])));
        REQUIRE(!files.contains("sam3-q8_0.gguf"));
        REQUIRE(dialog.requiredModelFiles(true).isEmpty());
    }
    mesh->setChecked(true);
    REQUIRE(dialog.requiredModelFiles().size() == 6);
    mesh->setChecked(false);
    rmbg->setChecked(true);
    REQUIRE(dialog.requiredModelFiles().contains("rmbg_q8.gguf"));
    rmbgQuant->setCurrentIndex(1);
    REQUIRE(dialog.requiredModelFiles().contains("rmbg_f16.gguf"));
    REQUIRE(!dialog.requiredModelFiles().contains("rmbg_q8.gguf"));
    scene->setChecked(true);
    prompts->setText("chair");
    masks->clear();
    const QString selectedSam = dialog.settings().sam3ModelFile;
    REQUIRE(dialog.requiredModelFiles(true) == QStringList{selectedSam});
    REQUIRE(!dialog.requiredModelFiles().contains("rmbg_f16.gguf"));
    prompts->clear();
    masks->setText(temporary.path());
    REQUIRE(dialog.requiredModelFiles(true).isEmpty());
    REQUIRE(!dialog.requiredModelFiles().contains(selectedSam));
    // UI-only ingestion fixtures: exercise the verified-cache continuation
    // without loading a neural network or claiming numeric inference evidence.
    scene->setChecked(false);
    rmbg->setChecked(false);
    quant->setCurrentIndex(2);
    const QString cache = QString::fromUtf8(aicore_sam3d_model_cache_dir());
    REQUIRE(QDir().mkpath(cache));
    for (const QString& name : dialog.requiredModelFiles()) {
        const QString path = QDir(cache).filePath(name);
        QFile file(path);
        REQUIRE(file.open(QIODevice::WriteOnly));
        REQUIRE(file.write(QByteArray("GGUF") + QByteArray(80 * 1024, 'x')) >
                0);
        file.close();
        const auto* entry =
                aicore_sam3d_model_by_filename(name.toUtf8().constData());
        REQUIRE(entry);
        ecvAssetIntegrity::markVerified(
                path, {QCryptographicHash::Sha256, QByteArray(entry->sha256)});
    }
    image->setText("ui-fixture.png");
    int runs = 0;
    QObject::connect(&dialog, &Sam3dDialog::runRequested, &app,
                     [&](const Sam3dDialog::Settings&) { ++runs; });
    REQUIRE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    REQUIRE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    REQUIRE(runs == 2);
    REQUIRE(!dialog.findChild<ecvModelDownloader*>()->isBusy());
    REQUIRE(!QFile::exists(QDir(cache).filePath("ss_generator-f16.gguf")));
    REQUIRE(QMetaObject::invokeMethod(&dialog, "onDownloadModels"));
    REQUIRE(dialog.findChild<QPushButton*>("downloadModels")->isEnabled());
    // Complete a transport boundary event with the selected file now cached.
    // This verifies that Generate resumes once and failure restores retry UI.
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
    auto* downloader = dialog.findChild<ecvModelDownloader*>();
    dialog.findChild<ecvModelDownloader*>()
            ->findChild<QNetworkAccessManager*>()
            ->setProxy(QNetworkProxy(QNetworkProxy::HttpProxy,
                                     QStringLiteral("127.0.0.1"), 9));
    REQUIRE(QFile::remove(selectedPath));
    REQUIRE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    REQUIRE(downloader->isBusy() && !quant->isEnabled());
    auto* cancel = dialog.findChild<QPushButton*>("cancelTask");
    REQUIRE(cancel && cancel->isEnabled());
    cancel->click();
    REQUIRE(!downloader->isBusy());
    REQUIRE(quant->isEnabled());
    REQUIRE(dialog.findChild<QPushButton*>("downloadModels")->isEnabled());
    REQUIRE(runs == 2);
    REQUIRE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    REQUIRE(downloader->isBusy());
    downloader->blockSignals(true);
    downloader->cancel();
    downloader->blockSignals(false);
    REQUIRE(restoreSelected());
    downloader->finished(true, selectedPath);
    REQUIRE(runs == 3 && quant->isEnabled());
    REQUIRE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    REQUIRE(runs == 4 && !downloader->isBusy());
    // Closing detaches the pending continuation, including a late success.
    REQUIRE(QFile::remove(selectedPath));
    REQUIRE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    REQUIRE(downloader->isBusy());
    dialog.reject();
    REQUIRE(!downloader->isBusy() && quant->isEnabled());
    REQUIRE(restoreSelected());
    downloader->finished(true, selectedPath);
    REQUIRE(runs == 4);
    REQUIRE(QMetaObject::invokeMethod(&dialog, "emitRun"));
    REQUIRE(runs == 5 && !downloader->isBusy());
    std::puts(
            "SAM3D selected dependency and repeated cached Run workflow passed "
            "(UI fixtures only)");
    return 0;
}
