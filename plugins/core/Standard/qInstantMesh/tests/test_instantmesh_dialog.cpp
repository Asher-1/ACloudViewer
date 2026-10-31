// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QAbstractItemView>
#include <QApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QJsonDocument>
#include <QJsonObject>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>
#include <QWindow>
#include <cstdio>

#include "CVLog.h"
#include "InstantMeshDialog.h"
#include "aicore/rmbg_capi.h"

class ConsoleCapture final : public CVLog {
public:
    QStringList messages;
    void logMessage(const QString& message, int) override {
        messages.append(message);
    }
};

// Explicit manual network validation; excluded from default CTest and the
// user's warm data root.
static int test_real_sample_cache(const QString& root) {
    if (root.isEmpty() || QDir(root).exists()) {
        std::fprintf(stderr,
                     "Real cold-cache validation requires a data root that "
                     "does not exist yet\n");
        return 1;
    }
    qputenv("CLOUDVIEWER_DATA_ROOT", root.toUtf8());
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       QDir(root).filePath("settings"));
    const auto kind = ecvTestDataRepository::Dataset::Image2Mesh;
    auto& repo = ecvTestDataRepository::instance();
    int failures = 0;
    const auto check = [&](bool ok, const char* reason) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s\n", reason);
            ++failures;
        }
    };
    check(!repo.isDatasetAvailable(kind),
          "The cold data root must contain neither samples nor a ZIP");
    QEventLoop loop;
    int ready = 0, downloaded = 0;
    bool success = false;
    QObject::connect(&repo, &ecvTestDataRepository::downloadFinished, &loop,
                     [&](bool ok, ecvTestDataRepository::Dataset dataset) {
                         if (dataset == kind && ok) ++downloaded;
                     });
    QObject::connect(&repo, &ecvTestDataRepository::downloadLogMessage, &loop,
                     [](const QString& message) {
                         std::fprintf(stderr, "%s\n",
                                      message.toUtf8().constData());
                     });
    QObject::connect(&repo, &ecvTestDataRepository::datasetReady, &loop,
                     [&](bool ok, ecvTestDataRepository::Dataset dataset) {
                         if (dataset != kind) return;
                         ++ready;
                         success = ok;
                         loop.quit();
                     });
    InstantMeshDialog dialog;
    InstantMeshDialog closed;
    const QString closedInput = closed.getSettings().inputPath;
    check(dialog.getSettings().inputPath.isEmpty(),
          "Old settings must not autofill a cold root");
    QTimer deadline;
    deadline.setSingleShot(true);
    QObject::connect(&deadline, &QTimer::timeout, &loop, [&] {
        repo.cancelDownload();
        loop.quit();
    });
    deadline.start(180000);
    check(QMetaObject::invokeMethod(&dialog, "onTestDataClicked",
                                    Qt::DirectConnection),
          "Invoke the real Use test data widget preparation path");
    QMetaObject::invokeMethod(&dialog, "onTestDataClicked",
                              Qt::DirectConnection);
    // Join the real in-flight shared preparation, then close its consumer.
    closed.show();
    QMetaObject::invokeMethod(&closed, "onTestDataClicked",
                              Qt::DirectConnection);
    closed.close();
    if (ready == 0) loop.exec();
    deadline.stop();
    check(success && ready == 1 && downloaded == 1,
          "Cold download and repeated clicks must coalesce once without "
          "skipping the real download");
    if (!success) return 1;
    const QString sample =
            QDir(ecvTestDataRepository::extractPath(kind))
                    .filePath("examples_images/cartoon_panda.png");
    const auto digest = [](const QString& path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) return QByteArray();
        QCryptographicHash hash(QCryptographicHash::Sha256);
        if (!hash.addData(&file)) return QByteArray();
        return hash.result().toHex();
    };
    const QString archive = ecvTestDataRepository::zipPath(kind);
    const QByteArray zipSha = digest(archive), sampleSha = digest(sample);
    check(QFileInfo(archive).size() == 54046585 &&
                  zipSha ==
                          "490f85a45c37f278bff8011109e91f66488d155c438015e3fdd7"
                          "b95bbc297cac",
          "Retain the release ZIP with its pinned size and digest");
    check(QFileInfo(sample).size() == 1517334 &&
                  sampleSha ==
                          "c82fea6ac66b782b2aa1c6bd133447b5f54f688c7eb44998c4b0"
                          "0f190d47b2b7",
          "The ZIP panda member must match the independently pinned manifest "
          "digest");
    check(dialog.getSettings().inputPath == sample && !QImage(sample).isNull(),
          "Autofill and decode the real default sample after a cold download");
    check(closed.getSettings().inputPath == closedInput,
          "A window closed during preparation must not consume stale sample "
          "results");
    closed.show();
    for (auto* button : closed.findChildren<QPushButton*>())
        if (button->text().contains("sample", Qt::CaseInsensitive))
            check(button->isEnabled(),
                  "Reopening after preparation must restore the sample button");
    closed.close();
    const QString screenshots =
            QString::fromLocal8Bit(qgetenv("ACV_UI_TEST_SCREENSHOTS"));
    if (!screenshots.isEmpty()) QDir().mkpath(screenshots);
    dialog.show();
    QElapsedTimer exposure;
    exposure.start();
    while (dialog.windowHandle() && !dialog.windowHandle()->isExposed() &&
           exposure.elapsed() < 3000)
        QApplication::processEvents(QEventLoop::AllEvents, 20);
    dialog.setProgressStage(1, "Diffusion", 5, 75);
    for (const QSize size : {QSize(640, 480), QSize(800, 600)}) {
        dialog.resize(size);
        QApplication::processEvents();
        check(dialog.size() == size,
              "The real panda UI must fit the requested window size");
        for (auto* scroll : dialog.findChildren<QScrollArea*>()) {
            check(scroll->horizontalScrollBar()->maximum() == 0,
                  "Real catalog and sample labels must not cause horizontal "
                  "clipping");
            for (int position : {0, scroll->verticalScrollBar()->maximum()}) {
                scroll->verticalScrollBar()->setValue(position);
                QApplication::processEvents();
                if (!screenshots.isEmpty())
                    check(dialog.grab().save(
                                  QStringLiteral("%1/instantmesh-%2x%3-%4.png")
                                          .arg(screenshots)
                                          .arg(size.width())
                                          .arg(size.height())
                                          .arg(position == 0 ? "top"
                                                             : "bottom")),
                          "Save native panda configuration and fixed progress "
                          "evidence");
            }
        }
    }
    dialog.setRunning(false);
    QMetaObject::invokeMethod(&dialog, "onTestDataClicked",
                              Qt::DirectConnection);
    check(success && ready == 2 && downloaded == 1,
          "Warm-cache reuse must complete synchronously without a second "
          "download");
    check(digest(archive) == zipSha && digest(sample) == sampleSha &&
                  !repo.isDownloadInProgress(),
          "Warm reuse must not change cache contents or leave a download in "
          "flight");
    QFile brokenMember(sample);
    check(brokenMember.open(QIODevice::WriteOnly),
          "Failed to write the corrupt-member negative fixture");
    brokenMember.write("invalid PNG");
    brokenMember.close();
    QMetaObject::invokeMethod(&dialog, "onTestDataClicked",
                              Qt::DirectConnection);
    const QString fallback = dialog.getSettings().inputPath;
    check(success && ready == 3 && downloaded == 1 && fallback != sample &&
                  !QImage(fallback).isNull(),
          "A corrupt default member must fall back to a decodable candidate, "
          "not fill a corrupt panda member");
    check(repo.extractDataset(kind),
          "The retained ZIP must restore corrupt members offline");
    QMetaObject::invokeMethod(&dialog, "onTestDataClicked",
                              Qt::DirectConnection);
    check(success && ready == 4 && downloaded == 1 &&
                  dialog.getSettings().inputPath == sample &&
                  digest(sample) == sampleSha && digest(archive) == zipSha,
          "Restoring from the retained ZIP must autofill the pinned default "
          "member again");
    QTemporaryDir bad(QDir(root).filePath("negative-XXXXXX"));
    check(bad.isValid(),
          "Failed to create the corrupt-cache negative directory");
    if (!bad.isValid()) return 1;
    qputenv("CLOUDVIEWER_DATA_ROOT", bad.path().toUtf8());
    QDir().mkpath(ecvTestDataRepository::downloadDir());
    QFile brokenZip(ecvTestDataRepository::zipPath(kind));
    check(brokenZip.open(QIODevice::WriteOnly),
          "Failed to write the corrupt-ZIP negative fixture");
    brokenZip.write("not a ZIP");
    brokenZip.close();
    check(!repo.isDatasetAvailable(kind) && !repo.extractDataset(kind),
          "A corrupt ZIP must be rejected for readiness and extraction");
    qputenv("CLOUDVIEWER_DATA_ROOT", root.toUtf8());
    QJsonObject receipt{{"kind", "real-qt-sample-cache-not-native-material"},
                        {"qpa_platform", QGuiApplication::platformName()},
                        {"qt_version", QString::fromLatin1(qVersion())},
                        {"data_root", root},
                        {"sample", sample},
                        {"zip_sha256", QString::fromLatin1(zipSha)},
                        {"sample_sha256", QString::fromLatin1(sampleSha)},
                        {"ready_count", ready},
                        {"downloads", downloaded},
                        {"failures", failures}};
    std::puts(
            QJsonDocument(receipt).toJson(QJsonDocument::Compact).constData());
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    ConsoleCapture console;
    CVLog::RegisterInstance(&console);
    app.setOrganizationName(QStringLiteral("InstantMeshContract"));
    app.setApplicationName(QStringLiteral("Dialog"));
    if (argc == 3 && QString::fromLocal8Bit(argv[1]) == "--sample-cache") {
        const int result =
                test_real_sample_cache(QString::fromLocal8Bit(argv[2]));
        CVLog::RegisterInstance(nullptr);
        return result;
    }
    QTemporaryDir tmp;
    qputenv("CLOUDVIEWER_DATA_ROOT", tmp.path().toUtf8());
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       tmp.filePath("settings"));
    const auto kind = ecvTestDataRepository::Dataset::Image2Mesh;
    const QString directory =
            ecvTestDataRepository::extractPath(kind) + "/examples_images";
    if (!QDir().mkpath(directory)) return 1;
    QImage sample(8, 8, QImage::Format_RGB888);
    sample.fill(Qt::green);
    QFile corrupt(directory + "/0.png");
    if (!corrupt.open(QIODevice::WriteOnly)) return 1;
    corrupt.write("invalid PNG");
    corrupt.close();
    int failures = 0;
    const auto check = [&](bool ok, const char* reason) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s\n", reason);
            ++failures;
        }
    };
    check(!ecvTestDataRepository::instance().isDatasetAvailable(kind),
          "Image2Mesh must not be ready with only corrupt images and no valid "
          "ZIP");
    if (!sample.save(directory + "/A.png")) return 1;
    check(ecvTestDataRepository::instance().isDatasetAvailable(kind),
          "Allow offline reuse when cartoon_panda.png is absent but a "
          "decodable candidate "
          "exists");
    {
        InstantMeshDialog dialog;
        console.messages.clear();
        dialog.appendLog("[InstantMesh] Console routing probe");
        check(console.messages.count("[InstantMesh] Console routing probe") ==
                      1,
              "A plugin log must reach the application console exactly once");
        dialog.show();
        QElapsedTimer exposure;
        exposure.start();
        while (dialog.windowHandle() && !dialog.windowHandle()->isExposed() &&
               exposure.elapsed() < 3000)
            QApplication::processEvents(QEventLoop::AllEvents, 20);
        dialog.setProgressStage(1, "Diffusion", 5, 75);
        for (const QSize size : {QSize(640, 480), QSize(800, 600)}) {
            dialog.resize(size);
            QApplication::processEvents();
            check(dialog.size() == size,
                  "The dialog must fit the requested window size");
            for (auto* scroll : dialog.findChildren<QScrollArea*>()) {
                if (scroll->horizontalScrollBar()->maximum() != 0) {
                    std::fprintf(stderr,
                                 "Window=%dx%d viewport=%d content_min=%d "
                                 "content=%d\n",
                                 size.width(), size.height(),
                                 scroll->viewport()->width(),
                                 scroll->widget()->minimumSizeHint().width(),
                                 scroll->widget()->width());
                    for (auto* group : scroll->findChildren<QGroupBox*>()) {
                        std::fprintf(stderr, "Group %s minimum=%d hint=%d\n",
                                     group->title().toUtf8().constData(),
                                     group->minimumSizeHint().width(),
                                     group->sizeHint().width());
                        if (auto* grid = qobject_cast<QGridLayout*>(
                                    group->layout())) {
                            for (int i = 0; i < grid->count(); ++i) {
                                int row, column, rows, columns;
                                grid->getItemPosition(i, &row, &column, &rows,
                                                      &columns);
                                std::fprintf(
                                        stderr,
                                        "Grid item r%d c%d span%dx%d "
                                        "minimum=%d\n",
                                        row, column, rows, columns,
                                        grid->itemAt(i)->minimumSize().width());
                            }
                        }
                    }
                    for (auto* combo : scroll->findChildren<QComboBox*>())
                        std::fprintf(stderr, "Combo %s minimum=%d hint=%d\n",
                                     combo->currentText().toUtf8().constData(),
                                     combo->minimumSizeHint().width(),
                                     combo->sizeHint().width());
                    for (auto* box : scroll->findChildren<QCheckBox*>())
                        std::fprintf(stderr, "Checkbox %s minimum=%d hint=%d\n",
                                     box->text().toUtf8().constData(),
                                     box->minimumSizeHint().width(),
                                     box->sizeHint().width());
                }
                check(scroll->horizontalScrollBar()->maximum() == 0,
                      "Compact configuration must not hide fields behind "
                      "horizontal scrolling");
                for (auto* field : scroll->widget()->findChildren<QWidget*>()) {
                    if (!(qobject_cast<QComboBox*>(field) ||
                          qobject_cast<QCheckBox*>(field) ||
                          qobject_cast<QAbstractSpinBox*>(field) ||
                          qobject_cast<QPushButton*>(field)) ||
                        !field->isVisibleTo(scroll->widget()))
                        continue;
                    const QPoint origin =
                            field->mapTo(scroll->viewport(), QPoint());
                    check(origin.x() >= 0 &&
                                  origin.x() + field->width() <=
                                          scroll->viewport()->width(),
                          "Every configuration field must fit horizontally "
                          "inside the viewport");
                }
                for (auto* combo :
                     scroll->widget()->findChildren<QComboBox*>()) {
                    scroll->ensureWidgetVisible(combo);
                    combo->showPopup();
                    QApplication::processEvents();
                    check(combo->view()->sizeHintForColumn(
                                  combo->modelColumn()) <=
                                  combo->view()->viewport()->width(),
                          "The popup must keep full catalog labels readable in "
                          "a narrow form");
                    combo->hidePopup();
                }
            }
            for (auto* button : dialog.findChildren<QPushButton*>()) {
                if (button->text() == "Generate 3D Mesh" ||
                    button->text() == "Cancel" ||
                    button->text().contains("sample", Qt::CaseInsensitive)) {
                    const QRect rect(button->mapTo(&dialog, QPoint()),
                                     button->size());
                    check(button->isVisible() && dialog.rect().contains(rect) &&
                                  !button->visibleRegion().isEmpty(),
                          "Sample and run/cancel actions must stay visible "
                          "without scrolling");
                }
            }
            auto* progress = dialog.findChild<QProgressBar*>();
            check(progress && progress->isVisible() &&
                          dialog.rect().contains(
                                  QRect(progress->mapTo(&dialog, QPoint()),
                                        progress->size())) &&
                          !progress->visibleRegion().isEmpty(),
                  "Active progress must stay visible without scrolling");
        }
        dialog.setRunning(false);
        check(dialog.getSettings().inputPath.endsWith("/A.png"),
              "Select the first sorted decodable candidate when "
              "cartoon_panda.png is "
              "absent");
        QComboBox* rmbg = nullptr;
        for (auto* combo : dialog.findChildren<QComboBox*>())
            if (combo->findData(QStringLiteral("rmbg_f16.gguf")) >= 0)
                rmbg = combo;
        check(rmbg != nullptr, "Missing RMBG selection widget");
        if (rmbg) {
            check(rmbg->count() == aicore_rmbg_model_count(),
                  "RMBG precision choices must project the entire public "
                  "catalog");
            for (int i = 0; i < rmbg->count(); ++i) {
                rmbg->setCurrentIndex(i);
                check(dialog.getSettings().rmbgModelPath.endsWith(
                              rmbg->currentData().toString()),
                      "Settings must carry the selected RMBG file");
            }
        }
        if (!sample.save(directory + "/cartoon_panda.png")) return 1;
        ecvTestDataRepository::instance().ensureDatasetReady(kind);
        check(dialog.getSettings().inputPath.endsWith("/cartoon_panda.png"),
              "Shared datasetReady must autofill the prepared sample");
        int canceled = 0;
        QObject::connect(&dialog, &InstantMeshDialog::cancelRequested,
                         [&] { ++canceled; });
        dialog.show();
        dialog.setRunning(true);
        dialog.close();
        check(!dialog.isVisible() && canceled == 1,
              "Closing while running must request cancellation and stop "
              "receiving results");
        dialog.setRunning(false);
    }
    if (!sample.save(directory + "/cartoon_panda.png") ||
        !sample.save(directory + "/T.png"))
        return 1;
    {
        InstantMeshDialog dialog;
        check(dialog.getSettings().inputPath.endsWith("/cartoon_panda.png"),
              "The default sample must be the decodable ZIP panda member");
        check(QImage(dialog.getSettings().inputPath).pixelColor(0, 0) ==
                      QColor(Qt::green),
              "Autofill must point to the actual sample pixels");
        const QString beforeReject = dialog.getSettings().inputPath;
        QMetaObject::invokeMethod(&dialog, "reject", Qt::DirectConnection);
        QFile broken(directory + "/cartoon_panda.png");
        if (!broken.open(QIODevice::WriteOnly)) return 1;
        broken.write("invalid PNG");
        broken.close();
        ecvTestDataRepository::instance().ensureDatasetReady(kind);
        check(dialog.getSettings().inputPath == beforeReject,
              "Escape/reject must detach the closed consumer from shared "
              "sample results");
        if (!sample.save(directory + "/cartoon_panda.png")) return 1;
    }
    std::puts(
            QJsonDocument(
                    QJsonObject{
                            {"kind",
                             "dialog-layout-workflow-not-native-material"},
                            {"qpa_platform", QGuiApplication::platformName()},
                            {"qt_version", QString::fromLatin1(qVersion())},
                            {"qt_scale_factor", QString::fromLocal8Bit(qgetenv(
                                                        "QT_SCALE_FACTOR"))},
                            {"failures", failures}})
                    .toJson(QJsonDocument::Compact)
                    .constData());
    CVLog::RegisterInstance(nullptr);
    return failures ? 1 : 0;
}
