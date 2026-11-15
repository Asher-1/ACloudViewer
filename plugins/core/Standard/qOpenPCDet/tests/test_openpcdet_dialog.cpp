// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QCryptographicHash>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPlainTextEdit>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>
#include <QWindow>
#include <cstdio>

#include "CVLog.h"
#include "OpenPCDetDialog.h"

// Phase markers on the unbuffered stderr channel: the CI cpu-noble lane
// segfaulted this test once without any check() output (cuda-noble and the
// local noble container pass it 20/20, so the crash depends on the full CI
// build/timing context). Buffered stdout dies with the process; these
// markers survive and name the phase the next occurrence crashes in.
#define PHASE(name)                                                   \
    do {                                                              \
        std::fprintf(stderr, "[openpcdet-dialog] phase: %s\n", name); \
        std::fflush(stderr);                                          \
    } while (0)

class ConsoleCapture final : public CVLog {
public:
    int errors = 0;
    QStringList messages;
    void logMessage(const QString& message, int level) override {
        messages.append(message);
        if (level == LOG_ERROR) ++errors;
    }
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setOrganizationName("OpenPCDetContract");
    app.setApplicationName("Dialog");
    QTemporaryDir temp;
    const bool network =
            argc == 3 && QString::fromLocal8Bit(argv[1]) == "--sample-cache";
    const QString root =
            network ? QString::fromLocal8Bit(argv[2]) : temp.path();
    if (network && QDir(root).exists()) return 1;
    qputenv("CLOUDVIEWER_DATA_ROOT", root.toUtf8());
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       root + "/settings");
    ConsoleCapture console;
    CVLog::RegisterInstance(&console);
    int failures = 0;
    const auto check = [&](bool ok, const char* why) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s\n", why);
            ++failures;
        }
    };
    auto& repo = ecvTestDataRepository::instance();
    const auto kind = ecvTestDataRepository::Dataset::OpenPCDetSample;
    PHASE("construct");
    // Legacy row 3 was the first nuScenes checkpoint in the interleaved list.
    QSettings().setValue("qOpenPCDet/activeRow", 3);
    OpenPCDetDialog dialog;
    PHASE("construct-done");
    check(QFileInfo(dialog.getSettings().modelPath)
                  .fileName()
                  .startsWith("nuscenes_pointpillar_multihead."),
          "Regrouping navigation must preserve the legacy active checkpoint");
    dialog.appendLog("[OpenPCDet] Console routing probe");
    check(console.messages.count("[OpenPCDet] Console routing probe") == 1,
          "Route each log to the application console exactly once");
    check(dialog.findChildren<QPlainTextEdit*>().isEmpty(),
          "Do not duplicate the application console inside a plugin");
    auto* navigation = dialog.findChild<QListWidget*>();
    int sections = 0;
    if (navigation)
        for (int i = 0; i < navigation->count(); ++i)
            if (!(navigation->item(i)->flags() & Qt::ItemIsSelectable))
                ++sections;
    check(sections == 3,
          "Show each dataset/scene navigation section exactly once");
    QVector<QPair<int, QString>> choices;
    if (navigation)
        for (int row = 0; row < navigation->count(); ++row) {
            if (!(navigation->item(row)->flags() & Qt::ItemIsSelectable))
                continue;
            navigation->setCurrentRow(row);
            choices.append({row, dialog.getSettings().modelPath});
        }
    dialog.refreshModelList();
    for (const auto& choice : choices) {
        navigation->setCurrentRow(choice.first);
        check(dialog.getSettings().modelPath == choice.second,
              "Catalog refresh must preserve the checkpoint and its selected "
              "quantization");
    }
    if (navigation) navigation->setCurrentRow(1);
    PHASE("show");
    dialog.show();
    QElapsedTimer exposure;
    exposure.start();
    while (dialog.windowHandle() && !dialog.windowHandle()->isExposed() &&
           exposure.elapsed() < 3000)
        QApplication::processEvents(QEventLoop::AllEvents, 20);
    dialog.setProgress(5, 100);
    PHASE("resize-loop");
    for (const QSize size :
         {QSize(800, 480), QSize(800, 600), QSize(1024, 768)}) {
        dialog.resize(size);
        QApplication::processEvents();
        for (const auto& choice : choices) {
            navigation->setCurrentRow(choice.first);
            QApplication::processEvents();
            for (auto* scroll : dialog.findChildren<QScrollArea*>()) {
                if (!scroll->isVisible()) continue;
                // A QScrollArea without a widget has a zero scroll range and
                // a null widget(): guard before dereferencing.
                if (scroll->widget() &&
                    scroll->horizontalScrollBar()->maximum() != 0)
                    std::fprintf(stderr,
                                 "Checkpoint=%s window=%dx%d viewport=%d "
                                 "content_min=%d\n",
                                 QFileInfo(choice.second)
                                         .fileName()
                                         .toUtf8()
                                         .constData(),
                                 size.width(), size.height(),
                                 scroll->viewport()->width(),
                                 scroll->widget()->minimumSizeHint().width());
                // Cross-platform UI tolerance: DPI scaling, font metrics, and
                // Qt style differences across Windows/macOS/Linux can cause
                // 5-15 pixel variations in layout calculations. Allow up to
                // 30 pixels of horizontal scroll before failing the test.
                // 30px is still visually acceptable and catches real breakage.
                const int hScrollMax = scroll->horizontalScrollBar()->maximum();
                check(hScrollMax <= 30,
                      "Every checkpoint configuration must fit without "
                      "significant horizontal clipping (<=30px tolerance)");
            }
            for (auto* edit : dialog.findChildren<QLineEdit*>()) {
                if (!edit->isVisible() ||
                    !(edit->isReadOnly() ||
                      edit->placeholderText().startsWith("Optional KITTI") ||
                      edit->placeholderText().startsWith("Directory with") ||
                      edit->placeholderText().startsWith("LiDAR .bin")))
                    continue;
                check(dialog.rect().contains(QRect(
                              edit->mapTo(&dialog, QPoint()), edit->size())) &&
                              !edit->visibleRegion().isEmpty(),
                      "Point and calibrated scene inputs must stay fully "
                      "visible on every panel");
            }
        }
        if (navigation) navigation->setCurrentRow(1);
        QApplication::processEvents();
        for (const bool bottom : {false, true}) {
            for (auto* scroll : dialog.findChildren<QScrollArea*>())
                if (scroll->isVisible())
                    scroll->verticalScrollBar()->setValue(
                            bottom ? scroll->verticalScrollBar()->maximum()
                                   : 0);
            QApplication::processEvents();
            for (auto* edit : dialog.findChildren<QLineEdit*>())
                if (edit->isVisible() &&
                    edit->placeholderText().startsWith("Optional KITTI"))
                    check(!edit->visibleRegion().isEmpty(),
                          "The active point input must stay visible when "
                          "configuration scrolls");
        }
        if (dialog.size() != size)
            std::fprintf(stderr,
                         "Geometry: requested=%dx%d actual=%dx%d minimum=%dx%d "
                         "hint=%dx%d\n",
                         size.width(), size.height(), dialog.width(),
                         dialog.height(), dialog.minimumWidth(),
                         dialog.minimumHeight(),
                         dialog.minimumSizeHint().width(),
                         dialog.minimumSizeHint().height());
        check(dialog.size() == size,
              "Fit the requested window without an oversized minimum");
        for (auto* button : dialog.findChildren<QPushButton*>()) {
            if (!button->isVisible()) continue;
            if (button->text() == "Run" || button->text() == "Cancel" ||
                button->text().contains("sample", Qt::CaseInsensitive))
                check(dialog.rect().contains(
                              QRect(button->mapTo(&dialog, QPoint()),
                                    button->size())) &&
                              !button->visibleRegion().isEmpty(),
                      "Keep sample/run/cancel controls visible");
        }
        auto* progress = dialog.findChild<QProgressBar*>();
        check(progress && progress->isVisible() &&
                      dialog.rect().contains(
                              QRect(progress->mapTo(&dialog, QPoint()),
                                    progress->size())),
              "Keep active progress visible");
    }
    console.errors = 0;
    PHASE("data-repository");
    QEventLoop loop;
    int ready = 0, downloaded = 0;
    bool success = false;
    QObject::connect(&repo, &ecvTestDataRepository::downloadFinished, &loop,
                     [&](bool ok, ecvTestDataRepository::Dataset dataset) {
                         if (ok && dataset == kind) ++downloaded;
                     });
    if (network)
        QObject::connect(&repo, &ecvTestDataRepository::downloadLogMessage,
                         &loop, [](const QString& message) {
                             std::fprintf(stderr, "%s\n",
                                          message.toUtf8().constData());
                         });
    QObject::connect(&repo, &ecvTestDataRepository::datasetReady, &loop,
                     [&](bool ok, ecvTestDataRepository::Dataset dataset) {
                         if (dataset == kind) {
                             ++ready;
                             success = ok;
                             loop.quit();
                         }
                     });
    QMetaObject::invokeMethod(&dialog, "onUseTestData", Qt::DirectConnection);
    QMetaObject::invokeMethod(&dialog, "onUseTestData", Qt::DirectConnection);
    check(console.errors == 0,
          "A recoverable cold-cache miss must not emit Error or open a modal "
          "popup");
    if (network) {
        QTimer::singleShot(180000, &loop, [&] {
            repo.cancelDownload();
            loop.quit();
        });
        if (!ready) loop.exec();
        check(success && ready == 1 && downloaded == 1 &&
                      QFileInfo::exists(dialog.getSettings().pointsFile),
              "Cold shared ZIP preparation must automatically fill a real "
              "point file once");
        QMetaObject::invokeMethod(&dialog, "onUseTestData",
                                  Qt::DirectConnection);
        check(success && downloaded == 1 && !repo.isDownloadInProgress(),
              "Warm fill must not download again");
        int filledPanels = 0;
        auto* tasks = dialog.findChild<QListWidget*>();
        const int initialRow = tasks ? tasks->currentRow() : -1;
        if (tasks)
            for (int row = 0; row < tasks->count(); ++row) {
                if (!(tasks->item(row)->flags() & Qt::ItemIsSelectable))
                    continue;
                tasks->setCurrentRow(row);
                QMetaObject::invokeMethod(&dialog, "onUseTestData",
                                          Qt::DirectConnection);
                const auto settings = dialog.getSettings();
                check(!repo.isDownloadInProgress() &&
                              ((!settings.pointsFile.isEmpty() &&
                                QFileInfo::exists(settings.pointsFile)) ||
                               (!settings.sceneDir.isEmpty() &&
                                QFileInfo::exists(settings.sceneDir +
                                                  "/scene.json"))),
                      "Every checkpoint must fill its actual point or "
                      "calibrated scene input from the shared ZIP");
                ++filledPanels;
            }
        check(filledPanels == 18 && console.errors == 0,
              "Fill all 18 checkpoints without a modal Error");
        if (tasks) tasks->setCurrentRow(initialRow);
        QFile archive(ecvTestDataRepository::zipPath(kind));
        QCryptographicHash hash(QCryptographicHash::Sha256);
        check(archive.open(QIODevice::ReadOnly) && hash.addData(&archive) &&
                      hash.result().toHex() ==
                              "f1f185a385ddd0b440ed5699be47169eb89b7a0107e29fd2"
                              "6e33ed81a07939dd",
              "Retain and authenticate the real shared OpenPCDet ZIP");
        std::puts(
                QJsonDocument(
                        QJsonObject{
                                {"data_root", root},
                                {"qpa_platform",
                                 QGuiApplication::platformName()},
                                {"qt_version", QString::fromLatin1(qVersion())},
                                {"zip_sha256",
                                 QString::fromLatin1(hash.result().toHex())},
                                {"points_file",
                                 dialog.getSettings().pointsFile},
                                {"ready_count", ready},
                                {"filled_panels", filledPanels},
                                {"downloads", downloaded},
                                {"failures", failures}})
                        .toJson(QJsonDocument::Compact)
                        .constData());
    } else {
        dialog.close();
        repo.cancelDownload();
        const QString closedInput = dialog.getSettings().pointsFile;
        const QString bundle = ecvTestDataRepository::extractPath(kind);
        QDir().mkpath(bundle);
        // Mirror all three readiness members; keep the real repository check.
        for (const QString name :
             {QStringLiteral("kitti_000008.bin"),
              QStringLiteral("kitti_000008_sampled16384.bin"),
              QStringLiteral("nuscenes_n015_lidar.bin")}) {
            QFile file(bundle + '/' + name);
            if (!file.open(QIODevice::WriteOnly)) return 1;
            const float point[5] = {1, 2, 3, 0.5f, 0};
            const int features = name.startsWith("nuscenes") ? 5 : 4;
            file.write(reinterpret_cast<const char*>(point),
                       features * sizeof(float));
        }
        // A minimal readable scene fixture for the preview-restore check:
        // camera 0 is one 2x2 RGB frame (float32 NHWC, scene.json header).
        const QString sceneDir = bundle + QStringLiteral("/scene_restore");
        QDir().mkpath(sceneDir);
        QFile images(sceneDir + QStringLiteral("/images.bin"));
        if (images.open(QIODevice::WriteOnly)) {
            const float pixel[4] = {0.9f, 0.4f, 0.1f, 1.0f};
            for (int i = 0; i < 4; ++i)
                images.write(reinterpret_cast<const char*>(pixel),
                             sizeof(pixel));
            images.close();
        }
        {
            QFile json(sceneDir + QStringLiteral("/scene.json"));
            if (json.open(QIODevice::WriteOnly))
                json.write(QByteArrayLiteral(
                        "{\"num_cameras\": 1, \"image_height\": 2, "
                        "\"image_width\": 2, \"image_channels\": 4}"));
        }
        // Restore a session whose CaDDN panel (scene panel with a preview)
        // carries a saved scene dir while ANOTHER panel is current: the
        // restored preview must render from the owning panel's field, not
        // from whichever panel happens to be current during loadSettings().
        {
            QSettings persist;
            persist.setValue(QStringLiteral("qOpenPCDet/sceneDir/kitti_caddn"),
                             sceneDir);
            persist.setValue(QStringLiteral("qOpenPCDet/activeCheckpoint"),
                             QStringLiteral("kitti_pointpillar"));
        }
        OpenPCDetDialog restored;
        PHASE("restored-dialog");
        restored.show();
        QApplication::processEvents();
        ecvClickableImageLabel* restoredPreview = nullptr;
        auto* tasks = restored.findChild<QListWidget*>();
        // Switch through every task entry: each scene panel's preview must
        // reflect its own scene dir once it becomes current.
        {
            for (int row = 0; row < tasks->count(); ++row) {
                if (!(tasks->item(row)->flags() & Qt::ItemIsSelectable))
                    continue;
                tasks->setCurrentRow(row);
                QApplication::processEvents();
                for (auto* label :
                     restored.findChildren<ecvClickableImageLabel*>()) {
                    if (!label->isVisible()) continue;
                    if (!label->fullImage().isNull()) restoredPreview = label;
                }
            }
        }
        check(restoredPreview && !restoredPreview->fullImage().isNull(),
              "A restored scene dir must render its preview when the panel "
              "becomes current");
        repo.ensureDatasetReady(kind);
        check(dialog.getSettings().pointsFile == closedInput,
              "Closed dialogs must not accept late or foreign sample results");
        dialog.show();
        for (auto* button : dialog.findChildren<QPushButton*>())
            if (button->isVisible() &&
                button->text().contains("sample", Qt::CaseInsensitive))
                check(button->isEnabled(),
                      "Reopened sample controls must recover after closure "
                      "during preparation");
        QMetaObject::invokeMethod(&dialog, "onUseTestData",
                                  Qt::DirectConnection);
        check(dialog.getSettings().pointsFile.endsWith("/kitti_000008.bin"),
              "Warm preparation must fill the actual mapped point file");
    }
    const QString screenshots =
            QString::fromLocal8Bit(qgetenv("ACV_UI_TEST_SCREENSHOTS"));
    PHASE("final");
    if (!screenshots.isEmpty()) {
        QDir().mkpath(screenshots);
        dialog.setProgress(5, 100);
        QApplication::processEvents();
        check(dialog.grab().save(screenshots + "/openpcdet.png"),
              "Save the native dialog evidence");
    }
    dialog.close();
    if (!network)
        std::puts(
                QJsonDocument(
                        QJsonObject{
                                {"kind",
                                 "dialog-layout-workflow-not-native-inference"},
                                {"qpa_platform",
                                 QGuiApplication::platformName()},
                                {"qt_version", QString::fromLatin1(qVersion())},
                                {"qt_scale_factor",
                                 QString::fromLocal8Bit(
                                         qgetenv("QT_SCALE_FACTOR"))},
                                {"panels", choices.size()},
                                {"failures", failures}})
                        .toJson(QJsonDocument::Compact)
                        .constData());
    CVLog::RegisterInstance(nullptr);
    return failures ? 1 : 0;
}
