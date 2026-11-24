// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QApplication>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPlainTextEdit>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QSpinBox>
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

namespace {
QString coldSampleRoot;
int observedPanelCount = 0;
}  // namespace

class OpenPCDetDialogTest : public ::testing::Test {
protected:
    QTemporaryDir temp;
    ConsoleCapture console;
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    static constexpr auto kind =
            ecvTestDataRepository::Dataset::OpenPCDetSample;
    QString root;
    void SetUp() override {
        ASSERT_TRUE(temp.isValid());
        root = temp.path();
        qputenv("CLOUDVIEWER_DATA_ROOT", root.toUtf8());
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                           root + "/settings");
        CVLog::RegisterInstance(&console);
    }
    void TearDown() override {
        repo.cancelDownload();
        CVLog::RegisterInstance(nullptr);
    }
};

TEST_F(OpenPCDetDialogTest,
       NavigationRefreshPreservesCheckpointAndModelDefaults) {
    PHASE("construct");
    // Legacy row 3 was the first nuScenes checkpoint in the interleaved list.
    QSettings().setValue("qOpenPCDet/activeRow", 3);
    // Stale follow-model display regression: positive threshold values
    // persisted by an older session must not override the GGUF-declared
    // defaults shown while "Use model defaults" is checked.
    QSettings().setValue("qOpenPCDet/score/nuscenes_pointpillar_multihead",
                         0.5);
    QSettings().setValue("qOpenPCDet/nms/nuscenes_pointpillar_multihead", 0.5);
    QSettings().setValue("qOpenPCDet/maxBoxes/nuscenes_pointpillar_multihead",
                         500);
    QSettings().setValue(
            "qOpenPCDet/useDefaults/nuscenes_pointpillar_multihead", true);
    OpenPCDetDialog dialog;
    PHASE("construct-done");
    EXPECT_TRUE((QFileInfo(dialog.getSettings().modelPath)
                         .fileName()
                         .startsWith("nuscenes_pointpillar_multihead.")))
            << "Regrouping navigation must preserve the legacy active "
               "checkpoint";
    dialog.appendLog("[OpenPCDet] Console routing probe");
    EXPECT_TRUE(
            (console.messages.count("[OpenPCDet] Console routing probe") == 1))
            << "Route each log to the application console exactly once";
    EXPECT_TRUE((dialog.findChildren<QPlainTextEdit*>().isEmpty()))
            << "Do not duplicate the application console inside a plugin";
    auto* navigation = dialog.findChild<QListWidget*>();
    int sections = 0;
    if (navigation)
        for (int i = 0; i < navigation->count(); ++i)
            if (!(navigation->item(i)->flags() & Qt::ItemIsSelectable))
                ++sections;
    EXPECT_TRUE((sections == 3))
            << "Show each dataset/scene navigation section exactly once";
    QVector<QPair<int, QString>> choices;
    if (navigation)
        for (int row = 0; row < navigation->count(); ++row) {
            if (!(navigation->item(row)->flags() & Qt::ItemIsSelectable))
                continue;
            navigation->setCurrentRow(row);
            choices.append({row, dialog.getSettings().modelPath});
        }
    dialog.refreshModelList();
    observedPanelCount = choices.size();
    for (const auto& choice : choices) {
        navigation->setCurrentRow(choice.first);
        EXPECT_TRUE((dialog.getSettings().modelPath == choice.second))
                << "Catalog refresh must preserve the checkpoint and its "
                   "selected "
                   "quantization";
    }
    if (navigation) navigation->setCurrentRow(1);
    PHASE("show");
    dialog.show();
    QElapsedTimer exposure;
    exposure.start();
    while (dialog.windowHandle() && !dialog.windowHandle()->isExposed() &&
           exposure.elapsed() < 3000)
        QApplication::processEvents(QEventLoop::AllEvents, 20);
    // The stale persisted overrides written before construction must not be
    // displayed while the multihead panel follows the model defaults (no
    // GGUF in this temp root, so the spins stay at the "model" placeholder).
    if (navigation)
        for (int row = 0; row < navigation->count(); ++row) {
            if (!(navigation->item(row)->flags() & Qt::ItemIsSelectable))
                continue;
            navigation->setCurrentRow(row);
            if (!QFileInfo(dialog.getSettings().modelPath)
                         .fileName()
                         .startsWith("nuscenes_pointpillar_multihead."))
                continue;
            for (auto* spin : dialog.findChildren<QDoubleSpinBox*>())
                if (spin->isVisible())
                    EXPECT_TRUE((spin->value() == 0.0))
                            << "Follow-model mode must not display stale "
                               "persisted "
                               "score/NMS overrides";
            for (auto* spin : dialog.findChildren<QSpinBox*>())
                if (spin->isVisible() &&
                    spin->specialValueText() == QStringLiteral("model"))
                    EXPECT_TRUE((spin->value() == 0))
                            << "Follow-model mode must not display a stale "
                               "persisted max-boxes override";
            break;
        }
}

TEST_F(OpenPCDetDialogTest,
       EveryCheckpointFitsInputsActionsAndProgressAtSupportedSizes) {
    OpenPCDetDialog dialog;
    auto* navigation = dialog.findChild<QListWidget*>();
    ASSERT_NE(navigation, nullptr);
    QVector<QPair<int, QString>> choices;
    if (navigation)
        for (int row = 0; row < navigation->count(); ++row) {
            if (!(navigation->item(row)->flags() & Qt::ItemIsSelectable))
                continue;
            navigation->setCurrentRow(row);
            choices.append({row, dialog.getSettings().modelPath});
        }
    dialog.refreshModelList();
    observedPanelCount = choices.size();
    for (const auto& choice : choices) {
        navigation->setCurrentRow(choice.first);
        EXPECT_TRUE((dialog.getSettings().modelPath == choice.second))
                << "Catalog refresh must preserve the checkpoint and its "
                   "selected "
                   "quantization";
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
                EXPECT_TRUE((hScrollMax <= 30))
                        << "Every checkpoint configuration must fit without "
                           "significant horizontal clipping (<=30px tolerance)";
            }
            for (auto* edit : dialog.findChildren<QLineEdit*>()) {
                if (!edit->isVisible() ||
                    !(edit->isReadOnly() ||
                      edit->placeholderText().startsWith("Optional KITTI") ||
                      edit->placeholderText().startsWith("Directory with") ||
                      edit->placeholderText().startsWith("LiDAR .bin")))
                    continue;
                EXPECT_TRUE((dialog.rect().contains(
                                     QRect(edit->mapTo(&dialog, QPoint()),
                                           edit->size())) &&
                             !edit->visibleRegion().isEmpty()))
                        << "Point and calibrated scene inputs must stay fully "
                           "visible on every panel";
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
                    EXPECT_TRUE((!edit->visibleRegion().isEmpty()))
                            << "The active point input must stay visible when "
                               "configuration scrolls";
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
        EXPECT_TRUE((dialog.size() == size))
                << "Fit the requested window without an oversized minimum";
        for (auto* button : dialog.findChildren<QPushButton*>()) {
            if (!button->isVisible()) continue;
            if (button->text() == "Run" || button->text() == "Cancel" ||
                button->text().contains("sample", Qt::CaseInsensitive))
                EXPECT_TRUE((dialog.rect().contains(
                                     QRect(button->mapTo(&dialog, QPoint()),
                                           button->size())) &&
                             !button->visibleRegion().isEmpty()))
                        << "Keep sample/run/cancel controls visible";
        }
        auto* progress = dialog.findChild<QProgressBar*>();
        EXPECT_TRUE((
                progress && progress->isVisible() &&
                dialog.rect().contains(QRect(progress->mapTo(&dialog, QPoint()),
                                             progress->size()))))
                << "Keep active progress visible";
    }
    const QString screenshots =
            QString::fromLocal8Bit(qgetenv("ACV_UI_TEST_SCREENSHOTS"));
    PHASE("final");
    if (!screenshots.isEmpty()) {
        QDir().mkpath(screenshots);
        dialog.setProgress(5, 100);
        QApplication::processEvents();
        EXPECT_TRUE((dialog.grab().save(screenshots + "/openpcdet.png")))
                << "Save the native dialog evidence";
    }
    dialog.close();
}

TEST_F(OpenPCDetDialogTest,
       ClosedSampleConsumerIgnoresLateResultsAndRestoresOwningScenePreview) {
    OpenPCDetDialog dialog;
    auto* navigation = dialog.findChild<QListWidget*>();
    ASSERT_NE(navigation, nullptr);
    navigation->setCurrentRow(1);
    dialog.show();
    console.errors = 0;
    PHASE("data-repository");
    QEventLoop loop;
    int ready = 0, downloaded = 0;
    bool success = false;
    QObject::connect(&repo, &ecvTestDataRepository::downloadFinished, &loop,
                     [&](bool ok, ecvTestDataRepository::Dataset dataset) {
                         if (ok && dataset == kind) ++downloaded;
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
    EXPECT_TRUE((console.errors == 0)) << "A recoverable cold-cache miss must "
                                          "not emit Error or open a modal "
                                          "popup";
    dialog.close();
    repo.cancelDownload();
    const QString closedInput = dialog.getSettings().pointsFile;
    const QString bundle = ecvTestDataRepository::extractPath(kind);
    QDir().mkpath(bundle);
    // Mirror the three readiness members in the v2 frames/<id>/ layout
    // (the recursive basename lookup keeps the repository check valid).
    for (const QString rel :
         {QStringLiteral("frames/kitti_000008/kitti_000008.bin"),
          QStringLiteral("frames/kitti_000008/kitti_000008_sampled16384.bin"),
          QStringLiteral("frames/nuscenes_n015/nuscenes_n015_lidar.bin")}) {
        const QString path = bundle + '/' + rel;
        QDir().mkpath(QFileInfo(path).absolutePath());
        QFile file(path);
        ASSERT_FALSE((!file.open(QIODevice::WriteOnly)));
        const float point[5] = {1, 2, 3, 0.5f, 0};
        const int features = rel.contains("nuscenes") ? 5 : 4;
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
            images.write(reinterpret_cast<const char*>(pixel), sizeof(pixel));
        images.close();
    }
    {
        QFile json(sceneDir + QStringLiteral("/scene.json"));
        if (json.open(QIODevice::WriteOnly))
            json.write(QByteArrayLiteral(
                    "{\"num_cameras\": 1, \"image_height\": 2, "
                    "\"image_width\": 2, \"image_channels\": 4}"));
    }
    // The multi-camera grid reads the full scene payload (calibration
    // matrices included); seed identity matrices so the restored scene
    // renders a camera cell.
    const auto writeIdentity = [&sceneDir](const QString& name) {
        QFile file(sceneDir + QLatin1Char('/') + name);
        if (file.open(QIODevice::WriteOnly)) {
            const float eye[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                                   0, 0, 1, 0, 0, 0, 0, 1};
            file.write(reinterpret_cast<const char*>(eye), sizeof(eye));
        }
    };
    writeIdentity(QStringLiteral("lidar_to_image.bin"));
    writeIdentity(QStringLiteral("image_aug.bin"));
    writeIdentity(QStringLiteral("lidar_aug.bin"));
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
            if (!(tasks->item(row)->flags() & Qt::ItemIsSelectable)) continue;
            tasks->setCurrentRow(row);
            QApplication::processEvents();
            // The camera grid is scroll content at the page bottom;
            // scroll every config page down so the current panel's
            // cells are reachable before scanning.
            for (auto* scroll : restored.findChildren<QScrollArea*>()) {
                scroll->verticalScrollBar()->setValue(
                        scroll->verticalScrollBar()->maximum());
            }
            QApplication::processEvents();
            for (auto* label :
                 restored.findChildren<ecvClickableImageLabel*>()) {
                if (!label->isVisible()) continue;
                if (!label->fullImage().isNull()) restoredPreview = label;
            }
        }
    }
    EXPECT_TRUE((restoredPreview && !restoredPreview->fullImage().isNull()))
            << "A restored scene dir must render its preview when the panel "
               "becomes current";
    repo.ensureDatasetReady(kind);
    EXPECT_TRUE((dialog.getSettings().pointsFile == closedInput))
            << "Closed dialogs must not accept late or foreign sample results";
    dialog.show();
    for (auto* button : dialog.findChildren<QPushButton*>())
        if (button->isVisible() &&
            button->text().contains("sample", Qt::CaseInsensitive))
            EXPECT_TRUE((button->isEnabled()))
                    << "Reopened sample controls must recover after closure "
                       "during preparation";
    QMetaObject::invokeMethod(&dialog, "onUseTestData", Qt::DirectConnection);
    EXPECT_TRUE((dialog.getSettings().pointsFile.endsWith("/kitti_000008.bin")))
            << "Warm preparation must fill the actual mapped point file";
}

TEST_F(OpenPCDetDialogTest,
       DatasetFramesDriveCameraGridOverlaysAndPersistSelection) {
    const QString bundle = ecvTestDataRepository::extractPath(kind);
    QDir().mkpath(bundle);
    // Mirror the three readiness members in the v2 frames/<id>/ layout
    // (the recursive basename lookup keeps the repository check valid).
    for (const QString rel :
         {QStringLiteral("frames/kitti_000008/kitti_000008.bin"),
          QStringLiteral("frames/kitti_000008/kitti_000008_sampled16384.bin"),
          QStringLiteral("frames/nuscenes_n015/nuscenes_n015_lidar.bin")}) {
        const QString path = bundle + '/' + rel;
        QDir().mkpath(QFileInfo(path).absolutePath());
        QFile file(path);
        ASSERT_FALSE((!file.open(QIODevice::WriteOnly)));
        const float point[5] = {1, 2, 3, 0.5f, 0};
        const int features = rel.contains("nuscenes") ? 5 : 4;
        file.write(reinterpret_cast<const char*>(point),
                   features * sizeof(float));
    }
    // ---- Multi-frame combo + multi-camera grid + result overlay ----
    PHASE("multiframe-grid");
    const QString bundle2 = ecvTestDataRepository::extractPath(kind);
    const auto writeScene = [&](const QString& dir, int n, int h, int w,
                                int c) {
        QDir().mkpath(dir);
        QFile img(dir + QStringLiteral("/images.bin"));
        if (img.open(QIODevice::WriteOnly)) {
            const float v = 0.5f;
            for (int i = 0; i < n * h * w * c; ++i)
                img.write(reinterpret_cast<const char*>(&v), sizeof(float));
        }
        QFile json(dir + QStringLiteral("/scene.json"));
        if (json.open(QIODevice::WriteOnly))
            json.write(
                    QStringLiteral("{\"num_cameras\":%1,\"image_height\":%2,"
                                   "\"image_width\":%3,\"image_channels\":%4}")
                            .arg(n)
                            .arg(h)
                            .arg(w)
                            .arg(c)
                            .toUtf8());
        const auto eye = [&](const QString& name, int count) {
            QFile f(dir + QLatin1Char('/') + name);
            if (f.open(QIODevice::WriteOnly)) {
                const float e[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                                     0, 0, 1, 0, 0, 0, 0, 1};
                for (int i = 0; i < count; ++i)
                    f.write(reinterpret_cast<const char*>(e), sizeof(e));
            }
        };
        eye(QStringLiteral("lidar_to_image.bin"), n);
        eye(QStringLiteral("image_aug.bin"), n);
        eye(QStringLiteral("lidar_aug.bin"), 1);
    };
    writeScene(bundle2 + QStringLiteral("/frames/kitti_000008/scene"), 1, 4, 8,
               3);
    writeScene(bundle2 + QStringLiteral("/frames/kitti_000002/scene"), 1, 4, 8,
               3);
    writeScene(bundle2 + QStringLiteral("/frames/nuscenes_n015/scene"), 6, 4, 8,
               3);
    {
        QDir().mkpath(bundle2 + QStringLiteral("/frames/kitti_000002"));
        QFile f(bundle2 +
                QStringLiteral("/frames/kitti_000002/kitti_000002.bin"));
        if (f.open(QIODevice::WriteOnly)) {
            const float point[4] = {1, 2, 3, 0.5f};
            f.write(reinterpret_cast<const char*>(point), sizeof(point));
        }
    }
    const auto selectCheckpoint = [&](OpenPCDetDialog& dlg,
                                      const QString& stem) {
        auto* list = dlg.findChild<QListWidget*>();
        for (int row = 0; row < list->count(); ++row) {
            if (!(list->item(row)->flags() & Qt::ItemIsSelectable)) continue;
            list->setCurrentRow(row);
            QApplication::processEvents();
            if (QFileInfo(dlg.getSettings().modelPath)
                        .fileName()
                        .startsWith(stem + QLatin1Char('.')))
                return true;
        }
        return false;
    };
    // The frame combo is the only visible combo whose first entry carries
    // empty data (the "Custom…" escape hatch).
    const auto frameComboOf = [](OpenPCDetDialog& dlg) -> QComboBox* {
        for (auto* combo : dlg.findChildren<QComboBox*>())
            if (combo->isVisible() && combo->count() > 0 &&
                combo->itemData(0).toString().isEmpty())
                return combo;
        return nullptr;
    };
    const auto gridLabels = [](OpenPCDetDialog& dlg) {
        QVector<ecvClickableImageLabel*> out;
        for (auto* l : dlg.findChildren<ecvClickableImageLabel*>())
            if (l->isVisible() && !l->fullImage().isNull()) out.append(l);
        return out;
    };
    OpenPCDetDialog gridDialog;
    gridDialog.show();
    QApplication::processEvents();
    EXPECT_TRUE(
            (selectCheckpoint(gridDialog, QStringLiteral("kitti_pointpillar"))))
            << "Navigate to the KITTI pointpillar panel";
    QComboBox* kittiCombo = frameComboOf(gridDialog);
    EXPECT_TRUE((kittiCombo != nullptr))
            << "The point panel exposes a frame combo";
    if (kittiCombo) {
        // Dataset-filtered: Custom + the two KITTI frames, no nuScenes.
        EXPECT_TRUE((kittiCombo->count() == 3))
                << "KITTI combo lists Custom + the KITTI frames only";
        EXPECT_TRUE(
                (kittiCombo->findData(QStringLiteral("kitti_000002")) >= 0 &&
                 kittiCombo->findData(QStringLiteral("kitti_000008")) >= 0 &&
                 kittiCombo->findData(QStringLiteral("nuscenes_n015")) < 0))
                << "KITTI combo carries the KITTI frames and no nuScenes frame";
        const int idx = kittiCombo->findData(QStringLiteral("kitti_000002"));
        kittiCombo->setCurrentIndex(idx);
        QApplication::processEvents();
        EXPECT_TRUE((gridDialog.getSettings().pointsFile.endsWith(
                "/kitti_000002.bin")))
                << "Selecting a frame fills the panel's point file";
        EXPECT_TRUE((gridLabels(gridDialog).size() == 1))
                << "A KITTI frame renders one camera cell";
    }
    // Overlay: a synthetic detection must change the rendered camera frame.
    const auto kittiCells = gridLabels(gridDialog);
    if (!kittiCells.isEmpty()) {
        const QImage base = kittiCells.first()->fullImage();
        OpenPCDetRunResult result;
        result.numClass = 3;
        OpenPCDetDetection det;
        det.x = 40.0f;
        det.y = 20.0f;
        det.z = 10.0f;
        det.dx = det.dy = det.dz = 2.0f;
        det.heading = 0.0f;
        det.score = 0.9f;
        det.label = 1;
        det.className = QStringLiteral("Car");
        result.detections.append(det);
        gridDialog.applyResultOverlays(gridDialog.currentPanelIndex(), result);
        QApplication::processEvents();
        EXPECT_TRUE((gridLabels(gridDialog).first()->fullImage() != base))
                << "A result paints a class-colored overlay onto the camera";
    }
    EXPECT_TRUE((
            selectCheckpoint(gridDialog, QStringLiteral("nuscenes_bevfusion"))))
            << "Navigate to the BEVFusion scene panel";
    QComboBox* nusCombo = frameComboOf(gridDialog);
    EXPECT_TRUE((nusCombo != nullptr && nusCombo->count() == 2 &&
                 nusCombo->findData(QStringLiteral("nuscenes_n015")) >= 0 &&
                 nusCombo->findData(QStringLiteral("kitti_000008")) < 0))
            << "nuScenes combo lists Custom + the nuScenes frame only";
    if (nusCombo) {
        // The combo already defaults to the only nuScenes frame, so fill
        // through the real "Use test data" path (the user's first fill).
        QMetaObject::invokeMethod(&gridDialog, "onUseTestData",
                                  Qt::DirectConnection);
        QApplication::processEvents();
        EXPECT_TRUE((gridLabels(gridDialog).size() == 6))
                << "A nuScenes frame renders all six camera cells";
        bool hasFront = false;
        for (auto* caption : gridDialog.findChildren<QLabel*>())
            if (caption->isVisible() &&
                caption->text() == QStringLiteral("CAM_FRONT"))
                hasFront = true;
        EXPECT_TRUE((hasFront))
                << "Camera cells are labeled with their camera name";
    }
    // Persistence: the selected frame survives a close/reopen cycle.
    selectCheckpoint(gridDialog, QStringLiteral("kitti_pointpillar"));
    if (QComboBox* combo = frameComboOf(gridDialog)) {
        combo->setCurrentIndex(0);  // Custom…
        combo->setCurrentIndex(combo->findData(QStringLiteral("kitti_000002")));
    }
    gridDialog.close();
    OpenPCDetDialog persistDialog;
    persistDialog.show();
    QApplication::processEvents();
    EXPECT_TRUE((selectCheckpoint(persistDialog,
                                  QStringLiteral("kitti_pointpillar"))))
            << "Reopen the KITTI pointpillar panel";
    QComboBox* restoredCombo = frameComboOf(persistDialog);
    EXPECT_TRUE((restoredCombo && restoredCombo->currentData().toString() ==
                                          QStringLiteral("kitti_000002")))
            << "The selected test frame persists across sessions";
    persistDialog.close();
}

TEST_F(OpenPCDetDialogTest,
       RealColdSampleDownloadAuthenticatesZipAndFillsEveryCheckpoint) {
    if (coldSampleRoot.isEmpty())
        GTEST_SKIP() << "Pass --sample-cache with a nonexistent data root for "
                        "real download evidence";
    root = coldSampleRoot;
    ASSERT_FALSE(QDir(root).exists()) << "Cold-cache root must not exist";
    qputenv("CLOUDVIEWER_DATA_ROOT", root.toUtf8());
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       root + "/settings");
    OpenPCDetDialog dialog;
    auto* navigation = dialog.findChild<QListWidget*>();
    ASSERT_NE(navigation, nullptr);
    navigation->setCurrentRow(1);
    dialog.show();
    console.errors = 0;
    PHASE("data-repository");
    QEventLoop loop;
    int ready = 0, downloaded = 0;
    bool success = false;
    QObject::connect(&repo, &ecvTestDataRepository::downloadFinished, &loop,
                     [&](bool ok, ecvTestDataRepository::Dataset dataset) {
                         if (ok && dataset == kind) ++downloaded;
                     });
    QObject::connect(&repo, &ecvTestDataRepository::downloadLogMessage, &loop,
                     [](const QString& message) {
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
    EXPECT_TRUE((console.errors == 0)) << "A recoverable cold-cache miss must "
                                          "not emit Error or open a modal "
                                          "popup";
    QTimer::singleShot(180000, &loop, [&] {
        repo.cancelDownload();
        loop.quit();
    });
    if (!ready) loop.exec();
    EXPECT_TRUE((success && ready == 1 && downloaded == 1 &&
                 QFileInfo::exists(dialog.getSettings().pointsFile)))
            << "Cold shared ZIP preparation must automatically fill a real "
               "point file once";
    QMetaObject::invokeMethod(&dialog, "onUseTestData", Qt::DirectConnection);
    EXPECT_TRUE((success && downloaded == 1 && !repo.isDownloadInProgress()))
            << "Warm fill must not download again";
    int filledPanels = 0;
    auto* tasks = dialog.findChild<QListWidget*>();
    const int initialRow = tasks ? tasks->currentRow() : -1;
    if (tasks)
        for (int row = 0; row < tasks->count(); ++row) {
            if (!(tasks->item(row)->flags() & Qt::ItemIsSelectable)) continue;
            tasks->setCurrentRow(row);
            QMetaObject::invokeMethod(&dialog, "onUseTestData",
                                      Qt::DirectConnection);
            const auto settings = dialog.getSettings();
            EXPECT_TRUE(
                    (!repo.isDownloadInProgress() &&
                     ((!settings.pointsFile.isEmpty() &&
                       QFileInfo::exists(settings.pointsFile)) ||
                      (!settings.sceneDir.isEmpty() &&
                       QFileInfo::exists(settings.sceneDir + "/scene.json")))))
                    << "Every checkpoint must fill its actual point or "
                       "calibrated scene input from the shared ZIP";
            ++filledPanels;
        }
    EXPECT_TRUE((filledPanels == 18 && console.errors == 0))
            << "Fill all 18 checkpoints without a modal Error";
    if (tasks) tasks->setCurrentRow(initialRow);
    QFile archive(ecvTestDataRepository::zipPath(kind));
    QCryptographicHash hash(QCryptographicHash::Sha256);
    EXPECT_TRUE((archive.open(QIODevice::ReadOnly) && hash.addData(&archive) &&
                 hash.result().toHex() ==
                         "9e60d9240b8f2f8cfdfc95da0d4cdc3990e7e9bb8375"
                         "531fc0abfc8e91b89db5"))
            << "Retain and authenticate the real shared OpenPCDet ZIP";
    std::puts(QJsonDocument(
                      QJsonObject{
                              {"data_root", root},
                              {"qpa_platform", QGuiApplication::platformName()},
                              {"qt_version", QString::fromLatin1(qVersion())},
                              {"zip_sha256",
                               QString::fromLatin1(hash.result().toHex())},
                              {"points_file", dialog.getSettings().pointsFile},
                              {"ready_count", ready},
                              {"filled_panels", filledPanels},
                              {"downloads", downloaded},
                              {"failures", HasFailure() ? 1 : 0}})
                      .toJson(QJsonDocument::Compact)
                      .constData());
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setOrganizationName("OpenPCDetContract");
    app.setApplicationName("Dialog");
    if (argc == 3 && QString::fromLocal8Bit(argv[1]) == "--sample-cache") {
        coldSampleRoot = QString::fromLocal8Bit(argv[2]);
        argc = 1;
        ::testing::GTEST_FLAG(filter) =
                "OpenPCDetDialogTest."
                "RealColdSampleDownloadAuthenticatesZipAndFillsEveryCheckpoint";
    }
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    if (coldSampleRoot.isEmpty()) {
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
                                {"panels", observedPanelCount},
                                {"failures", ::testing::UnitTest::GetInstance()
                                                     ->failed_test_count()}})
                        .toJson(QJsonDocument::Compact)
                        .constData());
    }
    return result;
}
