// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTableWidget>

#include "OpenPCDetModelCatalog.h"
#include "OpenPCDetWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ccPointCloud;
class ecvMainAppInterface;
class QSplitter;

/** One checkpoint panel: its own quantization combo (f32 / f16 / q8_0 of
 *  the panel's catalog checkpoint), threshold row and input rows. A dialog
 *  owns one panel per detector checkpoint (18) shown in a left-hand grouped
 *  task list (KITTI point cloud / nuScenes point cloud / multi-modal
 *  scene). Device / Threads are global controls rendered once above the
 *  task list. Layout mirrors the qYOLO task-panel pattern. */
struct OpenPCDetTaskPanel {
    QString checkpoint;  // catalog stem, e.g. "kitti_pointpillar"

    QWidget* tab = nullptr;
    QComboBox* quantCombo = nullptr;
    QLabel* modelInfoLabel = nullptr;
    // Optional custom GGUF override; when non-empty it takes precedence
    // over the checkpoint + quantization selection (mirrors the qYOLO
    // custom-model escape hatch, kept as a row instead of a combo entry
    // because the quant combo carries only the three catalog suffixes).
    QWidget* customModelRow = nullptr;
    QLineEdit* customModelPath = nullptr;
    // Threshold row (0 = follow the GGUF-declared value).
    QDoubleSpinBox* score = nullptr;
    QDoubleSpinBox* nms = nullptr;
    QSpinBox* maxBoxes = nullptr;
    // Point-cloud inputs: the active DB cloud (shown read-only, updated on
    // selection changes) plus an optional raw point file that takes
    // precedence when set (KITTI .bin / ASCII .txt — parsed with the exact
    // OpenPCDetHelpers semantics of the headless batch command).
    QLineEdit* cloudPath = nullptr;
    QWidget* pointsFileRow = nullptr;
    QLineEdit* pointsFileEdit = nullptr;
    // Multi-modal scene inputs (CaDDN / BEVFusion panels): scene directory
    // (images.bin + calibration + scene.json) with an image preview, plus
    // the LiDAR branch point source (required by BEVFusion, unused by
    // CaDDN).
    QWidget* sceneDirRow = nullptr;
    QLineEdit* sceneDirEdit = nullptr;
    ecvClickableImageLabel* previewLabel = nullptr;
    QWidget* scenePointsFileRow = nullptr;
    QLineEdit* scenePointsFileEdit = nullptr;
    QPushButton* testDataBtn = nullptr;
    QPushButton* runBtn = nullptr;
    QPushButton* cancelBtn = nullptr;
    QCheckBox* addBoxesCheck = nullptr;
    QCheckBox* annotatedCheck = nullptr;

    /** Catalog GGUF filename of the panel's checkpoint + quantization
     *  selection (empty when the combo carries no valid quant). */
    QString modelFilename() const;
    /** Resolved model path: the custom GGUF override when set, otherwise
     *  the cache path of modelFilename(). */
    QString modelPath() const;
};

class OpenPCDetDialog : public QDialog {
    Q_OBJECT

public:
    /** Values the user configures in the dialog; forwarded to the worker
     *  by qOpenPCDet. */
    struct Settings {
        QString modelPath;
        QString device = QStringLiteral("auto");
        int threads = 0;
        /** 0 = follow the GGUF-declared threshold. */
        float scoreThresh = 0.0f;
        /** 0 = follow the GGUF-declared NMS IoU. */
        float nmsThresh = 0.0f;
        /** 0 = follow the GGUF-declared box cap. */
        int maxBoxes = 0;
        /** Optional raw point file; takes precedence over the active
         *  cloud when set. */
        QString pointsFile;
        /** Optional multi-modal scene directory; non-empty routes the run
         *  through detect_scene. */
        QString sceneDir;
        /** Write the wireframe boxes into the DB tree on success. */
        bool addBoxesToDb = true;
        /** Export a per-class-colored copy of the source point cloud (box
         *  members tinted with the detection class color) into the DB
         *  tree on success; needs a point-cloud source (skipped for
         *  file-only or pointless-scene runs). */
        bool annotatedCloudToDb = true;
    };

    explicit OpenPCDetDialog(QWidget* parent = nullptr);
    ~OpenPCDetDialog() override;

    void setAppInterface(ecvMainAppInterface* app);
    /** The plugin owns the active-cloud lifecycle; the dialog only shows
     *  its name and refuses to run without one. */
    void setActiveCloud(ccPointCloud* cloud, const QString& name);
    Settings getSettings() const;

    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setRunning(bool running);
    void refreshModelList();
    /** Shows the per-class detection summary and keeps \p result for the
     *  result table / CSV export. */
    void showDetectionSummary(const OpenPCDetRunResult& result);

    static QString modelCacheDir();

signals:
    void runRequested(const OpenPCDetDialog::Settings& settings);
    void cancelRequested();

private slots:
    void onBrowseCustomModel();
    void onEnsureModel();
    void onUseTestData();
    void onTestDataProgress(int percent, const QString& statusText);
    void onTestDataLog(const QString& message);
    /** Settles the shared ensureDatasetReady chain (download→extract) and
     *  fills the requesting panel on success. */
    void onTestDataReady(bool success, ecvTestDataRepository::Dataset kind);
    void onRun();
    void onCancel();
    void onExportCsv();

protected:
    void closeEvent(QCloseEvent* event) override;
    /** Esc intercept: confirm before closing when a task is running. */
    void keyPressEvent(QKeyEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void reject() override;

private:
    enum class PendingAction { None, Run };

    void setupUi();
    void loadSettings();
    void saveSettings() const;
    /** Content-driven (font / DPI aware) ideal width of the task list. */
    int taskListIdealWidth() const;
    /** First-show allocation of the splitter's left/right panes (skipped
     *  when loadSettings() restored a user-saved splitter state). */
    void applyTaskListDefaultWidth();
    void populatePanels();
    QString resolveModelPath(const QString& filename) const;
    bool ensureModelAvailable(PendingAction action);
    void startDownload(const OpenPCDetModelEntry& model);
    /** Update the scene preview of \p panel (empty path or unreadable
     *  payload clears the preview). */
    void refreshScenePreview(OpenPCDetTaskPanel* panel,
                             const QString& sceneDir);
    /** Fill the shared detection result table + summary label. */
    void updateResultTable(const OpenPCDetRunResult& result);

    /** Fill panel \p panelIndex from the extracted openpcdet_test_data
     *  bundle (points file / scene dir / LiDAR branch per the checkpoint
     *  mapping) and load the KITTI sample frame into the DB tree. Returns
     *  false (with a logged error) when the bundle members are missing. */
    bool fillTestDataPanel(int panelIndex);
    /** Parse the KITTI .bin sample frame (OpenPCDetHelpers::parsePointsBin)
     *  into a ccPointCloud named "KITTI 000008 (sample)" with an intensity
     *  scalar field (the exact inverse of the worker's cloud input path),
     *  add it to the DB tree and select it so the active-cloud path can
     *  Run directly. No-op without an app interface. */
    void loadSampleCloudIntoDb(const QString& binPath);

    /** The task panel of the currently active task-list entry. */
    OpenPCDetTaskPanel* currentPanel() const;
    /** Find the panel whose model combo lists `filename` (may be nullptr). */
    OpenPCDetTaskPanel* panelForFilename(const QString& filename) const;

    // Left-hand task navigation: a grouped list driving the stack. Lives
    // in a splitter so users can drag the list wider/narrower (state is
    // persisted in QSettings).
    QListWidget* m_taskList = nullptr;
    QStackedWidget* m_taskStack = nullptr;
    QSplitter* m_bodySplitter = nullptr;
    bool m_splitterRestored = false;
    bool m_splitterSized = false;

    // Global runtime parameters rendered once above the task list.
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;

    // One panel per checkpoint (stack order; see populatePanels()).
    QVector<OpenPCDetTaskPanel> m_panels;

    QPushButton* m_ensureModelBtn = nullptr;
    QPushButton* m_exportCsvBtn = nullptr;
    QTableWidget* m_resultTable = nullptr;
    QLabel* m_summaryLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    ecvMainAppInterface* m_app = nullptr;
    ccPointCloud* m_cloud = nullptr;
    bool m_downloadInProgress = false;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
    bool m_taskRunning = false;
    OpenPCDetRunResult m_lastResult;

    // Shared openpcdet_test_data chain (ecvTestDataRepository singleton).
    bool m_testDataBusy = false;
    /** Panel index captured at request time: the async fill targets THIS
     *  panel even if the user switches the task list meanwhile. */
    int m_pendingTestDataPanel = -1;
};
