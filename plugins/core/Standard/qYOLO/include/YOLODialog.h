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
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QToolButton>

#include "YOLOLiveWidget.h"
#include "YOLOModelCatalog.h"
#include "YOLOVisualPromptLabel.h"
#include "YOLOWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;
class QSplitter;
class QShowEvent;

/** One task panel: its own model combo (filtered on the panel's task),
 *  threshold row, image input and Run button. A dialog owns one panel per
 *  task family (detect / segment / depth / pose / obb / classify /
 *  semantic / world / yoloe) shown in a left-hand task list; the Live page
 *  reuses all closed-set models. Device / Threads are global controls
 *  rendered once above the task list (they configure every panel and the
 *  Live widget). */
struct YOLOTaskPanel {
    QString task;  // "detect" | "segment" | "depth" | "pose" | "obb" |
                   // "classify" | "semantic" | "world" | "yoloe"

    QWidget* tab = nullptr;
    QComboBox* modelCombo = nullptr;
    // True once the user picked a model explicitly; legacy builds persisted
    // the auto index-0 default, which must not shadow the recommended one.
    bool explicitModelChoice = false;
    QLineEdit* customModelPath = nullptr;
    QWidget* customModelRow = nullptr;
    QWidget* thresholdRow = nullptr;  // Conf/IoU/Top-K (hidden for depth /
                                      // classify / semantic)
    QDoubleSpinBox* conf = nullptr;
    QDoubleSpinBox* iou = nullptr;
    QSpinBox* topK = nullptr;
    // Open-vocabulary row (world/yoloe tabs only): comma-separated class
    // list + text-encoder GGUF combo (default follows the detector family).
    QWidget* textRow = nullptr;
    QLineEdit* classesEdit = nullptr;
    QComboBox* textModelCombo = nullptr;
    // Visual-prompt row (yoloe tab only): prompt-mode selector + the box
    // drawing canvas shown while "Visual prompt" is active. Boxes live in
    // full-image pixel coordinates and become one SAVPE class embedding
    // each (object0..objectN-1, official semantics).
    QWidget* promptModeRow = nullptr;
    QComboBox* promptModeCombo = nullptr;
    YOLOVisualPromptLabel* vpLabel = nullptr;
    QPushButton* vpClearBtn = nullptr;
    // Set while the multilingual bridge tower is the panel's active text
    // encoder; drives the symmetric confidence restore when the user
    // switches back to a native tower.
    bool bridgeWasActive = false;
    // Last text tower seen by updateBridgeHints(): a conf recalibration
    // only fires on a real tower change (or a forced refresh), so unrelated
    // visibility/model-combo updates never disturb the threshold.
    QString lastTextTower;
    QLabel* bridgeHint = nullptr;  // conf guidance shown while the
                                   // multilingual bridge tower is selected
    QLineEdit* imagePath = nullptr;
    ecvClickableImageLabel* previewLabel = nullptr;
    QPushButton* runBtn = nullptr;
    QPushButton* cancelBtn = nullptr;
    QPushButton* testDataBtn = nullptr;
    QCheckBox* addAnnotatedCheck = nullptr;
    QToolButton* dbToggleBtn = nullptr;
    QWidget* dbContentWidget = nullptr;
    QListWidget* dbImageList = nullptr;

    QString modelPath() const;  // resolved path of modelCombo's selection
    /** Resolved path of textModelCombo's selection (empty when the tab has
     *  no text row). */
    QString textModelPath() const;
};

class YOLODialog : public QDialog {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = QStringLiteral("auto");
        float confThres = 0.25f;
        float iouThres = 0.7f;
        uint32_t topK = 300;
        bool addAnnotatedImageToDb = true;
        // Open-vocabulary (world/yoloe) tabs.
        QStringList classes;
        QString textModelPath;
        // YOLOE visual prompts (full-image pixel boxes); non-empty switches
        // the run to the SAVPE visual-prompt path (classes are ignored).
        QList<QRectF> visualPrompts;
    };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    explicit YOLODialog(QWidget* parent = nullptr);
    ~YOLODialog() override;

    void setAppInterface(ecvMainAppInterface* app);
    /** Applies the Chinese->English prompt translation for the text towers
     *  and returns the effective run settings. */
    Settings getSettings();
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setTaskStage(const QString& stage, int percent = -1);
    void setRunning(bool running);
    void enableResultButtons(bool hasResult);
    void setDbImages(const QList<DbImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);
    void refreshModelList();

    static QString modelCacheDir();

signals:
    void runRequested(const YOLODialog::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();
    void liveCaptureReady(const YOLORunResult& result);
    void liveDepthCaptureReady(const YOLODepthResult& result);

private slots:
    void onBrowseImage();
    void onBrowseCustomModel();
    void onModelComboChanged(int index);
    void onRun();
    void onCancel();
    void onDbListActivated(QListWidgetItem* item);
    void onLiveStart();
    void onLiveStop();
    void onLiveRestart();
    void onLiveCapture(const YOLORunResult& result);
    void onLiveDepthCapture(const YOLODepthResult& result);

protected:
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    enum class PendingAction { None, Run, LiveStart };
    enum class TestDataTarget { None, Image, Video };

    void setupUi();
    void loadSettings();
    void saveSettings() const;
    /** Content-driven (font / DPI aware) ideal width of the task list: the
     *  widest entry text plus item padding, frame and scrollbar room, so no
     *  entry is elided by default on any platform or screen resolution. */
    int taskListIdealWidth() const;
    /** First-show allocation of the splitter's left/right panes (skipped
     *  when loadSettings() restored a user-saved splitter state). */
    void applyTaskListDefaultWidth();
    void populateModelCombo(const QString& keepFilename = QString());
    /** Select the family-default text-encoder GGUF in the panel's text
     *  model combo (CLIP for World, MobileCLIP for YOLOE). No-op for tabs
     *  without a text row. */
    void selectDefaultTextModel(YOLOTaskPanel& panel) const;
    bool ensureModelAvailable(PendingAction action);
    void startDownload(const YOLOModelEntry& model);
    void cancelDownload();
    void updateImagePreview();
    void startLiveStream();
    /** Update custom-row / threshold-row visibility of one task panel. */
    void applyPanelVisibility(YOLOTaskPanel& panel);
    /** Show/hide the per-panel multilingual confidence hint (visible while
     *  the World panel's text tower is the mclip bridge, whose prompts
     *  score lower than native-English ones) and recalibrate the panel's
     *  confidence to the active tower's score band. force=true skips the
     *  tower-change guard (used after settings restore, where the restored
     *  combo entry may equal the populated default and never emit). */
    void updateBridgeHints(bool force = false);
    /** True when the panel's YOLOE prompt mode is "visual" (SAVPE boxes).
     *  Always false for non-yoloe panels (no mode row). */
    static bool panelUsesVisualPrompts(const YOLOTaskPanel& panel);

    void requestTestData(TestDataTarget target);
    /** Resume/queue driver for a sample-data request: fill the file when
     *  it is already cached, wait while the shared repository serves
     *  another download (the repo's finished signals re-enter this), or
     *  start the ObjectsDetection download/extract chain. Requests made
     *  while the repository is busy are QUEUED here instead of dropped. */
    void advancePendingTestData();
    /** Load the pending + follow-up queued requests (after a chain's
     *  extraction made the archive available) and clear both slots. */
    void servePendingTestData();
    /** Load one request's file into its panel / the Live widget. */
    bool loadTestDataFor(TestDataTarget target, const QString& task);
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);
    void setTestDataControlsEnabled(bool enabled);

    /** The task panel of the currently active task-list entry. */
    YOLOTaskPanel* currentTaskPanel() const;
    /** Find the panel whose task id is `task` (may be nullptr). */
    YOLOTaskPanel* panelForTask(const QString& task) const;
    /** Find the panel whose model combo lists `filename` (may be nullptr). */
    YOLOTaskPanel* panelForFilename(const QString& filename) const;

    // Left-hand task navigation: a grouped list driving the stack. Lives
    // in a splitter so users can drag the list wider/narrower (state is
    // persisted in QSettings); see taskListIdealWidth() for the default.
    QListWidget* m_taskList = nullptr;
    QStackedWidget* m_taskStack = nullptr;
    QSplitter* m_bodySplitter = nullptr;
    bool m_splitterRestored = false;
    bool m_splitterSized = false;
    // Global runtime parameters rendered once above the task list.
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QWidget* m_liveTab = nullptr;
    YOLOLiveWidget* m_liveWidget = nullptr;
    QPushButton* m_liveStartBtn = nullptr;
    QPushButton* m_liveStopBtn = nullptr;
    QPushButton* m_liveRestartBtn = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QComboBox* m_testVideoCombo = nullptr;

    // One panel per task family (stack order; see kPanelTasks() in
    // YOLODialog.cpp).
    QVector<YOLOTaskPanel> m_panels;

    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QLineEdit* m_imagePath = nullptr;
    ecvClickableImageLabel* m_previewLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    ecvModelDownloader* m_downloader = nullptr;
    ecvMainAppInterface* m_app = nullptr;
    bool m_downloadInProgress = false;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
    QString m_lastTaskError;

    bool m_testDataDownloadInProgress = false;
    // Two queued sample-data request slots: pending is served by the
    // running chain's completion, followup (a request made while a chain
    // was already serving pending) right after it. Later clicks overwrite
    // the followup slot (latest wins; bounded queue).
    TestDataTarget m_pendingTestDataTarget = TestDataTarget::None;
    QString m_pendingTestDataTask;
    TestDataTarget m_followupTestDataTarget = TestDataTarget::None;
    QString m_followupTestDataTask;

    QLabel* m_taskStatusLabel = nullptr;
};
