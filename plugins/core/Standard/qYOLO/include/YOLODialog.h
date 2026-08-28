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
#include "YOLOWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;

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
    };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    explicit YOLODialog(QWidget* parent = nullptr);
    ~YOLODialog() override;

    void setAppInterface(ecvMainAppInterface* app);
    Settings getSettings() const;
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

private:
    enum class PendingAction { None, Run, LiveStart };
    enum class TestDataTarget { None, Image, Video };

    void setupUi();
    void loadSettings();
    void saveSettings() const;
    void populateModelCombo(const QString& keepFilename = QString());
    bool selectModelByFilename(const QString& filename);
    /** Select the family-default text-encoder GGUF in the panel's text
     *  model combo (CLIP for World, MobileCLIP for YOLOE). No-op for tabs
     *  without a text row. */
    void selectDefaultTextModel(YOLOTaskPanel& panel) const;
    QString resolveModelPath() const;
    bool ensureModelAvailable(PendingAction action);
    void startDownload(const YOLOModelEntry& model);
    void cancelDownload();
    void updateImagePreview();
    void startLiveStream();
    /** Update custom-row / threshold-row visibility of one task panel. */
    void applyPanelVisibility(YOLOTaskPanel& panel);

    void requestTestData(TestDataTarget target);
    bool loadRequestedTestData();
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

    // Left-hand task navigation: a grouped list driving the stack.
    QListWidget* m_taskList = nullptr;
    QStackedWidget* m_taskStack = nullptr;
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
    QPushButton* m_imageTestDataBtn = nullptr;
    QToolButton* m_dbToggleBtn = nullptr;
    QWidget* m_dbContentWidget = nullptr;
    QListWidget* m_dbImageList = nullptr;
    ecvModelDownloader* m_downloader = nullptr;
    ecvMainAppInterface* m_app = nullptr;
    bool m_downloadInProgress = false;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
    bool m_taskRunning = false;
    QString m_lastTaskError;
    QString m_downloadTargetFilename;

    bool m_testDataDownloadInProgress = false;
    TestDataTarget m_pendingTestDataTarget = TestDataTarget::None;

    QLabel* m_taskStatusLabel = nullptr;
};
