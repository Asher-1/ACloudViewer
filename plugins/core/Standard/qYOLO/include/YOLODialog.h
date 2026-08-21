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
#include <QTabWidget>
#include <QToolButton>

#include "YOLOLiveWidget.h"
#include "YOLOModelCatalog.h"
#include "YOLOWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;

/** One task-tab panel: its own model combo (filtered on the tab's task),
 *  runtime params (device / threads), threshold row, image input and Run
 *  button. A dialog owns three of these (detect / segment / depth); the
 *  Live tab reuses all models. The per-panel device/threads controls are
 *  kept in sync across panels (they are shared runtime parameters rendered
 *  inside every tab, next to the model they configure). */
struct YOLOTaskPanel {
    QString task;  // "detect" | "segment" | "depth"

    QWidget* tab = nullptr;
    QComboBox* modelCombo = nullptr;
    QComboBox* deviceCombo = nullptr;
    QSpinBox* threads = nullptr;
    QLineEdit* customModelPath = nullptr;
    QWidget* customModelRow = nullptr;
    QWidget* thresholdRow = nullptr;  // Conf/IoU/Top-K (hidden for depth)
    QDoubleSpinBox* conf = nullptr;
    QDoubleSpinBox* iou = nullptr;
    QSpinBox* topK = nullptr;
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
    void changeEvent(QEvent* event) override;

private:
    enum class PendingAction { None, Run, LiveStart };
    enum class TestDataTarget { None, Image, Video };

    void setupUi();
    void loadSettings();
    void saveSettings() const;
    void populateModelCombo(const QString& keepFilename = QString());
    bool selectModelByFilename(const QString& filename);
    QString resolveModelPath() const;
    bool ensureModelAvailable(PendingAction action);
    void startDownload(const YOLOModelEntry& model);
    void cancelDownload();
    void updateImagePreview();
    void startLiveStream();
    void adaptTabWidgetHeight();
    /** Update custom-row / threshold-row visibility of one task panel. */
    void applyPanelVisibility(YOLOTaskPanel& panel);

    void requestTestData(TestDataTarget target);
    bool loadRequestedTestData();
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);
    void setTestDataControlsEnabled(bool enabled);

    /** The task panel of the currently active tab. */
    YOLOTaskPanel* currentTaskPanel() const;
    /** Find the panel whose tab is `tab`. */
    YOLOTaskPanel* panelForTab(QWidget* tab) const;
    /** Find the panel whose model combo lists `filename` (may be nullptr). */
    YOLOTaskPanel* panelForFilename(const QString& filename) const;

    QTabWidget* m_tabWidget = nullptr;
    QWidget* m_liveTab = nullptr;
    YOLOLiveWidget* m_liveWidget = nullptr;
    QPushButton* m_liveStartBtn = nullptr;
    QPushButton* m_liveStopBtn = nullptr;
    QPushButton* m_liveRestartBtn = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QComboBox* m_testVideoCombo = nullptr;

    // One panel per task (index 0=detect, 1=segment, 2=depth).
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
    int m_activeTabHeight = -1;

    bool m_testDataDownloadInProgress = false;
    TestDataTarget m_pendingTestDataTarget = TestDataTarget::None;

    QLabel* m_taskStatusLabel = nullptr;
};
