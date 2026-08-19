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

#include "RMBGLiveWidget.h"
#include "RMBGModelCatalog.h"
#include "RMBGWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;

class RMBGDialog : public QDialog {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = QStringLiteral("auto");
        /** Pixels with alpha below this value become fully transparent
         *  (0.0 disables the threshold pass). */
        float alphaThreshold = 0.5f;
        bool addResultToDb = true;
        QString savePngDir;  // empty = do not write PNG files
    };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    explicit RMBGDialog(QWidget* parent = nullptr);
    ~RMBGDialog() override;

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
    void runRequested(const RMBGDialog::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();
    void liveCaptureReady(const RMBGRunResult& result);

private slots:
    void onBrowseImage();
    void onBrowseCustomModel();
    void onBrowseSaveDir();
    void onModelComboChanged(int index);
    void onRun();
    void onCancel();
    void onDbListActivated(QListWidgetItem* item);
    void onLiveStart();
    void onLiveStop();
    void onLiveRestart();
    void onLiveCapture(const RMBGRunResult& result);

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
    void startDownload(const RMBGModelEntry& model);
    void cancelDownload();
    void updateImagePreview();
    void startLiveStream();
    void adaptTabWidgetHeight();
    void requestTestData(TestDataTarget target);
    bool loadRequestedTestData();
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);
    void setTestDataControlsEnabled(bool enabled);

    QTabWidget* m_tabWidget = nullptr;
    QWidget* m_imageTab = nullptr;
    QWidget* m_liveTab = nullptr;
    RMBGLiveWidget* m_liveWidget = nullptr;
    QPushButton* m_liveStartBtn = nullptr;
    QPushButton* m_liveStopBtn = nullptr;
    QPushButton* m_liveRestartBtn = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QComboBox* m_testVideoCombo = nullptr;

    QComboBox* m_modelCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QDoubleSpinBox* m_alphaThreshold = nullptr;
    QLineEdit* m_imagePath = nullptr;
    ecvClickableImageLabel* m_previewLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QPushButton* m_imageTestDataBtn = nullptr;
    QCheckBox* m_addDbCheck = nullptr;
    QCheckBox* m_savePngCheck = nullptr;
    QLineEdit* m_savePngDir = nullptr;
    QToolButton* m_dbToggleBtn = nullptr;
    QWidget* m_dbContentWidget = nullptr;
    QListWidget* m_dbImageList = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    ecvMainAppInterface* m_app = nullptr;
    bool m_downloadInProgress = false;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
    bool m_taskRunning = false;
    QString m_lastTaskError;
    int m_activeTabHeight = -1;
    bool m_testDataDownloadInProgress = false;
    TestDataTarget m_pendingTestDataTarget = TestDataTarget::None;

    QLabel* m_taskStatusLabel = nullptr;
};
