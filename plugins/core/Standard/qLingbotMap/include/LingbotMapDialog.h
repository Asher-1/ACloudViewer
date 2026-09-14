// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTextEdit>
#include <QVector>

#include "LingbotMapWorker.h"
#include "ecvTestDataRepository.h"

class QCheckBox;
class QComboBox;
class QGroupBox;
class QLineEdit;
class QDoubleSpinBox;

class ecvModelDownloader;
class ecvMainAppInterface;

/** qLingbotMap main dialog: catalog-backed model selection (with automatic
 *  digest-anchored download), image-sequence input, streaming options, and
 *  the log/progress section shared by all AICore plugins. */
class LingbotMapDialog : public QDialog {
    Q_OBJECT

public:
    explicit LingbotMapDialog(QWidget* parent = nullptr);
    ~LingbotMapDialog() override;

    void setAppInterface(ecvMainAppInterface* app) { m_app = app; }
    void appendLog(const QString& msg);
    /** Run/cancel button + progress visibility while a task is active. */
    void setTaskRunning(bool running);
    /** Worker progress in percent (-1 = indeterminate); hidden when idle. */
    void setProgress(int percent);

    LingbotMapWorker::Settings collectSettings() const;

signals:
    void runRequested(const LingbotMapWorker::Settings& settings);
    void cancelRequested();

public slots:
    /** Automatic download wiring shared with the AICore plugin family. */
    void onModelComboChanged();
    void onBrowseCustomModel();
    void onBrowseImageFolder();
    void onBrowseSkyMaskDir();
    void onBrowseVideoFile();

private:
    enum class PendingAction { None, Run };

    void loadSettings();
    void saveSettings() const;
    void populateCatalogs();
    void populateSkysegCombo(const QString& filename);
    void refreshInputRows();
    QString resolveModelPath() const;
    QString resolveSkysegPath() const;
    bool ensureModelAvailable(PendingAction action);
    void startDownload(const QString& filename);
    void onRun();
    void onCancel();

    ecvMainAppInterface* m_app = nullptr;

    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_skysegCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QComboBox* m_inputModeCombo = nullptr;
    QWidget* m_folderRow = nullptr;
    QWidget* m_videoRow = nullptr;
    QWidget* m_videoFpsRow = nullptr;
    QLineEdit* m_videoPath = nullptr;
    QSpinBox* m_videoFps = nullptr;
    QLineEdit* m_imageExt = nullptr;
    QLineEdit* m_imageFolder = nullptr;
    QSpinBox* m_maxFrames = nullptr;
    QSpinBox* m_imageSize = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QDoubleSpinBox* m_confThreshold = nullptr;
    QComboBox* m_skySourceCombo = nullptr;
    QWidget* m_skysegRow = nullptr;
    QLineEdit* m_skyMaskDir = nullptr;
    QWidget* m_skyMaskRow = nullptr;
    QLabel* m_skyHint = nullptr;
    QGroupBox* m_advancedBox = nullptr;
    QWidget* m_advancedContainer = nullptr;
    QSpinBox* m_kvScale = nullptr;
    QSpinBox* m_kvWindow = nullptr;
    QSpinBox* m_frameStride = nullptr;
    QCheckBox* m_rotate90 = nullptr;
    QCheckBox* m_addDbCheck = nullptr;
    QPushButton* m_runButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QPushButton* m_downloadButton = nullptr;
    QTextEdit* m_log = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_downloadLabel = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    bool m_downloadInProgress = false;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
    bool m_modelExplicit = false;
    /** Sky-source auto-switch bookkeeping: outdoor test scenes enable sky
     *  masking on selection, indoor resets it to none, custom folders only
     *  get a hint. A user-explicit combo choice always wins afterwards. */
    bool m_skySourceUserSet = false;
    bool m_skyAutoSwitching = false;
    bool m_customDataSelected = false;

    // ---- test data (shared ecvTestDataRepository, lingbot_map_data) ----
    void populateTestDataScenes();
    void applyTestSceneSelection();
    void requestTestData();
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);
    void finishTestDataFlow();
    void refreshSkySourceOptions();
    QComboBox* m_testSceneCombo = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    bool m_testDataInProgress = false;
    ecvTestDataRepository::Dataset m_pendingMaskDownload =
            ecvTestDataRepository::Dataset::LingbotMapCourthouse;
    bool m_maskDownloadPending = false;
};
