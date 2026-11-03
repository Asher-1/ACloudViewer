// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <ecvTestDataRepository.h>

#include <QDialog>
#include <QElapsedTimer>
#include <QString>
#include <QStringList>

#include "QuerySplatModelCatalog.h"
#include "QuerySplatWorker.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QListWidget;
class QPushButton;
class QProgressBar;
class QSpinBox;
class QStackedWidget;

class ecvModelDownloader;
class ecvMainAppInterface;
class ecvClickableImageLabel;

/** QuerySplat panel. UI contract: the input list, actions and progress stay
 *  visible OUTSIDE the configuration scroll at the minimum window size; all
 *  messages route to the application console through
 *  aicore_inference_log (no plugin-local log widget). */
class QuerySplatDialog : public QDialog {
    Q_OBJECT

public:
    explicit QuerySplatDialog(QWidget* parent = nullptr);
    ~QuerySplatDialog() override;

    void setAppInterface(ecvMainAppInterface* app);

    struct Settings {
        QString decoderPath;
        QString encoderPath;
        QStringList imagePaths;
        int threads = 0;
        QString device = QStringLiteral("auto");
        int imageSize = 0;
        bool ttoEnabled = false;
        int ttoSteps = 20;
        double ttoLr = 5e-3;
        double ttoLpipsWeight = 0.05;
        float opacityThreshold = 0.05f;
        float renderBackground = 0.0f;  // gsplat raster background (grayscale)
        bool addCameras = true;
        bool addDepthViews = true;
        bool addRenderViews = true;
        bool addDepthPointCloud = true;
        bool exportPly = true;
        QString outputDir;
    };

    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setTaskStage(const QString& stage, int percent = -1);
    void setRunning(bool running);
    void refreshModelList();
    void fillImages(const QStringList& paths);

signals:
    void runRequested(const QuerySplatDialog::Settings& settings);
    void cancelRequested();

private slots:
    void onAddFiles();
    void onRemoveSelected();
    void onClearImages();
    void onUseTestData();
    void onRun();
    void onCancel();

    // model download chain (shared ecvModelDownloader)
    void onModelDownloadProgress(qint64 received, qint64 total);
    void onModelDownloadLog(const QString& msg);
    void onModelDownloadFinished(bool ok, const QString& destPath);

    // shared test-data preparation (ecvTestDataRepository)
    void onTestDataReady(bool ok, ecvTestDataRepository::Dataset kind);
    void onTestDataProgress(int percent, const QString& statusText);
    void onTestDataLog(const QString& message);
    // input preview (shared ecvClickableImageLabel, MapAnything semantics)
    void updatePreview();

private:
    void buildUi();
    QWidget* buildInputPage();
    QWidget* buildConfigPage();
    bool ensureModelsAvailable();
    void startModelDownload(const QuerySplatModelEntry& model);
    void queueNextDownload();
    bool ensureEncoderAvailable();
    QString decoderCacheDir() const;

    ecvMainAppInterface* m_app = nullptr;

    // input (fixed area)
    QListWidget* m_imageList = nullptr;
    ecvClickableImageLabel* m_previewLabel = nullptr;
    QPushButton* m_addBtn = nullptr;
    QPushButton* m_removeBtn = nullptr;
    QPushButton* m_clearBtn = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QComboBox* m_testSceneCombo = nullptr;

    // configuration (scroll area)
    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;
    QSpinBox* m_imageSizeSpin = nullptr;
    QCheckBox* m_ttoCheck = nullptr;
    QSpinBox* m_ttoStepsSpin = nullptr;
    QDoubleSpinBox* m_ttoLrSpin = nullptr;
    QDoubleSpinBox* m_ttoLpipsSpin = nullptr;
    QDoubleSpinBox* m_opacitySpin = nullptr;
    QDoubleSpinBox* m_bgSpin = nullptr;
    QCheckBox* m_depthViewsCheck = nullptr;
    QCheckBox* m_camerasCheck = nullptr;
    QCheckBox* m_renderViewsCheck = nullptr;
    QCheckBox* m_pointCloudCheck = nullptr;
    QCheckBox* m_plyCheck = nullptr;

    // fixed action/progress area
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_taskStatusLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    QVector<QuerySplatModelEntry> m_pendingDownloads;
    QString m_pendingAfterDownload;
    bool m_downloadInProgress = false;
    bool m_testDataBusy = false;
    // The scene dataset the running shared preparation belongs to (the
    // repository signal is broadcast to every listener).
    ecvTestDataRepository::Dataset m_pendingTestDataset =
            ecvTestDataRepository::Dataset::MapAnythingCourtyard;
    QString m_lastTaskError;
};

// Queued-connection payload: register at plugin construction time
// (qQuerySplat::qQuerySplat) so worker -> GUI result delivery works.
Q_DECLARE_METATYPE(QuerySplatDialog::Settings)
