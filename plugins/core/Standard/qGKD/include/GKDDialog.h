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
#include <QTextEdit>

#include "GKDModelCatalog.h"
#include "GKDWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;

/** qGKD dialog: GKDT general keypoint detection on a still or DB image with
 *  text / 1-shot visual / multimodal prompts, plus an optional multi-object
 *  mode composed on top of the YOLO-World open-vocabulary detector. */
class GKDDialog : public QDialog {
    Q_OBJECT

public:
    explicit GKDDialog(QWidget* parent = nullptr);
    ~GKDDialog() override;

    void setAppInterface(ecvMainAppInterface* app);
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setTaskStage(const QString& stage, int percent = -1);
    void setRunning(bool running);
    void setDbImages(const QList<GKDImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);
    void refreshModelList();
    GKDWorker::Settings workerSettings() const;

    static QString modelCacheDir();

signals:
    void runRequested(const GKDWorker::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();

private slots:
    void onBrowseImage();
    void onBrowseSupportImage();
    void onBrowseCustomModel();
    void onBrowseSaveDir();
    void onModelComboChanged(int index);
    void onRun();
    void onCancel();
    void onDbListActivated(QListWidgetItem* item);
    void requestTestData();
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);

protected:
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    enum class PendingAction { None, Run };

    void setupUi();
    void loadSettings();
    void saveSettings() const;
    void populateModelCombo(const QString& keepFilename = QString());
    void populateYoloModelCombo(const QString& keepFilename = QString());
    void populateTestDataCombo();
    bool loadSelectedTestData();
    bool selectModelByFilename(const QString& filename);
    QString resolveModelPath() const;
    QString resolveYoloModelPath() const;
    bool ensureModelAvailable(PendingAction action);
    bool ensureYoloModelAvailable();
    void startDownload(const GKDModelEntry& model,
                       const QString& destDir = QString());
    void cancelDownload();
    void updateImagePreview();

    QComboBox* m_modelCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QLineEdit* m_imagePath = nullptr;
    ecvClickableImageLabel* m_previewLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QPushButton* m_saveResultBtn = nullptr;

    // Prompts
    QLineEdit* m_kpsTexts = nullptr;
    QLineEdit* m_supportImagePath = nullptr;
    QLineEdit* m_supportKps = nullptr;
    QLineEdit* m_roi = nullptr;
    QDoubleSpinBox* m_minScore = nullptr;

    // Multi-object (YOLO-World composition)
    QCheckBox* m_multiObjectCheck = nullptr;
    QWidget* m_multiObjectRow = nullptr;
    QLineEdit* m_objectClasses = nullptr;
    QComboBox* m_yoloModelCombo = nullptr;
    QDoubleSpinBox* m_yoloConf = nullptr;

    QCheckBox* m_addDbCheck = nullptr;
    QCheckBox* m_savePngCheck = nullptr;
    QLineEdit* m_savePngDir = nullptr;
    QTextEdit* m_log = nullptr;
    QLabel* m_taskStatusLabel = nullptr;

    // Sample data (shared ecvTestDataRepository, GKDT demo dataset)
    QComboBox* m_testDataCombo = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    bool m_testDataDownloadInProgress = false;

    // DB image picker
    QListWidget* m_dbImageList = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    ecvMainAppInterface* m_app = nullptr;
    bool m_downloadInProgress = false;
    bool m_modelExplicit = false;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
    bool m_taskRunning = false;
    bool m_firstShow = true;
};
