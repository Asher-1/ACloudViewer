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
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>

#include "TrellisModelCatalog.h"
#include "TrellisWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;

class TrellisDialog : public QDialog {
    Q_OBJECT

public:
    struct Settings {
        QStringList modelPaths;
        QString presetName;
        int pipelineType = 0;    // aicore_trellis_pipeline_type
        int backgroundMode = 0;  // aicore_trellis_background_mode
        QString inputPath;
        int steps = 0;
        double guidance = -1.0;
        int textureSteps = 0;
        uint64_t seed = 0;
        int threads = 0;
        QString device = QStringLiteral("auto");
        QString shapeDecPlacement = QStringLiteral("auto");
        bool useRmbg = false;
        bool textureEnabled = true;
        bool addResultToDb = true;
        /** Export the AI background-removed image as a ccImage entity to the
         *  DB tree (default off; requires useRmbg with a loaded RMBG model). */
        bool addRmbgImageToDb = false;
        QString saveGlbDir;  // empty = do not write GLB files
    };

    explicit TrellisDialog(QWidget* parent = nullptr);
    ~TrellisDialog() override;

    void setAppInterface(ecvMainAppInterface* app);
    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgressStage(int stageId,
                          const QString& stage,
                          int step,
                          int total);
    void setRunning(bool running);
    void setImagePreview(const QImage& image);
    void refreshModelState();

signals:
    void runRequested(const TrellisDialog::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();
    void browseImageRequested();

private slots:
    void onBrowseImage();
    void onBrowseSaveDir();
    void onPresetChanged(int index);
    void onRun();
    void onCancel();
    void onDownloadModels();
    void onTestDataClicked();
    void onTestImageSelected(int index);
    void onTestDataExtracted();

protected:
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    enum class PendingAction { None, Run };

    void setupUi();
    void loadSettings();
    void saveSettings() const;
    void updateModelStatus();
    QStringList missingPresetFiles() const;
    void downloadNextModel();
    void updateImagePreview();
    void ensureImage2MeshDataset();  // download/extract on demand
    void populateTestImages();       // fill combo from extracted dataset

    ecvMainAppInterface* m_app = nullptr;

    // Input.
    QLineEdit* m_imagePath = nullptr;
    ecvClickableImageLabel* m_imagePreview = nullptr;
    QPushButton* m_browseImageBtn = nullptr;
    QPushButton* m_useTestDataBtn = nullptr;
    QComboBox* m_testImageCombo = nullptr;  // pick one of the sample images

    // Model preset.
    QComboBox* m_presetCombo = nullptr;
    QComboBox* m_dinoCombo = nullptr;  // dino_q8 / dino_f16
    QComboBox* m_decCombo = nullptr;   // ss_dec_q8 / ss_dec_f16
    QLabel* m_modelStatus = nullptr;
    QPushButton* m_downloadBtn = nullptr;
    QCheckBox* m_textureCheck = nullptr;
    QCheckBox* m_rmbgCheck = nullptr;

    // Parameters (each row: "use default" checkbox + numeric spinbox).
    QCheckBox* m_stepsAuto = nullptr;
    QSpinBox* m_steps = nullptr;
    QCheckBox* m_guidanceAuto = nullptr;
    QDoubleSpinBox* m_guidance = nullptr;
    QCheckBox* m_textureStepsAuto = nullptr;
    QSpinBox* m_textureSteps = nullptr;
    QCheckBox* m_seedRandom = nullptr;
    QSpinBox* m_seed = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;

    // Output.
    QCheckBox* m_addToDbCheck = nullptr;
    QCheckBox* m_addRmbgToDbCheck = nullptr;
    QLineEdit* m_saveGlbDir = nullptr;

    // Progress / log.
    QLabel* m_stageLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_log = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    QStringList m_pendingDownloads;
    bool m_downloadInProgress = false;
    bool m_firstShow = true;  // lock the dialog size on first show (see §13.4)
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
};
