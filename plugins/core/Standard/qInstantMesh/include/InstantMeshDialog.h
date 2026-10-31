// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>

#include "InstantMeshModelCatalog.h"
#include "InstantMeshWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;

class InstantMeshDialog : public QDialog {
    Q_OBJECT

public:
    struct Settings {
        QStringList modelPaths;  // paths-field order (cond, unet, vae, dino,
                                 // transformer, synthesizer)
        QString presetName;
        QString inputPath;
        int steps = 0;  // <=0 -> pipeline default (75)
        uint64_t seed = 42;
        double guidance = -1.0;   // <=0 -> pipeline default (4.0)
        int gridRes = 0;          // <=0 -> pipeline default (64)
        double gridScale = -1.0;  // <=0 -> pipeline default (2.1)
        int textureRes = 0;       // <=0 -> pipeline default (2048)
        bool bakeTexture = true;
        int threads = 0;
        QString device = QStringLiteral("auto");
        bool useRmbg = false;
        QString rmbgModelPath;
        bool addResultToDb = true;
        QString quantization = QStringLiteral("f16");
    };

    explicit InstantMeshDialog(QWidget* parent = nullptr);
    ~InstantMeshDialog() override;

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
    void setLastResult(const InstantMeshRunResult& result) {
        m_lastResult = result;
    }
    const InstantMeshRunResult& lastResult() const { return m_lastResult; }

    /** Strip slot states (public: the plugin marks completion). */
    enum StageSlotState { kStagePending = 0, kStageActive = 1, kStageDone = 2 };
    void setStageState(int slot, int state);
    void resetStageStrip(const QImage& input);
    void applyResultToStrip(const InstantMeshRunResult& result);

signals:
    void runRequested(const InstantMeshDialog::Settings& settings);
    void cancelRequested();

private slots:
    void onBrowseImage();
    void onPresetChanged(int index);
    void onQuantChanged(int index);
    void onRun();
    void onCancel();
    void onDownloadModels();
    void onTestDataClicked();
    void onTestImageSelected(int index);
    void onTestDataExtracted();

protected:
    void closeEvent(QCloseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void reject() override;

private:
    enum class PendingAction { None, Run };

    void setupUi();
    void buildGeneratePage(QWidget* page);
    void loadSettings();
    void saveSettings() const;
    void updateModelStatus();
    QStringList missingPresetFiles() const;
    void downloadNextModel();
    void updateImagePreview();
    void ensureImage2MeshDataset();  // fill picker when already extracted
    void populateTestImages();       // fill combo from extracted dataset
    QString quantization() const;
    void setImagePreviewToSlot(int slot, const QImage& image);

    ecvMainAppInterface* m_app = nullptr;

    // Input.
    QLineEdit* m_imagePath = nullptr;
    ecvClickableImageLabel* m_imagePreview = nullptr;
    QImage m_lastPreviewImage;
    QPushButton* m_browseImageBtn = nullptr;
    QPushButton* m_useTestDataBtn = nullptr;
    QComboBox* m_testImageCombo = nullptr;

    // Model preset.
    QComboBox* m_presetCombo = nullptr;
    QComboBox* m_quantCombo = nullptr;  // f16 (default) / q8 / f32
    QCheckBox* m_rmbgCheck = nullptr;
    QComboBox* m_rmbgQuantCombo = nullptr;  // q8 (default) / f16
    QLabel* m_modelStatus = nullptr;
    QPushButton* m_downloadBtn = nullptr;

    // End-to-end step strip: four chips (source / views / mesh / GLB).
    QVector<ecvClickableImageLabel*> m_stageThumbs;
    QVector<QLabel*> m_stageCaptions;

    // Parameters.
    QCheckBox* m_stepsAuto = nullptr;
    QSpinBox* m_steps = nullptr;
    QCheckBox* m_guidanceAuto = nullptr;
    QDoubleSpinBox* m_guidance = nullptr;
    QCheckBox* m_seedRandom = nullptr;
    QSpinBox* m_seed = nullptr;
    QSpinBox* m_gridRes = nullptr;
    QDoubleSpinBox* m_gridScale = nullptr;
    QCheckBox* m_bakeTextureCheck = nullptr;
    QSpinBox* m_textureRes = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;

    // Output.
    QCheckBox* m_addToDbCheck = nullptr;

    // Fixed execution controls and feedback; logs belong to the application
    // console.
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QLabel* m_stageLabel = nullptr;
    QProgressBar* m_progress = nullptr;

    InstantMeshRunResult m_lastResult;
    ecvModelDownloader* m_downloader = nullptr;
    QStringList m_pendingDownloads;
    bool m_downloadInProgress = false;
    bool m_taskRunning = false;
    bool m_samplePreparing = false;
    bool m_acceptDataResults = true;
    bool m_firstShow = true;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
};
