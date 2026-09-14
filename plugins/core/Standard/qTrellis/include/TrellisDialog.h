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

#include "TrellisModelCatalog.h"
#include "TrellisWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;

/** Left-hand page ids of the two-page stack (qYOLO-style task list). */
enum class TrellisPage {
    Generate = 0,  ///< image -> 3D end-to-end (per-step preview + one-click)
    Export = 1     ///< GLB / print export of the last generation result
};

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
        /** Weight-precision chain ("q8" default | "f16"). Resolved by
         *  TrellisHelpers::resolvePresetFiles into concrete GGUF paths. */
        QString quantization = QStringLiteral("q8");
        /** Full (Generate + GLB) or GeometryOnly (Generate 3D — the GLB bake
         *  is skipped and can be run later from the Export page). */
        TrellisRunMode runMode = TrellisRunMode::Full;
    };

    explicit TrellisDialog(QWidget* parent = nullptr);
    ~TrellisDialog() override;

    void setAppInterface(ecvMainAppInterface* app);
    Settings getSettings() const;
    void runWithMode(TrellisRunMode mode);
    void appendLog(const QString& msg);
    void setProgressStage(int stageId,
                          const QString& stage,
                          int step,
                          int total);
    void setRunning(bool running);
    void setImagePreview(const QImage& image);
    void refreshModelState();

    /** Live intermediate-stage preview (voxel set / mesh keyframe) rendered
     *  on the worker thread; updates the matching step chip in the strip. */
    void setStagePreview(const TrellisStagePreview& preview);
    /** Reset the six-step strip and seed the first chip with the source
     *  image (call at generation start). */
    void resetStageStrip(const QImage& input);
    /** Strip slot states (public: the plugin marks completion). */
    enum StageSlotState { kStagePending = 0, kStageActive = 1, kStageDone = 2 };

    /** Mark a strip slot done / active (see the kStage* constants). */
    void setStageState(int slot, int state);
    /** Fill the strip slots that only exist after a generation completes
     *  (Preprocess = RMBG-matted image; Mesh = guaranteed final render;
     *  Texture / GLB = textured render). Safe to skip empty sources. */
    void applyResultToStrip(const TrellisRunResult& result);
    /** Open the orbit 3D viewer for a mesh-bearing strip slot (3 = Mesh,
     *  4 = Texture, 5 = GLB); falls back to the image zoom when no result
     *  mesh is available. */
    void openMesh3D(int slot);
    /** Remember the last generation result for the export page. */
    void setLastResult(const TrellisRunResult& result) {
        m_lastResult = result;
    }
    const TrellisRunResult& lastResult() const { return m_lastResult; }
    /** Export-page settings + info line. */
    int exportTextureSize() const {
        return m_exportTextureSize ? m_exportTextureSize->value() : 2048;
    }
    int exportComponentFilter() const {
        return m_exportComponentFilter ? m_exportComponentFilter->currentIndex()
                                       : 0;
    }
    /** Re-bake destination (persisted): DB-tree entity (default), GLB file
     *  only, or both. */
    enum ExportDestination {
        kExportDb = 0,
        kExportFile = 1,
        kExportDbAndFile = 2
    };
    int exportDestination() const {
        return m_exportDestination ? m_exportDestination->currentIndex()
                                   : kExportDb;
    }
    void updateExportInfo(const TrellisRunResult& result);
    /** Export-page busy state: both action buttons disable while a print
     *  wrap runs, restored per build-time availability when done. */
    void setPrintWrapRunning(bool running);
    /** Export-page bake progress: busy bar + AICore bake stage text (the
     *  bake reports stage boundaries with elapsed seconds, no percentage). */
    void setBakeProgress(const QString& stage, double elapsedS);
    /** Export-page busy state: shows the bake progress section while a bake
     *  runs and sets its final status text when it ends. */
    void setExportBusy(bool busy, const QString& finalText = QString());

signals:
    void runRequested(const TrellisDialog::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();
    void browseImageRequested();
    /** Re-bake request on the last result (export page). */
    void exportRequested();
    /** Watertight Alpha-Wrap print mesh of the last result (export page). */
    void printWrapRequested();

private slots:
    void onBrowseImage();
    void onBrowseSaveDir();
    void onPresetChanged(int index);
    void onQuantChanged(int index);
    void onRun();
    void onRunOneClick();
    void onCancel();
    void onDownloadModels();
    void onTestDataClicked();
    void onTestImageSelected(int index);
    void onTestDataExtracted();
    void onPageChanged(int row);

protected:
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    enum class PendingAction { None, Run, OneClick };

    void buildGeneratePage(QWidget* page);
    void buildExportPage(QWidget* page);
    void setupUi();
    void loadSettings();
    void saveSettings() const;
    void updateModelStatus();
    QStringList missingPresetFiles() const;
    void downloadNextModel();
    void updateImagePreview();
    void ensureImage2MeshDataset();  // download/extract on demand
    void populateTestImages();       // fill combo from extracted dataset
    /** The quantization chain id selected in the UI ("q8" | "f16"). */
    QString quantization() const;

    ecvMainAppInterface* m_app = nullptr;

    // Left-hand page navigation (qYOLO-style) driving the stack.
    QListWidget* m_pageList = nullptr;
    QStackedWidget* m_pageStack = nullptr;

    // Input.
    QLineEdit* m_imagePath = nullptr;
    ecvClickableImageLabel* m_imagePreview = nullptr;
    QImage m_lastPreviewImage;  // full-res copy for the Source strip chip
    QPushButton* m_browseImageBtn = nullptr;
    QPushButton* m_useTestDataBtn = nullptr;
    QComboBox* m_testImageCombo = nullptr;  // pick one of the sample images

    // Model preset.
    QComboBox* m_presetCombo = nullptr;
    QComboBox* m_quantCombo = nullptr;  // q8 (default) / f16 chain
    QLabel* m_modelStatus = nullptr;
    QPushButton* m_downloadBtn = nullptr;
    QCheckBox* m_textureCheck = nullptr;
    QCheckBox* m_rmbgCheck = nullptr;

    // End-to-end step strip: six chips (source / preprocess / voxels / mesh /
    // texture / GLB), each a clickable thumbnail + caption driven by the live
    // AICore preview callbacks and the stage progress; clicking a chip opens
    // the zoomable full-size preview (shared UI spec §13.3).
    QVector<ecvClickableImageLabel*> m_stageThumbs;
    QVector<QLabel*> m_stageCaptions;
    TrellisRunMode m_runMode = TrellisRunMode::Full;
    /** Raw T2VOX01 payload of the last voxel preview (Voxels chip 3D). */
    QByteArray m_lastVoxelBlob;
    // The Mesh/Texture/GLB chips' 3D viewer reads m_lastResult (the export
    // page already owned it) via applyResultToStrip's setLastResult().

    // Export page (last-result GLB / print-wrap controls).
    QSpinBox* m_exportTextureSize = nullptr;
    QComboBox* m_exportComponentFilter = nullptr;
    QComboBox* m_exportDestination = nullptr;
    QLabel* m_exportInfo = nullptr;
    QPushButton* m_rebakeBtn = nullptr;
    QPushButton* m_printWrapBtn = nullptr;
    /** Export-page bake progress section (busy bar + stage text + cancel). */
    QLabel* m_exportStageLabel = nullptr;
    QProgressBar* m_exportProgress = nullptr;
    QPushButton* m_exportCancelBtn = nullptr;
    // Build-time availability of the export-page actions (the AICore-enabled
    // re-bake and the CGAL print wrap), set by buildExportPage and honored
    // by setRunning / setPrintWrapRunning on every unlock path.
    bool m_rebakeAvailable = false;
    bool m_printWrapAvailable = false;

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

    TrellisRunResult m_lastResult;  // export-page source (last generation)
    ecvModelDownloader* m_downloader = nullptr;
    QStringList m_pendingDownloads;
    bool m_downloadInProgress = false;
    bool m_firstShow = true;  // lock the dialog size on first show (see §13.4)
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
};
