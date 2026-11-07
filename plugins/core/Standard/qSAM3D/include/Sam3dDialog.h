// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvMainAppInterface.h>

#include <QDialog>
#include <QStringList>
#include <QVector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QSpinBox;
class QLineEdit;
class QPushButton;
class QProgressBar;
class QLabel;

//! Result envelope delivered from the worker thread to the GUI thread.
struct Sam3dRunResult {
    QVector<float> vertices;      //!< xyz per vertex (FlexiCubes surface)
    QVector<uint32_t> triangles;  //!< triangle indices
    //! Official SLatMeshDecoder use_color payload: 6 floats per vertex —
    //! RGB at [0:3], learned normal at [3:6]. The color source of the
    //! official non-baked GLB path (smooth neural-field colors); empty
    //! falls the texture bake back to nearest-splat sampling.
    QVector<float> meshVertexAttrs;
    QVector<float> splatCenters;  //!< 3 * gaussianCount (world domain)
    QVector<float> splatRgb;      //!< 3 * gaussianCount (0..1 display RGB)
    QByteArray glb;       //!< textured GLB bytes (empty = bake off/failed)
    bool hasPbr = false;  //!< glb came from the official native PBR bake
    QString glbPath;      //!< GLB file path (empty = not exported)
    QString plyPath;      //!< Gaussian PLY export path (empty = none)
    QString sourceImage;  //!< source image path
    QString backend;      //!< resolved AICore backend
    QString dtype;
    double e2eMs = 0.0;  //!< end-to-end pipeline latency
    int gaussianCount = 0;
    int meshVertexCount = 0;
    int meshTriangleCount = 0;
    // ---- scene mode (per-object runs composed into one frame) -------------
    QString objectLabel;  //!< "obj_<i>" in scene mode (empty = single object)
    //! Scene-composer interchange attributes, PLY semantics (empty outside
    //! scene mode): raw f_dc, PLY log-scale, unnormalized PLY rotation and
    //! the PLY opacity logit.
    QVector<float> splatSh0;
    QVector<float> splatLogScale;
    QVector<float> splatRotPly;
    QVector<float> splatOpacityLogit;
    //! Official ScaleShiftInvariant pose receipt, 10 floats (rotation wxyz,
    //! translation, scale); empty when the run carried no pose.
    QVector<float> pose;
};

//! Multi-object scene envelope: the per-object results, each already placed
//! in the scene frame (official make_scene position action) so every entity
//! imports into the DB where it belongs. The AICore scene assembler stays
//! available at the ABI (exercised by tests/parity scripts); the plugin
//! keeps one independent entity per object instead of a merged cloud.
struct Sam3dSceneResult {
    QList<Sam3dRunResult> objects;
    QString sourceImage;
    QString backend;
    QString dtype;
    double e2eMs = 0.0;  //!< whole-scene wall time
};

//! Minimal, dialog-driven front end for the AICore sam3d task.
class Sam3dDialog : public QDialog {
    Q_OBJECT

public:
    struct Settings {
        QString imagePath;
        QString modelsDir;
        QString outputDir;
        //! Scene mode: directory of per-object binary masks (<n>.png, same
        //! resolution as the image). Empty = single-object run.
        QString masksDir;
        //! Scene mode, automatic segmentation: SAM3 text prompts
        //! (one instance set per prompt). Non-empty overrides the
        //! mask directory (fully end-to-end: no hand-drawn masks).
        QStringList conceptPrompts;
        //! Segmentation thresholds for the prompt path
        //! (<= 0 = the AICore context defaults).
        float samScoreThreshold = 0.0f;
        float samNmsThreshold = 0.0f;
        //! Cross-prompt duplicate rejection IoU (a detection from a
        //! different prompt covering an already accepted object);
        //! <= 0 keeps the balanced plugin default (0.5).
        float samDedupIouThreshold = 0.5f;
        QString device = QStringLiteral("auto");
        int dtypeIndex = 2;      //!< 0=f16 1=q8_0 2=q4_k
        int rmbgDtypeIndex = 0;  //!< 0=q8_0 1=f16
        int steps = 25;
        int seed = 42;
        bool useRmbg = true;
        // Output artifacts (pipeline gating): the colored point cloud is the
        // mandatory generation product; the textured mesh extends the
        // pipeline with the FlexiCubes decode + UV-atlas bake stages.
        bool outputPointCloud = true;    //!< colored splat cloud -> DB
        bool outputTexturedMesh = true;  //!< mesh decode + bake -> DB + GLB
        //! Official native PBR texture bake (AICore_USE_SAM3D_PBR builds,
        //! CUDA only): replaces the projection bake with the official
        //! 100-view render + 2500-step optimized base-color atlas.
        bool pbrBake = false;
        //! PBR bake quality knobs (reconstruction side; 0 = the AICore
        //! defaults: library-chosen texture size, 2500 optimization steps).
        //! Lower steps trade bake precision for speed (~20 ms/step/object).
        int pbrTextureSize = 0;
        int pbrBakeSteps = 0;
        //! Scene mode diagnostic: composite every accepted object mask over
        //! the source image (one color + label per object) and import the
        //! preview into the DB tree before the per-object runs start.
        bool segPreview = true;
        //! SAM3 segmentation model filename (text-prompt capable sam3
        //! family, AICore sam3 catalog; best default = q8_0). Selected in
        //! the dialog's "SAM3 model" combobox.
        QString sam3ModelFile = QStringLiteral("sam3-q8_0.gguf");
        //! Run-mode flag set by the "Generate mask preview" button (not a
        //! UI checkbox): the worker stops after the segmentation preview and
        //! skips the per-object reconstruction entirely.
        bool previewOnly = false;
        bool importToDb = true;  //!< deliver artifacts into the DB tree
        bool exportPly = false;  //!< optional Gaussian PLY file export
    };

    explicit Sam3dDialog(QWidget* parent = nullptr);
    ~Sam3dDialog() override;

    void setAppInterface(ecvMainAppInterface* app) { m_app = app; }

    Settings settings() const;
    //! Logs to the shared application console (no in-dialog log panel).
    //! The optional level maps onto ecvMainAppInterface::dispToConsole.
    void appendLog(const QString& message,
                   ecvMainAppInterface::ConsoleMessageLevel level =
                           ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    void setProgress(int percent);
    //! statusText overrides the default "Running the SAM 3D pipeline..."
    //! line (the preview-only run shows its own message).
    void setRunning(bool running, const QString& statusText = QString());

signals:
    void runRequested(const Sam3dDialog::Settings& settings);
    //! Two-step scene workflow, step 1: segmentation + preview ONLY (no
    //! reconstruction). The user iterates on prompts until the preview is
    //! satisfactory, then presses Generate for the full pipeline.
    void previewMasksRequested(const Sam3dDialog::Settings& settings);
    void cancelRequested();

private slots:
    void browseImage();
    void browseOutputDir();
    void emitRun();
    void emitPreviewMasks();
    //! "Try sample data": extract / download the official SAM 3D Objects
    //! scene set (notebook/images/<scene>/image.png) and fill the picker.
    void onTestDataClicked();
    //! "Download models": queue the current-dtype model set from the AICore
    //! catalog (HF Asher-1/SAM_3D_OBJECTS_GGUF) into the shared cache.
    void onDownloadModels();
    void downloadNextModel();

private:
    void restoreDefaults();
    void populateTestImage();

    ecvMainAppInterface* m_app = nullptr;
    QLineEdit* m_imageEdit = nullptr;
    QLineEdit* m_outputDirEdit = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QComboBox* m_dtypeCombo = nullptr;
    QComboBox* m_rmbgDtypeCombo = nullptr;
    QSpinBox* m_stepsSpin = nullptr;
    QSpinBox* m_seedSpin = nullptr;
    QCheckBox* m_rmbgCheck = nullptr;
    QCheckBox* m_pointCloudCheck = nullptr;
    QCheckBox* m_texturedMeshCheck = nullptr;
    QCheckBox* m_pbrCheck = nullptr;
    QComboBox* m_pbrTextureCombo = nullptr;
    QSpinBox* m_pbrStepsSpin = nullptr;
    QCheckBox* m_segPreviewCheck = nullptr;
    QCheckBox* m_importDbCheck = nullptr;
    QCheckBox* m_exportPlyCheck = nullptr;
    QCheckBox* m_sceneCheck = nullptr;
    QLineEdit* m_masksDirEdit = nullptr;
    QPushButton* m_browseMasksBtn = nullptr;
    QLineEdit* m_promptsEdit = nullptr;
    QDoubleSpinBox* m_samScoreSpin = nullptr;
    QDoubleSpinBox* m_samNmsSpin = nullptr;
    QDoubleSpinBox* m_samDedupSpin = nullptr;
    QComboBox* m_sam3ModelCombo = nullptr;
    QPushButton* m_runButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QPushButton* m_previewMasksBtn = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QPushButton* m_downloadBtn = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_status = nullptr;
    class ecvModelDownloader* m_downloader = nullptr;
    QStringList m_pendingDownloads;
    bool m_downloadInProgress = false;
    //! Mirrors the last setRunning() call: the preview button's enabled
    //! state is (scene mode && !running), and BOTH toggles must be able to
    //! recompute it (the scene checkbox can flip while no run is active,
    //! so setRunning alone cannot own this state).
    bool m_running = false;
};
