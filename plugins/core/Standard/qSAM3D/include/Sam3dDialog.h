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
class QSpinBox;
class QLineEdit;
class QPushButton;
class QProgressBar;
class QLabel;

//! Result envelope delivered from the worker thread to the GUI thread.
struct Sam3dRunResult {
    QVector<float> vertices;      //!< xyz per vertex (FlexiCubes surface)
    QVector<uint32_t> triangles;  //!< triangle indices
    QVector<float> splatCenters;  //!< 3 * gaussianCount (world domain)
    QVector<float> splatRgb;      //!< 3 * gaussianCount (0..1 display RGB)
    QByteArray glb;       //!< textured GLB bytes (empty = bake off/failed)
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

//! Multi-object scene envelope: per-object results composed into one world
//! frame by the AICore scene assembler (official make_scene semantics).
struct Sam3dSceneResult {
    QList<Sam3dRunResult> objects;
    QVector<float> sceneCenters;  //!< 3 * N composed splat centers (world)
    QVector<float> sceneRgb;      //!< 3 * N composed display RGB (0..1)
    int sceneSplatCount = 0;
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
        bool importToDb = true;          //!< deliver artifacts into the DB tree
        bool exportPly = false;          //!< optional Gaussian PLY file export
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
    void setRunning(bool running);

signals:
    void runRequested(const Sam3dDialog::Settings& settings);
    void cancelRequested();

private slots:
    void browseImage();
    void browseOutputDir();
    void emitRun();
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
    QCheckBox* m_importDbCheck = nullptr;
    QCheckBox* m_exportPlyCheck = nullptr;
    QCheckBox* m_sceneCheck = nullptr;
    QLineEdit* m_masksDirEdit = nullptr;
    QPushButton* m_browseMasksBtn = nullptr;
    QPushButton* m_runButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QPushButton* m_downloadBtn = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_status = nullptr;
    class ecvModelDownloader* m_downloader = nullptr;
    QStringList m_pendingDownloads;
    bool m_downloadInProgress = false;
};
