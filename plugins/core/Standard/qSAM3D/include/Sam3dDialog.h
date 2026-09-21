// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QStringList>
#include <QVector>

class QCheckBox;
class QComboBox;
class QSpinBox;
class QLineEdit;
class QPushButton;
class QPlainTextEdit;
class QProgressBar;
class QLabel;
class ecvMainAppInterface;

//! Result envelope delivered from the worker thread to the GUI thread.
struct Sam3dRunResult {
    QVector<float> vertices;      //!< xyz per vertex (FlexiCubes surface)
    QVector<uint32_t> triangles;  //!< triangle indices
    QString plyPath;              //!< Gaussian PLY export written by AICore
    QString sourceImage;          //!< source image path
    QString backend;              //!< resolved AICore backend
    QString dtype;
    double e2eMs = 0.0;  //!< end-to-end pipeline latency
    int gaussianCount = 0;
    int meshVertexCount = 0;
    int meshTriangleCount = 0;
};

//! Minimal, dialog-driven front end for the AICore sam3d task.
class Sam3dDialog : public QDialog {
    Q_OBJECT

public:
    struct Settings {
        QString imagePath;
        QString modelsDir;
        QString outputDir;
        QString device = QStringLiteral("auto");
        int dtypeIndex = 2;  //!< 0=f16 1=q8_0 2=q4_k
        int steps = 25;
        int seed = 42;
        bool useRmbg = true;
        bool generateMesh = true;
    };

    explicit Sam3dDialog(QWidget* parent = nullptr);
    ~Sam3dDialog() override;

    void setAppInterface(ecvMainAppInterface* app) { m_app = app; }

    Settings settings() const;
    void appendLog(const QString& message);
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
    QSpinBox* m_stepsSpin = nullptr;
    QSpinBox* m_seedSpin = nullptr;
    QCheckBox* m_rmbgCheck = nullptr;
    QCheckBox* m_meshCheck = nullptr;
    QPushButton* m_runButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QPushButton* m_downloadBtn = nullptr;
    QPlainTextEdit* m_log = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_status = nullptr;
    class ecvModelDownloader* m_downloader = nullptr;
    QStringList m_pendingDownloads;
    bool m_downloadInProgress = false;
};
