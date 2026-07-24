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
#include <QFile>
#include <QLabel>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTextEdit>

struct DA3BuiltinModel {
    QString displayName;
    QString filename;
    QString downloadUrl;
};

class DA3Dialog : public QDialog {
    Q_OBJECT

public:
    enum class Mode {
        DepthSingle,
        DepthPose,
        DepthMultiView,
        Reconstruct,
        ExportGLB,
        ExportCOLMAP,
        Quantize,
        ModelInfo
    };

    struct Settings {
        Mode mode = Mode::DepthSingle;
        QString modelPath;
        QString metricModelPath;
        QStringList inputPaths;
        QString outputDir;
        int threads = 0;
        QString device = "auto";  // auto | cpu | sycl | vulkan | cuda | metal
        bool invertDepth = true;
        bool unproject3D = true;
        int downsampleStep = 1;
        bool colmapBinary = true;
        QString quantInputPath;
        QString quantOutputPath;
        QString quantType;
        QString dbImageName;
    };

    explicit DA3Dialog(QWidget* parent = nullptr);

    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setRunning(bool running);
    void enableExportButtons(bool hasResults);

    void setDbImages(const QStringList& imageNames);

    static QString modelCacheDir();

signals:
    void runRequested(const DA3Dialog::Settings& settings);
    void cancelRequested();
    void exportDepthRequested();
    void exportAllDepthsRequested(const QString& outputDir);
    void refreshDbImagesRequested();

private slots:
    void onBrowseFile();
    void onBrowseFolder();
    void onBrowseOutput();
    void onModeChanged(int index);
    void onRun();
    void onModelComboChanged(int index);
    void onMetricModelComboChanged(int index);
    void onBrowseCustomModel();
    void onBrowseCustomMetricModel();
    void onDbImageSelected(int index);
    void onExportAllDepths();

private:
    void setupUi();
    void populateModelCombos(const QString& keepModelFilename = QString(),
                             const QString& keepMetricFilename = QString());
    QString resolveModelPath(QComboBox* combo, QLineEdit* customEdit) const;
    bool ensureAllModelsAvailable();
    void startDownload(const DA3BuiltinModel& model);
    void cancelDownload();
    void onCancel();
    bool selectComboItemByFilename(QComboBox* combo, const QString& filename);

    enum class ModelSize { Base, Large, Giant, Unknown };
    static ModelSize modelSizeFromFilename(const QString& filename);

    static QVector<DA3BuiltinModel> builtinModels();
    static QVector<DA3BuiltinModel> builtinMetricModels();
    static QString formatFileSize(qint64 bytes);
    void syncModeModelConstraints();

    QComboBox* m_modeCombo = nullptr;

    QComboBox* m_modelCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QPushButton* m_browseCustomModelBtn = nullptr;
    QWidget* m_customModelRow = nullptr;

    QComboBox* m_metricModelCombo = nullptr;
    QLineEdit* m_customMetricPath = nullptr;
    QPushButton* m_browseCustomMetricBtn = nullptr;
    QWidget* m_customMetricRow = nullptr;
    QLabel* m_metricLabel = nullptr;

    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QLineEdit* m_inputPath = nullptr;
    QLineEdit* m_outputDir = nullptr;
    QCheckBox* m_invertCheck = nullptr;
    QCheckBox* m_unprojectCheck = nullptr;
    QSpinBox* m_downsampleStep = nullptr;
    QLabel* m_downsampleLabel = nullptr;
    QCheckBox* m_colmapBinaryCheck = nullptr;

    QComboBox* m_dbImageCombo = nullptr;
    QLabel* m_dbImageLabel = nullptr;

    QWidget* m_quantGroup = nullptr;
    QLineEdit* m_quantInput = nullptr;
    QLineEdit* m_quantOutput = nullptr;
    QComboBox* m_quantType = nullptr;

    QTextEdit* m_logOutput = nullptr;
    QProgressBar* m_progressBar = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QPushButton* m_exportDepthBtn = nullptr;
    QPushButton* m_exportAllBtn = nullptr;
    QPushButton* m_closeBtn = nullptr;

    QNetworkAccessManager* m_netManager = nullptr;
    QNetworkReply* m_currentDownload = nullptr;
    QVector<DA3BuiltinModel> m_downloadQueue;
    bool m_autoRunAfterDownload = false;
    bool m_downloadInProgress = false;
    QString m_downloadTargetFilename;
    QFile* m_downloadOutFile = nullptr;
};
