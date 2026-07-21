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
#include <QFile>
#include <QHash>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTextEdit>

struct FreeSplatterBuiltinModel {
    QString displayName;
    QString filename;
    QString downloadUrl;
};

class FreeSplatterDialog : public QDialog {
    Q_OBJECT

public:
    enum class Mode { Reconstruct, ModelInfo };

    enum class ModelType {
        Scene,   // tuned for 2 views; at least 2 required
        Object,  // requires 3+ images
        Unknown
    };

    enum class ExportFieldMode {
        Basic,  // XYZ + RGB + Opacity scalar field
        Full    // XYZ + RGB + Opacity + SH + scale SF + thin-axis normals
    };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    struct Settings {
        Mode mode = Mode::Reconstruct;
        QString modelPath;
        QStringList inputPaths;
        int threads = 0;
        QString device = "auto";
        float opacityThreshold = 0.05f;
        ExportFieldMode exportFieldMode = ExportFieldMode::Basic;
        bool addToDb = true;
        bool estimatePoses = false;
    };

    explicit FreeSplatterDialog(QWidget* parent = nullptr);

    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setTaskStage(const QString& stage, int percent = -1);
    void setRunning(bool running);
    void enableResultButtons(bool hasResult);

    void setDbImages(const QList<DbImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);

    static QString modelCacheDir();

signals:
    void runRequested(const FreeSplatterDialog::Settings& settings);
    void cancelRequested();
    void visualizeRequested();
    void exportPlyRequested();
    void refreshDbImagesRequested();

private slots:
    void onBrowseFile();
    void onBrowseFolder();
    void onModeChanged(int index);
    void onRun();
    void onModelComboChanged(int index);
    void onBrowseCustomModel();
    void onDbListItemChanged(QListWidgetItem* item);
    void onClearInput();
    void onRemoveInputItem();
    void onVisualize();
    void onExportPly();

private:
    void setupUi();
    void populateModelCombo(const QString& keepFilename = QString());
    QString resolveModelPath() const;
    bool ensureModelAvailable();
    void startDownload(const FreeSplatterBuiltinModel& model);
    void cancelDownload();
    void onCancel();
    bool selectModelByFilename(const QString& filename);

    static QVector<FreeSplatterBuiltinModel> builtinModels();
    static QString formatFileSize(qint64 bytes);
    static ModelType modelTypeFromFilename(const QString& filename);
    ModelType currentModelType() const;
    int requiredImageCount() const;
    int currentImageCount() const;
    void updateImageCountStatus();
    void updateRunButtonState();
    void refreshThumbnailStrip();
    void addInputPaths(const QStringList& paths, bool replace);
    void removeInputPath(const QString& path);
    bool isModelReady() const;
    bool isInputValid() const;
    QImage previewForPath(const QString& path) const;

    QComboBox* m_modeCombo = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QPushButton* m_browseCustomModelBtn = nullptr;
    QWidget* m_customModelRow = nullptr;

    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QStringList m_inputPaths;
    QDoubleSpinBox* m_opacityThreshold = nullptr;
    QComboBox* m_exportFieldModeCombo = nullptr;
    QLabel* m_exportFieldLabel = nullptr;
    QCheckBox* m_addToDbCheck = nullptr;
    QCheckBox* m_estimatePosesCheck = nullptr;

    QLabel* m_imageCountLabel = nullptr;

    QListWidget* m_dbImageList = nullptr;
    QLabel* m_dbImageLabel = nullptr;

    QScrollArea* m_thumbScroll = nullptr;
    QWidget* m_thumbContainer = nullptr;

    QLabel* m_taskStatusLabel = nullptr;

    QTextEdit* m_logOutput = nullptr;
    QProgressBar* m_progressBar = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QPushButton* m_visualizeBtn = nullptr;
    QPushButton* m_exportPlyBtn = nullptr;
    QPushButton* m_closeBtn = nullptr;

    QNetworkAccessManager* m_netManager = nullptr;
    QNetworkReply* m_currentDownload = nullptr;
    bool m_autoRunAfterDownload = false;
    bool m_downloadInProgress = false;
    QString m_downloadTargetFilename;
    QString m_downloadTmpPath;
    QFile* m_downloadOutFile = nullptr;
    bool m_taskRunning = false;
    bool m_hasResult = false;

    QHash<QString, QImage> m_dbPreviews;
};
