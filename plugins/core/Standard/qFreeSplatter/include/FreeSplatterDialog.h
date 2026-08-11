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
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTabWidget>
#include <QToolButton>

#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

class ecvMainAppInterface;
class FaceCaptureWidget;

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
        struct IdentityInput {
            QString id;
            QString name;
            QStringList inputPaths;
        };

        Mode mode = Mode::Reconstruct;
        QString modelPath;
        QStringList inputPaths;
        int threads = 0;
        QString device = "auto";
        float opacityThreshold = 0.05f;
        ExportFieldMode exportFieldMode = ExportFieldMode::Basic;
        bool addToDb = true;
        bool estimatePoses = false;
        bool removeBackground = false;
        int maxViews = 0;  // 0 = auto (Scene:2, Object-3DGS:16, Object-2DGS:24)
        QString identityId;
        QString identityName;
        QList<IdentityInput> identityInputs;
    };

    explicit FreeSplatterDialog(QWidget* parent = nullptr);

    void setAppInterface(ecvMainAppInterface* app);
    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setTaskStage(const QString& stage, int percent = -1);
    void setRunning(bool running);
    void enableResultButtons(bool hasResult);
    // Captured face frames are task-private biometric data. The controller
    // calls this after the final identity task has consumed them.
    void clearFaceCaptureTransientInputs();

    void setDbImages(const QList<DbImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);

    static QString modelCacheDir();
    void refreshModelList();

signals:
    void runRequested(const FreeSplatterDialog::Settings& settings);
    void cancelRequested();
    void visualizeRequested();
    void exportPlyRequested();
    void refreshDbImagesRequested();

protected:
    void closeEvent(QCloseEvent* event) override;

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
    static bool isObject2dgsModel(const QString& filename);
    ModelType currentModelType() const;
    QString currentModelFilename() const;
    void updateObjectModelHint();
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
    void adaptTabWidgetHeight();

    QComboBox* m_modeCombo = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QPushButton* m_browseCustomModelBtn = nullptr;
    QWidget* m_customModelRow = nullptr;

    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QSpinBox* m_maxViewsSpin = nullptr;
    QStringList m_inputPaths;
    QList<Settings::IdentityInput> m_identityInputs;
    QString m_faceCaptureExportDir;
    QDoubleSpinBox* m_opacityThreshold = nullptr;
    QComboBox* m_exportFieldModeCombo = nullptr;
    QLabel* m_exportFieldLabel = nullptr;
    QCheckBox* m_addToDbCheck = nullptr;
    QCheckBox* m_estimatePosesCheck = nullptr;

    QLabel* m_imageCountLabel = nullptr;
    QLabel* m_objectHintLabel = nullptr;
    QCheckBox* m_removeBgCheck = nullptr;

    QToolButton* m_dbToggleBtn = nullptr;
    QWidget* m_dbContentWidget = nullptr;
    QListWidget* m_dbImageList = nullptr;

    QScrollArea* m_thumbScroll = nullptr;
    QWidget* m_thumbContainer = nullptr;

    QLabel* m_taskStatusLabel = nullptr;

    QProgressBar* m_progressBar = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QPushButton* m_visualizeBtn = nullptr;
    QPushButton* m_exportPlyBtn = nullptr;
    QPushButton* m_closeBtn = nullptr;

    ecvMainAppInterface* m_app = nullptr;
    ecvModelDownloader* m_downloader = nullptr;
    bool m_autoRunAfterDownload = false;
    bool m_downloadInProgress = false;
    QString m_downloadTargetFilename;
    bool m_taskRunning = false;
    bool m_hasResult = false;
    QString m_lastTaskError;

    // --- Test data download state ---
    bool m_testDataDownloadInProgress = false;
    int m_testDataDatasetKind = -1;  // ecvTestDataRepository::Dataset

    QHash<QString, QImage> m_dbPreviews;

    // --- Face capture tab (conditional on HAS_OPENCV_FACE_CAPTURE) ---
    QTabWidget* m_inputTabWidget = nullptr;
    QWidget* m_imagesTab = nullptr;
    QScrollArea* m_faceCaptureScroll = nullptr;
    int m_activeInputTabHeight = -1;
    FaceCaptureWidget* m_faceCaptureWidget = nullptr;
    QPushButton* m_faceStartBtn = nullptr;
    QPushButton* m_faceStopBtn = nullptr;
    QPushButton* m_faceRestartBtn = nullptr;
    QPushButton* m_faceResetBtn = nullptr;
    void onFaceStartCamera();
    void onFaceStopCamera();
    void onFaceRestart();
    void onFaceReset();
    void onFaceCaptureComplete();
    void clearFaceCaptureExportDir();

    // --- Test data: FriendsFaces (video) + Monstree (images) ---
    void ensureFriendsTestData();
    void ensureMonstreeTestData();
    void onTestDataDownloadFinished(bool success, int kind);
    void onTestDataExtractionFinished(bool success, int kind);
    void loadTestDataAfterExtract(int kind);
};
