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
#include <QGroupBox>
#include <QHash>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTextEdit>

struct LightGlueBuiltinModel {
    QString displayName;
    QString filename;
    QString downloadUrl;
    int matcherType = 2;
};

class LightGlueDialog : public QDialog {
    Q_OBJECT

public:
    enum class Mode { Match, ModelInfo };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    struct Settings {
        Mode mode = Mode::Match;
        QString modelPath;
        QStringList inputPaths;
        int threads = 0;
        QString device = "auto";
        double minScore = 0.1;
        int matcherType = 2;
        bool addResultToDb = true;
    };

    static constexpr int kSlotCount = 2;

    explicit LightGlueDialog(QWidget* parent = nullptr);

    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setRunning(bool running);
    void setTaskStage(const QString& stage, int percent = -1);
    void enableExportButton(bool enabled);

    void setDbImages(const QList<DbImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);

    static QString modelCacheDir();

signals:
    void runRequested(const LightGlueDialog::Settings& settings);
    void cancelRequested();
    void exportMatchesRequested();
    void refreshDbImagesRequested();

private slots:
    void onBrowseFile();
    void onBrowseFolder();
    void onBrowseCustomModel();
    void onBrowseSlotImage();
    void onClearSlot();
    void onDbListActivated(QListWidgetItem* item);
    void onFilePoolActivated(QListWidgetItem* item);
    void onAssignToSlot1();
    void onAssignToSlot2();
    void onModelComboChanged(int index);
    void onClearImages();
    void onModeChanged(int index);
    void onRun();
    void onCancel();
    void onExportMatches();

private:
    void setupUi();
    void populateModelCombo(const QString& keepFilename = QString());
    bool selectModelByFilename(const QString& filename);
    QString resolveModelPath() const;
    bool ensureModelAvailable();
    void startDownload(const LightGlueBuiltinModel& model);
    void cancelDownload();
    void updateImageStatus();
    void updateRunButtonState();
    void refreshSlotWidgets();
    void assignToSlot(int slot, const QString& path);
    void clearSlot(int slot);
    void clearAllSlots();
    void autoAssignPaths(const QStringList& paths);
    int senderSlotIndex() const;
    QImage previewForPath(const QString& path) const;
    QString displayNameForPath(const QString& path) const;
    void syncDbListHighlight();
    void setFilePoolPaths(const QStringList& paths);
    void refreshFilePoolList();
    void syncFilePoolHighlight();
    bool assignSelectedToSlot(int slot);
    bool isModelReady() const;
    bool isInputValid() const;

    static QVector<LightGlueBuiltinModel> builtinModels();
    static QString formatFileSize(qint64 bytes);

    QComboBox* m_modeCombo = nullptr;
    QGroupBox* m_ioGroup = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QPushButton* m_browseCustomModelBtn = nullptr;

    QString m_slotPaths[kSlotCount];

    QGroupBox* m_slotGroups[kSlotCount] = {nullptr, nullptr};
    QLabel* m_slotPreview[kSlotCount] = {nullptr, nullptr};
    QLabel* m_slotNameLabel[kSlotCount] = {nullptr, nullptr};
    QLabel* m_imageStatusLabel = nullptr;

    QWidget* m_filePoolGroup = nullptr;
    QListWidget* m_filePoolList = nullptr;
    QListWidget* m_dbImageList = nullptr;
    QStringList m_filePoolPaths;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QDoubleSpinBox* m_minScore = nullptr;
    QCheckBox* m_addToDbCheck = nullptr;

    QLabel* m_downloadLabel = nullptr;
    QLabel* m_taskStatusLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QTextEdit* m_log = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QPushButton* m_exportBtn = nullptr;

    QNetworkAccessManager* m_netManager = nullptr;
    QNetworkReply* m_currentDownload = nullptr;
    QFile* m_downloadOutFile = nullptr;
    bool m_downloadInProgress = false;
    bool m_taskRunning = false;
    bool m_autoRunAfterDownload = false;
    QString m_downloadTargetFilename;
    QString m_downloadTmpPath;

    QHash<QString, QImage> m_dbPreviews;
};
