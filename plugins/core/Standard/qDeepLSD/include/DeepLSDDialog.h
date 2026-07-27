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
#include <QListWidget>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTextEdit>
#include <QToolButton>
#include <QWidget>

struct DeepLSDBuiltinModel {
    QString displayName;
    QString filename;
    QString downloadUrl;
};

class DeepLSDDialog : public QDialog {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = "auto";
        bool addResultToDb = true;
    };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    explicit DeepLSDDialog(QWidget* parent = nullptr);

    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setRunning(bool running);
    void setDbImages(const QList<DbImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);
    void refreshModelList();

    static QString modelCacheDir();

signals:
    void runRequested(const DeepLSDDialog::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();

private slots:
    void onBrowseImage();
    void onBrowseCustomModel();
    void onModelComboChanged(int index);
    void onRun();
    void onCancel();
    void onDbListActivated(QListWidgetItem* item);

private:
    void setupUi();
    void populateModelCombo();
    QString resolveModelPath() const;
    bool ensureModelAvailable();
    void startDownload(const DeepLSDBuiltinModel& model);
    void cancelDownload();
    void updateImagePreview();
    static QVector<DeepLSDBuiltinModel> builtinModels();
    static QString formatFileSize(qint64 bytes);

    QComboBox* m_modelCombo = nullptr;
    QLabel* m_quantWarningLabel = nullptr;
    QLabel* m_variantHintLabel = nullptr;
    QLineEdit* m_imagePath = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QLabel* m_previewLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QTextEdit* m_log = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QCheckBox* m_addToDbCheck = nullptr;
    QToolButton* m_dbToggleBtn = nullptr;
    QWidget* m_dbContentWidget = nullptr;
    QListWidget* m_dbImageList = nullptr;

    QNetworkAccessManager* m_netManager = nullptr;
    QNetworkReply* m_currentDownload = nullptr;
    QFile* m_downloadOutFile = nullptr;
    bool m_downloadInProgress = false;
    bool m_autoRunAfterDownload = false;
    QString m_downloadTmpPath;
};
