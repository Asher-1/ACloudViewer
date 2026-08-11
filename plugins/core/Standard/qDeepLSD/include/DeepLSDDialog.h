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
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QToolButton>
#include <QWidget>

#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

class ecvMainAppInterface;

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
        float minSegmentScore = 0.0f;
        bool addLineVizToDb = true;
        bool addDistanceOverlayToDb = false;
        bool exportPolylinesToDb = false;
    };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    explicit DeepLSDDialog(QWidget* parent = nullptr);

    void setAppInterface(ecvMainAppInterface* app);
    Settings getSettings() const;
    void saveSettings();
    void restoreSettings();
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

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onBrowseImage();
    void onBrowseCustomModel();
    void onModelComboChanged(int index);
    void onRun();
    void onCancel();
    void onDbListActivated(QListWidgetItem* item);

private:
    void setupUi();
    void populateModelCombo(const QString& keepFilename = QString());
    bool selectModelByFilename(const QString& filename);
    QString resolveModelPath() const;
    bool ensureModelAvailable();
    void startDownload(const DeepLSDBuiltinModel& model);
    void cancelDownload();
    void updateImagePreview();
    static QVector<DeepLSDBuiltinModel> builtinModels();
    static QString formatFileSize(qint64 bytes);

    QComboBox* m_modelCombo = nullptr;
    QLabel* m_variantHintLabel = nullptr;
    QLineEdit* m_imagePath = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QDoubleSpinBox* m_minSegmentScore = nullptr;
    ecvClickableImageLabel* m_previewLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QCheckBox* m_addLineVizCheck = nullptr;
    QCheckBox* m_addDistanceOverlayCheck = nullptr;
    QCheckBox* m_exportPolylinesCheck = nullptr;
    QToolButton* m_dbToggleBtn = nullptr;
    QWidget* m_dbContentWidget = nullptr;
    QListWidget* m_dbImageList = nullptr;

    ecvMainAppInterface* m_app = nullptr;
    ecvModelDownloader* m_downloader = nullptr;
    bool m_downloadInProgress = false;
    bool m_autoRunAfterDownload = false;
};
