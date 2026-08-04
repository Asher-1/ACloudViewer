// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVector>
#include <QWidget>

#include "FaceDetectModelContext.h"
#include "FaceDetectTestData.h"
#include "FaceRegistryStore.h"
#include "ecvClickableImageLabel.h"

class FaceRegistryWidget : public QWidget {
    Q_OBJECT

public:
    explicit FaceRegistryWidget(QWidget* parent = nullptr);

    void setModelPath(const QString& path);
    void setDevice(const QString& device);
    void setThreads(int threads);
    QString modelFilename() const;
    QString deviceId() const;
    int threadCount() const;
    void syncModelControlsFrom(const QComboBox* modelCombo,
                               const QComboBox* deviceCombo,
                               const QSpinBox* threadsSpin);
    void rebuildModelCombo(const QStringList& labels,
                           const QStringList& filenames,
                           const QString& currentFilename);
    void rebuildDeviceCombo(const QComboBox* sourceDeviceCombo);
    bool exportAuthResultToDb() const;

    void setAuthThreshold(float value);
    float authThreshold() const;
    void setMinDetectionScore(float score);
    float minDetectionScore() const;
    void setRegistryPath(const QString& path, bool userChosen = false);
    bool isRegistryPathUserChosen() const { return m_registryPathUserChosen; }
    void releaseStoreConnection();
    void fillFriendsTestBundleFields(const FaceDetectFriendsBundle& bundle);
    int registerGalleryEntries(const QVector<FaceDetectGalleryEntry>& entries);
    void refreshList();
    void showVerifySummary(int faceCount, int matchedCount, float threshold);

    FaceRegistryStore* store() { return &m_store; }
    const FaceRegistryStore* store() const { return &m_store; }
    QString registryPath() const;

    void loadSettings();
    void saveSettings() const;

signals:
    void logMessage(const QString& msg);
    void registryChanged();
    void registryPathChanged(const QString& path);
    void authThresholdChanged(float value);
    void minDetectionScoreChanged(float value);
    void modelSelectionChanged(const QString& modelFilename);
    void deviceSelectionChanged(const QString& deviceId);
    void threadCountChanged(int threads);
    void authResultImageReady(const QImage& annotated, const QString& summary);
    void testDataRequested();

public slots:
    void registerFromImagePath(const QString& imagePath, const QImage& thumb);
    void authenticateFromImagePath(const QString& imagePath);

private slots:
    void onBrowseRegisterImage();
    void onBrowseRegistryDb();
    void onBrowseAuthImage();
    void onRegister();
    void onAuthenticate();
    void onRemove();
    void onClear();

private:
    void updateModelPathFromCombo();
    bool registerPersonFromImage(const QString& name,
                                 const QString& imagePath,
                                 float minDetectionScore);
    bool embedImage(const QString& imagePath,
                    std::vector<float>* out,
                    int* outDim,
                    QString* err,
                    float minDetectionScore);
    QString resolveModelPath() const;
    static QString registryPathForModel(const QString& baseDir,
                                        const QString& modelFilename);
    void updateAuthPreview();

    FaceRegistryStore m_store{QStringLiteral("")};
    QString m_modelPath;
    QString m_device = QStringLiteral("auto");
    int m_threadCount = 0;
    float m_minDetectionScore = 0.5f;
    bool m_registryPathUserChosen = false;
    bool m_syncingModelControls = false;

    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;
    QLineEdit* m_registryPathEdit = nullptr;
    QLineEdit* m_nameEdit = nullptr;
    QLineEdit* m_registerImagePath = nullptr;
    QLineEdit* m_authImagePath = nullptr;
    QDoubleSpinBox* m_authThreshold = nullptr;
    QDoubleSpinBox* m_minDetectionScoreSpin = nullptr;
    QCheckBox* m_exportAuthToDbCheck = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QListWidget* m_entryList = nullptr;
    QPlainTextEdit* m_authResultLabel = nullptr;
    QLabel* m_dbStatusLabel = nullptr;
    ecvClickableImageLabel* m_authPreviewLabel = nullptr;

#ifdef AICore_ENABLED
    FaceDetectModelContext m_embedContext;
#endif
};
