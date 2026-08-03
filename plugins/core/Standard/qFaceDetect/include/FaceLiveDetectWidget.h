// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QThread>
#include <QTimer>
#include <QWidget>

#include <memory>

#include "FaceDetectWorker.h"
#include "FaceLiveDetectInferWorker.h"
#include "FaceRegistryStore.h"
#include "ecvClickableImageLabel.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/videoio.hpp>
#endif

/** Live camera / video preview with throttled AICore face detection. */
class FaceLiveDetectWidget : public QWidget {
    Q_OBJECT

public:
    enum class StreamMode { Detect, Recognize };

    struct Config {
        QString modelPath;
        QString device = QStringLiteral("auto");
        int threads = 0;
        float minDetectionScore = 0.5f;
        FaceDetectWorker::Mode mode = FaceDetectWorker::Mode::Detect;
        StreamMode streamMode = StreamMode::Detect;
        float recognizeMaxDistance = 0.52f;
        FaceRegistryStore* registry = nullptr;
    };

    explicit FaceLiveDetectWidget(QWidget* parent = nullptr);
    ~FaceLiveDetectWidget() override;

    void setConfig(const Config& config);
    Config config() const { return m_config; }

    void setRegistryStore(FaceRegistryStore* store);

    bool startCamera(int deviceIndex = 0);
    bool startVideoFile(const QString& path);
    void stopStream();
    bool isActive() const;

    bool hasSnapshot() const;
    FaceDetectRunResult lastSnapshot() const;

    enum class InputSource { Camera, VideoFile };

    InputSource inputSource() const;
    int selectedCameraIndex() const;
    QString videoFilePath() const;
    void setVideoFilePath(const QString& path, bool userChosen = true);
    void selectVideoFileSource();

    void setRegistryPath(const QString& path, bool userChosen = false);
    QString registryPath() const;
    bool isVideoPathUserChosen() const { return m_videoPathUserChosen; }
    bool isRegistryPathUserChosen() const { return m_registryPathUserChosen; }

    void setStreamMode(StreamMode mode);
    void setMatchThreshold(float value);
    void setMinDetectionScore(float value);

    void syncModelControlsFrom(const QComboBox* modelCombo,
                               const QComboBox* deviceCombo,
                               const QSpinBox* threadsSpin);
    void rebuildModelCombo(const QStringList& labels, const QStringList& filenames,
                           const QString& currentFilename);
    void rebuildDeviceCombo(const QComboBox* sourceDeviceCombo);
    void setModelPath(const QString& path);
    void setDevice(const QString& device);
    void setThreads(int threads);
    QString modelFilename() const;
    QString deviceId() const;
    int threadCount() const;
    QString resolveModelPath() const;

    void loadSettings();
    void saveSettings() const;

    static bool isAvailable();

signals:
    void logMessage(const QString& msg);
    void snapshotUpdated(const FaceDetectRunResult& result);
    void captureToDbRequested(const FaceDetectRunResult& result);
    void streamStarted();
    void streamStopped();
    void streamModeChanged(StreamMode mode);
    void matchThresholdChanged(float value);
    void minDetectionScoreChanged(float value);
    void modelSelectionChanged(const QString& modelFilename);
    void deviceSelectionChanged(const QString& deviceId);
    void threadCountChanged(int threads);
    void testDataRequested();
    void registryPathEdited(const QString& path);

public slots:
    void captureSnapshotToDb();

private slots:
    void onSourceChanged(int index);
    void onBrowseVideo();
    void onStreamModeChanged(int index);
    void processFrame();
    void onInferComplete(FaceLiveDetectInferWorker::Result result);

private:
    void setupUi();
    void updateThresholdUi();
    void updateRegistryUi();
    void updateModelPathFromCombo();
    void submitInferJob(const QImage& displayRgb, const QImage& inferRgb,
                        float inferScale);
    void shutdownInferThread();

#ifdef HAS_OPENCV_FACE_CAPTURE
    cv::VideoCapture m_capture;
#endif

    Config m_config;
    ecvClickableImageLabel* m_previewLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;
    QComboBox* m_modeCombo = nullptr;
    QComboBox* m_sourceCombo = nullptr;
    QComboBox* m_cameraCombo = nullptr;
    QWidget* m_cameraRow = nullptr;
    QWidget* m_videoRow = nullptr;
    QLineEdit* m_videoPathEdit = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    QDoubleSpinBox* m_recognizeThreshold = nullptr;
    QDoubleSpinBox* m_minDetectionScore = nullptr;
    QLabel* m_matchDistLabel = nullptr;
    QLabel* m_minScoreLabel = nullptr;
    QWidget* m_registryRow = nullptr;
    QLineEdit* m_registryPathEdit = nullptr;
    QPushButton* m_captureBtn = nullptr;

    bool m_videoPathUserChosen = false;
    bool m_registryPathUserChosen = false;
    bool m_syncingModelControls = false;

    QTimer* m_frameTimer = nullptr;
    bool m_streamActive = false;
    bool m_camerasEnumerated = false;
    int m_inferSkip = 0;
    bool m_inferBusy = false;

    QThread* m_inferThread = nullptr;
    FaceLiveDetectInferWorker* m_inferWorker = nullptr;

    FaceDetectRunResult m_lastSnapshot;
    bool m_hasSnapshot = false;
};
