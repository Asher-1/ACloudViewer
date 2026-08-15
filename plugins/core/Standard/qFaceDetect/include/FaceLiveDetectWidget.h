// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QAtomicInt>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFutureWatcher>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMutex>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QThread>
#include <QTimer>
#include <QVector>
#include <QWidget>
#include <memory>
#include <utility>

#include "FaceDetectWorker.h"
#include "FaceLiveDetectInferWorker.h"
#include "FaceRegistryStore.h"
#include "VideoPlaybackWidget.h"

/** Live camera / video preview with throttled AICore face detection.
 *  The playback panel (preview, source selection, seek/speed controls and
 *  the background decode pipeline) is inherited from VideoPlaybackWidget;
 *  this widget only adds the model controls, inference worker and overlays. */
class FaceLiveDetectWidget : public VideoPlaybackWidget {
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
        float recognizeMaxDistance = 0.65f;
        FaceRegistryStore* registry = nullptr;
    };

    explicit FaceLiveDetectWidget(QWidget* parent = nullptr);
    ~FaceLiveDetectWidget() override;

    void setConfig(const Config& config);
    Config config() const { return m_config; }

    void setRegistryStore(FaceRegistryStore* store);

    // Playback control is inherited from VideoPlaybackWidget
    // (startCamera / startVideoFile / restartVideoFile / stopStream /
    //  resumePlayback / inputSource / selectedCameraIndex / videoFilePath /
    //  isActive ...).
    // Compatibility overload kept for FaceDetectDialog:
    using VideoPlaybackWidget::setVideoFilePath;
    void setVideoFilePath(const QString& path, bool userChosen);

    bool hasSnapshot() const;
    FaceDetectRunResult lastSnapshot() const;

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
    void rebuildModelCombo(const QStringList& labels,
                           const QStringList& filenames,
                           const QString& currentFilename);
    void rebuildDeviceCombo(const QComboBox* sourceDeviceCombo);
    void setModelPath(const QString& path);
    void setDevice(const QString& device);
    void setThreads(int threads);
    QString modelFilename() const;
    QString deviceId() const;
    int threadCount() const;
    QString resolveModelPath() const;
    QPushButton* testDataButton() const { return m_testDataBtn; }

    void loadSettings();
    void saveSettings() const;

    static bool isAvailable();

signals:
    void logMessage(const QString& msg);
    void snapshotUpdated(const FaceDetectRunResult& result);
    void captureToDbRequested(const FaceDetectRunResult& result);
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
    void onStreamModeChanged(int index);
    void onInferComplete(FaceLiveDetectInferWorker::Result result);

protected:
    // ---- video_base hooks -------------------------------------------------
    void onFrameDecoded(cv::Mat& frame, int frameIndex) override;
    void onDisplayFrame(QImage& display, int frameIndex) override;
    void onVideoLooped() override;
    void onStreamReset() override;
    void onStreamResumed() override;
    void onStreamStopping() override;
    bool onPrepareStream() override;
    void onSourceChanged(InputSource source) override;

private:
    void setupUi();
    void updateThresholdUi();
    void updateRegistryUi();
    void updateModelPathFromCombo();
    void submitInferJob(const QImage& inferRgb, float inferScale);
    void shutdownInferThread();
    void drawLiveOverlay(QImage& frame);

    Config m_config;
    ecvClickableImageLabel* m_previewLabel = nullptr;  // cached base accessor
    QLabel* m_statusLabel = nullptr;                   // cached base accessor
    QProgressBar* m_preloadProgress = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;
    QComboBox* m_modeCombo = nullptr;
    QDoubleSpinBox* m_recognizeThreshold = nullptr;
    QDoubleSpinBox* m_minDetectionScore = nullptr;
    QLabel* m_matchDistLabel = nullptr;
    QLabel* m_minScoreLabel = nullptr;
    QWidget* m_registryRow = nullptr;
    QLineEdit* m_registryPathEdit = nullptr;
    QPushButton* m_captureBtn = nullptr;
    QPushButton* m_testDataBtn = nullptr;

    bool m_videoPathUserChosen = false;
    bool m_registryPathUserChosen = false;
    bool m_syncingModelControls = false;

    bool m_inferBusy = false;
    bool m_preloadingModel = false;
    quint64 m_streamGeneration = 0;

    QThread* m_inferThread = nullptr;
    FaceLiveDetectInferWorker* m_inferWorker = nullptr;

    FaceDetectRunResult m_lastSnapshot;
    bool m_hasSnapshot = false;

    // Cached overlay data — drawn on every frame to prevent flicker.
    std::vector<FaceDetectBox> m_overlayFaces;
    QVector<QString> m_overlayLabels;
    QSize m_overlayInferSize;
    qint64 m_overlayFrameNum =
            0;  // video frame number when overlay was generated
    qint64 m_lastSubmitFrameNum =
            0;  // video frame number of last inference submission
    QSize m_lastFrameSize;  // original frame size of the last decode

    // Inference timing — for latency display.
    QElapsedTimer m_inferSubmitTime;
    qint64 m_lastInferLatencyMs = 0;
    qint64 m_overlayTimestampMs = 0;  // ms since epoch of last infer complete
};
