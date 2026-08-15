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
        float recognizeMaxDistance = 0.65f;
        FaceRegistryStore* registry = nullptr;
    };

    explicit FaceLiveDetectWidget(QWidget* parent = nullptr);
    ~FaceLiveDetectWidget() override;

    void setConfig(const Config& config);
    Config config() const { return m_config; }

    void setRegistryStore(FaceRegistryStore* store);

    bool startCamera(int deviceIndex = 0);
    bool startVideoFile(const QString& path);
    void restartVideoFile();
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
    void onVideoSeekSliderChanged(int value);
    void onPlaybackSpeedChanged(int index);

private:
    void setupUi();
    void updateThresholdUi();
    void updateRegistryUi();
    void updateModelPathFromCombo();
    void submitInferJob(const QImage& inferRgb, float inferScale);
    void shutdownInferThread();
    void drawLiveOverlay(QImage& frame);
    void beginFrameProcessing();
    void updateVideoTimeLabel(int frameIndex);
    void showSeekPreview(int frameIndex);
    // Async seek-preview decode: Windows MSMF/DirectShow seek+read can
    // block for 100ms+ — running it on the UI thread would freeze both the
    // slider and the playing video.
    void onSeekPreviewReady();
    void closePreviewCapture();
    bool eventFilter(QObject* obj, QEvent* event) override;
    void showEvent(QShowEvent* event) override;

#ifdef HAS_OPENCV_FACE_CAPTURE
    cv::VideoCapture m_previewCapture;  // independent decode path for
                                        // scrub/hover preview (worker thread
                                        // only, guarded by m_previewMutex)
    QMutex m_previewMutex;
    // Async decode result carries the decoded frame index so the cache key
    // matches the actual frame (not the latest slider position).
    QFutureWatcher<QPair<int, QPixmap>>* m_seekPreviewWatcher = nullptr;
    // Latest frame the slider asked for; a stale async result is ignored.
    int m_pendingPreviewFrame = -1;
    // Bumped every time the video changes; async preview results from an
    // older video are discarded (frame numbers can collide across videos).
    // Atomic: read from the decode worker thread, written on the UI thread.
    QAtomicInt m_previewGeneration{0};
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
    QProgressBar* m_preloadProgress = nullptr;
    QSlider* m_videoSeekSlider = nullptr;
    QComboBox* m_playbackSpeedCombo = nullptr;
    QWidget* m_videoControlsRow = nullptr;
    QLabel* m_videoTimeLabel = nullptr;
    QLabel* m_seekPreviewLabel =
            nullptr;  // thumbnail above slider during scrub/hover
    int m_totalVideoFrames = 0;
    double m_videoFps = 0.0;  // cached FPS from CAP_PROP_FPS
    double m_playbackSpeed = 1.0;
    int m_baseTimerInterval = 33;
    bool m_userSeeking = false;
    int m_sliderUpdateSkip =
            0;  // suppress processFrame slider updates after user seek
    qint64 m_lastPreviewTimeMs = 0;  // throttle hover preview updates

    bool m_videoPathUserChosen = false;
    bool m_registryPathUserChosen = false;
    bool m_syncingModelControls = false;

    QTimer* m_frameTimer = nullptr;
    QTimer* m_frameReadTimer = nullptr;  // drives background frame reader
    // VideoFrameReader: reads cv::VideoCapture frames on a background thread
    // so that OpenCV's MSMF/DirectShow backend (Windows) does not block the
    // Qt main thread, which would cause stuttering / UI freezes.
#ifdef HAS_OPENCV_FACE_CAPTURE
    QThread* m_frameReaderThread = nullptr;
    QObject* m_frameReader = nullptr;
    cv::Mat m_latestFrame;  // most recently decoded frame (GUI thread)
    QMutex m_frameMutex;    // guards m_latestFrame
    bool m_frameReaderReady = false;  // background reader has opened source
    QAtomicInt m_frameReaderRunning{0};
    QAtomicInt m_frameReaderSeekTo{-1};
#endif

    bool m_streamActive = false;
    bool m_videoPaused = false;  // video paused (not released) for resume
    QString m_videoFilePath;     // path of currently opened video
    bool m_camerasEnumerated = false;
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

    // Inference timing — for latency display.
    QElapsedTimer m_inferSubmitTime;
    qint64 m_lastInferLatencyMs = 0;
    qint64 m_overlayTimestampMs = 0;  // ms since epoch of last infer complete

    /// Compute timer interval from video FPS and playback speed.
    int computeTimerInterval() const;
};
