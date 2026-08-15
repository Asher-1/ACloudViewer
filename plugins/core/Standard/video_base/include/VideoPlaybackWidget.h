// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// VideoPlaybackWidget: self-contained camera / video-file playback panel
// shared by all GGML-based inference plugins (qFreeSplatter, qFaceDetect,
// future plugins).
//
// Responsibilities (all owned here, one maintenance copy):
//  - background-thread frame decode (VideoFrameReader + QThread), so OpenCV's
//    MSMF/DirectShow backend (Windows) never blocks the Qt main thread
//  - pipeline-driven reads with backpressure (queueNextRead) + a fallback
//    pacemaker timer
//  - playback controls: play / pause / resume / stop / seek / speed / loop
//  - seek preview thumbnails (async decode + QPixmapCache, generation-guarded)
//  - camera enumeration and source switching
//
// Subclassing contract: reimplement the protected virtual hooks below; the
// frame pipeline, UI and threading are fully managed by the base class.
// ----------------------------------------------------------------------------

#pragma once

#include <QAtomicInt>
#include <QImage>
#include <QMutex>
#include <QPair>
#include <QPixmap>
#include <QString>
#include <QWidget>

#include <memory>
#include <vector>

#include "ecvClickableImageLabel.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/videoio.hpp>
#endif

class QComboBox;
class QLabel;
class QLineEdit;
class QSlider;
class QTimer;
class QVBoxLayout;
template <typename T>
class QFutureWatcher;
class VideoFrameReader;

class VideoPlaybackWidget : public QWidget {
    Q_OBJECT

public:
    enum class InputSource { Camera, VideoFile };

    explicit VideoPlaybackWidget(QWidget* parent = nullptr);
    ~VideoPlaybackWidget() override;

    // ---- Playback control -------------------------------------------------

    bool startCamera(int deviceIndex = 0);
    bool startVideoFile(const QString& path);
    void restartVideoFile();
    // Video file: pause (keep the capture open for resume).  Camera: stop
    // and release the device.
    void stopStream();
    void resumePlayback();
    void seekToFrame(int frameIndex);
    void setPlaybackSpeed(double speed);
    double playbackSpeed() const { return m_playbackSpeed; }
    bool isActive() const { return m_streamActive; }

    InputSource inputSource() const;
    int selectedCameraIndex() const;
    QString videoFilePath() const;
    void setVideoFilePath(const QString& path);
    void selectVideoFileSource();
    void setInputSource(InputSource source);

    // ---- Stream metadata --------------------------------------------------

    int totalVideoFrames() const { return m_totalVideoFrames; }
    double videoFps() const { return m_videoFps; }
    int currentFrameNumber() const { return m_currentFrameNum; }

    // ---- Utilities --------------------------------------------------------

    // True when OpenCV video capture support is compiled in.
    static bool isAvailable();

    // BGR cv::Mat -> RGB888 QImage with a single deep copy.
    static QImage cvMatToQImage(const cv::Mat& mat);

    // File picker for video inputs; remembers the last directory under
    // <settingsPrefix>/lastVideoDir (ecvPS convention).
    QString browseVideoFile(const QString& settingsPrefix);

    // >0: fixed preview height (e.g. qFaceDetect 300px).
    // 0 / negative: 16:9 adaptive height clamped to [180, 360].
    void setPreviewFixedHeight(int height);

signals:
    void streamStarted();
    void streamStopped();
    void streamError(const QString& error);
    void logMessage(const QString& message);

protected:
    // ---- Subclass hooks (all invoked on the GUI thread) -------------------

    // Raw BGR frame before any scaling; the subclass submits inference /
    // detection jobs here (throttling is the subclass's own responsibility).
    // frame is a local copy owned by the pipeline — safe to keep references.
    virtual void onFrameDecoded(cv::Mat& frame, int frameIndex);

    // Display-resolution RGB frame (already scaled to the preview label);
    // the subclass draws overlays (boxes, labels) here, coordinates must be
    // scaled from the original frame size.
    virtual void onDisplayFrame(QImage& display, int frameIndex);

    // Video reached EOF and looped back to frame 0 (throttle reset etc.).
    virtual void onVideoLooped();

    // restartVideoFile() / resumePlayback(): the stream restarts from 0 or
    // resumes; invalidate stale async inference results here.
    virtual void onStreamReset();

    // resumePlayback(): the stream resumes from a paused video; reset
    // lightweight throttle state here (keep overlays from the paused frame).
    virtual void onStreamResumed();

    // stopStream(): the stream is stopping (paused for video files, released
    // for cameras); clear subclass inference/detection state here.
    virtual void onStreamStopping();

    // Input source switched (Camera <-> VideoFile).
    virtual void onSourceChanged(InputSource source);

    // Called before the base class opens the capture device / video file;
    // return false to abort the start (e.g. missing model / settings).
    virtual bool onPrepareStream();

    // ---- UI accessors for subclasses --------------------------------------

    ecvClickableImageLabel* previewLabel() const { return m_previewLabel; }
    QLabel* statusLabel() const { return m_statusLabel; }
    QVBoxLayout* mainLayout() const { return m_mainLayout; }
    QComboBox* sourceCombo() const { return m_sourceCombo; }
    QComboBox* cameraCombo() const { return m_cameraCombo; }
    QLineEdit* videoPathEdit() const { return m_videoPathEdit; }
    QSlider* videoSeekSlider() const { return m_videoSeekSlider; }
    QComboBox* playbackSpeedCombo() const { return m_playbackSpeedCombo; }
    QLabel* videoTimeLabel() const { return m_videoTimeLabel; }
    QLabel* seekPreviewLabel() const { return m_seekPreviewLabel; }
    QWidget* videoControlsRow() const { return m_videoControlsRow; }
    QWidget* cameraRow() const { return m_cameraRow; }
    QWidget* videoRow() const { return m_videoRow; }

    // True when a video file is loaded and its controls should be enabled.
    bool videoFileLoaded() const;

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Most recently decoded BGR frame (GUI thread; may be empty).
    // Subclasses use it for snapshot capture (e.g. face capture export).
    const cv::Mat& latestFrame() const { return m_latestFrame; }
#endif

private:
    void setupUi();
    void setupFrameReader();

    void onSourceChangedInternal(int index);
    void onBrowseVideoFile();
    void onVideoSeekSliderChanged(int value);
    void onPlaybackSpeedChanged(int index);
    void updateVideoTimeLabel(int frameIndex);
    void showSeekPreview(int frameIndex);
    void onSeekPreviewReady();
    void closePreviewCapture();
    void beginFrameProcessing();
    void queueNextRead();
    void processFrame();
    int computeTimerInterval() const;
    bool enumerateCamerasIfNeeded(int* firstAvailableIndex);

    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* event) override;

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

    // Background frame reader (VideoFrameReader on m_frameReaderThread).
    VideoFrameReader* m_frameReader = nullptr;
    QThread* m_frameReaderThread = nullptr;
    cv::Mat m_latestFrame;  // most recently decoded frame (GUI thread)
    QMutex m_frameMutex;    // guards m_latestFrame
    bool m_frameReaderReady = false;  // background reader has opened source
    QAtomicInt m_frameReaderRunning{0};
    QAtomicInt m_frameReaderSeekTo{-1};
#endif

    // UI
    ecvClickableImageLabel* m_previewLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QComboBox* m_sourceCombo = nullptr;
    QComboBox* m_cameraCombo = nullptr;
    QLineEdit* m_videoPathEdit = nullptr;
    QSlider* m_videoSeekSlider = nullptr;
    QComboBox* m_playbackSpeedCombo = nullptr;
    QLabel* m_videoTimeLabel = nullptr;
    QLabel* m_seekPreviewLabel =
            nullptr;  // thumbnail above slider during scrub/hover
    QWidget* m_videoControlsRow = nullptr;
    QWidget* m_cameraRow = nullptr;
    QWidget* m_videoRow = nullptr;
    QVBoxLayout* m_mainLayout = nullptr;

    // Playback state
    QTimer* m_frameTimer = nullptr;
    QTimer* m_frameReadTimer = nullptr;  // drives background frame reader
    bool m_streamActive = false;
    bool m_videoPaused = false;  // video paused (not released) for resume
    InputSource m_inputSource = InputSource::Camera;
    QString m_videoFilePath;
    int m_totalVideoFrames = 0;
    double m_videoFps = 0.0;  // cached FPS from CAP_PROP_FPS
    double m_playbackSpeed = 1.0;
    int m_baseTimerInterval = 30;
    int m_currentFrameNum = 0;
    bool m_userSeeking = false;
    int m_sliderUpdateSkip =
            0;  // suppress processFrame slider updates after user seek
    qint64 m_lastPreviewTimeMs = 0;  // throttle hover preview updates
    bool m_camerasEnumerated = false;
    int m_previewFixedHeight = -1;  // <0: 16:9 adaptive; >0: fixed
};
