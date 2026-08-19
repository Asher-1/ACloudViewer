// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoPlaybackWidget.h"

#include <QtCompat.h>
#include <cvFileDialog.h>
#include <ecvPersistentSettings.h>

#include <QComboBox>
#include <QDateTime>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmapCache>
#include <QPushButton>
#include <QSettings>
#include <QShowEvent>
#include <QSlider>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <algorithm>
#include <cmath>

#include "VideoFrameReader.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {
#ifdef HAS_OPENCV_FACE_CAPTURE
// Playback speed presets (kept in sync with the speed combo box).
constexpr double kSpeedPresets[] = {0.25, 0.5, 1.0, 2.0, 4.0};
constexpr int kSpeedPresetCount = 5;

QString normalizedVideoPath(const QString& path) {
    const QString trimmed = path.trimmed();
    return trimmed.isEmpty() ? QString()
                             : QFileInfo(trimmed).absoluteFilePath();
}

// Drains decoded frames until the capture position reaches `target` (used
// by the seek-preview worker).  Mirrors VideoFrameReader::readFrame()'s
// exact-seek alignment: FFmpeg backends land on a keyframe boundary and
// decode forward, so VFR videos or B-frame reordering can land a few frames
// short.  Bounded so a backend that cannot report POS_FRAMES degrades to
// returning the first decoded frame.
bool grabToExactFrame(cv::VideoCapture& cap, int target, cv::Mat& out) {
    if (!cap.read(out) || out.empty()) return false;
    int64_t pos = static_cast<int64_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
    int guard = 0;
    while (pos >= 0 && pos < target && guard < 30) {
        if (!cap.grab()) return false;
        pos = static_cast<int64_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
        ++guard;
    }
    if (guard > 0) {
        if (!cap.retrieve(out) || out.empty()) return false;
    }
    return true;
}
#endif
}  // namespace

VideoPlaybackWidget::VideoPlaybackWidget(QWidget* parent) : QWidget(parent) {
    setupUi();
    setupFrameReader();
    if (m_statusLabel) {
        m_statusLabel->setText(
                isAvailable() ? tr("Ready \u2014 choose a source, then start "
                                   "playback")
                              : tr("Video input unavailable (build with OpenCV "
                                   "videoio)"));
    }
}

VideoPlaybackWidget::~VideoPlaybackWidget() {
    stopStream();
#ifdef HAS_OPENCV_FACE_CAPTURE
    // Tear down the background frame reader. QThread::finished is emitted
    // from the reader thread itself while m_frameReader lives on that same
    // thread, so the finished→deleteLater connection (setupFrameReader) is
    // delivered as a DIRECT call during the thread teardown — the same
    // affinity trap that double-deleted the RMBG live inference worker. Drop
    // the connection first: this destructor is the sole owner of the
    // reader's lifetime.
    if (m_frameReaderThread && m_frameReader) {
        disconnect(m_frameReaderThread, &QThread::finished, m_frameReader,
                   &QObject::deleteLater);
        if (m_frameReaderThread->isRunning()) {
            QMetaObject::invokeMethod(m_frameReader, "release",
                                      Qt::QueuedConnection);
            m_frameReaderThread->quit();
            // Unbounded wait: the queued release() is handled before the
            // queued quit(), so the only upper bound is a single in-flight
            // read. A timed-out wait() would instead fall through and
            // destroy the still-running QThread (parented to this widget)
            // below — qFatal "Destroyed while thread is still running".
            m_frameReaderThread->wait();
        }
        delete m_frameReader;
        m_frameReader = nullptr;
    }
    {
        QMutexLocker lock(&m_frameMutex);
        m_latestFrame.release();
    }
    closePreviewCapture();
#endif
}

bool VideoPlaybackWidget::isAvailable() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    return true;
#else
    return false;
#endif
}

QImage VideoPlaybackWidget::cvMatToQImage(const cv::Mat& mat) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (mat.empty()) return QImage();

    if (mat.channels() == 3) {
        // Single deep copy: interpret the BGR buffer as RGB888 and swap
        // R/B once.  rgbSwapped() returns a self-owned image, so no
        // second .copy() is needed (cvtColor + copy was 2 full-frame
        // copies per frame).
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                      QImage::Format_RGB888)
                .rgbSwapped();
    }

    cv::Mat rgb;
    if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGBA);
    } else {
        return QImage();
    }

    // Use QImage(rgb.data, ...).copy() to share the pixel data briefly and
    // then perform a single deep copy, which is ~3x faster than the per-row
    // memcpy loop.
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  rgb.channels() == 4 ? QImage::Format_RGBA8888
                                      : QImage::Format_RGB888)
            .copy();
#else
    Q_UNUSED(mat);
    return QImage();
#endif
}

QString VideoPlaybackWidget::browseVideoFile(const QString& settingsPrefix) {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, settingsPrefix,
                             QStringLiteral("lastVideoDir"), QDir::homePath());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select video file"), lastDir,
            tr("Video files (*.mp4 *.avi *.mkv *.mov *.webm *.m4v *.wmv *.ts "
               "*.mpg *.mpeg);;All files (*.*)"));
    if (path.isEmpty()) return QString();
    ecvPS::saveBrowseDir(settings, settingsPrefix,
                         QStringLiteral("lastVideoDir"), path);
    return path;
}

void VideoPlaybackWidget::setPreviewFixedHeight(int height) {
    m_previewFixedHeight = height;
    if (!m_previewLabel) return;
    if (height > 0) {
        m_previewLabel->setMinimumHeight(height);
    } else {
        const int previewWidth = std::max(320, contentsRect().width() - 8);
        const int previewHeight = qBound(180, previewWidth * 9 / 16, 360);
        m_previewLabel->setMinimumHeight(previewHeight);
    }
}

bool VideoPlaybackWidget::videoFileLoaded() const {
    return m_frameReaderReady && m_inputSource == InputSource::VideoFile;
}

void VideoPlaybackWidget::setupUi() {
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(4, 4, 4, 4);
    m_mainLayout->setSpacing(6);

    m_previewLabel = new ecvClickableImageLabel(this);
    m_previewLabel->setMinimumSize(320, 180);
    m_previewLabel->setSizePolicy(QSizePolicy::Expanding,
                                  QSizePolicy::Expanding);
    m_previewLabel->setStyleSheet(
            QStringLiteral("QLabel { background-color: #1a1a1a; "
                           "border: 1px solid #444; border-radius: 4px; }"));
    m_previewLabel->setText(tr("Video preview"));
    m_mainLayout->addWidget(m_previewLabel, 1);

    // Fixed gap between the video preview and the controls below so the
    // input / camera comboboxes never visually invade the display area.
    m_mainLayout->addSpacing(8);

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Seek preview thumbnail — child of m_previewLabel so it overlays on
    // the video area without affecting the controls layout.
    m_seekPreviewLabel = new QLabel(m_previewLabel);
    m_seekPreviewLabel->setFixedSize(160, 90);
    m_seekPreviewLabel->setVisible(false);
    m_seekPreviewLabel->setStyleSheet(
            QStringLiteral("QLabel { border: 1px solid #444; "
                           "background: #1a1a1a; }"));
    m_seekPreviewLabel->setAlignment(Qt::AlignCenter);
    m_seekPreviewLabel->raise();

    // Input source selector: live camera vs video file.
    auto* sourceRow = new QWidget(this);
    auto* sourceLayout = new QHBoxLayout(sourceRow);
    sourceLayout->setContentsMargins(0, 0, 0, 0);
    m_sourceCombo = new QComboBox(sourceRow);
    m_sourceCombo->addItem(tr("Live camera"),
                           static_cast<int>(InputSource::Camera));
    m_sourceCombo->addItem(tr("Video file"),
                           static_cast<int>(InputSource::VideoFile));
    sourceLayout->addWidget(new QLabel(tr("Input:"), sourceRow));
    sourceLayout->addWidget(m_sourceCombo, 1);
    // (addWidget moved below slider, see after m_videoControlsRow)
    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &VideoPlaybackWidget::onSourceChangedInternal);

    // Camera device row (visible in camera mode).
    m_cameraRow = new QWidget(this);
    m_cameraRow->setVisible(false);  // hidden until camera source selected
    auto* camLayout = new QHBoxLayout(m_cameraRow);
    camLayout->setContentsMargins(0, 0, 0, 0);
    m_cameraCombo = new QComboBox(m_cameraRow);
    m_cameraCombo->addItem(tr("Default (0)"), 0);
    camLayout->addWidget(new QLabel(tr("Camera:"), m_cameraRow));
    camLayout->addWidget(m_cameraCombo, 1);
    // (addWidget moved below slider, see after m_videoControlsRow)
    connect(m_cameraCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (!m_streamActive || m_inputSource != InputSource::Camera) {
                    return;
                }
                const int idx = m_cameraCombo->currentData().toInt();
                stopStream();
                startCamera(idx);
            });

    // Video file row (visible in video-file mode).
    m_videoRow = new QWidget(this);
    auto* vidLayout = new QHBoxLayout(m_videoRow);
    vidLayout->setContentsMargins(0, 0, 0, 0);
    m_videoPathEdit = new QLineEdit(m_videoRow);
    m_videoPathEdit->setPlaceholderText(
            tr("Path to video (mp4, avi, mkv, mov, webm, \u2026)"));
    auto* browseBtn = new QPushButton(tr("Browse\u2026"), m_videoRow);
    vidLayout->addWidget(m_videoPathEdit, 1);
    vidLayout->addWidget(browseBtn);
    m_videoRow->setVisible(false);
    // (addWidget moved below slider, see after m_videoControlsRow)
    connect(browseBtn, &QPushButton::clicked, this,
            &VideoPlaybackWidget::onBrowseVideoFile);
    connect(m_videoPathEdit, &QLineEdit::textChanged, this,
            [this](const QString& text) {
                m_videoFilePath = normalizedVideoPath(text);
            });
    connect(m_videoPathEdit, &QLineEdit::editingFinished, this,
            [this]() { setVideoFilePath(m_videoPathEdit->text()); });

    // Playback controls: seek slider + time label + speed (video files only).
    m_videoControlsRow = new QWidget(this);
    auto* videoCtrlMainLayout = new QVBoxLayout(m_videoControlsRow);
    videoCtrlMainLayout->setContentsMargins(0, 0, 0, 0);
    videoCtrlMainLayout->setSpacing(2);

    auto* sliderRow = new QWidget(m_videoControlsRow);
    auto* videoCtrlLayout = new QHBoxLayout(sliderRow);
    videoCtrlLayout->setContentsMargins(0, 0, 0, 0);

    m_videoSeekSlider = new QSlider(Qt::Horizontal, m_videoControlsRow);
    m_videoSeekSlider->setRange(0, 0);
    m_videoSeekSlider->setEnabled(false);
    videoCtrlLayout->addWidget(m_videoSeekSlider, 1);

    m_videoTimeLabel = new QLabel(QStringLiteral("0:00"), m_videoControlsRow);
    m_videoTimeLabel->setMinimumWidth(110);
    m_videoTimeLabel->setAlignment(Qt::AlignCenter);
    videoCtrlLayout->addWidget(m_videoTimeLabel);

    m_playbackSpeedCombo = new QComboBox(m_videoControlsRow);
    m_playbackSpeedCombo->addItems(
            {QStringLiteral("0.25\u00d7"), QStringLiteral("0.5\u00d7"),
             QStringLiteral("1\u00d7"), QStringLiteral("2\u00d7"),
             QStringLiteral("4\u00d7")});
    m_playbackSpeedCombo->setCurrentIndex(2);  // 1\u00d7 default
    m_playbackSpeedCombo->setFixedWidth(70);
    m_playbackSpeedCombo->setEnabled(false);  // disabled until video loaded
    videoCtrlLayout->addWidget(m_playbackSpeedCombo);

    videoCtrlMainLayout->addWidget(sliderRow);
    m_videoControlsRow->setVisible(false);  // hidden until a video is loaded
    m_mainLayout->addWidget(m_videoControlsRow);

    // Input/camera/video rows (below slider, so seek preview overlays
    // the preview area without being occluded by input controls).
    m_mainLayout->addWidget(sourceRow);
    m_mainLayout->addWidget(m_cameraRow);
    m_mainLayout->addWidget(m_videoRow);

    // Install event filter for hover/click preview on slider
    m_videoSeekSlider->installEventFilter(this);
    m_videoSeekSlider->setMouseTracking(true);

    connect(m_videoSeekSlider, &QSlider::sliderPressed, this,
            [this]() { m_userSeeking = true; });
    connect(m_videoSeekSlider, &QSlider::sliderReleased, this, [this]() {
        m_userSeeking = false;
        // Explicitly perform the seek now that m_userSeeking is false.
        // We cannot rely on valueChanged firing after sliderReleased —
        // if the slider value was already set during the drag, valueChanged
        // will not fire again, and the seek would never happen.
        onVideoSeekSliderChanged(m_videoSeekSlider->value());
        // Suppress processFrame slider updates for the next 2 ticks so the
        // video has time to seek to the new position before processFrame
        // starts overwriting the slider value again.
        m_sliderUpdateSkip = 2;
        if (m_seekPreviewLabel) m_seekPreviewLabel->setVisible(false);
    });
    connect(m_videoSeekSlider, &QSlider::valueChanged, this,
            &VideoPlaybackWidget::onVideoSeekSliderChanged);
    connect(m_playbackSpeedCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &VideoPlaybackWidget::onPlaybackSpeedChanged);
#endif

    m_statusLabel = new QLabel(this);
    m_statusLabel->setAlignment(Qt::AlignCenter);
    m_statusLabel->setWordWrap(true);
    m_mainLayout->addWidget(m_statusLabel);

    if (!isAvailable()) {
        setEnabled(false);
    }
}

void VideoPlaybackWidget::setupFrameReader() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    m_frameTimer = new QTimer(this);
    m_frameTimer->setInterval(m_baseTimerInterval);
    connect(m_frameTimer, &QTimer::timeout, this,
            &VideoPlaybackWidget::processFrame);

    // Async seek-preview decode (Windows MSMF seek/read must not run on
    // the UI thread — it would freeze the slider and the playing video).
    m_seekPreviewWatcher = new QFutureWatcher<QPair<int, QPixmap>>(this);
    connect(m_seekPreviewWatcher,
            &QFutureWatcher<QPair<int, QPixmap>>::finished, this,
            &VideoPlaybackWidget::onSeekPreviewReady);

    // Background frame reader: reads frames on a dedicated thread so
    // OpenCV's MSMF/DirectShow backend (Windows) does not block the UI.
    // cv::Mat must be registered so queued frameReady() deliveries work
    // (unregistered types are silently dropped by Qt).
    qRegisterMetaType<cv::Mat>("cv::Mat");
    m_frameReaderThread = new QThread(this);
    m_frameReader = new VideoFrameReader();  // no parent — moved to thread
    m_frameReader->moveToThread(m_frameReaderThread);
    connect(m_frameReaderThread, &QThread::finished, m_frameReader,
            &QObject::deleteLater);
    connect(m_frameReader, &VideoFrameReader::frameReady, this,
            [this](const cv::Mat& rgbFrame, int frameIndex) {
                {
                    QMutexLocker lock(&m_frameMutex);
                    // Shallow copy (refcount bump): the signal argument owns
                    // the decoded buffer exclusively after delivery, so the
                    // member can share it without a full-frame deep copy.
                    m_latestFrame = rgbFrame;
                    m_frameReaderSeekTo.store(frameIndex);
                }
                if (m_streamActive &&
                    m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven &&
                    !m_waitingForConsumer) {
                    QTimer::singleShot(0, this, [this]() { processFrame(); });
                }
            });
    connect(m_frameReader, &VideoFrameReader::frameReadFailed, this, [this]() {
        // Video end-of-file: loop back to start.
        if (m_inputSource == InputSource::VideoFile && m_frameReaderReady &&
            !m_userSeeking) {
            QMetaObject::invokeMethod(m_frameReader, "seekToFrame",
                                      Qt::QueuedConnection, Q_ARG(int, 0));
            m_currentFrameNum = 0;
            m_sliderUpdateSkip = 0;
            if (m_videoSeekSlider) {
                m_videoSeekSlider->blockSignals(true);
                m_videoSeekSlider->setValue(0);
                m_videoSeekSlider->blockSignals(false);
            }
            updateVideoTimeLabel(0);
            onVideoLooped();
            if (m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven) {
                queueNextRead();
            }
        } else if (m_inputSource == InputSource::Camera &&
                   m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven &&
                   m_frameReaderReady && m_streamActive) {
            // Clock-driven streams retry on the next timer tick. Consumer-
            // driven streams have no timer, so explicitly recover from a
            // transient camera read failure without creating a busy loop.
            QTimer::singleShot(m_baseTimerInterval, this, [this]() {
                if (m_inputSource == InputSource::Camera &&
                    m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven &&
                    m_frameReaderReady && m_streamActive) {
                    queueNextRead();
                }
            });
        }
    });
    m_frameReaderThread->start();
#endif
}

// ---------------------------------------------------------------------------
// Subclass hooks (default no-ops)
// ---------------------------------------------------------------------------

void VideoPlaybackWidget::onFrameDecoded(cv::Mat& /*frame*/,
                                         int /*frameIndex*/) {}
void VideoPlaybackWidget::onDisplayFrame(QImage& /*display*/,
                                         int /*frameIndex*/) {}
void VideoPlaybackWidget::onVideoLooped() {}
void VideoPlaybackWidget::onStreamReset() {}
void VideoPlaybackWidget::onStreamResumed() {}
void VideoPlaybackWidget::onStreamStopping() {}
bool VideoPlaybackWidget::onPrepareStream() { return true; }
void VideoPlaybackWidget::onSourceChanged(InputSource /*source*/) {}

void VideoPlaybackWidget::setFrameAdvanceMode(FrameAdvanceMode mode) {
    if (m_streamActive || m_videoPaused) {
        emit logMessage(
                tr("Frame advance mode can only change while stopped."));
        return;
    }
    m_frameAdvanceMode = mode;
}

void VideoPlaybackWidget::completeFrameProcessing(
        const QImage& processedImage) {
    if (m_frameAdvanceMode != FrameAdvanceMode::ConsumerDriven ||
        !m_waitingForConsumer) {
        return;
    }

    m_waitingForConsumer = false;
    if (!processedImage.isNull() && m_previewLabel) {
        const QImage display = processedImage.scaled(m_previewLabel->size(),
                                                     Qt::KeepAspectRatio,
                                                     Qt::SmoothTransformation);
        m_previewLabel->setPixmap(QPixmap::fromImage(display));
    }
    if (m_streamActive) queueNextRead();
}

// ---------------------------------------------------------------------------
// Playback control
// ---------------------------------------------------------------------------

bool VideoPlaybackWidget::enumerateCamerasIfNeeded(int* firstAvailableIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_camerasEnumerated) return true;
    m_camerasEnumerated = true;
    if (!m_cameraCombo) return true;

    m_cameraCombo->blockSignals(true);
    m_cameraCombo->clear();

    namespace cvlog = cv::utils::logging;
    const auto prevLevel = cvlog::getLogLevel();
    cvlog::setLogLevel(cvlog::LOG_LEVEL_SILENT);

    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture testCap(i, cv::CAP_ANY);
        if (testCap.isOpened()) {
            m_cameraCombo->addItem(tr("Camera %1").arg(i), i);
            testCap.release();
        }
    }

    cvlog::setLogLevel(prevLevel);

    if (m_cameraCombo->count() == 0) {
        m_cameraCombo->addItem(tr("No camera found"), -1);
        m_cameraCombo->blockSignals(false);
        return false;
    }
    if (firstAvailableIndex) {
        *firstAvailableIndex = m_cameraCombo->itemData(0).toInt();
    }
    m_cameraCombo->blockSignals(false);
    return true;
#else
    Q_UNUSED(firstAvailableIndex);
    return false;
#endif
}

bool VideoPlaybackWidget::startCamera(int deviceIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    stopStream();
    if (!m_camerasEnumerated) {
        int firstAvailable = 0;
        if (!enumerateCamerasIfNeeded(&firstAvailable)) {
            m_statusLabel->setText(tr("No camera devices detected"));
            return false;
        }
        // Default request resolves to the first enumerated device.
        if (deviceIndex == 0 && m_cameraCombo && m_cameraCombo->count() > 0) {
            deviceIndex = m_cameraCombo->itemData(0).toInt();
        }
    }

    if (!onPrepareStream()) return false;
    onStreamReset();

    auto* reader = static_cast<VideoFrameReader*>(m_frameReader);
    bool opened = false;
    const bool consumerDriven =
            m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven;
    QMetaObject::invokeMethod(
            reader,
            [reader, consumerDriven, deviceIndex, &opened]() {
                reader->setConsumerDriven(consumerDriven);
                opened = reader->openCamera(deviceIndex, cv::CAP_ANY);
            },
            Qt::BlockingQueuedConnection);
    if (!opened) {
        m_frameReaderReady = false;
        m_streamActive = false;
        const QString error =
                tr("Failed to open camera device %1").arg(deviceIndex);
        m_statusLabel->setText(error);
        emit streamError(error);
        return false;
    }
    m_openVideoPath.clear();
    m_videoPaused = false;
    m_frameReaderReady = true;
    m_streamActive = true;
    m_statusLabel->setText(tr("Camera active"));
    beginFrameProcessing();
    emit streamStarted();
    return true;
#else
    Q_UNUSED(deviceIndex);
    return false;
#endif
}

bool VideoPlaybackWidget::startVideoFile(const QString& path) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    const QString requestedPath = normalizedVideoPath(path);
    if (requestedPath.isEmpty()) return false;

    // Resume from paused state if same video
    if (m_videoPaused && m_frameReaderReady &&
        m_openVideoPath == requestedPath &&
        m_inputSource == InputSource::VideoFile) {
        resumePlayback();
        m_statusLabel->setText(tr("Resuming video"));
        return true;
    }

    // Cancel and drain the previous session before issuing any work for the
    // new video.
    stopStream();
    closePreviewCapture();

    if (!onPrepareStream()) return false;
    onStreamReset();

    m_inputSource = InputSource::VideoFile;
    m_videoFilePath = requestedPath;
    if (m_videoPathEdit) m_videoPathEdit->setText(requestedPath);

    auto* reader = static_cast<VideoFrameReader*>(m_frameReader);
    bool opened = false;
    qint64 frameCount = 0;
    double fps = 0.0;
    const bool consumerDriven =
            m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven;
    const std::string requestedPathUtf8 = requestedPath.toStdString();
    m_openVideoPath.clear();
    QMetaObject::invokeMethod(
            reader,
            [reader, consumerDriven, requestedPathUtf8, &opened, &frameCount,
             &fps]() {
                reader->setConsumerDriven(consumerDriven);
                opened = reader->openVideo(requestedPathUtf8, cv::CAP_FFMPEG) ||
                         reader->openVideo(requestedPathUtf8, cv::CAP_ANY);
                if (opened) {
                    frameCount = reader->getFrameCount();
                    fps = reader->getFps();
                }
            },
            Qt::BlockingQueuedConnection);
    if (!opened) {
        const QString err =
                tr("Failed to open video (rebuild OpenCV with FFmpeg / "
                   "WITH_FFMPEG=ON): %1")
                        .arg(requestedPath);
        m_statusLabel->setText(err);
        emit streamError(err);
        m_frameReaderReady = false;
        m_videoPaused = false;
        return false;
    }
    m_openVideoPath = requestedPath;
    m_videoPaused = false;
    m_frameReaderReady = true;
    m_streamActive = true;

    // Initialize video seek slider from background reader metadata
    m_totalVideoFrames = static_cast<int>(frameCount);
    m_videoFps = fps;
    if (m_videoFps <= 0)
        m_videoFps = 30.0;  // fallback for codecs that don't report FPS
    if (m_videoSeekSlider) {
        m_videoSeekSlider->blockSignals(true);
        m_videoSeekSlider->setRange(0, std::max(0, m_totalVideoFrames - 1));
        m_videoSeekSlider->setValue(0);
        m_videoSeekSlider->setEnabled(m_totalVideoFrames > 0);
        m_videoSeekSlider->blockSignals(false);
    }
    if (m_playbackSpeedCombo) {
        m_playbackSpeedCombo->setEnabled(true);
    }
    // Show playback controls now that a video is loaded.
    if (m_videoControlsRow) m_videoControlsRow->setVisible(true);
    updateVideoTimeLabel(0);

    beginFrameProcessing();
    emit streamStarted();
    return true;
#else
    Q_UNUSED(path);
    return false;
#endif
}

void VideoPlaybackWidget::restartVideoFile() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    const QString requestedPath = normalizedVideoPath(m_videoFilePath);
    if (requestedPath.isEmpty()) return;
    if (!m_frameReaderReady || m_openVideoPath != requestedPath) {
        startVideoFile(requestedPath);
        return;
    }

    // Reset ALL stream state so the pipeline restarts fresh.
    onStreamReset();

    // Clear any stale decoded frame — the reader's clock may have delivered
    // a frame from the old position (near EOF) between the seek being queued
    // and being processed.  Consuming it would poison the subclass's
    // frame-index-based throttle (frame index jumps backwards after seek).
    {
        QMutexLocker lock(&m_frameMutex);
        m_latestFrame.release();
    }

    QMetaObject::invokeMethod(m_frameReader, "seekToFrame",
                              Qt::QueuedConnection, Q_ARG(int, 0));
    m_currentFrameNum = 0;
    m_sliderUpdateSkip = 0;
    if (m_videoSeekSlider) {
        m_videoSeekSlider->blockSignals(true);
        m_videoSeekSlider->setValue(0);
        m_videoSeekSlider->blockSignals(false);
    }
    updateVideoTimeLabel(0);

    // Always restart the pipeline even if streamActive is still true:
    // the seek invalidates the current frame position, so we stop the
    // timer and re-enter beginFrameProcessing to queue a fresh read.
    if (m_frameTimer) m_frameTimer->stop();
    m_videoPaused = false;
    m_streamActive = true;
    beginFrameProcessing();
    m_statusLabel->setText(tr("Restarted video"));
#endif
}

void VideoPlaybackWidget::stopStream() {
    if (m_frameTimer) m_frameTimer->stop();
    m_waitingForConsumer = false;
#ifdef HAS_OPENCV_FACE_CAPTURE
    // Halt the decode clock before pausing/releasing so no further frames
    // are decoded while the stream tears down.
    QMetaObject::invokeMethod(m_frameReader, "stopClockReading",
                              Qt::QueuedConnection);
    // For video files: pause (don't release) so we can resume from same
    // position.  The background reader keeps its video open.
    const bool isVideoFile = m_inputSource == InputSource::VideoFile;
    if (isVideoFile && m_frameReaderReady) {
        // Just pause — keep reader open for resume.  The mpv backend
        // pauses A/V playback via the pause property; OpenCV ignores it.
        m_videoPaused = true;
        QMetaObject::invokeMethod(m_frameReader, "setPaused",
                                  Qt::QueuedConnection, Q_ARG(bool, true));
    } else if (m_frameReaderReady) {
        // Release background reader for camera mode
        QMetaObject::invokeMethod(m_frameReader, "release",
                                  Qt::QueuedConnection);
        m_frameReaderReady = false;
        m_videoFilePath.clear();
        m_openVideoPath.clear();
        m_videoPaused = false;
        closePreviewCapture();
        {
            QMutexLocker lock(&m_frameMutex);
            m_latestFrame.release();
        }
    }
#endif

    // For video files that are paused (not released), keep the slider and
    // metadata intact so the user can still seek and preview.
    const bool videoStillLoaded =
            isVideoFile && m_frameReaderReady && !m_openVideoPath.isEmpty();
    if (!videoStillLoaded) {
        m_totalVideoFrames = 0;
        if (m_videoSeekSlider) {
            m_videoSeekSlider->blockSignals(true);
            m_videoSeekSlider->setRange(0, 0);
            m_videoSeekSlider->setValue(0);
            m_videoSeekSlider->setEnabled(false);
            m_videoSeekSlider->blockSignals(false);
        }
        if (m_playbackSpeedCombo) {
            m_playbackSpeedCombo->setEnabled(false);
        }
        updateVideoTimeLabel(0);
    }
    m_currentFrameNum = 0;
    if (m_seekPreviewLabel) m_seekPreviewLabel->setVisible(false);

    if (m_streamActive) {
        m_streamActive = false;
        onStreamStopping();
        emit streamStopped();
    }
}

void VideoPlaybackWidget::resumePlayback() {
    if (!m_videoPaused || !m_frameReaderReady) return;
    m_videoPaused = false;
    m_streamActive = true;
    // mpv backend: resume A/V playback; OpenCV ignores this.
    QMetaObject::invokeMethod(m_frameReader, "setPaused", Qt::QueuedConnection,
                              Q_ARG(bool, false));
    onStreamResumed();
    beginFrameProcessing();
    emit streamStarted();
}

void VideoPlaybackWidget::seekToFrame(int frameIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_frameReaderReady || m_inputSource != InputSource::VideoFile) return;
    onStreamReset();
    m_waitingForConsumer = false;
    {
        QMutexLocker lock(&m_frameMutex);
        m_latestFrame.release();
    }
    QMetaObject::invokeMethod(m_frameReader, "seekToFrame",
                              Qt::QueuedConnection, Q_ARG(int, frameIndex));
    m_currentFrameNum = frameIndex;
    if (m_streamActive &&
        m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven) {
        queueNextRead();
    }
#else
    Q_UNUSED(frameIndex);
#endif
}

void VideoPlaybackWidget::setPlaybackSpeed(double speed) {
    m_playbackSpeed = std::max(0.1, speed);
    if (m_playbackSpeedCombo) {
        const int index =
                std::min(kSpeedPresetCount - 1,
                         std::max(0, static_cast<int>(std::lround(
                                             std::log2(speed) * 2.0 + 2.0))));
        m_playbackSpeedCombo->blockSignals(true);
        m_playbackSpeedCombo->setCurrentIndex(index);
        m_playbackSpeedCombo->blockSignals(false);
    }
    // Recompute both clocks for clock-driven playback: the UI tick and the
    // reader-thread decode clock advance at fps × speed.  Consumer-driven
    // mode intentionally advances only when its subclass completes
    // inference, but the decode interval is still tracked for a later
    // switch back.  Camera pacing ignores the speed (base interval).
    const bool isVideo = m_inputSource == InputSource::VideoFile;
    const int interval = computeTimerInterval();
    if (m_frameTimer && isVideo) m_frameTimer->setInterval(interval);
#ifdef HAS_OPENCV_FACE_CAPTURE
    QMetaObject::invokeMethod(
            m_frameReader, "setClockInterval", Qt::QueuedConnection,
            Q_ARG(int, isVideo ? interval : m_baseTimerInterval));
    // mpv backend: apply the speed to the A/V clock; OpenCV ignores it
    // (pacing stays with the decode clock above).
    QMetaObject::invokeMethod(m_frameReader, "setPlaybackSpeed",
                              Qt::QueuedConnection, Q_ARG(double, speed));
#endif
}

VideoPlaybackWidget::InputSource VideoPlaybackWidget::inputSource() const {
    return m_inputSource;
}

int VideoPlaybackWidget::selectedCameraIndex() const {
    if (!m_cameraCombo || m_cameraCombo->count() == 0) return 0;
    return m_cameraCombo->currentData().toInt();
}

QString VideoPlaybackWidget::videoFilePath() const {
    if (m_videoPathEdit) {
        const QString edited = m_videoPathEdit->text().trimmed();
        if (!edited.isEmpty()) return edited;
    }
    return m_videoFilePath;
}

void VideoPlaybackWidget::setVideoFilePath(const QString& path) {
    const QString requestedPath = normalizedVideoPath(path);
    if (m_streamActive && m_inputSource == InputSource::VideoFile &&
        !m_openVideoPath.isEmpty() && m_openVideoPath != requestedPath) {
        stopStream();
    }
    m_videoFilePath = requestedPath;
    if (m_videoPathEdit) {
        m_videoPathEdit->setText(requestedPath);
    }
}

void VideoPlaybackWidget::selectVideoFileSource() {
    if (!m_sourceCombo) return;
    const int idx =
            m_sourceCombo->findData(static_cast<int>(InputSource::VideoFile));
    if (idx >= 0) m_sourceCombo->setCurrentIndex(idx);
}

void VideoPlaybackWidget::setInputSource(InputSource source) {
    if (m_sourceCombo) {
        const int idx = m_sourceCombo->findData(static_cast<int>(source));
        if (idx >= 0) m_sourceCombo->setCurrentIndex(idx);
    } else {
        m_inputSource = source;
    }
}

// ---------------------------------------------------------------------------
// Frame pipeline
// ---------------------------------------------------------------------------

void VideoPlaybackWidget::beginFrameProcessing() {
    if (!m_streamActive) return;
    m_waitingForConsumer = false;
    // Use video FPS × speed for video files; camera uses base interval.
    const bool isVideo = m_inputSource == InputSource::VideoFile;
    const int interval = isVideo ? computeTimerInterval() : m_baseTimerInterval;
    m_frameTimer->setInterval(interval);
    if (m_frameAdvanceMode == FrameAdvanceMode::ClockDriven) {
        m_frameTimer->start();
        // Clock-paced decode: the reader thread decodes at the playback
        // frame rate in parallel with UI-side consumption.  This is what
        // makes playback speed effective — the video timeline advances on
        // the decode clock, while the UI tick consumes the newest decoded
        // frame (dropping frames when processing is slower than the clock).
        if (m_frameReaderReady) {
            QMetaObject::invokeMethod(m_frameReader, "startClockReading",
                                      Qt::QueuedConnection,
                                      Q_ARG(int, interval));
        }
    } else {
        m_frameTimer->stop();
        // Consumer-driven mode decodes one frame per request (the subclass's
        // inference completion drives advancement).
        if (m_frameReaderReady) {
            QMetaObject::invokeMethod(m_frameReader, "readFrame",
                                      Qt::QueuedConnection);
        }
    }
    if (isVideo) {
        m_statusLabel->setText(tr("Playing video"));
    } else {
        m_statusLabel->setText(tr("Camera active"));
    }
}

void VideoPlaybackWidget::queueNextRead() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_frameReaderReady && m_streamActive) {
        QMetaObject::invokeMethod(m_frameReader, "readFrame",
                                  Qt::QueuedConnection);
    }
#endif
}

void VideoPlaybackWidget::processFrame() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    // Clock-driven decode is self-pacing (startClockReading); a missing
    // frame simply means the decode clock has not delivered yet — keep the
    // previous pixmap and wait for the next tick.  Consumer-driven mode has
    // no clock, so its early-return paths must re-queue a background read
    // or the pipeline would stall permanently.
    if (!m_streamActive || !m_frameReaderReady) {
        return;
    }

    cv::Mat frame;
    int curFrameNum = 0;
    {
        QMutexLocker lock(&m_frameMutex);
        if (m_latestFrame.empty()) {
            return;
        }
        // O(1) swap instead of a full-frame deep copy: the member hands
        // ownership to the local, the next frameReady repopulates it.
        // (std::swap uses move semantics; Mat::swap is unavailable in some
        // distro OpenCV builds, e.g. Debian 4.2.)
        std::swap(frame, m_latestFrame);
        curFrameNum = m_frameReaderSeekTo.load();
    }
    m_currentFrameNum = curFrameNum;

    // Update video seek slider and time label
    if (m_inputSource == InputSource::VideoFile && !m_userSeeking) {
        // Skip slider updates immediately after a user seek so the video
        // has time to reach the target position.
        if (m_sliderUpdateSkip > 0) {
            --m_sliderUpdateSkip;
        } else if (m_videoSeekSlider && m_totalVideoFrames > 0) {
            m_videoSeekSlider->blockSignals(true);
            m_videoSeekSlider->setValue(curFrameNum);
            m_videoSeekSlider->blockSignals(false);
        }
        updateVideoTimeLabel(curFrameNum);
    }

    // Consumer-driven callers own frame completion from this point onward.
    if (m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven) {
        m_waitingForConsumer = true;
    }

    // Subclass inference / detection hook (raw BGR frame, GUI thread).
    onFrameDecoded(frame, curFrameNum);

    if (m_frameAdvanceMode == FrameAdvanceMode::ConsumerDriven) {
        return;
    }

    // Display path: scale in the cv domain first, then convert — the
    // previous pipeline converted the full-resolution frame (full-frame
    // rgbSwapped copy) and immediately shrank it with a full-frame Qt
    // scale. Scaling first also lets the subclass overlay hooks paint on an
    // INTER_AREA result instead of nearest-neighbour.
    QImage display = scaledDisplayImage(frame);
    if (display.isNull()) {
        return;
    }

    onDisplayFrame(display, curFrameNum);
    m_previewLabel->setPixmap(QPixmap::fromImage(display));
#endif
}

QImage VideoPlaybackWidget::scaledDisplayImage(const cv::Mat& frame) const {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (frame.empty() || !m_previewLabel) return QImage();
    // QSize::scaled shares its arithmetic with QImage::scaled(
    // KeepAspectRatio), so the output size (and therefore subclass overlay
    // coordinate scaling) is identical to the previous
    // rgb.scaled(previewLabel->size(), KeepAspectRatio) pipeline.
    const QSize target =
            QSize(frame.cols, frame.rows)
                    .scaled(m_previewLabel->size(), Qt::KeepAspectRatio);
    if (target.isEmpty()) return QImage();
    cv::Mat scaled;
    if (target.width() != frame.cols || target.height() != frame.rows) {
        cv::resize(frame, scaled, cv::Size(target.width(), target.height()), 0,
                   0, cv::INTER_AREA);
    } else {
        scaled = frame;  // shallow — Mat refcount bump only
    }
    return cvMatToQImage(scaled);
#else
    Q_UNUSED(frame);
    return QImage();
#endif
}

int VideoPlaybackWidget::computeTimerInterval() const {
    // For video files: match the video's native frame rate × playback speed.
    // This ensures one timer tick ≈ one video frame advance.
    if (m_inputSource == InputSource::VideoFile && m_videoFps > 0) {
        const double interval = 1000.0 / (m_videoFps * m_playbackSpeed);
        return std::max(1, static_cast<int>(std::lround(interval)));
    }
    // For camera: base interval adjusted by speed (if speed control is used).
    return std::max(1, static_cast<int>(m_baseTimerInterval / m_playbackSpeed));
}

// ---------------------------------------------------------------------------
// Source switching / file browsing
// ---------------------------------------------------------------------------

void VideoPlaybackWidget::onSourceChangedInternal(int index) {
    if (!m_sourceCombo) return;
    const InputSource nextSource =
            static_cast<InputSource>(m_sourceCombo->itemData(index).toInt());
    // Stop while the old source is still selected. stopStream() uses the
    // source type to decide whether it can pause or must release the reader.
    if (m_streamActive && m_inputSource != nextSource) {
        stopStream();
    }
    m_inputSource = nextSource;

    if (m_videoRow) {
        m_videoRow->setVisible(m_inputSource == InputSource::VideoFile);
    }
    if (m_cameraRow) {
        m_cameraRow->setVisible(m_inputSource == InputSource::Camera);
    }
    // Hide playback controls in camera mode — they are meaningless for
    // live capture.
    if (m_videoControlsRow) {
        m_videoControlsRow->setVisible(m_inputSource == InputSource::VideoFile);
        if (m_videoSeekSlider) m_videoSeekSlider->setEnabled(false);
        if (m_playbackSpeedCombo) m_playbackSpeedCombo->setEnabled(false);
    }
    if (m_streamActive) {
        stopStream();
    }
    onSourceChanged(m_inputSource);
    if (m_inputSource == InputSource::VideoFile) {
        m_statusLabel->setText(tr("Select a video file, then start playback"));
    } else {
        m_statusLabel->setText(
                tr("Ready \u2014 choose a source, then start playback"));
    }
}

void VideoPlaybackWidget::onBrowseVideoFile() {
    const QString path = browseVideoFile(QStringLiteral("video_base"));
    if (path.isEmpty()) return;
    setVideoFilePath(path);
}

// ---------------------------------------------------------------------------
// Seek + playback speed
// ---------------------------------------------------------------------------

void VideoPlaybackWidget::onVideoSeekSliderChanged(int value) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_frameReaderReady || m_inputSource != InputSource::VideoFile) return;
    if (m_userSeeking) {
        // During drag: show preview thumbnail via independent decode path
        showSeekPreview(value);
        // Update time label to show target position (only when timer is
        // inactive; during playback processFrame overwrites it each tick).
        if (m_frameTimer && !m_frameTimer->isActive()) {
            updateVideoTimeLabel(value);
        }
        return;
    }
    // Use the public seek path so subclasses can invalidate in-flight
    // inference and consumer-driven playback can request the target frame.
    // Repainting m_latestFrame here would show the pre-seek frame under the
    // new timestamp and can make overlays appear temporally misaligned.
    seekToFrame(value);
#endif
}

void VideoPlaybackWidget::onPlaybackSpeedChanged(int index) {
    if (index < 0 || index >= kSpeedPresetCount) return;
    setPlaybackSpeed(kSpeedPresets[index]);
}

void VideoPlaybackWidget::updateVideoTimeLabel(int frameIndex) {
    if (!m_videoTimeLabel) return;
    if (m_totalVideoFrames <= 0) {
        m_videoTimeLabel->setText(QStringLiteral("0:00"));
        return;
    }
    const double fps = m_videoFps > 0 ? m_videoFps : 30.0;
    const int totalSec = static_cast<int>(frameIndex / fps);
    const int totalAllSec = static_cast<int>(m_totalVideoFrames / fps);
    auto fmt = [](int sec) -> QString {
        const int h = sec / 3600;
        const int m = (sec % 3600) / 60;
        const int s = sec % 60;
        if (h > 0)
            return QStringLiteral("%1:%2:%3")
                    .arg(h)
                    .arg(m, 2, 10, QLatin1Char('0'))
                    .arg(s, 2, 10, QLatin1Char('0'));
        return QStringLiteral("%1:%2").arg(m).arg(s, 2, 10, QLatin1Char('0'));
    };
    m_videoTimeLabel->setText(fmt(totalSec) + QStringLiteral(" / ") +
                              fmt(totalAllSec));
}

// ---------------------------------------------------------------------------
// Seek preview (async decode on the global thread pool)
// ---------------------------------------------------------------------------

void VideoPlaybackWidget::showSeekPreview(int frameIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_videoFilePath.isEmpty() || m_totalVideoFrames <= 0) return;
    if (frameIndex < 0) frameIndex = 0;
    if (frameIndex >= m_totalVideoFrames) frameIndex = m_totalVideoFrames - 1;

    if (!m_seekPreviewLabel) return;

    // Check QPixmapCache first — avoids re-decoding the same frame.
    // Key includes a path hash so different videos of equal length do not
    // collide (length-only keys showed the old video's frame).
    const QString cacheKey = QStringLiteral("vb_seekpreview_%1_%2")
                                     .arg(qHash(m_videoFilePath))
                                     .arg(frameIndex);
    QPixmap cached;
    if (QPixmapCache::find(cacheKey, &cached)) {
        m_seekPreviewLabel->setPixmap(cached);
    } else {
        // ASYNC decode: Windows MSMF/DirectShow seek+read() can block for
        // 100 ms+ — doing it synchronously here would freeze the slider AND
        // the playing video (the main thread is shared).  Decode on the
        // global thread pool; the mutex serializes access to the shared
        // preview capture, and only the LATEST request wins.
        m_pendingPreviewFrame.storeRelaxed(frameIndex);
        if (m_seekPreviewWatcher && !m_seekPreviewWatcher->isRunning()) {
            const QString path = m_videoFilePath;
            // Snapshot the generation: results decoded for a previous video
            // must not be painted over the new one.
            const int gen = m_previewGeneration.loadRelaxed();
            m_seekPreviewWatcher->setFuture(QtConcurrent::run(
                    [this, path, frameIndex, gen]() -> QPair<int, QPixmap> {
                        if (gen != m_previewGeneration.loadRelaxed()) {
                            return {frameIndex, QPixmap()};
                        }
                        QMutexLocker lock(&m_previewMutex);
                        if (!m_previewCapture.isOpened()) {
                            // The preview capture stays open for the whole
                            // video session (released only by
                            // closePreviewCapture() on stop / source switch
                            // / destruction): opening a file + initializing
                            // the decoder costs 50-200 ms on Windows, which
                            // is what made hover scrubbing stutter.
                            if (!OpenCVFrameSource::openVideoWithHw(
                                        m_previewCapture, path.toStdString(),
                                        cv::CAP_FFMPEG) &&
                                !m_previewCapture.open(path.toStdString(),
                                                       cv::CAP_ANY)) {
                                return {frameIndex, QPixmap()};
                            }
                            m_previewCapture.set(cv::CAP_PROP_BUFFERSIZE, 1);
                        }
                        // Follow the LATEST drag position: while the user
                        // keeps scrubbing, decoding the stale request is
                        // wasted work that delays the preview catching up.
                        const int target =
                                m_pendingPreviewFrame.loadRelaxed() >= 0
                                        ? m_pendingPreviewFrame.loadRelaxed()
                                        : frameIndex;
                        m_previewCapture.set(cv::CAP_PROP_POS_FRAMES, target);
                        cv::Mat frame;
                        if (!grabToExactFrame(m_previewCapture, target,
                                              frame)) {
                            return {target, QPixmap()};
                        }
                        // Scale to thumbnail size (160×90)
                        cv::Mat thumb;
                        cv::resize(frame, thumb, cv::Size(160, 90), 0, 0,
                                   cv::INTER_AREA);
                        return {target,
                                QPixmap::fromImage(cvMatToQImage(thumb))};
                    }));
        }
    }

    // Position the thumbnail centered above the slider handle,
    // overlaying on the video preview area (like modern video players).
    if (m_videoSeekSlider && m_previewLabel) {
        const int sliderWidth = m_videoSeekSlider->width();
        const int range = m_totalVideoFrames - 1;
        const int handleX =
                (range > 0) ? static_cast<int>(static_cast<qint64>(frameIndex) *
                                               (sliderWidth - 1) / range)
                            : 0;
        // Map slider handle position to preview label coordinates.
        const QPoint sliderGlobal =
                m_videoSeekSlider->mapToGlobal(QPoint(handleX, 0));
        const QPoint localPos = m_previewLabel->mapFromGlobal(sliderGlobal);
        // Center horizontally on the handle, place just above the slider.
        const int previewW = m_seekPreviewLabel->width();
        const int previewH = m_seekPreviewLabel->height();
        int x = localPos.x() - previewW / 2;
        int y = localPos.y() - previewH - 4;
        // Clamp within preview bounds.
        x = qBound(0, x, m_previewLabel->width() - previewW);
        y = qMax(0, y);
        m_seekPreviewLabel->move(x, y);
    }
    m_seekPreviewLabel->raise();
    m_seekPreviewLabel->setVisible(true);
#else
    Q_UNUSED(frameIndex);
#endif
}

void VideoPlaybackWidget::onSeekPreviewReady() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_seekPreviewWatcher || !m_seekPreviewLabel) return;
    const QPair<int, QPixmap> result = m_seekPreviewWatcher->result();
    if (result.second.isNull()) return;
    // Cache under the frame that was ACTUALLY decoded, not the latest
    // slider position (they differ while the user keeps dragging).
    const QString cacheKey = QStringLiteral("vb_seekpreview_%1_%2")
                                     .arg(qHash(m_videoFilePath))
                                     .arg(result.first);
    QPixmapCache::insert(cacheKey, result.second);
    // Only paint if this result still matches the latest request.
    if (m_seekPreviewLabel->isVisible() &&
        result.first == m_pendingPreviewFrame.loadRelaxed()) {
        m_seekPreviewLabel->setPixmap(result.second);
    }

    // Limit cache size: periodic trim.
    static int s_cacheCheckCounter = 0;
    if (++s_cacheCheckCounter % 50 == 0 && QPixmapCache::cacheLimit() > 0) {
        QPixmapCache::setCacheLimit(32768);  // 32 MB cap
    }
#endif
}

void VideoPlaybackWidget::closePreviewCapture() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    // Invalidate in-flight async previews from the current video.
    m_previewGeneration.fetchAndAddRelaxed(1);
    // Cancel any pending async decode, then release under the mutex so we
    // never destroy the capture while a worker is reading it.
    if (m_seekPreviewWatcher) {
        m_seekPreviewWatcher->future().cancel();
        m_seekPreviewWatcher->waitForFinished();
    }
    {
        QMutexLocker lock(&m_previewMutex);
        if (m_previewCapture.isOpened()) {
            m_previewCapture.release();
        }
    }
#endif
    if (m_seekPreviewLabel) {
        m_seekPreviewLabel->clear();
        m_seekPreviewLabel->setVisible(false);
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

void VideoPlaybackWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (!m_previewLabel) return;

    if (m_previewFixedHeight > 0) {
        return;  // fixed-height mode (e.g. qFaceDetect) — geometry stable
    }
    // The camera image remains legible while the surrounding form scrolls on
    // short displays.  Limit its height so a wide desktop does not make the
    // capture controls unnecessarily far away.
    const int previewWidth = std::max(320, contentsRect().width() - 8);
    const int previewHeight = qBound(180, previewWidth * 9 / 16, 360);
    if (m_previewLabel->height() != previewHeight) {
        m_previewLabel->setFixedHeight(previewHeight);
    }
}

void VideoPlaybackWidget::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    // Restore video controls visibility when the widget is shown again
    // (e.g., after minimize/restore or plugin reopen).
#ifdef HAS_OPENCV_FACE_CAPTURE
    const bool videoLoaded = videoFileLoaded();
    if (m_videoControlsRow) m_videoControlsRow->setVisible(videoLoaded);
    if (m_videoSeekSlider) m_videoSeekSlider->setEnabled(videoLoaded);
    if (m_playbackSpeedCombo) m_playbackSpeedCombo->setEnabled(videoLoaded);
#endif
}

bool VideoPlaybackWidget::eventFilter(QObject* obj, QEvent* event) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (obj == m_videoSeekSlider && m_totalVideoFrames > 0) {
        switch (event->type()) {
            case QEvent::MouseButtonPress: {
                // Click-to-seek: when the user clicks on the slider track (not
                // just the handle), seek immediately to the clicked position.
                auto* me = static_cast<QMouseEvent*>(event);
                if (me->button() == Qt::LeftButton) {
                    const int sliderWidth = m_videoSeekSlider->width();
                    if (sliderWidth > 0) {
                        const int frame = static_cast<int>(
                                me->pos().x() * (m_totalVideoFrames - 1) /
                                sliderWidth);
                        const int clamped =
                                qBound(0, frame, m_totalVideoFrames - 1);
                        // Show preview at clicked position
                        showSeekPreview(clamped);
                        // Seek background reader
                        if (m_frameReaderReady &&
                            m_inputSource == InputSource::VideoFile) {
                            QMetaObject::invokeMethod(
                                    m_frameReader, "seekToFrame",
                                    Qt::QueuedConnection, Q_ARG(int, clamped));
                        }
                        // Update slider value
                        m_videoSeekSlider->blockSignals(true);
                        m_videoSeekSlider->setValue(clamped);
                        m_videoSeekSlider->blockSignals(false);
                        updateVideoTimeLabel(clamped);
                        // Update preview display (important when paused).
                        if (m_frameTimer && !m_frameTimer->isActive()) {
                            cv::Mat frame;
                            {
                                QMutexLocker lock(&m_frameMutex);
                                if (!m_latestFrame.empty())
                                    m_latestFrame.copyTo(frame);
                            }
                            if (!frame.empty()) {
                                QImage display = scaledDisplayImage(frame);
                                if (!display.isNull()) {
                                    onDisplayFrame(display, clamped);
                                    m_previewLabel->setPixmap(
                                            QPixmap::fromImage(display));
                                }
                            }
                        }
                        // Preview stays visible until Leave event or next
                        // drag.  Do NOT use QTimer::singleShot to auto-hide —
                        // it would dismiss the preview while the mouse is
                        // still on the slider.
                    }
                }
                break;
            }
            case QEvent::MouseMove: {
                auto* me = static_cast<QMouseEvent*>(event);
                // Throttle hover preview: 66ms (~15fps) on Linux/macOS,
                // 100ms on Windows where MSMF decode is slower.
                constexpr qint64 kHoverThrottleMs =
#ifdef Q_OS_WIN
                        100
#else
                        66
#endif
                        ;
                const qint64 now = QDateTime::currentMSecsSinceEpoch();
                if (!m_userSeeking &&
                    now - m_lastPreviewTimeMs < kHoverThrottleMs)
                    return QWidget::eventFilter(obj, event);
                m_lastPreviewTimeMs = now;

                // Map mouse x to frame index
                const int sliderWidth = m_videoSeekSlider->width();
                if (sliderWidth <= 0) break;
                const int frame = static_cast<int>(
                        me->pos().x() * (m_totalVideoFrames - 1) / sliderWidth);
                showSeekPreview(qBound(0, frame, m_totalVideoFrames - 1));
                break;
            }
            case QEvent::Leave:
                if (!m_userSeeking && m_seekPreviewLabel)
                    m_seekPreviewLabel->setVisible(false);
                break;
            default:
                break;
        }
    }
#endif
    return QWidget::eventFilter(obj, event);
}
