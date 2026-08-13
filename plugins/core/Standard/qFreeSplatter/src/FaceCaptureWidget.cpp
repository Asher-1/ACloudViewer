// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceCaptureWidget.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/core/utils/logger.hpp>
#endif

#include <QtCompat.h>
#include <cvFileDialog.h>

#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QFrame>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QPixmapCache>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSettings>
#include <QShowEvent>
#include <QTimer>
#ifdef HAS_OPENCV_FACE_CAPTURE
#if defined(HAS_QT_SQL)
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#endif
#endif
#include <QStandardPaths>
#include <QTemporaryFile>
#include <QUuid>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "aicore/runtime_capi.h"
#include "ecvPersistentSettings.h"

namespace {

QString facedetectModelCacheDir() {
    char* dir = aicore_facedetect_model_cache_dir();
    if (dir) {
        QString result = QString::fromUtf8(dir);
        aicore_facedetect_free_string(dir);
        return result;
    }
    return QDir::homePath() +
           QStringLiteral("/cloudViewer_data/extract/facedetect_models");
}

QString facedetectCachePath(const QString& filename) {
    return facedetectModelCacheDir() + QLatin1Char('/') + filename;
}

/// Default registry DB path following qFaceDetect convention:
/// face_registry_<model_stem>.db under the model cache directory.
/// Falls back to face_registry.db when no GGUF model is selected.
QString defaultRegistryDbPath(const QString& detectorModelFilename) {
    const QString baseDir = facedetectModelCacheDir();
    if (detectorModelFilename.isEmpty() ||
        detectorModelFilename == QStringLiteral("opencv")) {
        return baseDir + QStringLiteral("/face_registry.db");
    }
    const QString stem = QFileInfo(detectorModelFilename).completeBaseName();
    if (stem.isEmpty()) {
        return baseDir + QStringLiteral("/face_registry.db");
    }
    return baseDir + QStringLiteral("/face_registry_%1.db").arg(stem);
}

class AICoreInferenceGuard {
public:
    AICoreInferenceGuard(aicore_cancel_token* cancelToken,
                         const QString& device)
        : m_cancelToken(cancelToken) {
        m_locked = aicore_device_task_lock_cancelable(
                           device.toUtf8().constData(), m_cancelToken) == 0;
        if (!m_locked) return;
        aicore_cancel_scope_begin(m_cancelToken);
    }
    ~AICoreInferenceGuard() {
        if (!m_locked) return;
        aicore_cancel_scope_end(m_cancelToken);
        aicore_device_task_unlock();
    }

    bool locked() const { return m_locked; }

private:
    aicore_cancel_token* m_cancelToken = nullptr;
    bool m_locked = false;
};

// ---------------------------------------------------------------------------
// VideoFrameReader: owns a cv::VideoCapture on a dedicated QThread so that
// OpenCV's MSMF/DirectShow backend (Windows) does not block the Qt main
// thread during read() calls.  Communicates frames as RGB cv::Mat via
// emitted signal for direct use by face detection.
// ---------------------------------------------------------------------------
#ifdef HAS_OPENCV_FACE_CAPTURE
class VideoFrameReader : public QObject {
    Q_OBJECT
public:
    explicit VideoFrameReader(QObject* parent = nullptr) : QObject(parent) {}

    ~VideoFrameReader() override { release(); }

    bool openVideo(const std::string& path, int backend = cv::CAP_ANY) {
        release();
        return m_cap.open(path, backend) || m_cap.open(path, cv::CAP_ANY);
    }

    bool openCamera(int deviceIndex, int backend = cv::CAP_ANY) {
        release();
        m_cap.open(deviceIndex, backend);
        if (m_cap.isOpened()) {
            m_cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        }
        return m_cap.isOpened();
    }

    bool isOpened() const { return m_cap.isOpened(); }

    Q_INVOKABLE void release() {
        if (m_cap.isOpened()) m_cap.release();
    }

    int64_t getFrameCount() const {
        return m_cap.isOpened() ? static_cast<int64_t>(
                                          m_cap.get(cv::CAP_PROP_FRAME_COUNT))
                                : 0;
    }

    double getFps() const {
        return m_cap.isOpened() ? m_cap.get(cv::CAP_PROP_FPS) : 0.0;
    }

    int getFrameWidth() const {
        return m_cap.isOpened()
                       ? static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH))
                       : 0;
    }

    int getFrameHeight() const {
        return m_cap.isOpened()
                       ? static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                       : 0;
    }

    int currentFrameNum() const {
        return m_cap.isOpened()
                       ? static_cast<int>(m_cap.get(cv::CAP_PROP_POS_FRAMES))
                       : 0;
    }

public slots:
    void readFrame() {
        if (!m_cap.isOpened()) return;
        cv::Mat frame;
        if (!m_cap.read(frame) || frame.empty()) {
            emit frameReadFailed();
            return;
        }
        // Emit the raw OpenCV frame (BGR). Downstream consumers
        // (cvMatToQImage / detection) perform the single BGR→RGB conversion;
        // converting here would double-swap the channels and corrupt colors.
        emit frameReady(frame, currentFrameNum());
    }

    void seekToFrame(int frameIndex) {
        if (m_cap.isOpened()) {
            m_cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
        }
    }

signals:
    void frameReady(const cv::Mat& rgbFrame, int frameIndex);
    void frameReadFailed();

private:
    cv::VideoCapture m_cap;
};
#endif

}  // namespace

FaceCaptureWidget::FaceCaptureWidget(QWidget* parent) : QWidget(parent) {
    m_inferenceCancelToken = aicore_cancel_token_new();
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &FaceCaptureWidget::logMessage);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                if (m_downloadProgress) {
                    m_downloadProgress->setMaximum(
                            total > 0 ? static_cast<int>(total) : 0);
                    m_downloadProgress->setValue(static_cast<int>(received));
                }
                if (m_downloadLabel && total > 0) {
                    m_downloadLabel->setText(
                            tr("Downloading %1 — %2")
                                    .arg(currentGgmlFilename())
                                    .arg(ecvModelDownloader::
                                                 formatDownloadProgress(
                                                         received, total)));
                }
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& dest) {
                m_downloadInProgress = false;
                if (m_downloadProgress) m_downloadProgress->setVisible(false);
                if (m_downloadLabel) m_downloadLabel->setVisible(false);
                if (ok) {
                    emit logMessage(
                            tr("[FaceCapture] Downloaded model: %1").arg(dest));
                    populateDetectorCombo();
                    if (m_autoStartAfterDownload) {
                        m_autoStartAfterDownload = false;
                        startCamera(m_pendingCameraIndex);
                    }
                } else {
                    m_autoStartAfterDownload = false;
                    emit cameraError(tr("Model download failed: %1")
                                             .arg(currentGgmlFilename()));
                }
            });

    setupUi();
    loadFaceCaptureSettings();
    populateDetectorCombo();

    m_ggmlLoadWatcher = new QFutureWatcher<aicore_facedetect_ctx*>(this);
    connect(m_ggmlLoadWatcher,
            &QFutureWatcher<aicore_facedetect_ctx*>::finished, this, [this]() {
                if (!m_ggmlModelLoading) return;
                m_ggmlModelLoading = false;
                aicore_facedetect_ctx* ctx = m_ggmlLoadWatcher->result();
                const QString loadedPath = m_pendingGgmlPath;
                m_pendingGgmlPath.clear();
                const QString requestedPath =
                        facedetectCachePath(currentGgmlFilename());
                if (m_detectorKind != DetectorKind::Ggml ||
                    loadedPath != requestedPath) {
                    if (ctx) aicore_facedetect_free(ctx);
                    if (m_detectorKind == DetectorKind::Ggml &&
                        !currentGgmlFilename().isEmpty() &&
                        ecvModelDownloader::isValidCachedFile(requestedPath)) {
                        scheduleGgmlModelLoad(requestedPath);
                    }
                    return;
                }
                if (!aicore_facedetect_is_ready(ctx)) {
                    QString detail;
                    if (ctx) {
                        if (const char* error =
                                    aicore_facedetect_last_error(ctx)) {
                            detail = QString::fromUtf8(error);
                        }
                        aicore_facedetect_free(ctx);
                    }
                    m_statusLabel->setText(
                            tr("Failed to load face detector model"));
                    emit cameraError(
                            detail.isEmpty()
                                    ? tr("Failed to load face detector GGUF")
                                    : tr("Failed to load face detector GGUF: "
                                         "%1")
                                              .arg(detail));
                    return;
                }
                releaseGgmlModel();
                m_ggmlCtx = ctx;
                m_loadedGgmlPath = loadedPath;
                emit logMessage(
                        tr("[FaceCapture] Loaded face detector: %1")
                                .arg(QFileInfo(m_loadedGgmlPath).fileName()));
                if (m_cameraActive) {
                    m_statusLabel->setText(
                            m_inputSource == InputSource::VideoFile
                                    ? tr("Playing video — preview + face "
                                         "overlay")
                                    : tr("Camera active — detecting faces"));
                }
                if (m_detectorKind == DetectorKind::Ggml &&
                    !currentGgmlFilename().isEmpty() &&
                    requestedPath != m_loadedGgmlPath &&
                    ecvModelDownloader::isValidCachedFile(requestedPath)) {
                    scheduleGgmlModelLoad(requestedPath);
                }
            });
}

FaceCaptureWidget::~FaceCaptureWidget() {
    requestInferenceCancel();
    stopCamera();
#ifdef HAS_OPENCV_FACE_CAPTURE
    // Clean up background frame reader thread
    m_frameReaderRunning.store(0);
    m_frameReaderReady = false;
    if (m_frameReadTimer) m_frameReadTimer->stop();
    if (m_frameReaderThread && m_frameReaderThread->isRunning()) {
        QMetaObject::invokeMethod(m_frameReader, "release",
                                  Qt::QueuedConnection);
        m_frameReaderThread->quit();
        m_frameReaderThread->wait(2000);
    }
    {
        QMutexLocker lock(&m_frameMutex);
        m_latestFrame.release();
    }
#endif
    if (m_ggmlLoadWatcher && m_ggmlLoadWatcher->isRunning()) {
        m_ggmlLoadWatcher->waitForFinished();
    }
    if (m_ggmlLoadWatcher && m_ggmlModelLoading) {
        if (aicore_facedetect_ctx* ctx = m_ggmlLoadWatcher->result()) {
            aicore_facedetect_free(ctx);
        }
        m_ggmlModelLoading = false;
        m_pendingGgmlPath.clear();
    }
    releaseGgmlModel();
    aicore_cancel_token_free(m_inferenceCancelToken);
    m_inferenceCancelToken = nullptr;
}

bool FaceCaptureWidget::isAvailable() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    return true;
#else
    return false;
#endif
}

void FaceCaptureWidget::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(4, 4, 4, 4);
    mainLayout->setSpacing(4);

    m_previewLabel = new ecvClickableImageLabel(this);
    m_previewLabel->setMinimumSize(320, 180);
    m_previewLabel->setFixedHeight(180);
    m_previewLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_previewLabel->setStyleSheet(
            QStringLiteral("QLabel { background-color: #1a1a1a; "
                           "border: 1px solid #444; border-radius: 4px; }"));
    m_previewLabel->setText(tr("Camera preview"));

    // Preview label — added directly to the main layout.
    mainLayout->addWidget(m_previewLabel, 1);

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Video playback controls — placed below the preview area.
    // Using a regular QWidget + QVBoxLayout (no overlay / QStackedLayout)
    // so that child widgets render with native macOS styling.
    m_videoControlsRow = new QWidget(this);
    auto* videoCtrlMainLayout = new QVBoxLayout(m_videoControlsRow);
    videoCtrlMainLayout->setContentsMargins(0, 0, 0, 0);
    videoCtrlMainLayout->setSpacing(2);

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
    m_videoControlsRow->setVisible(false);  // Hidden until video file is loaded
    mainLayout->addWidget(m_videoControlsRow);  // below preview

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
        if (m_seekPreviewLabel) m_seekPreviewLabel->setVisible(false);
    });
    connect(m_videoSeekSlider, &QSlider::valueChanged, this,
            &FaceCaptureWidget::onVideoSeekSliderChanged);
    connect(m_playbackSpeedCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &FaceCaptureWidget::onPlaybackSpeedChanged);
#endif

    m_angleLabel = new QLabel(this);
    m_angleLabel->setAlignment(Qt::AlignCenter);
    m_angleLabel->hide();
    mainLayout->addWidget(m_angleLabel);

    m_statusLabel = new QLabel(this);
    m_statusLabel->setAlignment(Qt::AlignCenter);
    m_statusLabel->setWordWrap(false);
    m_statusLabel->setFixedHeight(m_statusLabel->fontMetrics().height() + 2);
    mainLayout->addWidget(m_statusLabel);

    m_captureProgress = new QProgressBar(this);
    m_captureProgress->setTextVisible(true);
    m_captureProgress->setFormat(tr("%v / %m faces"));
    m_captureProgress->setValue(0);
    mainLayout->addWidget(m_captureProgress);

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Keep the action adjacent to the video state. A file source has no camera
    // device to select, but must still offer the same explicit Capture action.
    m_cameraControlsRow = new QWidget(this);
    auto* controlsLayout = new QHBoxLayout(m_cameraControlsRow);
    controlsLayout->setContentsMargins(0, 0, 0, 0);
    controlsLayout->setSpacing(6);
    m_cameraDeviceLabel = new QLabel(tr("Device:"), m_cameraControlsRow);
    controlsLayout->addWidget(m_cameraDeviceLabel);

    m_cameraCombo = new QComboBox(m_cameraControlsRow);
    m_cameraCombo->addItem(tr("Default (0)"), 0);
    controlsLayout->addWidget(m_cameraCombo, 1);

    m_captureBtn = new QPushButton(tr("Capture"), m_cameraControlsRow);
    m_captureBtn->setEnabled(false);
    controlsLayout->addWidget(m_captureBtn);
    mainLayout->addWidget(m_cameraControlsRow);
#endif

    m_capturedGalleryScroll = new QScrollArea(this);
    m_capturedGalleryScroll->setWidgetResizable(true);
    m_capturedGalleryScroll->setFixedHeight(56);
    m_capturedGalleryScroll->setHorizontalScrollBarPolicy(
            Qt::ScrollBarAsNeeded);
    m_capturedGalleryScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_capturedGalleryScroll->setFrameShape(QFrame::NoFrame);
    m_capturedGalleryScroll->setVisible(false);
    m_capturedGalleryRow = new QWidget(m_capturedGalleryScroll);
    auto* galleryLayout = new QHBoxLayout(m_capturedGalleryRow);
    galleryLayout->setContentsMargins(0, 0, 0, 0);
    galleryLayout->setSpacing(4);
    m_capturedGalleryScroll->setWidget(m_capturedGalleryRow);
    mainLayout->addWidget(m_capturedGalleryScroll);

#ifdef HAS_OPENCV_FACE_CAPTURE
    auto* detectorInputRow = new QHBoxLayout();
    detectorInputRow->setSpacing(6);
    detectorInputRow->addWidget(new QLabel(tr("Face detector:"), this));
    m_detectorCombo = new QComboBox(this);
    detectorInputRow->addWidget(m_detectorCombo, 2);
    detectorInputRow->addWidget(new QLabel(tr("Input:"), this));
    m_sourceCombo = new QComboBox(this);
    m_sourceCombo->addItem(tr("Live camera"),
                           static_cast<int>(InputSource::Camera));
    m_sourceCombo->addItem(tr("Video file"),
                           static_cast<int>(InputSource::VideoFile));
    detectorInputRow->addWidget(m_sourceCombo, 1);
    mainLayout->addLayout(detectorInputRow);
    connect(m_detectorCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &FaceCaptureWidget::onDetectorComboChanged);
    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FaceCaptureWidget::onSourceChanged);

    auto* settingsRow = new QHBoxLayout();
    settingsRow->setSpacing(6);
    settingsRow->addWidget(new QLabel(tr("Min score:"), this));
    m_minScoreSpin = new QDoubleSpinBox(this);
    m_minScoreSpin->setRange(0.0, 1.0);
    m_minScoreSpin->setSingleStep(0.05);
    m_minScoreSpin->setValue(0.5);
    m_minScoreSpin->setToolTip(
            tr("Ignore detections below this confidence when capturing."));
    settingsRow->addWidget(m_minScoreSpin);

    settingsRow->addWidget(new QLabel(tr("Min captures:"), this));
    m_minCapturesSpin = new QSpinBox(this);
    m_minCapturesSpin->setRange(1, 20);
    m_minCapturesSpin->setValue(2);
    m_minCapturesSpin->setToolTip(
            tr("Minimum face snapshots required before reconstruction "
               "completes. Angle guides cycle until this count is reached."));
    settingsRow->addWidget(m_minCapturesSpin);

    settingsRow->addWidget(new QLabel(tr("Max distance:"), this));
    m_maxDistanceSpin = new QDoubleSpinBox(this);
    m_maxDistanceSpin->setRange(0.01, 2.0);
    m_maxDistanceSpin->setSingleStep(0.05);
    m_maxDistanceSpin->setDecimals(2);
    m_maxDistanceSpin->setValue(kDefaultSamePersonMaxDistance);
    m_maxDistanceSpin->setToolTip(
            tr("Cosine distance threshold for matching the same person across "
               "frames and registry identities. Lower = stricter matching."));
    settingsRow->addWidget(m_maxDistanceSpin);

    settingsRow->addWidget(new QLabel(tr("Face pick:"), this));
    m_faceStrategyCombo = new QComboBox(this);
    m_faceStrategyCombo->addItem(
            tr("Track same person"),
            static_cast<int>(FacePickStrategy::TrackSamePerson));
    m_faceStrategyCombo->addItem(
            tr("Largest face"),
            static_cast<int>(FacePickStrategy::LargestFace));
    m_faceStrategyCombo->addItem(
            tr("Highest score"),
            static_cast<int>(FacePickStrategy::HighestScore));
    m_faceStrategyCombo->setToolTip(tr(
            "When multiple faces appear, choose which one to capture. "
            "In Track same person mode, one identity is kept across frames."));
    settingsRow->addWidget(m_faceStrategyCombo, 1);
    mainLayout->addLayout(settingsRow);

    connect(m_minScoreSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { saveFaceCaptureSettings(); });
    connect(m_minCapturesSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int) {
                saveFaceCaptureSettings();
                updateCaptureProgressUi();
            });
    connect(m_faceStrategyCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { saveFaceCaptureSettings(); });

    auto* registryPathRow = new QHBoxLayout();
    registryPathRow->setSpacing(6);
    registryPathRow->addWidget(new QLabel(tr("Face registry:"), this));
    m_registryPathEdit = new QLineEdit(this);
    m_registryPathEdit->setPlaceholderText(tr("qFaceDetect registry database"));
    registryPathRow->addWidget(m_registryPathEdit, 1);
    auto* browseRegistryBtn = new QPushButton(tr("Browse..."), this);
    auto* reloadRegistryBtn = new QPushButton(tr("Reload"), this);
    registryPathRow->addWidget(browseRegistryBtn);
    registryPathRow->addWidget(reloadRegistryBtn);
    mainLayout->addLayout(registryPathRow);

    m_registryFilterEdit = new QLineEdit(this);
    m_registryFilterEdit->setPlaceholderText(
            tr("Filter registered identities by id or name"));
    mainLayout->addWidget(m_registryFilterEdit);
    m_registryList = new QListWidget(this);
    m_registryList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_registryList->setMaximumHeight(96);
    m_registryList->setAlternatingRowColors(true);
    mainLayout->addWidget(m_registryList);
    m_registryStatusLabel = new QLabel(this);
    mainLayout->addWidget(m_registryStatusLabel);

    connect(browseRegistryBtn, &QPushButton::clicked, this,
            &FaceCaptureWidget::onBrowseRegistry);
    connect(reloadRegistryBtn, &QPushButton::clicked, this,
            &FaceCaptureWidget::reloadRegistry);
    connect(m_registryPathEdit, &QLineEdit::editingFinished, this, [this]() {
        m_registryPathUserChosen = true;
        saveFaceCaptureSettings();
        reloadRegistry();
    });
    connect(m_registryFilterEdit, &QLineEdit::textChanged, this,
            &FaceCaptureWidget::filterRegistry);
#else
    auto* detectorLayout = new QHBoxLayout();
    detectorLayout->addWidget(new QLabel(tr("Face detector:"), this));
    m_detectorCombo = new QComboBox(this);
    detectorLayout->addWidget(m_detectorCombo, 1);
    mainLayout->addLayout(detectorLayout);
    connect(m_detectorCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &FaceCaptureWidget::onDetectorComboChanged);
#endif

    m_downloadLabel = new QLabel(this);
    m_downloadLabel->setAlignment(Qt::AlignCenter);
    m_downloadLabel->setVisible(false);
    mainLayout->addWidget(m_downloadLabel);

    m_downloadProgress = new QProgressBar(this);
    m_downloadProgress->setVisible(false);
    m_downloadProgress->setTextVisible(false);
    mainLayout->addWidget(m_downloadProgress);

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_videoFileRow = new QWidget(this);
    auto* videoLayout = new QHBoxLayout(m_videoFileRow);
    videoLayout->setContentsMargins(0, 0, 0, 0);
    m_videoPathEdit = new QLineEdit(m_videoFileRow);
    m_videoPathEdit->setPlaceholderText(
            tr("Path to video (mp4, avi, mkv, mov, webm, …)"));
    videoLayout->addWidget(m_videoPathEdit, 1);
    m_browseVideoBtn = new QPushButton(tr("Browse…"), m_videoFileRow);
    connect(m_browseVideoBtn, &QPushButton::clicked, this,
            &FaceCaptureWidget::onBrowseVideoFile);
    videoLayout->addWidget(m_browseVideoBtn);
    m_videoFileRow->setVisible(false);
    mainLayout->addWidget(m_videoFileRow);
    connect(m_videoPathEdit, &QLineEdit::textChanged, this,
            [this](const QString& text) {
                m_videoFilePath = text.trimmed();
                saveFaceCaptureSettings();
            });

    m_frameTimer = new QTimer(this);
    m_frameTimer->setInterval(30);

    m_frameReadTimer = new QTimer(this);
    // m_frameReadTimer interval will be set when video/camera starts.

    connect(m_frameTimer, &QTimer::timeout, this,
            &FaceCaptureWidget::processFrame);

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
    connect(m_frameReadTimer, &QTimer::timeout,
            static_cast<VideoFrameReader*>(m_frameReader),
            &VideoFrameReader::readFrame);
    connect(static_cast<VideoFrameReader*>(m_frameReader),
            &VideoFrameReader::frameReady, this,
            [this](const cv::Mat& rgbFrame, int frameIndex) {
                QMutexLocker lock(&m_frameMutex);
                rgbFrame.copyTo(m_latestFrame);
                m_frameReaderSeekTo.store(frameIndex);
            });
    connect(static_cast<VideoFrameReader*>(m_frameReader),
            &VideoFrameReader::frameReadFailed, this, [this]() {
                // Video end-of-file: loop back to start.
                // Camera transient failures are ignored (retry on next tick).
                if (m_inputSource == InputSource::VideoFile &&
                    m_frameReaderReady && !m_userSeeking) {
                    QMetaObject::invokeMethod(m_frameReader, "seekToFrame",
                                              Qt::QueuedConnection,
                                              Q_ARG(int, 0));
                    m_lastDetectedFrameNum = 0;
                    m_ggmlFrameSkip = 0;
                    if (m_videoSeekSlider) {
                        m_videoSeekSlider->blockSignals(true);
                        m_videoSeekSlider->setValue(0);
                        m_videoSeekSlider->blockSignals(false);
                    }
                    updateVideoTimeLabel(0);
                }
            });
    m_frameReaderThread->start();
    connect(m_captureBtn, &QPushButton::clicked, this,
            &FaceCaptureWidget::captureCurrentFrame);
    connect(m_cameraCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                if (!m_cameraActive) return;
                const int idx = m_cameraCombo->currentData().toInt();
                stopCamera();
                startCamera(idx);
            });

    m_statusLabel->setText(
            tr("Ready \u2014 choose a detector, then start the camera"));
    updateCaptureProgressUi();
#else
    m_statusLabel->setText(
            tr("Face capture unavailable (OpenCV not built with videoio "
               "and objdetect)"));
#endif
}

void FaceCaptureWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (!m_previewLabel) return;

    // The camera image remains legible while the surrounding form scrolls on
    // short displays.  Limit its height so a wide desktop does not make the
    // capture controls unnecessarily far away.
    const int previewWidth = std::max(320, contentsRect().width() - 8);
    const int previewHeight = qBound(180, previewWidth * 9 / 16, 360);
    if (m_previewLabel->height() != previewHeight) {
        m_previewLabel->setFixedHeight(previewHeight);
    }

#ifdef HAS_OPENCV_FACE_CAPTURE
    // The seek preview thumbnail is now managed by the layout system
    // (no manual positioning needed).
    Q_UNUSED(m_seekPreviewLabel)
    Q_UNUSED(m_videoSeekSlider)
    Q_UNUSED(m_videoControlsRow)
#endif
}

void FaceCaptureWidget::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    // Restore video controls visibility when the widget is shown again
    // (e.g., after minimize/restore or plugin reopen).
#ifdef HAS_OPENCV_FACE_CAPTURE
    const bool videoLoaded =
            m_frameReaderReady && m_inputSource == InputSource::VideoFile;
    if (m_videoControlsRow) m_videoControlsRow->setVisible(videoLoaded);
    if (m_videoSeekSlider) m_videoSeekSlider->setEnabled(videoLoaded);
    if (m_playbackSpeedCombo) m_playbackSpeedCombo->setEnabled(videoLoaded);
#endif
}

void FaceCaptureWidget::onSourceChanged(int index) {
    if (!m_sourceCombo) return;
    m_inputSource =
            static_cast<InputSource>(m_sourceCombo->itemData(index).toInt());
    if (m_videoFileRow) {
        m_videoFileRow->setVisible(m_inputSource == InputSource::VideoFile);
    }
    if (m_videoControlsRow) {
        // Hide playback controls in camera mode — they are meaningless for
        // live capture.  Re-enable when a video file is opened.
        m_videoControlsRow->setVisible(m_inputSource == InputSource::VideoFile);
        if (m_videoSeekSlider) m_videoSeekSlider->setEnabled(false);
        if (m_playbackSpeedCombo) m_playbackSpeedCombo->setEnabled(false);
    }
    if (m_cameraControlsRow) {
        m_cameraControlsRow->setVisible(true);
    }
    if (m_cameraDeviceLabel) {
        m_cameraDeviceLabel->setVisible(m_inputSource == InputSource::Camera);
    }
    if (m_cameraCombo) {
        m_cameraCombo->setVisible(m_inputSource == InputSource::Camera);
    }
    if (m_cameraActive) {
        stopCapture();
    }
    if (m_inputSource == InputSource::VideoFile) {
        m_statusLabel->setText(tr("Select a video file, then start playback"));
    } else {
        m_statusLabel->setText(
                tr("Ready \u2014 choose a detector, then start the camera"));
    }
}

void FaceCaptureWidget::onBrowseVideoFile() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qFreeSplatter"),
                             QStringLiteral("lastVideoDir"), QDir::homePath());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select video file"), lastDir,
            tr("Video files (*.mp4 *.avi *.mkv *.mov *.webm *.m4v *.wmv *.ts "
               "*.mpg *.mpeg);;All files (*.*)"));
    if (path.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qFreeSplatter"),
                         QStringLiteral("lastVideoDir"), path);
    if (m_videoPathEdit) m_videoPathEdit->setText(path);
    m_videoFilePath = path;
}

void FaceCaptureWidget::onVideoSeekSliderChanged(int value) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_frameReaderReady || m_inputSource != InputSource::VideoFile) return;
    if (m_userSeeking) {
        // During drag: show preview thumbnail via independent decode path
        showSeekPreview(value);
        // Update time label to show target position (only when timer is
        // inactive; during playback processFrame overwrites it each tick).
        if (!m_frameTimer->isActive()) {
            updateVideoTimeLabel(value);
        }
        return;
    }
    // Forward seek to background reader
    QMetaObject::invokeMethod(m_frameReader, "seekToFrame",
                              Qt::QueuedConnection, Q_ARG(int, value));
    // Reset detection throttle so detection resumes immediately after seek.
    // Without this, a backward seek would make curFrameNum <
    // m_lastDetectedFrameNum, causing the video-time delta to be negative and
    // timeForDetection=false forever.
    m_lastDetectedFrameNum = value;
    // When the timer is not running (video paused / stopped) the preview
    // label would otherwise keep showing the old frame.  Read the cached
    // frame and push it to the display.
    if (!m_frameTimer->isActive()) {
        cv::Mat frame;
        {
            QMutexLocker lock(&m_frameMutex);
            if (!m_latestFrame.empty()) m_latestFrame.copyTo(frame);
        }
        if (!frame.empty()) {
            QImage img = cvMatToQImage(frame);
            m_previewLabel->setPixmap(QPixmap::fromImage(
                    img.scaled(m_previewLabel->size(), Qt::KeepAspectRatio,
                               Qt::FastTransformation)));
        }
    }
#endif
}

int FaceCaptureWidget::computeTimerInterval() const {
    // For video files: match the video's native frame rate × playback speed.
    // This ensures one timer tick ≈ one video frame advance.
    if (m_inputSource == InputSource::VideoFile && m_videoFps > 0) {
        const double interval = 1000.0 / (m_videoFps * m_playbackSpeed);
        return std::max(1, static_cast<int>(std::lround(interval)));
    }
    // For camera: base interval adjusted by speed (if speed control is used).
    return std::max(1, static_cast<int>(m_baseTimerInterval / m_playbackSpeed));
}

void FaceCaptureWidget::onPlaybackSpeedChanged(int index) {
    // Speed values: 0.25, 0.5, 1.0, 2.0, 4.0
    static constexpr double speeds[] = {0.25, 0.5, 1.0, 2.0, 4.0};
    if (index < 0 || index >= 5) return;
    m_playbackSpeed = speeds[index];
    // Recompute BOTH timers: m_frameReadTimer drives the background reader
    // (the actual video read rate) and m_frameTimer drives display/inference.
    const int interval = computeTimerInterval();
    if (m_frameReadTimer) m_frameReadTimer->setInterval(interval);
    if (m_frameTimer) m_frameTimer->setInterval(interval);
}

void FaceCaptureWidget::updateVideoTimeLabel(int frameIndex) {
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

void FaceCaptureWidget::showSeekPreview(int frameIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_videoFilePath.isEmpty() || m_totalVideoFrames <= 0) return;
    if (frameIndex < 0) frameIndex = 0;
    if (frameIndex >= m_totalVideoFrames) frameIndex = m_totalVideoFrames - 1;

    if (!m_seekPreviewLabel) return;

    // Check QPixmapCache first — avoids re-decoding the same frame.
    const QString cacheKey = QStringLiteral("qfs_seekpreview_%1_%2")
                                     .arg(m_videoFilePath.size())
                                     .arg(frameIndex);
    QPixmap cached;
    if (QPixmapCache::find(cacheKey, &cached)) {
        m_seekPreviewLabel->setPixmap(cached);
    } else {
        // Open preview capture if not already open (independent decode path)
        if (!m_previewCapture.isOpened()) {
            if (!m_previewCapture.open(m_videoFilePath.toStdString(),
                                       cv::CAP_FFMPEG) &&
                !m_previewCapture.open(m_videoFilePath.toStdString(),
                                       cv::CAP_ANY)) {
                return;  // silently fail — preview is non-critical
            }
        }

        m_previewCapture.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
        cv::Mat frame;
        if (!m_previewCapture.read(frame) || frame.empty()) return;

        // Scale to thumbnail size (160×90)
        cv::Mat thumb;
        cv::resize(frame, thumb, cv::Size(160, 90), 0, 0, cv::INTER_AREA);
        QImage img = cvMatToQImage(thumb);
        QPixmap pixmap = QPixmap::fromImage(img);
        QPixmapCache::insert(cacheKey, pixmap);
        m_seekPreviewLabel->setPixmap(pixmap);

        // Limit cache size: if we have more than 300 entries, trim.
        // Each entry is ~160×90×4 ≈ 57 KB, so 300 entries ≈ 17 MB.
        static int s_cacheCheckCounter = 0;
        if (++s_cacheCheckCounter % 50 == 0 && QPixmapCache::cacheLimit() > 0) {
            QPixmapCache::setCacheLimit(32768);  // 32 MB cap
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

void FaceCaptureWidget::closePreviewCapture() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_previewCapture.isOpened()) {
        m_previewCapture.release();
    }
#endif
    if (m_seekPreviewLabel) {
        m_seekPreviewLabel->clear();
        m_seekPreviewLabel->setVisible(false);
    }
}

bool FaceCaptureWidget::eventFilter(QObject* obj, QEvent* event) {
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
                        // Seek main capture via background reader
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
                        if (!m_frameTimer->isActive()) {
                            cv::Mat evFrame;
                            {
                                QMutexLocker lock(&m_frameMutex);
                                if (!m_latestFrame.empty())
                                    m_latestFrame.copyTo(evFrame);
                            }
                            if (!evFrame.empty()) {
                                QImage img = cvMatToQImage(evFrame);
                                m_previewLabel->setPixmap(QPixmap::fromImage(
                                        img.scaled(m_previewLabel->size(),
                                                   Qt::KeepAspectRatio,
                                                   Qt::FastTransformation)));
                            }
                        }
                        // Preview stays visible until Leave event or next drag.
                        // Do NOT use QTimer::singleShot to auto-hide — it would
                        // dismiss the preview while the mouse is still on the
                        // slider.
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
    return QWidget::eventFilter(obj, event);
}

void FaceCaptureWidget::onBrowseRegistry() {
    const QString current =
            m_registryPathEdit
                    ? QFileInfo(m_registryPathEdit->text()).absolutePath()
                    : facedetectModelCacheDir();
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select qFaceDetect registry"), current,
            tr("SQLite databases (*.db *.sqlite *.sqlite3);;All files (*.*)"));
    if (path.isEmpty()) return;
    m_registryPathUserChosen = true;
    m_registryPathEdit->setText(path);
    saveFaceCaptureSettings();
    reloadRegistry();
}

std::vector<float> FaceCaptureWidget::normalizeEmbedding(
        const std::vector<float>& embedding) {
    double norm2 = 0.0;
    for (float value : embedding) {
        if (!std::isfinite(value)) return {};
        norm2 += static_cast<double>(value) * value;
    }
    if (embedding.empty() || norm2 <= 0.0) return {};
    const float inv = static_cast<float>(1.0 / std::sqrt(norm2));
    std::vector<float> normalized(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        normalized[i] = embedding[i] * inv;
    }
    return normalized;
}

void FaceCaptureWidget::reloadRegistry() {
    m_registryIdentities.clear();
    if (m_registryList) m_registryList->clear();
    const QString path = m_registryPathEdit
                                 ? m_registryPathEdit->text().trimmed()
                                 : QString();
    if (path.isEmpty() || !QFileInfo(path).isFile()) {
        if (m_registryStatusLabel) {
            m_registryStatusLabel->setText(tr("No face registry loaded"));
        }
        return;
    }

#if defined(HAS_OPENCV_FACE_CAPTURE) && defined(HAS_QT_SQL)
    const QString connection =
            QStringLiteral("qfs_face_registry_") +
            QUuid::createUuid().toString(QUuid::WithoutBraces);
    QString error;
    {
        QSqlDatabase db = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"),
                                                    connection);
        db.setConnectOptions(QStringLiteral("QSQLITE_OPEN_READONLY"));
        db.setDatabaseName(path);
        if (!db.open()) {
            error = db.lastError().text();
        } else {
            QSqlQuery query(db);
            if (!query.exec(QStringLiteral(
                        "SELECT id,name,model,dim,embedding FROM faces "
                        "ORDER BY name COLLATE NOCASE,id"))) {
                error = query.lastError().text();
            } else {
                while (query.next()) {
                    RegistryIdentity identity;
                    identity.id = query.value(0).toString();
                    identity.name = query.value(1).toString();
                    identity.modelFile = query.value(2).toString();
                    const int dim = query.value(3).toInt();
                    const QByteArray blob = query.value(4).toByteArray();
                    if (identity.id.isEmpty() || identity.name.isEmpty() ||
                        dim <= 0 || blob.size() != dim * int(sizeof(float))) {
                        continue;
                    }
                    identity.embedding.resize(static_cast<size_t>(dim));
                    std::memcpy(identity.embedding.data(), blob.constData(),
                                static_cast<size_t>(blob.size()));
                    identity.embedding = normalizeEmbedding(identity.embedding);
                    if (!identity.embedding.empty()) {
                        m_registryIdentities.push_back(std::move(identity));
                    }
                }
            }
            db.close();
        }
    }
    QSqlDatabase::removeDatabase(connection);

    if (!error.isEmpty()) {
        if (m_registryStatusLabel) {
            m_registryStatusLabel->setText(
                    tr("Registry load failed: %1").arg(error));
        }
        return;
    }
#else
    if (m_registryStatusLabel) {
        m_registryStatusLabel->setText(
                tr("Registry loading requires Qt SQL support (not available in "
                   "this build)."));
    }
    return;
#endif

    if (m_registryList) {
        for (size_t i = 0; i < m_registryIdentities.size(); ++i) {
            const RegistryIdentity& identity = m_registryIdentities[i];
            auto* item = new QListWidgetItem(
                    QStringLiteral("%1  [%2]")
                            .arg(identity.name, identity.id.left(12)),
                    m_registryList);
            item->setData(Qt::UserRole, static_cast<int>(i));
            item->setToolTip(tr("id: %1\nmodel: %2\ndimension: %3")
                                     .arg(identity.id, identity.modelFile)
                                     .arg(identity.embedding.size()));
        }
    }
    filterRegistry(m_registryFilterEdit ? m_registryFilterEdit->text()
                                        : QString());
    if (m_registryStatusLabel) {
        m_registryStatusLabel->setText(
                tr("%1 registered identities; select one or more to track")
                        .arg(m_registryIdentities.size()));
    }
}

void FaceCaptureWidget::filterRegistry(const QString& text) {
    if (!m_registryList) return;
    const QString needle = text.trimmed();
    for (int row = 0; row < m_registryList->count(); ++row) {
        QListWidgetItem* item = m_registryList->item(row);
        const int index = item->data(Qt::UserRole).toInt();
        const bool match =
                index >= 0 &&
                static_cast<size_t>(index) < m_registryIdentities.size() &&
                (needle.isEmpty() ||
                 m_registryIdentities[index].id.contains(needle,
                                                         Qt::CaseInsensitive) ||
                 m_registryIdentities[index].name.contains(
                         needle, Qt::CaseInsensitive));
        item->setHidden(!match);
    }
}

int FaceCaptureWidget::selectedCameraIndex() const {
    if (!m_cameraCombo) return 0;
    return m_cameraCombo->currentData().toInt();
}

QString FaceCaptureWidget::videoFilePath() const {
    if (m_videoPathEdit) {
        const QString edited = m_videoPathEdit->text().trimmed();
        if (!edited.isEmpty()) return edited;
    }
    return m_videoFilePath;
}

void FaceCaptureWidget::setVideoFilePath(const QString& path) {
    m_videoFilePath = path;
    if (m_videoPathEdit) {
        m_videoPathEdit->setText(path);
    }
    saveFaceCaptureSettings();
}

void FaceCaptureWidget::setInputSource(InputSource source) {
    if (m_sourceCombo) {
        const int idx = m_sourceCombo->findData(static_cast<int>(source));
        if (idx >= 0) m_sourceCombo->setCurrentIndex(idx);
    }
}

bool FaceCaptureWidget::startVideoFile(const QString& path) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (path.isEmpty()) return false;

    qInfo() << "[FaceCaptureWidget] startVideoFile:" << path;

    // Resume from paused state if same video
    if (m_videoPaused && m_frameReaderReady && m_videoFilePath == path &&
        m_inputSource == InputSource::VideoFile) {
        m_videoPaused = false;
        m_cameraActive = true;
        m_ggmlFrameSkip = 0;
        aicore_cancel_token_reset(m_inferenceCancelToken);
        const int interval = computeTimerInterval();
        m_frameReadTimer->setInterval(interval);
        m_frameReadTimer->start();
        m_frameTimer->setInterval(interval);
        m_frameTimer->start();
        m_statusLabel->setText(tr("Resuming video"));
        emit cameraStarted();
        return true;
    }

    // Cancel and drain the previous session before issuing any work for the
    // new video. Doing this after scheduleGgmlModelLoad would cancel the new
    // model load through the shared task token.
    qInfo() << "[FaceCaptureWidget] Stopping previous capture...";
    stopCapture();
    closePreviewCapture();
    aicore_cancel_token_reset(m_inferenceCancelToken);
    qInfo() << "[FaceCaptureWidget] Configuring detector...";
    if (!configureDetectorForRegistrySelection()) {
        qWarning() << "[FaceCaptureWidget] "
                      "configureDetectorForRegistrySelection failed";
        return false;
    }
    m_inputSource = InputSource::VideoFile;
    m_videoFilePath = path;
    if (m_videoPathEdit) m_videoPathEdit->setText(path);

    qInfo() << "[FaceCaptureWidget] Detector kind:"
            << static_cast<int>(m_detectorKind);
    if (m_detectorKind == DetectorKind::Ggml) {
        if (!ensureGgmlModelReady()) {
            m_statusLabel->setText(
                    tr("Downloading face detector — video preview starting…"));
        } else {
            scheduleGgmlModelLoad(facedetectCachePath(currentGgmlFilename()));
        }
    } else if (m_detectorKind == DetectorKind::OpenCV) {
        releaseGgmlModel();
        loadCascade();
    }

    qInfo() << "[FaceCaptureWidget] Opening video with OpenCV...";
    auto* reader = static_cast<VideoFrameReader*>(m_frameReader);
    if (!reader->openVideo(path.toStdString(), cv::CAP_FFMPEG) &&
        !reader->openVideo(path.toStdString(), cv::CAP_ANY)) {
        const QString err =
                tr("Failed to open video (rebuild OpenCV with FFmpeg / "
                   "WITH_FFMPEG=ON): %1")
                        .arg(path);
        m_statusLabel->setText(err);
        emit cameraError(err);
        m_frameReaderReady = false;
        return false;
    }
    m_frameReaderReady = true;

    qInfo() << "[FaceCaptureWidget] Video opened successfully, starting frame "
               "timer...";
    m_cameraActive = true;
    m_ggmlFrameSkip = 0;
    m_lastDetectedFrameNum = 0;

    // Initialize video seek slider from background reader metadata
    m_totalVideoFrames = static_cast<int>(reader->getFrameCount());
    m_videoFps = reader->getFps();
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

    // Set timer interval from video FPS × speed (ensures 1 tick ≈ 1 video
    // frame).
    const int interval = computeTimerInterval();
    m_frameReadTimer->setInterval(interval);
    m_frameReadTimer->start();
    m_frameTimer->setInterval(interval);
    m_frameTimer->start();
    m_statusLabel->setText(tr("Playing video — preview + face overlay"));
    emit cameraStarted();
    return true;
#else
    Q_UNUSED(path);
    return false;
#endif
}

void FaceCaptureWidget::restartVideoFile() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_frameReaderReady || m_videoFilePath.isEmpty()) return;
    QMetaObject::invokeMethod(m_frameReader, "seekToFrame",
                              Qt::QueuedConnection, Q_ARG(int, 0));
    // Reset detection throttle so detection resumes immediately after restart.
    m_lastDetectedFrameNum = 0;
    m_ggmlFrameSkip = 0;
    if (m_videoSeekSlider) {
        m_videoSeekSlider->blockSignals(true);
        m_videoSeekSlider->setValue(0);
        m_videoSeekSlider->blockSignals(false);
    }
    updateVideoTimeLabel(0);
    if (!m_cameraActive) {
        m_videoPaused = false;
        m_cameraActive = true;
        m_ggmlFrameSkip = 0;
        m_lastDetectedFrameNum = 0;
        aicore_cancel_token_reset(m_inferenceCancelToken);
        const int interval = computeTimerInterval();
        m_frameReadTimer->setInterval(interval);
        m_frameReadTimer->start();
        m_frameTimer->setInterval(interval);
        m_frameTimer->start();
        m_statusLabel->setText(tr("Restarted video"));
        emit cameraStarted();
    }
#endif
}

void FaceCaptureWidget::stopCapture() {
    requestInferenceCancel();
    stopCamera();
}

void FaceCaptureWidget::requestInferenceCancel() {
    aicore_cancel_token_request(m_inferenceCancelToken);
}

void FaceCaptureWidget::setInferenceDevice(const QString& device) {
    const QString normalized =
            device.isEmpty() ? QStringLiteral("auto") : device;
    if (normalized == m_inferenceDevice) return;
    requestInferenceCancel();
    if (m_ggmlLoadWatcher && m_ggmlLoadWatcher->isRunning()) {
        m_ggmlLoadWatcher->waitForFinished();
        if (aicore_facedetect_ctx* ctx = m_ggmlLoadWatcher->result()) {
            aicore_facedetect_free(ctx);
        }
        m_ggmlModelLoading = false;
        m_pendingGgmlPath.clear();
    }
    m_inferenceDevice = normalized;
    releaseGgmlModel();
    if (m_cameraActive && m_detectorKind == DetectorKind::Ggml) {
        scheduleGgmlModelLoad(facedetectCachePath(currentGgmlFilename()));
    }
}

void FaceCaptureWidget::releaseGpuResources() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    requestInferenceCancel();
    stopCamera();
    if (m_ggmlLoadWatcher && m_ggmlLoadWatcher->isRunning()) {
        m_ggmlLoadWatcher->waitForFinished();
    }
    if (m_ggmlLoadWatcher && m_ggmlModelLoading) {
        if (aicore_facedetect_ctx* ctx = m_ggmlLoadWatcher->result()) {
            aicore_facedetect_free(ctx);
        }
        m_ggmlModelLoading = false;
        m_pendingGgmlPath.clear();
    }
    releaseGgmlModel();
#endif
}

bool FaceCaptureWidget::isCaptureActive() const { return m_cameraActive; }

void FaceCaptureWidget::populateDetectorCombo() {
    if (!m_detectorCombo) return;

    const QString current = m_detectorCombo->currentData().toString();
    m_detectorCombo->blockSignals(true);
    m_detectorCombo->clear();

    m_detectorCombo->addItem(tr("OpenCV Haar Cascade"),
                             QStringLiteral("opencv"));

    const QString cache = facedetectModelCacheDir();
    for (int i = 0; i < aicore_facedetect_detector_model_count(); ++i) {
        const aicore_facedetect_model_entry* m =
                aicore_facedetect_detector_model_at(i);
        if (!m) continue;
        const QFileInfo fi(cache + QLatin1Char('/') +
                           QString::fromUtf8(m->filename));
        const QString suffix =
                ecvModelDownloader::isValidCachedFile(fi.absoluteFilePath())
                        ? QString(" [%1] \u2713")
                                  .arg(ecvModelDownloader::formatFileSize(
                                          fi.size()))
                        : QString(" [download]");
        m_detectorCombo->addItem(QCoreApplication::translate("FaceDetectModels",
                                                             m->display_name) +
                                         suffix,
                                 QString::fromUtf8(m->filename));
    }

    int restore = 0;
    for (int i = 0; i < m_detectorCombo->count(); ++i) {
        if (m_detectorCombo->itemData(i).toString() == current) {
            restore = i;
            break;
        }
    }
    if (restore == 0 && m_detectorCombo->count() > 1) {
        for (int i = 0; i < m_detectorCombo->count(); ++i) {
            if (m_detectorCombo->itemData(i).toString() ==
                QStringLiteral("buffalo_l.gguf")) {
                restore = i;
                break;
            }
        }
    }
    m_detectorCombo->setCurrentIndex(restore);
    m_detectorCombo->blockSignals(false);
    onDetectorComboChanged(m_detectorCombo->currentIndex());
}

void FaceCaptureWidget::refreshDetectorList() { populateDetectorCombo(); }

void FaceCaptureWidget::onDetectorComboChanged(int index) {
    Q_UNUSED(index);
    releaseGgmlModel();

    const QString data = m_detectorCombo->currentData().toString();
    if (data == QStringLiteral("opencv")) {
        m_detectorKind = DetectorKind::OpenCV;
    } else if (!data.isEmpty()) {
        m_detectorKind = DetectorKind::Ggml;
    } else {
        m_detectorKind = DetectorKind::None;
    }

    // When the user has not manually chosen a registry path, auto-switch
    // to the model-specific default so the DB matches the active detector.
    if (!m_registryPathUserChosen && m_registryPathEdit) {
        const QString newDefault = defaultRegistryDbPath(data);
        m_registryPathEdit->setText(newDefault);
        reloadRegistry();
    }

    if (m_cameraActive) {
        stopCamera();
        startCamera(m_pendingCameraIndex);
    }
}

QString FaceCaptureWidget::currentGgmlFilename() const {
    if (!m_detectorCombo) return {};
    const QString data = m_detectorCombo->currentData().toString();
    if (data == QStringLiteral("opencv")) return {};
    return data;
}

bool FaceCaptureWidget::detectorReady() const {
    if (m_detectorKind == DetectorKind::OpenCV) return m_cascadeLoaded;
    if (m_detectorKind == DetectorKind::Ggml)
        return aicore_facedetect_is_ready(m_ggmlCtx) != 0;
    return false;
}

bool FaceCaptureWidget::ensureGgmlModelReady() {
    const QString filename = currentGgmlFilename();
    if (filename.isEmpty()) return true;

    const QString path = facedetectCachePath(filename);
    if (ecvModelDownloader::isValidCachedFile(path)) return true;

    const aicore_facedetect_model_entry* model =
            aicore_facedetect_model_by_filename(filename.toUtf8().constData());
    if (!model) return false;

    if (m_downloadInProgress) return false;

    startModelDownload(model);
    return false;
}

void FaceCaptureWidget::startModelDownload(
        const aicore_facedetect_model_entry* model) {
    if (!model || m_downloadInProgress) return;
    QDir().mkpath(facedetectModelCacheDir());
    const QString dest =
            facedetectCachePath(QString::fromUtf8(model->filename));

    m_downloadInProgress = true;
    m_autoStartAfterDownload = true;
    if (m_downloadLabel) {
        m_downloadLabel->setVisible(true);
        m_downloadLabel->setText(
                tr("Downloading %1 ...")
                        .arg(QString::fromUtf8(model->filename)));
    }
    if (m_downloadProgress) {
        m_downloadProgress->setVisible(true);
        m_downloadProgress->setValue(0);
    }
    emit logMessage(tr("[FaceCapture] Downloading %1 ...")
                            .arg(QString::fromUtf8(model->filename)));

    ecvModelDownloader::Request req;
    req.url = QString::fromUtf8(model->download_url);
    req.destPath = dest;
    m_downloader->download(req);
}

void FaceCaptureWidget::releaseGgmlModel() {
    if (m_ggmlCtx) {
        aicore_facedetect_free(m_ggmlCtx);
        m_ggmlCtx = nullptr;
    }
    m_loadedGgmlPath.clear();
}

bool FaceCaptureWidget::loadGgmlModel(const QString& path) {
    if (path.isEmpty()) return false;
    if (m_ggmlCtx && m_loadedGgmlPath == path) return true;

    releaseGgmlModel();

    aicore_cancel_token_reset(m_inferenceCancelToken);
    AICoreInferenceGuard guard(m_inferenceCancelToken, m_inferenceDevice);
    if (!guard.locked()) return false;
    aicore_facedetect_options* opts = aicore_facedetect_options_new();
    if (!opts) return false;
    aicore_facedetect_options_set_device(
            opts, m_inferenceDevice.toUtf8().constData());
    aicore_facedetect_options_set_threads(opts, 0);
    m_ggmlCtx = aicore_facedetect_load_opts(path.toUtf8().constData(), opts);
    aicore_facedetect_options_free(opts);

    if (!aicore_facedetect_is_ready(m_ggmlCtx)) {
        releaseGgmlModel();
        emit logMessage(tr("[FaceCapture] Failed to load GGUF: %1").arg(path));
        return false;
    }
    m_loadedGgmlPath = path;
    emit logMessage(tr("[FaceCapture] Loaded face detector: %1")
                            .arg(QFileInfo(path).fileName()));
    return true;
}

void FaceCaptureWidget::scheduleGgmlModelLoad(const QString& path) {
    if (path.isEmpty()) return;
    if (m_ggmlCtx && m_loadedGgmlPath == path) return;
    if (m_ggmlLoadWatcher && m_ggmlLoadWatcher->isRunning()) return;

    m_ggmlModelLoading = true;
    aicore_cancel_token_reset(m_inferenceCancelToken);
    m_pendingGgmlPath = path;
    m_statusLabel->setText(tr("Loading face detector model..."));
    m_ggmlLoadWatcher->setFuture(QtConcurrent::run(
            [path, token = m_inferenceCancelToken,
             device = m_inferenceDevice]() -> aicore_facedetect_ctx* {
                AICoreInferenceGuard guard(token, device);
                if (!guard.locked()) return nullptr;
                aicore_facedetect_options* opts =
                        aicore_facedetect_options_new();
                if (!opts) return nullptr;
                aicore_facedetect_options_set_device(
                        opts, device.toUtf8().constData());
                aicore_facedetect_options_set_threads(opts, 0);
                aicore_facedetect_ctx* ctx = aicore_facedetect_load_opts(
                        path.toUtf8().constData(), opts);
                aicore_facedetect_options_free(opts);
                return ctx;
            }));
}

bool FaceCaptureWidget::loadCascade() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_cascadeLoaded) return true;

    const QString qrcPath = QStringLiteral(
            ":/CC/plugin/qFreeSplatter/"
            "data/haarcascade_frontalface_alt2.xml");
    if (QFile::exists(qrcPath)) {
        // Keep the bundled cascade in the per-user application data directory.
        // A shared system temp path can race between processes and is cleaned
        // unpredictably by the OS. AppLocalDataLocation maps to the native
        // cache/data location on Linux, Windows, and macOS.
        QString cacheDir = QStandardPaths::writableLocation(
                QStandardPaths::AppLocalDataLocation);
        if (cacheDir.isEmpty()) {
            cacheDir = QCoreApplication::applicationDirPath() +
                       QStringLiteral("/cloudViewer_data");
        }
        cacheDir += QStringLiteral("/qFreeSplatter");
        QDir().mkpath(cacheDir);
        const QString tmpPath =
                cacheDir + QStringLiteral("/haarcascade_frontalface_alt2.xml");
        if (!QFile::exists(tmpPath)) {
            QFile::copy(qrcPath, tmpPath);
            QFile::setPermissions(
                    tmpPath, QFileDevice::ReadOwner | QFileDevice::WriteOwner);
        }
        if (m_faceCascade.load(tmpPath.toStdString())) {
            m_cascadeLoaded = true;
            return true;
        }
    }

#ifdef OPENCV_DATA_DIR
    {
        const QString path =
                QString(OPENCV_DATA_DIR) +
                QStringLiteral(
                        "/haarcascades/haarcascade_frontalface_alt2.xml");
        if (QFile::exists(path) && m_faceCascade.load(path.toStdString())) {
            m_cascadeLoaded = true;
            return true;
        }
    }
#endif

    const QStringList systemPaths = {
            QCoreApplication::applicationDirPath() +
                    QStringLiteral("/../share/opencv4/haarcascades/"
                                   "haarcascade_frontalface_alt2.xml"),
            QStringLiteral("/usr/share/opencv4/haarcascades/"
                           "haarcascade_frontalface_alt2.xml"),
            QStringLiteral("/usr/local/share/opencv4/haarcascades/"
                           "haarcascade_frontalface_alt2.xml"),
            QStringLiteral("/opt/homebrew/share/opencv4/haarcascades/"
                           "haarcascade_frontalface_alt2.xml"),
    };
    for (const QString& p : systemPaths) {
        if (QFile::exists(p) && m_faceCascade.load(p.toStdString())) {
            m_cascadeLoaded = true;
            return true;
        }
    }

    return false;
#else
    return false;
#endif
}

bool FaceCaptureWidget::startCamera(int deviceIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    stopCapture();
    aicore_cancel_token_reset(m_inferenceCancelToken);
    m_pendingCameraIndex = deviceIndex;
    if (!configureDetectorForRegistrySelection()) return false;

    if (m_detectorKind == DetectorKind::Ggml) {
        if (!ensureGgmlModelReady()) {
            m_statusLabel->setText(tr("Downloading face detector model..."));
            return false;
        }
        scheduleGgmlModelLoad(facedetectCachePath(currentGgmlFilename()));
        m_cascadeLoaded = false;
    } else if (m_detectorKind == DetectorKind::OpenCV) {
        releaseGgmlModel();
        if (!loadCascade()) {
            m_statusLabel->setText(
                    tr("Warning: OpenCV cascade not found \u2014 "
                       "capture without detection"));
        }
    }

    if (!m_camerasEnumerated && m_cameraCombo) {
        m_camerasEnumerated = true;
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
            m_statusLabel->setText(tr("No camera devices detected"));
            return false;
        }
        if (deviceIndex == 0 && m_cameraCombo->count() > 0) {
            deviceIndex = m_cameraCombo->itemData(0).toInt();
            m_pendingCameraIndex = deviceIndex;
        }
        m_cameraCombo->blockSignals(false);
    }

    auto* reader = static_cast<VideoFrameReader*>(m_frameReader);
    if (!reader->openCamera(deviceIndex, cv::CAP_ANY)) {
        m_frameReaderReady = false;
        m_cameraActive = false;
        const QString error =
                tr("Failed to open camera device %1").arg(deviceIndex);
        m_statusLabel->setText(error);
        emit cameraError(error);
        return false;
    }
    m_frameReaderReady = true;

    if (detectorReady()) {
        m_statusLabel->setText(tr("Camera active — detecting faces"));
    } else {
        m_statusLabel->setText(
                tr("Camera active — no face detector (full-frame crop)"));
    }

    m_cameraActive = true;
    m_ggmlFrameSkip = 0;
    m_lastDetectedFrameNum = 0;
    m_frameReadTimer->setInterval(computeTimerInterval());
    m_frameReadTimer->start();
    m_frameTimer->setInterval(computeTimerInterval());
    m_frameTimer->start();
    emit cameraStarted();
    return true;
#else
    Q_UNUSED(deviceIndex);
    return false;
#endif
}

void FaceCaptureWidget::stopCamera() {
    if (m_frameTimer) m_frameTimer->stop();
    if (m_frameReadTimer) m_frameReadTimer->stop();

#ifdef HAS_OPENCV_FACE_CAPTURE
    // For video files: pause (don't release) so we can resume from same
    // position.  The background reader keeps its video open.
    if (m_inputSource == InputSource::VideoFile && m_frameReaderReady) {
        // Just pause — keep reader open for resume
        m_videoPaused = true;
    } else if (m_frameReaderReady) {
        // Release background reader for camera mode
        QMetaObject::invokeMethod(m_frameReader, "release",
                                  Qt::QueuedConnection);
        m_frameReaderReady = false;
        m_videoFilePath.clear();
        m_videoPaused = false;
        closePreviewCapture();
        {
            QMutexLocker lock(&m_frameMutex);
            m_latestFrame.release();
        }
    }
#endif

    if (m_cameraActive) {
        m_cameraActive = false;
        emit cameraStopped();
    }

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_lastFaceRect = cv::Rect();
    m_lastFaceScore = 0.f;

    // Ensure video controls remain functional after stop.
    // The slider and speed combo stay enabled as long as a video is loaded,
    // regardless of whether playback is active.
    const bool videoLoaded =
            m_frameReaderReady && m_inputSource == InputSource::VideoFile;
    if (m_videoSeekSlider) m_videoSeekSlider->setEnabled(videoLoaded);
    if (m_playbackSpeedCombo) m_playbackSpeedCombo->setEnabled(videoLoaded);
#endif
    m_consecutiveDetections = 0;
    if (m_seekPreviewLabel) m_seekPreviewLabel->setVisible(false);
}

bool FaceCaptureWidget::isCameraActive() const { return m_cameraActive; }

void FaceCaptureWidget::startGuidedCapture(
        const std::vector<CaptureAngle>& angles) {
    resetCapture();
    m_targetAngles = angles;

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Validate registry when identity tracking is requested.
    const bool needsRegistry =
            m_registryList && !m_registryList->selectedItems().isEmpty();
    if (needsRegistry && m_registryIdentities.empty()) {
        const QString registryPath =
                m_registryPathEdit ? m_registryPathEdit->text().trimmed()
                                   : QString();
        const bool pathMissing =
                registryPath.isEmpty() || !QFileInfo(registryPath).isFile();
        if (pathMissing) {
            QMessageBox::warning(
                    this, tr("Face Registry Required"),
                    tr("No face registry database is configured.\n\n"
                       "Please use the <b>qFaceDetect</b> plugin to register "
                       "faces first, then set the registry path here.\n\n"
                       "Identity-based tracking and reconstruction depend on "
                       "a pre-registered face database."));
        } else {
            QMessageBox::warning(
                    this, tr("Face Registry Empty or Invalid"),
                    tr("The face registry at:\n%1\n\n"
                       "contains no identities compatible with the current "
                       "detector model.\n\n"
                       "Please use the <b>qFaceDetect</b> plugin to register "
                       "faces with the same detector model, or select a "
                       "different registry database.")
                            .arg(registryPath));
        }
        emit logMessage(
                tr("[FaceCapture] Aborted: face registry dependency not "
                   "satisfied — use qFaceDetect to register faces first."));
        return;
    }

    if (m_registryList && !m_registryList->selectedItems().isEmpty()) {
        for (QListWidgetItem* item : m_registryList->selectedItems()) {
            const int index = item->data(Qt::UserRole).toInt();
            if (index < 0 ||
                static_cast<size_t>(index) >= m_registryIdentities.size()) {
                continue;
            }
            const RegistryIdentity& identity = m_registryIdentities[index];
            if (identity.modelFile != currentGgmlFilename()) continue;
            IdentityTrack track;
            track.identity = identity;
            m_identityTracks.push_back(std::move(track));
        }
        if (m_identityTracks.empty()) {
            m_statusLabel->setText(
                    tr("Selected registry identities do not match the active "
                       "FaceDetect model"));
            emit cameraError(m_statusLabel->text());
            return;
        }
        emit logMessage(
                tr("[FaceCapture] Tracking %1 selected registry identities")
                        .arg(m_identityTracks.size()));
    }
#endif
    m_capturingMode = true;
    m_currentAngleIndex = currentGuideAngleIndex();

    if (m_captureBtn) {
#ifdef HAS_OPENCV_FACE_CAPTURE
        // Registry tracks are captured independently after a stable identity
        // match. A generic manual capture has no identity owner and would be
        // silently excluded from the reconstruction batches.
        m_captureBtn->setEnabled(m_cameraActive && m_identityTracks.empty());
#else
        m_captureBtn->setEnabled(m_cameraActive);
#endif
    }

    const int target = minCapturesBeforeComplete();
    if (!m_targetAngles.empty()) {
        setAngleGuideText(
                tr("Angle: %1 (capture 1/%2)")
                        .arg(angleToString(m_targetAngles[static_cast<size_t>(
                                m_currentAngleIndex)]))
                        .arg(target));
    } else {
        setAngleGuideText(tr("Capture face snapshots (1/%1)").arg(target));
    }
    m_statusLabel->setText(tr("Position your face and capture each angle"));
    updateCaptureProgressUi();
}

bool FaceCaptureWidget::configureDetectorForRegistrySelection() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_registryList || m_registryList->selectedItems().isEmpty())
        return true;
    QString model;
    for (QListWidgetItem* item : m_registryList->selectedItems()) {
        const int index = item->data(Qt::UserRole).toInt();
        if (index < 0 ||
            static_cast<size_t>(index) >= m_registryIdentities.size()) {
            continue;
        }
        const QString entryModel = m_registryIdentities[index].modelFile;
        if (entryModel.isEmpty()) continue;
        if (model.isEmpty()) {
            model = entryModel;
        } else if (model != entryModel) {
            const QString message =
                    tr("Selected identities use different embedding models; "
                       "select identities registered with one model");
            m_statusLabel->setText(message);
            emit cameraError(message);
            return false;
        }
    }
    if (model.isEmpty()) return true;
    const int detectorIndex = m_detectorCombo->findData(model);
    if (detectorIndex < 0) {
        const QString message =
                tr("Registry model is unavailable: %1").arg(model);
        m_statusLabel->setText(message);
        emit cameraError(message);
        return false;
    }
    if (m_detectorCombo->currentIndex() != detectorIndex) {
        m_detectorCombo->setCurrentIndex(detectorIndex);
    }
    return true;
#else
    return false;
#endif
}

void FaceCaptureWidget::captureCurrentFrame() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_capturingMode) return;

    const int target = minCapturesBeforeComplete();
    if (static_cast<int>(m_capturedFrames.size()) >= target) return;

    if (detectorReady()) {
        const int trigger = (m_inputSource == InputSource::VideoFile)
                                    ? kVideoAutoCaptureTrigger
                                    : kAutoCaptureTrigger;
        if (m_consecutiveDetections < trigger) {
            m_statusLabel->setText(tr("Hold still \u2014 face not stable yet"));
            return;
        }
    } else if (m_inputSource == InputSource::VideoFile) {
        m_statusLabel->setText(tr("Face detector not ready"));
        return;
    }

    const cv::Mat& sourceFrame = detectorReady() && m_lastFaceRect.width > 0
                                         ? m_lastDetectedFrame
                                         : m_latestFrame;
    if (sourceFrame.empty()) {
        emit cameraError(tr("No captured frame is available"));
        return;
    }
    const cv::Mat frame = sourceFrame.clone();

    const int angleCount = std::max(1, static_cast<int>(m_targetAngles.size()));
    const int angleIdx = static_cast<int>(m_capturedFrames.size()) % angleCount;
    const auto angle = m_targetAngles.empty()
                               ? CaptureAngle::Front
                               : m_targetAngles[static_cast<size_t>(angleIdx)];

    CapturedFrame captured;
    captured.image = cvMatToQImage(frame);
    captured.angle = angle;

    if (detectorReady() && m_lastFaceRect.width > 0) {
        captured.croppedFace = cropAndResizeFace(frame, m_lastFaceRect, 512);
        captured.faceRect = QRect(m_lastFaceRect.x, m_lastFaceRect.y,
                                  m_lastFaceRect.width, m_lastFaceRect.height);
    } else {
        cv::Mat resized;
        int side = std::min(frame.cols, frame.rows);
        int x = (frame.cols - side) / 2;
        int y = (frame.rows - side) / 2;
        cv::Mat cropped = frame(cv::Rect(x, y, side, side)).clone();
        cv::resize(cropped, resized, cv::Size(512, 512));
        captured.croppedFace = cvMatToQImage(resized);
    }
    captured.valid = !captured.croppedFace.isNull();

    if (!captured.valid) {
        m_statusLabel->setText(tr("Failed to capture frame"));
        return;
    }

    m_capturedFrames.push_back(captured);

    const int index = static_cast<int>(m_capturedFrames.size());
    emit frameCaptured(index, target);
    updateCaptureProgressUi();
    refreshCapturedGallery();
    m_statusLabel->setText(tr("Captured %1/%2 faces").arg(index).arg(target));

    m_consecutiveDetections = 0;

    if (index >= target) {
        m_capturingMode = false;
        if (m_captureBtn) m_captureBtn->setEnabled(false);
        setAngleGuideText(tr("Capture complete"));
        emit captureComplete();
        return;
    }

    m_currentAngleIndex = index % angleCount;
    if (!m_targetAngles.empty()) {
        const auto nextAngle =
                m_targetAngles[static_cast<size_t>(m_currentAngleIndex)];
        setAngleGuideText(tr("Angle: %1 (capture %2/%3)")
                                  .arg(angleToString(nextAngle))
                                  .arg(index + 1)
                                  .arg(target));
    } else {
        setAngleGuideText(tr("Capture face snapshots (%1/%2)")
                                  .arg(index + 1)
                                  .arg(target));
    }
    if (m_captureBtn) m_captureBtn->setEnabled(false);
#endif
}

void FaceCaptureWidget::resetCapture() {
    // Reset capture progress but preserve identity tracking setup.
    // Identity tracks represent the user's tracking intent (which registered
    // persons to follow) — clearing them would force the user to re-select
    // identities after every reset.  Only per-track progress is cleared.
#ifdef HAS_OPENCV_FACE_CAPTURE
    for (auto& track : m_identityTracks) {
        track.frames.clear();
        track.cooldown = 0;
        track.consecutiveDetections = 0;
        track.lastRect = cv::Rect();
        track.lastDistance = 1.f;
    }
#endif

    m_targetAngles.clear();
    m_capturedFrames.clear();
    m_currentAngleIndex = 0;
    m_capturingMode = false;
    m_consecutiveDetections = 0;
    m_postCaptureCooldown = 0;
    m_noCascadeCounter = 0;

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_lastFaceRect = cv::Rect();
    m_latestFrame.release();
    m_lastDetectedFrame.release();
    m_lastFaceScore = 0.f;
#endif

    if (m_captureBtn) m_captureBtn->setEnabled(false);
    setAngleGuideText(QString());
    refreshCapturedGallery();
    updateCaptureProgressUi();

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Ensure video controls remain functional after reset.
    const bool videoLoaded =
            m_frameReaderReady && m_inputSource == InputSource::VideoFile;
    if (m_videoSeekSlider) m_videoSeekSlider->setEnabled(videoLoaded);
    if (m_playbackSpeedCombo) m_playbackSpeedCombo->setEnabled(videoLoaded);
#endif
    if (m_statusLabel) {
        m_statusLabel->setText(
                m_cameraActive ? tr("Camera active \u2014 detecting faces")
                               : tr("Ready"));
    }
}

std::vector<FaceCaptureWidget::CapturedFrame>
FaceCaptureWidget::capturedFrames() const {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_identityTracks.empty()) {
        std::vector<CapturedFrame> frames;
        for (const IdentityTrack& track : m_identityTracks) {
            frames.insert(frames.end(), track.frames.begin(),
                          track.frames.end());
        }
        return frames;
    }
#endif
    return m_capturedFrames;
}

QStringList FaceCaptureWidget::exportCapturedImages(
        const QString& outputDir) const {
    QStringList paths;
    QDir dir(outputDir);
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) return paths;

    for (size_t i = 0; i < m_capturedFrames.size(); ++i) {
        const CapturedFrame& f = m_capturedFrames[i];
        if (!f.valid || f.croppedFace.isNull()) continue;

        QString tag = angleToString(f.angle);
        tag.replace(QLatin1Char(' '), QLatin1String("_"));
        tag.replace(QStringLiteral("\u00B0"), QStringLiteral("deg"));

        const QString filename =
                QStringLiteral("face_%1_%2.png")
                        .arg(static_cast<int>(i), 2, 10, QChar('0'))
                        .arg(tag);
        const QString path = dir.filePath(filename);
        if (f.croppedFace.save(path, "PNG")) {
            paths << path;
        }
    }
    return paths;
}

std::vector<FaceCaptureWidget::IdentityImageBatch>
FaceCaptureWidget::exportCapturedIdentityImages(
        const QString& outputDir) const {
    std::vector<IdentityImageBatch> batches;
#ifdef HAS_OPENCV_FACE_CAPTURE
    for (const IdentityTrack& track : m_identityTracks) {
        IdentityImageBatch batch;
        batch.id = track.identity.id;
        batch.name = track.identity.name;
        QString segment = track.identity.name + QLatin1Char('_') +
                          track.identity.id.left(12);
        for (QChar& c : segment) {
            if (!c.isLetterOrNumber() && c != QLatin1Char('-') &&
                c != QLatin1Char('_')) {
                c = QLatin1Char('_');
            }
        }
        QDir dir(QDir(outputDir).filePath(segment));
        if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) continue;
        for (size_t i = 0; i < track.frames.size(); ++i) {
            const CapturedFrame& frame = track.frames[i];
            if (!frame.valid || frame.croppedFace.isNull()) continue;
            const QString path = dir.filePath(
                    QStringLiteral("face_%1.png")
                            .arg(static_cast<int>(i), 2, 10, QChar('0')));
            if (frame.croppedFace.save(path, "PNG"))
                batch.paths.push_back(path);
        }
        if (!batch.paths.isEmpty()) batches.push_back(std::move(batch));
    }
#endif
    if (batches.empty()) {
        IdentityImageBatch anonymous;
        anonymous.name = tr("Anonymous");
        anonymous.paths = exportCapturedImages(outputDir);
        if (!anonymous.paths.isEmpty()) batches.push_back(std::move(anonymous));
    }
    return batches;
}

int FaceCaptureWidget::capturedCount() const {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_identityTracks.empty()) {
        int total = 0;
        for (const IdentityTrack& track : m_identityTracks) {
            total += static_cast<int>(track.frames.size());
        }
        return total;
    }
#endif
    return static_cast<int>(m_capturedFrames.size());
}

int FaceCaptureWidget::targetCount() const {
    int identities = 1;
#ifdef HAS_OPENCV_FACE_CAPTURE
    identities = std::max(1, static_cast<int>(m_identityTracks.size()));
#endif
    return minCapturesBeforeComplete() * identities;
}

float FaceCaptureWidget::minDetectionScore() const {
    return m_minScoreSpin ? static_cast<float>(m_minScoreSpin->value()) : 0.5f;
}

int FaceCaptureWidget::minCapturesBeforeComplete() const {
    return m_minCapturesSpin ? m_minCapturesSpin->value() : 2;
}

float FaceCaptureWidget::maxSamePersonDistance() const {
    return m_maxDistanceSpin ? static_cast<float>(m_maxDistanceSpin->value())
                             : kDefaultSamePersonMaxDistance;
}

FaceCaptureWidget::FacePickStrategy FaceCaptureWidget::facePickStrategy()
        const {
    if (!m_faceStrategyCombo) return FacePickStrategy::TrackSamePerson;
    return static_cast<FacePickStrategy>(
            m_faceStrategyCombo->currentData().toInt());
}

void FaceCaptureWidget::processFrame() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_cameraActive || !m_frameReaderReady) return;

    cv::Mat frame;
    int curFrameNumForSlider = 0;
    {
        QMutexLocker lock(&m_frameMutex);
        if (m_latestFrame.empty()) return;
        m_latestFrame.copyTo(frame);
        curFrameNumForSlider = m_frameReaderSeekTo.load();
    }

    // Update video seek slider and time label
    if (m_inputSource == InputSource::VideoFile && !m_userSeeking) {
        if (m_videoSeekSlider && m_totalVideoFrames > 0) {
            m_videoSeekSlider->blockSignals(true);
            m_videoSeekSlider->setValue(curFrameNumForSlider);
            m_videoSeekSlider->blockSignals(false);
        }
        updateVideoTimeLabel(curFrameNumForSlider);
    }

    QImage preview = cvMatToQImage(frame);

    // Determine whether to run detection based on video-time elapsed since last
    // detection.  This keeps detection frequency consistent across all playback
    // speeds: at any speed, detection runs once per ~kGgmlDetectInterval video
    // frames of content.
    const int curFrameNum = curFrameNumForSlider;
    const bool timeForDetection = [&]() -> bool {
        if (m_inputSource == InputSource::VideoFile && m_videoFps > 0) {
            // Video-time throttle: detect every kGgmlDetectInterval frames of
            // video content, regardless of playback speed.
            const double videoTimeMs = curFrameNum / m_videoFps * 1000.0;
            const double thresholdMs =
                    kGgmlDetectInterval / m_videoFps * 1000.0;
            if (videoTimeMs - (m_lastDetectedFrameNum / m_videoFps * 1000.0) >=
                thresholdMs - 1.0) {  // -1ms tolerance for FP rounding
                return true;
            }
            return false;
        }
        // Camera mode: fall back to tick-based skip.
        if (m_ggmlFrameSkip <= 0) {
            m_ggmlFrameSkip = kGgmlDetectInterval;
            return true;
        }
        --m_ggmlFrameSkip;
        return false;
    }();

    if (!m_identityTracks.empty() && detectorReady() &&
        m_detectorKind == DetectorKind::Ggml && !m_ggmlModelLoading) {
        if (timeForDetection) {
            const std::vector<ScoredFace> faces = detectFacesGgml(frame);
            processRegistryIdentities(frame, faces, &preview);
            m_lastDetectedFrameNum = curFrameNum;
            m_lastDetectedFrame = frame.clone();
        } else {
            QPainter painter(&preview);
            painter.setPen(QPen(QColor(0, 200, 255), 3));
            for (const IdentityTrack& track : m_identityTracks) {
                if (track.lastRect.width > 0) {
                    painter.drawRect(track.lastRect.x, track.lastRect.y,
                                     track.lastRect.width,
                                     track.lastRect.height);
                }
            }
        }
        if (m_capturingMode && !m_targetAngles.empty()) {
            drawAngleGuide(preview, m_targetAngles[static_cast<size_t>(
                                            currentGuideAngleIndex())]);
        }
        if (!preview.isNull()) {
            m_previewLabel->setPixmap(QPixmap::fromImage(
                    preview.scaled(m_previewLabel->size(), Qt::KeepAspectRatio,
                                   Qt::FastTransformation)));
        }
        return;
    }

    cv::Rect faceRect;
    bool freshDetection = false;
    if (detectorReady() && !m_ggmlModelLoading) {
        if (m_detectorKind == DetectorKind::Ggml) {
            if (timeForDetection) {
                faceRect = detectFaceGgml(frame);
                m_lastDetectedFrameNum = curFrameNum;
                freshDetection = true;
            } else {
                if (m_lastFaceRect.width > 0) faceRect = m_lastFaceRect;
            }
        } else {
            faceRect = detectFaceOpenCv(frame);
            freshDetection = true;
        }

        if (faceRect.width > 0 && faceRect.height > 0) {
            ++m_consecutiveDetections;
            m_lastFaceRect = faceRect;
            if (freshDetection) m_lastDetectedFrame = frame.clone();
            emit faceDetected(QRect(faceRect.x, faceRect.y, faceRect.width,
                                    faceRect.height));
        } else {
            m_consecutiveDetections = 0;
            m_lastFaceRect = cv::Rect();
            m_lastDetectedFrame.release();
            m_lastFaceScore = 0.f;
            emit faceNotDetected();
        }
    }

    if (!preview.isNull()) {
        if (detectorReady() && faceRect.width > 0) {
            drawOverlay(preview, faceRect);
        }
        if (m_capturingMode && !m_targetAngles.empty()) {
            const int guideIdx = currentGuideAngleIndex();
            drawAngleGuide(preview,
                           m_targetAngles[static_cast<size_t>(guideIdx)]);
        }
        m_previewLabel->setPixmap(QPixmap::fromImage(
                preview.scaled(m_previewLabel->size(), Qt::KeepAspectRatio,
                               Qt::FastTransformation)));
    }

    if (m_capturingMode) {
        const int trigger = (m_inputSource == InputSource::VideoFile)
                                    ? kVideoAutoCaptureTrigger
                                    : kAutoCaptureTrigger;
        const int target = minCapturesBeforeComplete();
        const int captured = static_cast<int>(m_capturedFrames.size());
        if (m_postCaptureCooldown > 0) {
            --m_postCaptureCooldown;
            if (m_inputSource == InputSource::Camera) {
                m_statusLabel->setText(
                        tr("Repositioning... (%1)")
                                .arg(m_postCaptureCooldown / 30 + 1));
            } else {
                m_statusLabel->setText(
                        tr("Waiting for next frame... (%1)")
                                .arg(m_postCaptureCooldown / 30 + 1));
            }
            if (m_captureBtn) m_captureBtn->setEnabled(false);
        } else if (detectorReady()) {
            const bool stable = m_consecutiveDetections >= trigger;
            if (stable) {
                captureCurrentFrame();
                m_postCaptureCooldown = kPostCaptureCooldown;
                return;
            }
            if (m_captureBtn && m_inputSource == InputSource::Camera) {
                m_captureBtn->setEnabled(m_consecutiveDetections >= 3);
            }
            if (faceRect.width > 0) {
                int pct =
                        std::min(100, m_consecutiveDetections * 100 / trigger);
                m_statusLabel->setText(
                        tr("Stabilizing... %1% (%2/%3 faces captured)")
                                .arg(pct)
                                .arg(captured)
                                .arg(target));
            } else {
                m_statusLabel->setText(
                        tr("No face detected — %1/%2 faces captured")
                                .arg(captured)
                                .arg(target));
            }
        } else if (m_inputSource == InputSource::Camera) {
            ++m_noCascadeCounter;
            if (m_noCascadeCounter >= kNoCascadeAutoInterval) {
                m_noCascadeCounter = 0;
                captureCurrentFrame();
                m_postCaptureCooldown = kPostCaptureCooldown;
                return;
            }
            if (m_captureBtn) m_captureBtn->setEnabled(true);
            m_statusLabel->setText(
                    tr("Auto-capture in %1s (or click Capture)")
                            .arg((kNoCascadeAutoInterval - m_noCascadeCounter) /
                                         30 +
                                 1));
        } else {
            m_statusLabel->setText(tr("Loading face detector..."));
        }
    }
#endif
}

#ifdef HAS_OPENCV_FACE_CAPTURE

QImage FaceCaptureWidget::cvMatToQImage(const cv::Mat& mat) {
    if (mat.empty()) return QImage();

    cv::Mat rgb;
    if (mat.channels() == 1)
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    else if (mat.channels() == 3)
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else if (mat.channels() == 4)
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGBA);
    else
        return QImage();

    // Use QImage(rgb.data, ...).copy() to share the pixel data briefly and
    // then perform a single deep copy, which is ~3x faster than the per-row
    // memcpy loop.
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  rgb.channels() == 4 ? QImage::Format_RGBA8888
                                      : QImage::Format_RGB888)
            .copy();
}

cv::Rect FaceCaptureWidget::detectFaceOpenCv(const cv::Mat& frame) {
    return pickFace(frame, detectFacesOpenCv(frame));
}

std::vector<FaceCaptureWidget::ScoredFace> FaceCaptureWidget::detectFacesOpenCv(
        const cv::Mat& frame) {
    std::vector<ScoredFace> out;
    if (m_faceCascade.empty() || frame.empty()) return out;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    m_faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(80, 80));
    out.reserve(faces.size());
    for (const cv::Rect& rect : faces) {
        out.push_back({rect, 1.0f});
    }
    return out;
}

std::vector<FaceCaptureWidget::ScoredFace> FaceCaptureWidget::detectFacesGgml(
        const cv::Mat& frame) {
    std::vector<ScoredFace> out;
    if (!m_ggmlCtx || frame.empty()) return out;

    cv::Mat rgb;
    if (frame.channels() == 1)
        cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
    else if (frame.channels() == 3)
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    else if (frame.channels() == 4)
        cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGB);
    else
        return out;

    if (!rgb.isContinuous()) rgb = rgb.clone();

    AICoreInferenceGuard guard(m_inferenceCancelToken, m_inferenceDevice);
    if (!guard.locked()) return out;
    char* json = aicore_facedetect_detect_rgb_json(m_ggmlCtx, rgb.data,
                                                   rgb.cols, rgb.rows);
    if (!json) return out;

    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray(json));
    aicore_facedetect_free_string(json);
    if (!doc.isObject()) return out;

    const QJsonArray faces =
            doc.object().value(QStringLiteral("faces")).toArray();
    out.reserve(static_cast<size_t>(faces.size()));
    for (const QJsonValue& v : faces) {
        const QJsonObject obj = v.toObject();
        const QJsonArray box = obj.value(QStringLiteral("box")).toArray();
        if (box.size() != 4) continue;
        const float score = static_cast<float>(
                obj.value(QStringLiteral("score")).toDouble());
        const int x = static_cast<int>(std::floor(box.at(0).toDouble()));
        const int y = static_cast<int>(std::floor(box.at(1).toDouble()));
        const int w = static_cast<int>(
                std::ceil(box.at(2).toDouble() - box.at(0).toDouble()));
        const int h = static_cast<int>(
                std::ceil(box.at(3).toDouble() - box.at(1).toDouble()));
        if (w <= 0 || h <= 0) continue;
        ScoredFace face;
        face.rect = cv::Rect(x, y, w, h);
        face.score = score;
        const QJsonArray landmarks =
                obj.value(QStringLiteral("landmarks")).toArray();
        if (landmarks.size() >= 5) {
            face.hasLandmarks = true;
            for (int k = 0; k < 5; ++k) {
                const QJsonArray point = landmarks.at(k).toArray();
                if (point.size() < 2) {
                    face.hasLandmarks = false;
                    break;
                }
                face.landmarks[k * 2] =
                        static_cast<float>(point.at(0).toDouble());
                face.landmarks[k * 2 + 1] =
                        static_cast<float>(point.at(1).toDouble());
            }
        }
        out.push_back(face);
    }
    return out;
}

std::vector<FaceCaptureWidget::ScoredFace> FaceCaptureWidget::detectFaces(
        const cv::Mat& frame) {
    if (m_detectorKind == DetectorKind::Ggml) return detectFacesGgml(frame);
    return detectFacesOpenCv(frame);
}

bool FaceCaptureWidget::embedFaceCrop(const cv::Mat& frame,
                                      const cv::Rect& rect,
                                      std::vector<float>* embedding) {
    if (!embedding || !m_ggmlCtx || frame.empty() || rect.width <= 0 ||
        rect.height <= 0) {
        return false;
    }
    const int x = std::max(0, rect.x);
    const int y = std::max(0, rect.y);
    const int w = std::min(frame.cols - x, rect.width);
    const int h = std::min(frame.rows - y, rect.height);
    if (w <= 8 || h <= 8) return false;

    cv::Mat crop = frame(cv::Rect(x, y, w, h)).clone();
    cv::Mat rgb;
    if (crop.channels() == 1)
        cv::cvtColor(crop, rgb, cv::COLOR_GRAY2RGB);
    else if (crop.channels() == 3)
        cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
    else if (crop.channels() == 4)
        cv::cvtColor(crop, rgb, cv::COLOR_BGRA2RGB);
    else
        return false;
    if (!rgb.isContinuous()) rgb = rgb.clone();

    float* vec = nullptr;
    int dim = 0;
    AICoreInferenceGuard guard(m_inferenceCancelToken, m_inferenceDevice);
    if (!guard.locked()) return false;
    if (aicore_facedetect_embed_rgb(m_ggmlCtx, rgb.data, rgb.cols, rgb.rows,
                                    0.f, &vec, &dim) != 0 ||
        vec == nullptr || dim <= 0) {
        return false;
    }
    embedding->assign(vec, vec + dim);
    aicore_facedetect_free_vec(vec);
    return true;
}

bool FaceCaptureWidget::embedScoredFace(const cv::Mat& frame,
                                        const ScoredFace& face,
                                        std::vector<float>* embedding) {
    if (!embedding || !m_ggmlCtx || frame.empty()) return false;
    if (!face.hasLandmarks) {
        return embedFaceCrop(frame, face.rect, embedding);
    }

    cv::Mat rgb;
    if (frame.channels() == 1)
        cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
    else if (frame.channels() == 3)
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    else if (frame.channels() == 4)
        cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGB);
    else
        return false;
    if (!rgb.isContinuous()) rgb = rgb.clone();

    float* vec = nullptr;
    int dim = 0;
    AICoreInferenceGuard guard(m_inferenceCancelToken, m_inferenceDevice);
    if (!guard.locked()) return false;
    if (aicore_facedetect_embed_rgb_landmarks(m_ggmlCtx, rgb.data, rgb.cols,
                                              rgb.rows, face.landmarks, &vec,
                                              &dim) != 0 ||
        vec == nullptr || dim <= 0) {
        return false;
    }
    embedding->assign(vec, vec + dim);
    aicore_facedetect_free_vec(vec);
    *embedding = normalizeEmbedding(*embedding);
    return !embedding->empty();
}

bool FaceCaptureWidget::captureIdentityFrame(IdentityTrack* track,
                                             const cv::Mat& frame,
                                             const cv::Rect& rect) {
    if (!track || frame.empty() || rect.width <= 0) return false;
    const int target = minCapturesBeforeComplete();
    if (static_cast<int>(track->frames.size()) >= target) return false;

    const int angleCount = std::max(1, static_cast<int>(m_targetAngles.size()));
    const int angleIndex = static_cast<int>(track->frames.size()) % angleCount;
    CapturedFrame captured;
    captured.image = cvMatToQImage(frame);
    captured.croppedFace = cropAndResizeFace(frame, rect, 512);
    captured.angle = m_targetAngles.empty()
                             ? CaptureAngle::Front
                             : m_targetAngles[static_cast<size_t>(angleIndex)];
    captured.faceRect = QRect(rect.x, rect.y, rect.width, rect.height);
    captured.valid = !captured.croppedFace.isNull();
    if (!captured.valid) return false;

    track->frames.push_back(std::move(captured));
    track->consecutiveDetections = 0;
    track->cooldown = kPostCaptureCooldown;
    emit logMessage(
            tr("[FaceCapture] Captured %1 [%2] %3/%4")
                    .arg(track->identity.name, track->identity.id.left(12))
                    .arg(track->frames.size())
                    .arg(target));
    emit frameCaptured(capturedCount(), targetCount());
    refreshCapturedGallery();
    updateCaptureProgressUi();
    return true;
}

bool FaceCaptureWidget::processRegistryIdentities(
        const cv::Mat& frame,
        const std::vector<ScoredFace>& faces,
        QImage* preview) {
    if (m_identityTracks.empty() || !m_ggmlCtx || frame.empty()) return false;

    std::vector<const ScoredFace*> candidates;
    std::vector<float> queryMatrix;
    const int dim = static_cast<int>(
            m_identityTracks.front().identity.embedding.size());
    for (const ScoredFace& face : faces) {
        if (face.score < minDetectionScore()) continue;
        std::vector<float> embedding;
        if (!embedScoredFace(frame, face, &embedding) ||
            static_cast<int>(embedding.size()) != dim) {
            continue;
        }
        candidates.push_back(&face);
        queryMatrix.insert(queryMatrix.end(), embedding.begin(),
                           embedding.end());
    }

    std::vector<float> gallery;
    gallery.reserve(m_identityTracks.size() * static_cast<size_t>(dim));
    for (const IdentityTrack& track : m_identityTracks) {
        if (static_cast<int>(track.identity.embedding.size()) != dim) {
            return false;
        }
        gallery.insert(gallery.end(), track.identity.embedding.begin(),
                       track.identity.embedding.end());
    }

    struct Pair {
        int face = -1;
        int identity = -1;
        float distance = 1.f;
    };
    std::vector<Pair> pairs;
    if (!candidates.empty()) {
        std::vector<float> distances(candidates.size() *
                                     m_identityTracks.size());
        if (aicore_facedetect_cosine_distance_matrix(
                    queryMatrix.data(), static_cast<int>(candidates.size()),
                    gallery.data(), static_cast<int>(m_identityTracks.size()),
                    dim, distances.data()) == 0) {
            for (size_t f = 0; f < candidates.size(); ++f) {
                for (size_t i = 0; i < m_identityTracks.size(); ++i) {
                    const float distance =
                            distances[f * m_identityTracks.size() + i];
                    if (distance <= maxSamePersonDistance()) {
                        pairs.push_back({static_cast<int>(f),
                                         static_cast<int>(i), distance});
                    }
                }
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
        return a.distance < b.distance;
    });

    std::vector<int> assignedFace(m_identityTracks.size(), -1);
    std::vector<bool> faceUsed(candidates.size(), false);
    for (const Pair& pair : pairs) {
        if (assignedFace[static_cast<size_t>(pair.identity)] >= 0 ||
            faceUsed[static_cast<size_t>(pair.face)]) {
            continue;
        }
        assignedFace[static_cast<size_t>(pair.identity)] = pair.face;
        faceUsed[static_cast<size_t>(pair.face)] = true;
        m_identityTracks[static_cast<size_t>(pair.identity)].lastDistance =
                pair.distance;
    }

    const int trigger = m_inputSource == InputSource::VideoFile
                                ? kVideoAutoCaptureTrigger
                                : kAutoCaptureTrigger;
    bool anyMatched = false;
    for (size_t i = 0; i < m_identityTracks.size(); ++i) {
        IdentityTrack& track = m_identityTracks[i];
        if (track.cooldown > 0) --track.cooldown;
        const int faceIndex = assignedFace[i];
        if (faceIndex < 0) {
            track.consecutiveDetections = 0;
            track.lastRect = cv::Rect();
            continue;
        }
        const ScoredFace& face = *candidates[static_cast<size_t>(faceIndex)];
        anyMatched = true;
        track.lastRect = face.rect;
        ++track.consecutiveDetections;

        if (preview && !preview->isNull()) {
            QPainter painter(preview);
            painter.setPen(QPen(QColor(0, 200, 255), 3));
            painter.drawRect(face.rect.x, face.rect.y, face.rect.width,
                             face.rect.height);
            const QString label = QStringLiteral("%1 d=%2")
                                          .arg(track.identity.name)
                                          .arg(track.lastDistance, 0, 'f', 3);
            painter.fillRect(
                    face.rect.x, std::max(0, face.rect.y - 22),
                    std::max(90, QTCOMPAT_FONTMETRICS_WIDTH(
                                         painter.fontMetrics(), label) +
                                         8),
                    22, QColor(0, 0, 0, 180));
            painter.setPen(Qt::white);
            painter.drawText(face.rect.x + 4, std::max(16, face.rect.y - 6),
                             label);
        }

        if (m_capturingMode && track.cooldown == 0 &&
            track.consecutiveDetections >= trigger) {
            captureIdentityFrame(&track, frame, face.rect);
        }
    }

    if (m_capturingMode) {
        const int target = minCapturesBeforeComplete();
        const bool complete = std::all_of(
                m_identityTracks.begin(), m_identityTracks.end(),
                [target](const IdentityTrack& track) {
                    return static_cast<int>(track.frames.size()) >= target;
                });
        if (complete) {
            m_capturingMode = false;
            if (m_captureBtn) m_captureBtn->setEnabled(false);
            setAngleGuideText(tr("Capture complete for all identities"));
            emit captureComplete();
        } else {
            m_statusLabel->setText(
                    anyMatched
                            ? tr("Tracking selected identities: %1/%2 captures")
                                      .arg(capturedCount())
                                      .arg(targetCount())
                            : tr("Selected identities not found in this "
                                 "frame"));
        }
    }
    return anyMatched;
}

float FaceCaptureWidget::embeddingDistance(const std::vector<float>& a,
                                           const std::vector<float>& b) const {
    if (a.size() != b.size() || a.empty()) return 1.0f;
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        normA += static_cast<double>(a[i]) * a[i];
        normB += static_cast<double>(b[i]) * b[i];
    }
    if (normA <= 0.0 || normB <= 0.0) return 1.0f;
    const double cosine = dot / std::sqrt(normA * normB);
    return static_cast<float>(1.0 - std::clamp(cosine, -1.0, 1.0));
}

void FaceCaptureWidget::resetIdentityTrack() {
    m_referenceEmbedding.clear();
    m_hasReferenceEmbedding = false;
    m_identityTracks.clear();
}

cv::Rect FaceCaptureWidget::pickFace(const cv::Mat& frame,
                                     const std::vector<ScoredFace>& faces) {
    m_lastFaceScore = 0.f;
    if (faces.empty()) return cv::Rect();

    const float minScore = minDetectionScore();
    std::vector<ScoredFace> candidates;
    candidates.reserve(faces.size());
    for (const ScoredFace& face : faces) {
        if (face.score >= minScore) candidates.push_back(face);
    }
    if (candidates.empty()) return cv::Rect();

    const FacePickStrategy strategy = facePickStrategy();
    if (strategy == FacePickStrategy::LargestFace) {
        const auto it =
                std::max_element(candidates.begin(), candidates.end(),
                                 [](const ScoredFace& a, const ScoredFace& b) {
                                     return a.rect.area() < b.rect.area();
                                 });
        m_lastFaceScore = it->score;
        return it->rect;
    }
    if (strategy == FacePickStrategy::HighestScore) {
        const auto it =
                std::max_element(candidates.begin(), candidates.end(),
                                 [](const ScoredFace& a, const ScoredFace& b) {
                                     return a.score < b.score;
                                 });
        m_lastFaceScore = it->score;
        return it->rect;
    }

    if (m_detectorKind != DetectorKind::Ggml || !m_ggmlCtx) {
        const auto it =
                std::max_element(candidates.begin(), candidates.end(),
                                 [](const ScoredFace& a, const ScoredFace& b) {
                                     return a.rect.area() < b.rect.area();
                                 });
        m_lastFaceScore = it->score;
        return it->rect;
    }

    if (!m_hasReferenceEmbedding) {
        const auto seedIt =
                std::max_element(candidates.begin(), candidates.end(),
                                 [](const ScoredFace& a, const ScoredFace& b) {
                                     return a.rect.area() < b.rect.area();
                                 });
        if (seedIt == candidates.end()) return cv::Rect();
        if (embedScoredFace(frame, *seedIt, &m_referenceEmbedding)) {
            m_hasReferenceEmbedding = true;
        }
        m_lastFaceScore = seedIt->score;
        return seedIt->rect;
    }

    const ScoredFace* best = nullptr;
    float bestDistance = maxSamePersonDistance();
    for (const ScoredFace& face : candidates) {
        std::vector<float> emb;
        if (!embedScoredFace(frame, face, &emb)) continue;
        const float dist = embeddingDistance(m_referenceEmbedding, emb);
        if (dist < bestDistance) {
            bestDistance = dist;
            best = &face;
        }
    }
    if (best) {
        m_lastFaceScore = best->score;
        return best->rect;
    }
    return cv::Rect();
}

cv::Rect FaceCaptureWidget::detectFaceGgml(const cv::Mat& frame) {
    return pickFace(frame, detectFacesGgml(frame));
}

cv::Rect FaceCaptureWidget::detectFace(const cv::Mat& frame) {
    if (m_detectorKind == DetectorKind::Ggml) return detectFaceGgml(frame);
    return detectFaceOpenCv(frame);
}

QImage FaceCaptureWidget::cropAndResizeFace(const cv::Mat& frame,
                                            const cv::Rect& faceRect,
                                            int targetSize) {
    if (frame.empty() || faceRect.width <= 0 || targetSize <= 0)
        return QImage();

    const int expandW = static_cast<int>(faceRect.width * 0.5);
    const int expandH = static_cast<int>(faceRect.height * 0.5);
    int side = std::max(faceRect.width + 2 * expandW,
                        faceRect.height + 2 * expandH);

    int cx = faceRect.x + faceRect.width / 2;
    int cy = faceRect.y + faceRect.height / 2;
    int x = std::max(0, cx - side / 2);
    int y = std::max(0, cy - side / 2);
    if (x + side > frame.cols) side = frame.cols - x;
    if (y + side > frame.rows) side = frame.rows - y;
    if (side <= 0) return QImage();

    cv::Mat cropped = frame(cv::Rect(x, y, side, side)).clone();
    cv::Mat resized;
    int interp =
            (cropped.cols > targetSize) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(cropped, resized, cv::Size(targetSize, targetSize), 0, 0,
               interp);
    return cvMatToQImage(resized);
}

void FaceCaptureWidget::drawOverlay(QImage& image, const cv::Rect& faceRect) {
    if (image.isNull() || faceRect.width <= 0) return;

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);
    QPen pen(m_detectorKind == DetectorKind::Ggml ? QColor(0, 200, 255)
                                                  : QColor(0, 220, 80));
    pen.setWidth(3);
    painter.setPen(pen);
    painter.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);

    if (m_lastFaceScore > 0.f) {
        QFont font = painter.font();
        font.setPointSize(11);
        font.setBold(true);
        painter.setFont(font);

        const QString scoreText =
                tr("Score: %1").arg(QString::number(m_lastFaceScore, 'f', 2));
        const QFontMetrics fm(font);
        const int pad = 4;
        const int textW = QTCOMPAT_FONTMETRICS_WIDTH(fm, scoreText) + 2 * pad;
        const int textH = fm.height() + pad;
        int textX = faceRect.x;
        int textY = faceRect.y - textH - 2;
        if (textY < 0) textY = faceRect.y + faceRect.height + 2;
        if (textX + textW > image.width()) {
            textX = std::max(0, image.width() - textW);
        }

        painter.fillRect(textX, textY, textW, textH, QColor(0, 0, 0, 180));
        painter.setPen(Qt::white);
        painter.drawText(textX + pad, textY + fm.ascent() + pad / 2, scoreText);
    }
}

void FaceCaptureWidget::drawAngleGuide(QImage& image, CaptureAngle angle) {
    if (image.isNull()) return;

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);

    QFont font = painter.font();
    font.setPointSize(16);
    font.setBold(true);
    painter.setFont(font);

    const int margin = 12;
    const QFontMetrics fm(font);
    const QRect textRect(margin, margin, image.width() - 2 * margin,
                         fm.height() + 16);
    painter.fillRect(textRect, QColor(0, 0, 0, 160));
    painter.setPen(Qt::white);
    painter.drawText(textRect, Qt::AlignCenter, angleToString(angle));
}

#endif  // HAS_OPENCV_FACE_CAPTURE

int FaceCaptureWidget::currentGuideAngleIndex() const {
    const int angleCount = std::max(1, static_cast<int>(m_targetAngles.size()));
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_identityTracks.empty()) {
        size_t fewest = m_identityTracks.front().frames.size();
        for (const IdentityTrack& track : m_identityTracks) {
            fewest = std::min(fewest, track.frames.size());
        }
        return static_cast<int>(fewest) % angleCount;
    }
#endif
    return static_cast<int>(m_capturedFrames.size()) % angleCount;
}

void FaceCaptureWidget::updateCaptureProgressUi() {
    const int captured = capturedCount();
    const int target = targetCount();
    if (m_captureProgress) {
        m_captureProgress->setMaximum(target);
        m_captureProgress->setValue(std::min(captured, target));
    }
}

void FaceCaptureWidget::setAngleGuideText(const QString& text) {
    if (!m_angleLabel) return;
    m_angleLabel->setText(text);
    m_angleLabel->setVisible(!text.isEmpty());
}

void FaceCaptureWidget::refreshCapturedGallery() {
    if (!m_capturedGalleryRow) return;

    QLayout* layout = m_capturedGalleryRow->layout();
    if (!layout) return;

    while (QLayoutItem* item = layout->takeAt(0)) {
        if (QWidget* widget = item->widget()) widget->deleteLater();
        delete item;
    }

    const std::vector<CapturedFrame> frames = capturedFrames();
    if (m_capturedGalleryScroll) {
        m_capturedGalleryScroll->setVisible(!frames.empty());
    }
    for (size_t i = 0; i < frames.size(); ++i) {
        const CapturedFrame& frame = frames[i];
        if (!frame.valid || frame.croppedFace.isNull()) continue;

        auto* thumb = new QLabel(m_capturedGalleryRow);
        thumb->setFixedSize(48, 48);
        thumb->setAlignment(Qt::AlignCenter);
        thumb->setStyleSheet(QStringLiteral(
                "QLabel { background-color: #222; "
                "border: 1px solid #555; border-radius: 3px; }"));
        thumb->setPixmap(QPixmap::fromImage(frame.croppedFace)
                                 .scaled(48, 48, Qt::KeepAspectRatio,
                                         Qt::SmoothTransformation));
        thumb->setToolTip(tr("Capture %1 — %2")
                                  .arg(static_cast<int>(i) + 1)
                                  .arg(angleToString(frame.angle)));
        layout->addWidget(thumb);
    }
    if (auto* box = qobject_cast<QHBoxLayout*>(layout)) {
        box->addStretch();
    }
}

void FaceCaptureWidget::loadFaceCaptureSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qFreeSplatter"));

#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_minScoreSpin) {
        m_minScoreSpin->blockSignals(true);
        m_minScoreSpin->setValue(
                settings.value(QStringLiteral("faceMinScore"), 0.5).toDouble());
        m_minScoreSpin->blockSignals(false);
    }
    if (m_minCapturesSpin) {
        m_minCapturesSpin->blockSignals(true);
        m_minCapturesSpin->setValue(
                settings.value(QStringLiteral("faceMinCaptures"), 2).toInt());
        m_minCapturesSpin->blockSignals(false);
    }
    if (m_maxDistanceSpin) {
        m_maxDistanceSpin->blockSignals(true);
        m_maxDistanceSpin->setValue(
                settings.value(QStringLiteral("faceMaxDistance"),
                               kDefaultSamePersonMaxDistance)
                        .toDouble());
        m_maxDistanceSpin->blockSignals(false);
    }
    if (m_faceStrategyCombo) {
        m_faceStrategyCombo->blockSignals(true);
        const int strategy =
                settings.value(QStringLiteral("faceStrategy"),
                               static_cast<int>(
                                       FacePickStrategy::TrackSamePerson))
                        .toInt();
        for (int i = 0; i < m_faceStrategyCombo->count(); ++i) {
            if (m_faceStrategyCombo->itemData(i).toInt() == strategy) {
                m_faceStrategyCombo->setCurrentIndex(i);
                break;
            }
        }
        m_faceStrategyCombo->blockSignals(false);
    }
    const QString videoPath =
            settings.value(QStringLiteral("faceVideoPath")).toString();
    if (m_videoPathEdit && !videoPath.isEmpty()) {
        m_videoPathEdit->blockSignals(true);
        m_videoPathEdit->setText(videoPath);
        m_videoPathEdit->blockSignals(false);
        m_videoFilePath = videoPath;
    }
    const QString defaultPath = defaultRegistryDbPath(currentGgmlFilename());
    const QString savedRegistryPath =
            settings.value(QStringLiteral("faceRegistryPath")).toString();
    m_registryPathUserChosen = !savedRegistryPath.isEmpty();
    const QString registryPath =
            m_registryPathUserChosen ? savedRegistryPath : defaultPath;
    if (m_registryPathEdit) m_registryPathEdit->setText(registryPath);
#endif

    settings.endGroup();
    reloadRegistry();
    updateCaptureProgressUi();
}

void FaceCaptureWidget::saveFaceCaptureSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qFreeSplatter"));

#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_minScoreSpin) {
        settings.setValue(QStringLiteral("faceMinScore"),
                          m_minScoreSpin->value());
    }
    if (m_minCapturesSpin) {
        settings.setValue(QStringLiteral("faceMinCaptures"),
                          m_minCapturesSpin->value());
    }
    if (m_maxDistanceSpin) {
        settings.setValue(QStringLiteral("faceMaxDistance"),
                          m_maxDistanceSpin->value());
    }
    if (m_faceStrategyCombo) {
        settings.setValue(QStringLiteral("faceStrategy"),
                          m_faceStrategyCombo->currentData());
    }
    settings.setValue(QStringLiteral("faceVideoPath"), videoFilePath());
    if (m_registryPathEdit) {
        const QString path = m_registryPathEdit->text().trimmed();
        // Only persist the path when the user explicitly chose it;
        // otherwise clear so the default follows the active detector model.
        if (m_registryPathUserChosen && !path.isEmpty()) {
            settings.setValue(QStringLiteral("faceRegistryPath"), path);
        } else {
            settings.remove(QStringLiteral("faceRegistryPath"));
        }
    }
#endif

    settings.endGroup();
}

QString FaceCaptureWidget::angleToString(CaptureAngle angle) const {
    switch (angle) {
        case CaptureAngle::Front:
            return tr("Look straight ahead");
        case CaptureAngle::Left45:
            return tr("Turn head 45\u00B0 left");
        case CaptureAngle::Right45:
            return tr("Turn head 45\u00B0 right");
        case CaptureAngle::Left90:
            return tr("Turn head 90\u00B0 left");
        case CaptureAngle::Right90:
            return tr("Turn head 90\u00B0 right");
        case CaptureAngle::Up15:
            return tr("Tilt head up ~15\u00B0");
        case CaptureAngle::Down15:
            return tr("Tilt head down ~15\u00B0");
    }
    return tr("Unknown angle");
}

#include "FaceCaptureWidget.moc"
