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

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFrame>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QScrollArea>
#include <QStandardPaths>
#include <QSettings>
#include <QTemporaryFile>
#include <algorithm>
#include <cstring>

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

}  // namespace

FaceCaptureWidget::FaceCaptureWidget(QWidget* parent) : QWidget(parent) {
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
                                    .arg(ecvModelDownloader::formatDownloadProgress(
                                            received, total)));
                }
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& dest) {
                m_downloadInProgress = false;
                if (m_downloadProgress) m_downloadProgress->setVisible(false);
                if (m_downloadLabel) m_downloadLabel->setVisible(false);
                if (ok) {
                    emit logMessage(tr("[FaceCapture] Downloaded model: %1")
                                            .arg(dest));
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
    connect(m_ggmlLoadWatcher, &QFutureWatcher<aicore_facedetect_ctx*>::finished,
            this, [this]() {
                m_ggmlModelLoading = false;
                aicore_facedetect_ctx* ctx = m_ggmlLoadWatcher->result();
                if (ctx == nullptr) {
                    m_statusLabel->setText(tr("Failed to load face detector model"));
                    emit cameraError(tr("Failed to load face detector GGUF"));
                    return;
                }
                releaseGgmlModel();
                m_ggmlCtx = ctx;
                m_loadedGgmlPath = facedetectCachePath(currentGgmlFilename());
                emit logMessage(tr("[FaceCapture] Loaded face detector: %1")
                                        .arg(QFileInfo(m_loadedGgmlPath).fileName()));
                if (m_cameraActive) {
                    m_statusLabel->setText(
                            m_inputSource == InputSource::VideoFile
                                    ? tr("Playing video — preview + face overlay")
                                    : tr("Camera active — detecting faces"));
                }
            });
}

FaceCaptureWidget::~FaceCaptureWidget() {
    stopCamera();
    if (m_ggmlLoadWatcher && m_ggmlLoadWatcher->isRunning()) {
        m_ggmlLoadWatcher->waitForFinished();
    }
    releaseGgmlModel();
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
    m_previewLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_previewLabel->setStyleSheet(
            QStringLiteral("QLabel { background-color: #1a1a1a; "
                           "border: 1px solid #444; border-radius: 4px; }"));
    m_previewLabel->setText(tr("Camera preview"));
    mainLayout->addWidget(m_previewLabel, 1);

    m_angleLabel = new QLabel(this);
    m_angleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(m_angleLabel);

    m_statusLabel = new QLabel(this);
    m_statusLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(m_statusLabel);

    m_captureProgressLabel = new QLabel(this);
    m_captureProgressLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(m_captureProgressLabel);

    m_captureProgress = new QProgressBar(this);
    m_captureProgress->setTextVisible(true);
    m_captureProgress->setFormat(tr("%v / %m faces"));
    m_captureProgress->setValue(0);
    mainLayout->addWidget(m_captureProgress);

    auto* galleryScroll = new QScrollArea(this);
    galleryScroll->setWidgetResizable(true);
    galleryScroll->setFixedHeight(56);
    galleryScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    galleryScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    galleryScroll->setFrameShape(QFrame::NoFrame);
    m_capturedGalleryRow = new QWidget(galleryScroll);
    auto* galleryLayout = new QHBoxLayout(m_capturedGalleryRow);
    galleryLayout->setContentsMargins(0, 0, 0, 0);
    galleryLayout->setSpacing(4);
    galleryScroll->setWidget(m_capturedGalleryRow);
    mainLayout->addWidget(galleryScroll);

#ifdef HAS_OPENCV_FACE_CAPTURE
    auto* detectorInputRow = new QHBoxLayout();
    detectorInputRow->setSpacing(6);
    detectorInputRow->addWidget(new QLabel(tr("Face detector:"), this));
    m_detectorCombo = new QComboBox(this);
    detectorInputRow->addWidget(m_detectorCombo, 2);
    detectorInputRow->addWidget(new QLabel(tr("Input:"), this));
    m_sourceCombo = new QComboBox(this);
    m_sourceCombo->addItem(tr("Live camera"), static_cast<int>(InputSource::Camera));
    m_sourceCombo->addItem(tr("Video file"), static_cast<int>(InputSource::VideoFile));
    detectorInputRow->addWidget(m_sourceCombo, 1);
    mainLayout->addLayout(detectorInputRow);
    connect(m_detectorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FaceCaptureWidget::onDetectorComboChanged);
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

    settingsRow->addWidget(new QLabel(tr("Face pick:"), this));
    m_faceStrategyCombo = new QComboBox(this);
    m_faceStrategyCombo->addItem(
            tr("Track same person"),
            static_cast<int>(FacePickStrategy::TrackSamePerson));
    m_faceStrategyCombo->addItem(tr("Largest face"),
                                 static_cast<int>(FacePickStrategy::LargestFace));
    m_faceStrategyCombo->addItem(
            tr("Highest score"),
            static_cast<int>(FacePickStrategy::HighestScore));
    m_faceStrategyCombo->setToolTip(
            tr("When multiple faces appear, choose which one to capture. "
               "In Track same person mode, one identity is kept across frames."));
    settingsRow->addWidget(m_faceStrategyCombo, 1);
    mainLayout->addLayout(settingsRow);

    connect(m_minScoreSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double) { saveFaceCaptureSettings(); });
    connect(m_minCapturesSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) {
                saveFaceCaptureSettings();
                updateCaptureProgressUi();
            });
    connect(m_faceStrategyCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { saveFaceCaptureSettings(); });
#else
    auto* detectorLayout = new QHBoxLayout();
    detectorLayout->addWidget(new QLabel(tr("Face detector:"), this));
    m_detectorCombo = new QComboBox(this);
    detectorLayout->addWidget(m_detectorCombo, 1);
    mainLayout->addLayout(detectorLayout);
    connect(m_detectorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FaceCaptureWidget::onDetectorComboChanged);
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

    m_cameraControlsRow = new QWidget(this);
    auto* controlsLayout = new QHBoxLayout(m_cameraControlsRow);
    controlsLayout->setContentsMargins(0, 0, 0, 0);
    controlsLayout->addWidget(new QLabel(tr("Device:"), this));

    m_cameraCombo = new QComboBox(this);
    m_cameraCombo->addItem(tr("Default (0)"), 0);
    controlsLayout->addWidget(m_cameraCombo, 1);

    m_captureBtn = new QPushButton(tr("Capture"), this);
    m_captureBtn->setEnabled(false);
    controlsLayout->addWidget(m_captureBtn);

    mainLayout->addWidget(m_cameraControlsRow);

    m_frameTimer = new QTimer(this);
    m_frameTimer->setInterval(30);

    connect(m_frameTimer, &QTimer::timeout, this,
            &FaceCaptureWidget::processFrame);
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

void FaceCaptureWidget::onSourceChanged(int index) {
    if (!m_sourceCombo) return;
    m_inputSource = static_cast<InputSource>(m_sourceCombo->itemData(index).toInt());
    if (m_videoFileRow) {
        m_videoFileRow->setVisible(m_inputSource == InputSource::VideoFile);
    }
    if (m_cameraControlsRow) {
        m_cameraControlsRow->setVisible(m_inputSource == InputSource::Camera);
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
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qFreeSplatter"),
            QStringLiteral("lastVideoDir"), QDir::homePath());
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select video file"), lastDir,
            tr("Video files (*.mp4 *.avi *.mkv *.mov *.webm *.m4v *.wmv *.ts "
               "*.mpg *.mpeg);;All files (*.*)"));
    if (path.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qFreeSplatter"),
                                       QStringLiteral("lastVideoDir"),
                                       path);
    if (m_videoPathEdit) m_videoPathEdit->setText(path);
    m_videoFilePath = path;
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

bool FaceCaptureWidget::startVideoFile(const QString& path) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (path.isEmpty()) return false;
    m_inputSource = InputSource::VideoFile;
    m_videoFilePath = path;
    if (m_videoPathEdit) m_videoPathEdit->setText(path);

    if (m_detectorKind == DetectorKind::Ggml) {
        if (!ensureGgmlModelReady()) {
            m_statusLabel->setText(tr("Downloading face detector — video preview starting…"));
        } else {
            scheduleGgmlModelLoad(facedetectCachePath(currentGgmlFilename()));
        }
    } else if (m_detectorKind == DetectorKind::OpenCV) {
        releaseGgmlModel();
        loadCascade();
    }

    stopCapture();
    if (!m_camera.open(path.toStdString(), cv::CAP_FFMPEG) &&
        !m_camera.open(path.toStdString(), cv::CAP_ANY)) {
        const QString err =
                tr("Failed to open video (rebuild OpenCV with FFmpeg / "
                   "WITH_FFMPEG=ON): %1")
                        .arg(path);
        m_statusLabel->setText(err);
        emit cameraError(err);
        return false;
    }

    m_cameraActive = true;
    m_ggmlFrameSkip = 0;
    m_frameTimer->start();
    m_statusLabel->setText(tr("Playing video — preview + face overlay"));
    emit cameraStarted();
    return true;
#else
    Q_UNUSED(path);
    return false;
#endif
}

void FaceCaptureWidget::stopCapture() { stopCamera(); }

void FaceCaptureWidget::releaseGpuResources() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    stopCamera();
    if (m_ggmlLoadWatcher && m_ggmlLoadWatcher->isRunning()) {
        m_ggmlLoadWatcher->waitForFinished();
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

    m_detectorCombo->addItem(tr("OpenCV Haar Cascade"), QStringLiteral("opencv"));

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
        m_detectorCombo->addItem(
                QCoreApplication::translate("FaceDetectModels", m->display_name) +
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
    if (m_detectorKind == DetectorKind::Ggml) return m_ggmlCtx != nullptr;
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
    const QString dest = facedetectCachePath(QString::fromUtf8(model->filename));

    m_downloadInProgress = true;
    m_autoStartAfterDownload = true;
    if (m_downloadLabel) {
        m_downloadLabel->setVisible(true);
        m_downloadLabel->setText(
                tr("Downloading %1 ...").arg(QString::fromUtf8(model->filename)));
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

    aicore_facedetect_options* opts = aicore_facedetect_options_new();
    aicore_facedetect_options_set_device(opts, "auto");
    aicore_facedetect_options_set_threads(opts, 0);
    m_ggmlCtx = aicore_facedetect_load_opts(path.toUtf8().constData(), opts);
    aicore_facedetect_options_free(opts);

    if (!m_ggmlCtx) {
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
    m_statusLabel->setText(tr("Loading face detector model..."));
    m_ggmlLoadWatcher->setFuture(QtConcurrent::run([path]() -> aicore_facedetect_ctx* {
        aicore_facedetect_options* opts = aicore_facedetect_options_new();
        aicore_facedetect_options_set_device(opts, "auto");
        aicore_facedetect_options_set_threads(opts, 0);
        aicore_facedetect_ctx* ctx =
                aicore_facedetect_load_opts(path.toUtf8().constData(), opts);
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
        const QString tmpDir =
                QStandardPaths::writableLocation(QStandardPaths::TempLocation);
        const QString tmpPath =
                tmpDir + QStringLiteral("/cv_haarcascade_frontalface_alt2.xml");
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
    m_pendingCameraIndex = deviceIndex;

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

    stopCamera();

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

    if (!m_camera.open(deviceIndex, cv::CAP_ANY)) {
        m_cameraActive = false;
        const QString error =
                tr("Failed to open camera device %1").arg(deviceIndex);
        m_statusLabel->setText(error);
        emit cameraError(error);
        return false;
    }

    m_camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    m_camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (detectorReady()) {
        m_statusLabel->setText(tr("Camera active \u2014 detecting faces"));
    } else {
        m_statusLabel->setText(
                tr("Camera active \u2014 no face detector (full-frame crop)"));
    }

    m_cameraActive = true;
    m_ggmlFrameSkip = 0;
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

#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_camera.isOpened()) m_camera.release();
#endif

    if (m_cameraActive) {
        m_cameraActive = false;
        emit cameraStopped();
    }

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_lastFaceRect = cv::Rect();
    m_lastFaceScore = 0.f;
#endif
    m_consecutiveDetections = 0;
}

bool FaceCaptureWidget::isCameraActive() const { return m_cameraActive; }

void FaceCaptureWidget::startGuidedCapture(
        const std::vector<CaptureAngle>& angles) {
    resetCapture();
    m_targetAngles = angles;
    m_capturingMode = true;
    m_currentAngleIndex = currentGuideAngleIndex();

    if (m_captureBtn) {
        m_captureBtn->setEnabled(m_cameraActive);
    }

    const int target = minCapturesBeforeComplete();
    if (!m_targetAngles.empty()) {
        m_angleLabel->setText(
                tr("Angle: %1 (capture 1/%2)")
                        .arg(angleToString(
                                m_targetAngles[static_cast<size_t>(
                                        m_currentAngleIndex)]))
                        .arg(target));
    } else {
        m_angleLabel->setText(tr("Capture face snapshots (1/%1)").arg(target));
    }
    m_statusLabel->setText(tr("Position your face and capture each angle"));
    updateCaptureProgressUi();
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

    cv::Mat frame;
    if (!m_camera.read(frame) || frame.empty()) {
        emit cameraError(tr("Failed to read frame from camera"));
        return;
    }

    const int angleCount =
            std::max(1, static_cast<int>(m_targetAngles.size()));
    const int angleIdx =
            static_cast<int>(m_capturedFrames.size()) % angleCount;
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
        m_angleLabel->setText(tr("Capture complete"));
        emit captureComplete();
        return;
    }

    m_currentAngleIndex = index % angleCount;
    if (!m_targetAngles.empty()) {
        const auto nextAngle =
                m_targetAngles[static_cast<size_t>(m_currentAngleIndex)];
        m_angleLabel->setText(tr("Angle: %1 (capture %2/%3)")
                                      .arg(angleToString(nextAngle))
                                      .arg(index + 1)
                                      .arg(target));
    } else {
        m_angleLabel->setText(
                tr("Capture face snapshots (%1/%2)").arg(index + 1).arg(target));
    }
    if (m_captureBtn) m_captureBtn->setEnabled(false);
#endif
}

void FaceCaptureWidget::resetCapture() {
    resetIdentityTrack();
    m_targetAngles.clear();
    m_capturedFrames.clear();
    m_currentAngleIndex = 0;
    m_capturingMode = false;
    m_consecutiveDetections = 0;
    m_postCaptureCooldown = 0;
    m_noCascadeCounter = 0;

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_lastFaceRect = cv::Rect();
    m_lastFaceScore = 0.f;
#endif

    if (m_captureBtn) m_captureBtn->setEnabled(false);
    if (m_angleLabel) m_angleLabel->setText(QString());
    refreshCapturedGallery();
    updateCaptureProgressUi();
    if (m_statusLabel) {
        m_statusLabel->setText(
                m_cameraActive ? tr("Camera active \u2014 detecting faces")
                               : tr("Ready"));
    }
}

std::vector<FaceCaptureWidget::CapturedFrame>
FaceCaptureWidget::capturedFrames() const {
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

int FaceCaptureWidget::capturedCount() const {
    return static_cast<int>(m_capturedFrames.size());
}

int FaceCaptureWidget::targetCount() const {
    return minCapturesBeforeComplete();
}

float FaceCaptureWidget::minDetectionScore() const {
    return m_minScoreSpin ? static_cast<float>(m_minScoreSpin->value()) : 0.5f;
}

int FaceCaptureWidget::minCapturesBeforeComplete() const {
    return m_minCapturesSpin ? m_minCapturesSpin->value() : 2;
}

FaceCaptureWidget::FacePickStrategy FaceCaptureWidget::facePickStrategy() const {
    if (!m_faceStrategyCombo) return FacePickStrategy::TrackSamePerson;
    return static_cast<FacePickStrategy>(
            m_faceStrategyCombo->currentData().toInt());
}

void FaceCaptureWidget::processFrame() {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_cameraActive || !m_camera.isOpened()) return;

    cv::Mat frame;
    if (!m_camera.read(frame) || frame.empty()) {
        if (m_inputSource == InputSource::VideoFile) {
            m_camera.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!m_camera.read(frame) || frame.empty()) {
                stopCapture();
                m_statusLabel->setText(tr("Video finished"));
                return;
            }
        } else {
            return;
        }
    }

    QImage preview = cvMatToQImage(frame);
    if (!preview.isNull()) {
        m_previewLabel->setPreviewImage(preview, m_previewLabel->size());
    }

    cv::Rect faceRect;
    if (detectorReady() && !m_ggmlModelLoading) {
        if (m_detectorKind == DetectorKind::Ggml) {
            if (m_ggmlFrameSkip <= 0) {
                faceRect = detectFaceGgml(frame);
                m_ggmlFrameSkip = kGgmlDetectInterval;
            } else {
                --m_ggmlFrameSkip;
                if (m_lastFaceRect.width > 0) faceRect = m_lastFaceRect;
            }
        } else {
            faceRect = detectFaceOpenCv(frame);
        }

        if (faceRect.width > 0 && faceRect.height > 0) {
            ++m_consecutiveDetections;
            m_lastFaceRect = faceRect;
            emit faceDetected(QRect(faceRect.x, faceRect.y, faceRect.width,
                                    faceRect.height));
        } else {
            m_consecutiveDetections = 0;
            m_lastFaceRect = cv::Rect();
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
            drawAngleGuide(
                    preview,
                    m_targetAngles[static_cast<size_t>(guideIdx)]);
        }
        m_previewLabel->setPreviewImage(preview, m_previewLabel->size());
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
                int pct = std::min(100, m_consecutiveDetections * 100 / trigger);
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

    QImage image(rgb.cols, rgb.rows, QImage::Format_RGB888);
    for (int y = 0; y < rgb.rows; ++y) {
        std::memcpy(image.scanLine(y), rgb.ptr<uchar>(y),
                    static_cast<size_t>(rgb.cols) * 3);
    }
    return image;
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

    char* json = aicore_facedetect_detect_rgb_json(
            m_ggmlCtx, rgb.data, rgb.cols, rgb.rows);
    if (!json) return out;

    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray(json));
    aicore_facedetect_free_string(json);
    if (!doc.isObject()) return out;

    const QJsonArray faces = doc.object().value(QStringLiteral("faces")).toArray();
    out.reserve(static_cast<size_t>(faces.size()));
    for (const QJsonValue& v : faces) {
        const QJsonObject obj = v.toObject();
        const QJsonArray box = obj.value(QStringLiteral("box")).toArray();
        if (box.size() != 4) continue;
        const float score =
                static_cast<float>(obj.value(QStringLiteral("score")).toDouble());
        const int x = static_cast<int>(std::floor(box.at(0).toDouble()));
        const int y = static_cast<int>(std::floor(box.at(1).toDouble()));
        const int w = static_cast<int>(
                std::ceil(box.at(2).toDouble() - box.at(0).toDouble()));
        const int h = static_cast<int>(
                std::ceil(box.at(3).toDouble() - box.at(1).toDouble()));
        if (w <= 0 || h <= 0) continue;
        out.push_back({cv::Rect(x, y, w, h), score});
    }
    return out;
}

std::vector<FaceCaptureWidget::ScoredFace> FaceCaptureWidget::detectFaces(
        const cv::Mat& frame) {
    if (m_detectorKind == DetectorKind::Ggml) return detectFacesGgml(frame);
    return detectFacesOpenCv(frame);
}

bool FaceCaptureWidget::embedFaceCrop(const cv::Mat& frame, const cv::Rect& rect,
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
    if (aicore_facedetect_embed_rgb(m_ggmlCtx, rgb.data, rgb.cols, rgb.rows, 0.f,
                                    &vec, &dim) != 0 ||
        vec == nullptr || dim <= 0) {
        return false;
    }
    embedding->assign(vec, vec + dim);
    aicore_facedetect_free_vec(vec);
    return true;
}

float FaceCaptureWidget::embeddingDistance(const std::vector<float>& a,
                                           const std::vector<float>& b) const {
    if (a.size() != b.size() || a.empty()) return 1.0f;
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dot += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(1.0 - dot);
}

void FaceCaptureWidget::resetIdentityTrack() {
    m_referenceEmbedding.clear();
    m_hasReferenceEmbedding = false;
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
        const auto it = std::max_element(
                candidates.begin(), candidates.end(),
                [](const ScoredFace& a, const ScoredFace& b) {
                    return a.rect.area() < b.rect.area();
                });
        m_lastFaceScore = it->score;
        return it->rect;
    }
    if (strategy == FacePickStrategy::HighestScore) {
        const auto it = std::max_element(
                candidates.begin(), candidates.end(),
                [](const ScoredFace& a, const ScoredFace& b) {
                    return a.score < b.score;
                });
        m_lastFaceScore = it->score;
        return it->rect;
    }

    if (m_detectorKind != DetectorKind::Ggml || !m_ggmlCtx) {
        const auto it = std::max_element(
                candidates.begin(), candidates.end(),
                [](const ScoredFace& a, const ScoredFace& b) {
                    return a.rect.area() < b.rect.area();
                });
        m_lastFaceScore = it->score;
        return it->rect;
    }

    if (!m_hasReferenceEmbedding) {
        const auto seedIt = std::max_element(
                candidates.begin(), candidates.end(),
                [](const ScoredFace& a, const ScoredFace& b) {
                    return a.rect.area() < b.rect.area();
                });
        if (seedIt == candidates.end()) return cv::Rect();
        if (embedFaceCrop(frame, seedIt->rect, &m_referenceEmbedding)) {
            m_hasReferenceEmbedding = true;
        }
        m_lastFaceScore = seedIt->score;
        return seedIt->rect;
    }

    const ScoredFace* best = nullptr;
    float bestDistance = kSamePersonMaxDistance;
    for (const ScoredFace& face : candidates) {
        std::vector<float> emb;
        if (!embedFaceCrop(frame, face.rect, &emb)) continue;
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
        const int textW = fm.horizontalAdvance(scoreText) + 2 * pad;
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
    return static_cast<int>(m_capturedFrames.size()) % angleCount;
}

void FaceCaptureWidget::updateCaptureProgressUi() {
    const int captured = capturedCount();
    const int target = minCapturesBeforeComplete();
    if (m_captureProgress) {
        m_captureProgress->setMaximum(target);
        m_captureProgress->setValue(std::min(captured, target));
    }
    if (m_captureProgressLabel) {
        m_captureProgressLabel->setText(
                tr("Captured %1/%2 faces").arg(captured).arg(target));
    }
}

void FaceCaptureWidget::refreshCapturedGallery() {
    if (!m_capturedGalleryRow) return;

    QLayout* layout = m_capturedGalleryRow->layout();
    if (!layout) return;

    while (QLayoutItem* item = layout->takeAt(0)) {
        if (QWidget* widget = item->widget()) widget->deleteLater();
        delete item;
    }

    for (size_t i = 0; i < m_capturedFrames.size(); ++i) {
        const CapturedFrame& frame = m_capturedFrames[i];
        if (!frame.valid || frame.croppedFace.isNull()) continue;

        auto* thumb = new QLabel(m_capturedGalleryRow);
        thumb->setFixedSize(48, 48);
        thumb->setAlignment(Qt::AlignCenter);
        thumb->setStyleSheet(
                QStringLiteral("QLabel { background-color: #222; "
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
    if (m_faceStrategyCombo) {
        m_faceStrategyCombo->blockSignals(true);
        const int strategy =
                settings.value(QStringLiteral("faceStrategy"),
                               static_cast<int>(FacePickStrategy::TrackSamePerson))
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
#endif

    settings.endGroup();
    updateCaptureProgressUi();
}

void FaceCaptureWidget::saveFaceCaptureSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qFreeSplatter"));

#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_minScoreSpin) {
        settings.setValue(QStringLiteral("faceMinScore"), m_minScoreSpin->value());
    }
    if (m_minCapturesSpin) {
        settings.setValue(QStringLiteral("faceMinCaptures"),
                          m_minCapturesSpin->value());
    }
    if (m_faceStrategyCombo) {
        settings.setValue(QStringLiteral("faceStrategy"),
                          m_faceStrategyCombo->currentData());
    }
    settings.setValue(QStringLiteral("faceVideoPath"), videoFilePath());
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
