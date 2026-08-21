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
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QMouseEvent>
#include <QMutex>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QPixmapCache>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSettings>
#include <QShowEvent>
#include <QTimer>
#include <QtConcurrent>
#include <atomic>
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
        aicore_facedetect_free_buffer(dir);
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

}  // namespace

FaceCaptureWidget::FaceCaptureWidget(QWidget* parent)
    : VideoPlaybackWidget(parent) {
    // video_base owns the playback panel; cache the base status label so
    // existing code keeps working, and forward the stream signals under the
    // historical names used by FreeSplatterDialog.
    m_statusLabel = statusLabel();
    connect(this, &VideoPlaybackWidget::streamStarted, this,
            &FaceCaptureWidget::cameraStarted);
    connect(this, &VideoPlaybackWidget::streamStopped, this,
            &FaceCaptureWidget::cameraStopped);
    connect(this, &VideoPlaybackWidget::streamError, this,
            &FaceCaptureWidget::cameraError);

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
                if (isActive()) {
                    m_statusLabel->setText(
                            inputSource() == InputSource::VideoFile
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
    stopCamera();  // stops the video_base stream; the background reader
                   // thread is owned and torn down by the base class
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
    return VideoPlaybackWidget::isAvailable();
}

void FaceCaptureWidget::setupUi() {
    // The playback panel (preview + source selection + seek/speed controls)
    // is built by the video_base base class; this method only appends the
    // face-capture specific controls below the base layout.
    auto* layout = mainLayout();

    // Angle guide label (guided capture).
    m_angleLabel = new QLabel(this);
    m_angleLabel->setAlignment(Qt::AlignCenter);
    m_angleLabel->hide();
    layout->addWidget(m_angleLabel);

    m_captureProgress = new QProgressBar(this);
    m_captureProgress->setTextVisible(true);
    m_captureProgress->setFormat(tr("%v / %m faces"));
    m_captureProgress->setValue(0);
    layout->addWidget(m_captureProgress);

#ifdef HAS_OPENCV_FACE_CAPTURE
    // Manual capture action (camera mode) / guided-capture snapshot button.
    m_captureBtn = new QPushButton(tr("Capture"), this);
    m_captureBtn->setEnabled(false);
    layout->addWidget(m_captureBtn);
    connect(m_captureBtn, &QPushButton::clicked, this,
            &FaceCaptureWidget::captureCurrentFrame);
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
    layout->addWidget(m_capturedGalleryScroll);

#ifdef HAS_OPENCV_FACE_CAPTURE
    auto* detectorInputRow = new QHBoxLayout();
    detectorInputRow->setSpacing(6);
    detectorInputRow->addWidget(new QLabel(tr("Face detector:"), this));
    m_detectorCombo = new QComboBox(this);
    detectorInputRow->addWidget(m_detectorCombo, 1);
    layout->addLayout(detectorInputRow);
    connect(m_detectorCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &FaceCaptureWidget::onDetectorComboChanged);

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
    layout->addLayout(settingsRow);

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
    layout->addLayout(registryPathRow);

    m_registryFilterEdit = new QLineEdit(this);
    m_registryFilterEdit->setPlaceholderText(
            tr("Filter registered identities by id or name"));
    layout->addWidget(m_registryFilterEdit);
    m_registryList = new QListWidget(this);
    m_registryList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_registryList->setMaximumHeight(96);
    m_registryList->setAlternatingRowColors(true);
    layout->addWidget(m_registryList);
    m_registryStatusLabel = new QLabel(this);
    layout->addWidget(m_registryStatusLabel);

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
    layout->addLayout(detectorLayout);
    connect(m_detectorCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &FaceCaptureWidget::onDetectorComboChanged);
#endif

    m_downloadLabel = new QLabel(this);
    m_downloadLabel->setAlignment(Qt::AlignCenter);
    m_downloadLabel->setVisible(false);
    layout->addWidget(m_downloadLabel);

    m_downloadProgress = new QProgressBar(this);
    m_downloadProgress->setVisible(false);
    m_downloadProgress->setTextVisible(false);
    layout->addWidget(m_downloadProgress);

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_statusLabel->setText(
            tr("Ready \u2014 choose a detector, then start the camera"));
    updateCaptureProgressUi();
#else
    m_statusLabel->setText(
            tr("Face capture unavailable (OpenCV not built with videoio "
               "and objdetect)"));
#endif
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

void FaceCaptureWidget::resumeCapture() {
    // Resume guided capture WITHOUT clearing collected faces.
    // Used when playback resumes after Stop (paused) — the user keeps
    // everything gathered so far and capture continues where it left off.
    if (m_capturedFrames.size() >= minCapturesBeforeComplete()) {
        m_capturingMode = false;
        return;
    }
    m_capturingMode = true;
    m_consecutiveDetections = 0;
    m_postCaptureCooldown = 0;
    const int index = static_cast<int>(m_capturedFrames.size());
    const int target = minCapturesBeforeComplete();
    if (!m_targetAngles.empty()) {
        setAngleGuideText(
                tr("Angle: %1 (capture %2/%3)")
                        .arg(angleToString(m_targetAngles[static_cast<size_t>(
                                m_currentAngleIndex)]))
                        .arg(index + 1)
                        .arg(target));
    } else {
        setAngleGuideText(tr("Capture face snapshots (%1/%2)")
                                  .arg(index + 1)
                                  .arg(target));
    }
    updateCaptureProgressUi();
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
    if (isActive() && m_detectorKind == DetectorKind::Ggml) {
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

    if (isActive()) {
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

// ---------------------------------------------------------------------------
// video_base hooks
// ---------------------------------------------------------------------------

bool FaceCaptureWidget::onPrepareStream() {
    // Cancel and drain the previous inference session before issuing any
    // work for the new stream.  Doing this after scheduleGgmlModelLoad would
    // cancel the new model load through the shared task token.
    requestInferenceCancel();
    aicore_cancel_token_reset(m_inferenceCancelToken);

    if (!configureDetectorForRegistrySelection()) {
        qWarning() << "[FaceCaptureWidget] "
                      "configureDetectorForRegistrySelection failed";
        return false;
    }

    if (m_detectorKind == DetectorKind::Ggml) {
        if (!ensureGgmlModelReady()) {
            m_statusLabel->setText(
                    tr("Downloading face detector — video preview starting…"));
        } else {
            scheduleGgmlModelLoad(facedetectCachePath(currentGgmlFilename()));
        }
    } else if (m_detectorKind == DetectorKind::OpenCV) {
        releaseGgmlModel();
        if (!loadCascade()) {
            m_statusLabel->setText(
                    tr("Warning: OpenCV cascade not found \u2014 "
                       "capture without detection"));
        }
    }
    return true;
}

void FaceCaptureWidget::onStreamStopping() {
    // Reset per-stream detection state (base class owns the stream itself).
    m_lastFaceRect = cv::Rect();
    m_lastFaceScore = 0.f;
    m_consecutiveDetections = 0;
}

void FaceCaptureWidget::onStreamReset() {
    // Restart: clear ALL detection state so the pipeline restarts fresh —
    // stale face rect / cached frame disappear immediately and the next
    // frame forces a fresh detection (m_lastDetectedFrameNum = -1).
    m_lastFaceRect = cv::Rect();
    m_lastDetectedFrame.release();
    m_lastFaceScore = 0.f;
    m_consecutiveDetections = 0;
    m_ggmlFrameSkip = 0;
    m_lastDetectedFrameNum = -1;
    aicore_cancel_token_reset(m_inferenceCancelToken);
}

void FaceCaptureWidget::onStreamResumed() {
    // Resume from pause: keep collected faces/overlays, only reset the
    // detection throttle so detection resumes immediately.
    m_ggmlFrameSkip = 0;
    aicore_cancel_token_reset(m_inferenceCancelToken);
}

void FaceCaptureWidget::onVideoLooped() {
    // Video EOF looped back to frame 0 — reset the video-time detection
    // throttle so detection resumes on the first loop frame.
    m_lastDetectedFrameNum = 0;
    m_ggmlFrameSkip = 0;
}

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
        m_captureBtn->setEnabled(isActive() && m_identityTracks.empty());
#else
        m_captureBtn->setEnabled(isActive());
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
        const int trigger = (inputSource() == InputSource::VideoFile)
                                    ? kVideoAutoCaptureTrigger
                                    : kAutoCaptureTrigger;
        if (m_consecutiveDetections < trigger) {
            m_statusLabel->setText(tr("Hold still \u2014 face not stable yet"));
            return;
        }
    } else if (inputSource() == InputSource::VideoFile) {
        m_statusLabel->setText(tr("Face detector not ready"));
        return;
    }

    const cv::Mat& sourceFrame = detectorReady() && m_lastFaceRect.width > 0
                                         ? m_lastDetectedFrame
                                         : latestFrame();
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
    captured.image = VideoPlaybackWidget::cvMatToQImage(frame);
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
        captured.croppedFace = VideoPlaybackWidget::cvMatToQImage(resized);
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
    m_lastDetectedFrame.release();
    m_lastFaceScore = 0.f;
#endif

    if (m_captureBtn) m_captureBtn->setEnabled(false);
    setAngleGuideText(QString());
    refreshCapturedGallery();
    updateCaptureProgressUi();

    // Ensure video controls remain functional after reset (base class
    // keeps them enabled as long as a video is loaded).
    if (m_statusLabel) {
        m_statusLabel->setText(
                isActive() ? tr("Camera active \u2014 detecting faces")
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

void FaceCaptureWidget::onFrameDecoded(cv::Mat& frame, int frameIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    // Record the original frame size so onDisplayFrame can scale overlay
    // coordinates from original resolution to the displayed resolution.
    m_lastFrameSize = frame.size();

    // Determine whether to run detection based on video-time elapsed since
    // last detection.  This keeps detection frequency consistent across all
    // playback speeds: at any speed, detection runs once per
    // ~kGgmlDetectInterval video frames of content.
    const bool timeForDetection = [&]() -> bool {
        // After a stream reset (restart / reset button), no detection has
        // run yet — force immediate detection on the very first frame so
        // the overlay appears without a one-frame gap.
        if (m_lastDetectedFrameNum < 0) return true;
        if (inputSource() == InputSource::VideoFile && videoFps() > 0) {
            // Video-time throttle: detect every kGgmlDetectInterval frames of
            // video content, regardless of playback speed.
            // Also handle backwards frame index (after seek / restart): if
            // the current frame is before the last detected frame, a seek
            // occurred and we should detect immediately.
            if (frameIndex < m_lastDetectedFrameNum) return true;
            const double videoTimeMs = frameIndex / videoFps() * 1000.0;
            const double thresholdMs =
                    kGgmlDetectInterval / videoFps() * 1000.0;
            if (videoTimeMs - (m_lastDetectedFrameNum / videoFps() * 1000.0) >=
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

    // Identity-tracking mode: run registry matching (state only; the boxes
    // and labels are painted by onDisplayFrame).
    if (!m_identityTracks.empty() && detectorReady() &&
        m_detectorKind == DetectorKind::Ggml && !m_ggmlModelLoading) {
        if (timeForDetection) {
            const std::vector<ScoredFace> faces = detectFacesGgml(frame);
            processRegistryIdentities(frame, faces);
            m_lastDetectedFrameNum = frameIndex;
            m_lastDetectedFrame = frame.clone();
        }
        return;
    }

    cv::Rect faceRect;
    bool freshDetection = false;
    if (detectorReady() && !m_ggmlModelLoading) {
        if (m_detectorKind == DetectorKind::Ggml) {
            if (timeForDetection) {
                faceRect = detectFaceGgml(frame);
                m_lastDetectedFrameNum = frameIndex;
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
            if (freshDetection && m_capturingMode) {
                // Keep the detected frame only while capturing —
                // captureCurrentFrame needs a frame matching m_lastFaceRect.
                // Outside capture mode the per-frame full-res clone is pure
                // waste (OpenCV detects on every frame).
                m_lastDetectedFrame = frame.clone();
            }
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

    // Guided-capture auto-trigger logic (uses the detection state above).
    if (m_capturingMode) {
        const int trigger = (inputSource() == InputSource::VideoFile)
                                    ? kVideoAutoCaptureTrigger
                                    : kAutoCaptureTrigger;
        const int target = minCapturesBeforeComplete();
        const int captured = static_cast<int>(m_capturedFrames.size());
        if (m_postCaptureCooldown > 0) {
            --m_postCaptureCooldown;
            if (inputSource() == InputSource::Camera) {
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
            if (m_captureBtn && inputSource() == InputSource::Camera) {
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
        } else if (inputSource() == InputSource::Camera) {
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

void FaceCaptureWidget::onDisplayFrame(QImage& display, int frameIndex) {
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (display.isNull()) return;

    // Scale a rectangle from original frame coordinates to the displayed
    // resolution (the display is the original frame scaled with
    // KeepAspectRatio by the base class).
    auto scaledRect = [&](const cv::Rect& r) -> cv::Rect {
        if (m_lastFrameSize.width <= 0 || m_lastFrameSize.height <= 0) {
            return r;
        }
        const qreal sx =
                static_cast<qreal>(display.width()) / m_lastFrameSize.width;
        const qreal sy =
                static_cast<qreal>(display.height()) / m_lastFrameSize.height;
        return cv::Rect(static_cast<int>(std::lround(r.x * sx)),
                        static_cast<int>(std::lround(r.y * sy)),
                        static_cast<int>(std::lround(r.width * sx)),
                        static_cast<int>(std::lround(r.height * sy)));
    };

    // Identity-tracking mode: boxes/labels are cached in m_identityTracks
    // (original frame coordinates) — scale them to the display.
    if (!m_identityTracks.empty() && detectorReady() &&
        m_detectorKind == DetectorKind::Ggml && !m_ggmlModelLoading) {
        QPainter painter(&display);
        painter.setPen(QPen(QColor(0, 200, 255), 3));
        for (const IdentityTrack& track : m_identityTracks) {
            if (track.lastRect.width <= 0) continue;
            const cv::Rect box = scaledRect(track.lastRect);
            painter.drawRect(box.x, box.y, box.width, box.height);
            const QString label = QStringLiteral("%1 d=%2")
                                          .arg(track.identity.name)
                                          .arg(track.lastDistance, 0, 'f', 3);
            painter.fillRect(
                    box.x, std::max(0, box.y - 22),
                    std::max(90, QTCOMPAT_FONTMETRICS_WIDTH(
                                         painter.fontMetrics(), label) +
                                         8),
                    22, QColor(0, 0, 0, 180));
            painter.setPen(Qt::white);
            painter.drawText(box.x + 4, std::max(16, box.y - 6), label);
        }
        if (m_capturingMode && !m_targetAngles.empty()) {
            drawAngleGuide(display, m_targetAngles[static_cast<size_t>(
                                            currentGuideAngleIndex())]);
        }
        return;
    }

    if (detectorReady() && m_lastFaceRect.width > 0) {
        drawOverlay(display, scaledRect(m_lastFaceRect));
    }
    if (m_capturingMode && !m_targetAngles.empty()) {
        const int guideIdx = currentGuideAngleIndex();
        drawAngleGuide(display, m_targetAngles[static_cast<size_t>(guideIdx)]);
    }
#endif
}

#ifdef HAS_OPENCV_FACE_CAPTURE

cv::Rect FaceCaptureWidget::detectFaceOpenCv(const cv::Mat& frame) {
    return pickFace(frame, detectFacesOpenCv(frame));
}

std::vector<FaceCaptureWidget::ScoredFace> FaceCaptureWidget::detectFacesOpenCv(
        const cv::Mat& frame) {
    std::vector<ScoredFace> out;
    if (m_faceCascade.empty() || frame.empty()) return out;

    // Downscale large frames before cascade detection — detectMultiScale on
    // a full 1080p frame costs ~20-40 ms while a ≤640px copy is ~5-10 ms.
    // Boxes are mapped back to full-res coordinates below, so callers keep
    // working in original-frame space (m_lastFaceRect, capture crops).
    constexpr int kOpenCvDetectMaxDim = 640;
    const int maxDim = std::max(frame.cols, frame.rows);
    const float scale =
            maxDim > kOpenCvDetectMaxDim
                    ? static_cast<float>(kOpenCvDetectMaxDim) / maxDim
                    : 1.f;
    cv::Mat detectFrame = frame;
    if (scale < 1.f) {
        cv::resize(frame, detectFrame, cv::Size(), scale, scale,
                   cv::INTER_AREA);
    }

    cv::Mat gray;
    cv::cvtColor(detectFrame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    m_faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(80, 80));
    out.reserve(faces.size());
    const float inv = 1.f / scale;
    for (const cv::Rect& rect : faces) {
        const cv::Rect fullRes(
                static_cast<int>(std::lround(rect.x * inv)),
                static_cast<int>(std::lround(rect.y * inv)),
                static_cast<int>(std::lround(rect.width * inv)),
                static_cast<int>(std::lround(rect.height * inv)));
        out.push_back({fullRes, 1.0f});
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
    aicore_facedetect_free_buffer(json);
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
    aicore_facedetect_free_buffer(vec);
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
    aicore_facedetect_free_buffer(vec);
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
    captured.image = VideoPlaybackWidget::cvMatToQImage(frame);
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
        const cv::Mat& frame, const std::vector<ScoredFace>& faces) {
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

    const int trigger = inputSource() == InputSource::VideoFile
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

        // NOTE: the match box/label is painted by onDisplayFrame (scaled
        // from original frame coordinates to the displayed resolution).
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
    return VideoPlaybackWidget::cvMatToQImage(resized);
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
    QLineEdit* pathEdit = videoPathEdit();
    if (pathEdit && !videoPath.isEmpty()) {
        pathEdit->blockSignals(true);
        pathEdit->setText(videoPath);
        pathEdit->blockSignals(false);
        setVideoFilePath(videoPath);
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
