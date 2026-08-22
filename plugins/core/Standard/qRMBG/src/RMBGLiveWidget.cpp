// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RMBGLiveWidget.h"

#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMetaObject>
#include <QPainter>
#include <QSettings>
#include <QSizePolicy>
#include <QSpinBox>
#include <QThread>
#include <algorithm>

#include "RMBGLiveInferWorker.h"
#include "RMBGModelCatalog.h"
#include "ecvPersistentSettings.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/imgproc.hpp>
#endif

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/rmbg_capi.h"
#endif

namespace {

QString formatLatency(qint64 ms) {
    return ms >= 0 ? QStringLiteral("%1 ms").arg(ms) : QStringLiteral("--");
}

}  // namespace

RMBGLiveWidget::RMBGLiveWidget(QWidget* parent) : VideoPlaybackWidget(parent) {
    // Clock-driven playback: the video advances at its native frame rate
    // (× speed) while inference runs asynchronously and its alpha mask is
    // composited onto the live frames — decoupling display smoothness from
    // inference latency.
    setupUi();
    setPreviewFixedHeight(300);

    m_inferThread = new QThread(this);
    m_inferWorker = new RMBGLiveInferWorker;
    m_inferWorker->moveToThread(m_inferThread);
    connect(m_inferThread, &QThread::finished, m_inferWorker,
            &QObject::deleteLater);
    connect(m_inferWorker, &RMBGLiveInferWorker::inferComplete, this,
            &RMBGLiveWidget::onInferComplete, Qt::QueuedConnection);
    m_inferThread->start();
}

RMBGLiveWidget::~RMBGLiveWidget() {
    stopStream();
    shutdownInferThread();
}

bool RMBGLiveWidget::isAvailable() {
    return VideoPlaybackWidget::isAvailable();
}

void RMBGLiveWidget::setConfig(const Config& config) { m_config = config; }

void RMBGLiveWidget::setVideoFilePath(const QString& path, bool userChosen) {
    VideoPlaybackWidget::setVideoFilePath(path);
    m_videoPathUserChosen = userChosen;
}

void RMBGLiveWidget::setupUi() {
    m_statusLabel = statusLabel();

    auto* controls = new QHBoxLayout;
    controls->setContentsMargins(0, 4, 0, 0);

    controls->addWidget(new QLabel(tr("Model:"), this));
    m_modelCombo = new QComboBox(this);
    m_modelCombo->setMinimumContentsLength(20);
    m_modelCombo->setSizeAdjustPolicy(
            QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_modelCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    controls->addWidget(m_modelCombo, 1);

    controls->addWidget(new QLabel(tr("Device:"), this));
    m_deviceCombo = new QComboBox(this);
    controls->addWidget(m_deviceCombo);

    controls->addWidget(new QLabel(tr("Threads:"), this));
    m_threadsSpin = new QSpinBox(this);
    m_threadsSpin->setRange(0, 64);
    m_threadsSpin->setValue(0);
    m_threadsSpin->setToolTip(tr("0 = auto"));
    controls->addWidget(m_threadsSpin);

    controls->addStretch();
    mainLayout()->insertLayout(0, controls);

    // Model selection mirrors the batch tab.
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { updateModelPathFromCombo(); });
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { emit deviceSelectionChanged(deviceId()); });
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int v) { emit threadCountChanged(v); });
}

QString RMBGLiveWidget::modelFilename() const {
    return m_modelCombo ? m_modelCombo->currentData().toString() : QString();
}

QString RMBGLiveWidget::deviceId() const {
    return m_deviceCombo ? m_deviceCombo->currentData().toString()
                         : QStringLiteral("auto");
}

int RMBGLiveWidget::threadCount() const {
    return m_threadsSpin ? m_threadsSpin->value() : 0;
}

QString RMBGLiveWidget::resolveModelPath() const {
    const QString selection = modelFilename();
    if (selection.isEmpty()) return QString();
    const QFileInfo selectedFile(selection);
    if (selectedFile.isAbsolute()) return selectedFile.absoluteFilePath();
    const QString dir = RMBGHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return QDir(dir).filePath(selection);
}

void RMBGLiveWidget::setModelPath(const QString& path) {
    m_config.modelPath = path;
    // Sync the internal combo: select the matching entry or add a custom one.
    if (m_modelCombo) {
        const QString normalizedPath = QFileInfo(path).absoluteFilePath();
        int idx = -1;
        for (int i = 0; i < m_modelCombo->count(); ++i) {
            const QString stored = m_modelCombo->itemData(i).toString();
            const QString candidate =
                    QFileInfo(stored).isAbsolute()
                            ? QFileInfo(stored).absoluteFilePath()
                            : QDir(RMBGHelpers::modelCacheDir())
                                      .absoluteFilePath(stored);
            if (candidate == normalizedPath) {
                idx = i;
                break;
            }
        }
        if (idx >= 0) {
            m_syncingModelControls = true;
            m_modelCombo->setCurrentIndex(idx);
            m_syncingModelControls = false;
        } else if (!path.isEmpty() && QFileInfo::exists(path)) {
            m_syncingModelControls = true;
            m_modelCombo->blockSignals(true);
            m_modelCombo->addItem(QFileInfo(path).fileName(), path);
            m_modelCombo->setCurrentIndex(m_modelCombo->count() - 1);
            m_modelCombo->blockSignals(false);
            m_syncingModelControls = false;
        }
    }
}

void RMBGLiveWidget::setDevice(const QString& device) {
    m_config.device = device;
    if (m_deviceCombo) {
        const int idx = m_deviceCombo->findData(device);
        if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    }
}

void RMBGLiveWidget::setThreads(int threads) {
    m_config.threads = threads;
    if (m_threadsSpin) m_threadsSpin->setValue(threads);
}

void RMBGLiveWidget::rebuildModelCombo(const QStringList& labels,
                                       const QStringList& filenames,
                                       const QString& currentFilename) {
    m_syncingModelControls = true;
    m_modelCombo->clear();
    for (int i = 0; i < labels.size(); ++i) {
        m_modelCombo->addItem(labels.at(i), filenames.at(i));
    }
    const int idx = m_modelCombo->findData(currentFilename);
    if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    m_syncingModelControls = false;
    updateModelPathFromCombo();
}

void RMBGLiveWidget::rebuildDeviceCombo(const QComboBox* sourceDeviceCombo) {
    if (!sourceDeviceCombo) return;
    m_syncingModelControls = true;
    m_deviceCombo->clear();
    for (int i = 0; i < sourceDeviceCombo->count(); ++i) {
        m_deviceCombo->addItem(sourceDeviceCombo->itemText(i),
                               sourceDeviceCombo->itemData(i));
    }
    if (m_deviceCombo->count() > 0) m_deviceCombo->setCurrentIndex(0);
    m_syncingModelControls = false;
}

void RMBGLiveWidget::syncModelControlsFrom(const QComboBox* modelCombo,
                                           const QComboBox* deviceCombo,
                                           const QSpinBox* threadsSpin) {
    if (!modelCombo || !deviceCombo || !threadsSpin) return;
    const QString currentModel = modelCombo->currentData().toString();
    const QString currentDevice = deviceCombo->currentData().toString();
    m_syncingModelControls = true;
    m_modelCombo->clear();
    for (int i = 0; i < modelCombo->count(); ++i) {
        m_modelCombo->addItem(modelCombo->itemText(i), modelCombo->itemData(i));
    }
    m_deviceCombo->clear();
    for (int i = 0; i < deviceCombo->count(); ++i) {
        m_deviceCombo->addItem(deviceCombo->itemText(i),
                               deviceCombo->itemData(i));
    }
    const int modelIndex = m_modelCombo->findData(currentModel);
    if (modelIndex >= 0) m_modelCombo->setCurrentIndex(modelIndex);
    const int deviceIndex = m_deviceCombo->findData(currentDevice);
    if (deviceIndex >= 0) m_deviceCombo->setCurrentIndex(deviceIndex);
    m_threadsSpin->setRange(threadsSpin->minimum(), threadsSpin->maximum());
    m_threadsSpin->setValue(threadsSpin->value());
    m_syncingModelControls = false;
    m_config.device = deviceId();
    m_config.threads = threadCount();
    updateModelPathFromCombo();
}

void RMBGLiveWidget::updateModelPathFromCombo() {
    if (m_syncingModelControls) return;
    m_config.modelPath = resolveModelPath();
    emit modelSelectionChanged(modelFilename());
}

void RMBGLiveWidget::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRMBG/live"));
    m_threadsSpin->setValue(
            settings.value(QStringLiteral("threads"), 0).toInt());
    settings.endGroup();
}

void RMBGLiveWidget::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRMBG/live"));
    settings.setValue(QStringLiteral("threads"), m_threadsSpin->value());
    settings.endGroup();
}

// ---- video_base hooks -----------------------------------------------------

bool RMBGLiveWidget::onPrepareStream() {
    // Model must exist before the stream starts.
    if (m_config.modelPath.isEmpty() ||
        !QFileInfo::exists(m_config.modelPath)) {
        emit logMessage(
                tr("[RMBG] Model not available — download it in "
                   "the Image tab first."));
        return false;
    }
#ifdef AICore_ENABLED
    if (aicore_rmbg_warmup_backend(m_config.device.toUtf8().constData()) != 0) {
        emit logMessage(
                tr("[RMBG] Backend unavailable, falling back to "
                   "CPU for this stream."));
        m_config.device = QStringLiteral("cpu");
    }
#endif
    return true;
}

void RMBGLiveWidget::onFrameDecoded(cv::Mat& frame, int frameIndex) {
    Q_UNUSED(frameIndex);
    // Inference paces itself: frames decoded while the worker is busy are
    // skipped (the overlay lags 1-2 frames behind, imperceptible for a
    // background-removal preview).  The RGB conversion only runs when a job
    // is actually submitted — it is a full-frame copy.
    if (m_inferBusy) return;

#ifdef HAS_OPENCV_FACE_CAPTURE
    // The model resamples its input to a fixed square (input_size, e.g.
    // 1024x1024) regardless of the source resolution, so frames larger
    // than this edge only inflate the preprocess (full-frame RGBA +
    // normalize), the postprocess (bicubic alpha upsample) and the
    // cross-thread copies — inference time itself is unchanged.  Downscale
    // once in the cv domain (INTER_AREA) before the RGB copy: the alpha
    // matte never carries more than input_size^2 real detail, so the mask
    // quality is identical.  Captured snapshots are capped at this edge.
    constexpr int kInferMaxEdge = 1280;
    cv::Mat inferFrame = frame;
    cv::Mat downscaled;
    if (std::max(frame.cols, frame.rows) > kInferMaxEdge) {
        const double scale = static_cast<double>(kInferMaxEdge) /
                             std::max(frame.cols, frame.rows);
        cv::resize(frame, downscaled,
                   cv::Size(cvRound(frame.cols * scale),
                            cvRound(frame.rows * scale)),
                   0, 0, cv::INTER_AREA);
        inferFrame = downscaled;
    }
    const QImage rgb = VideoPlaybackWidget::cvMatToQImage(inferFrame)
                               .convertToFormat(QImage::Format_RGB888);
#else
    QImage rgb(frame.cols, frame.rows, QImage::Format_RGB888);
#endif
    if (rgb.isNull()) return;

    // AICore owns model-size preprocessing. inferFrame keeps the decoded
    // aspect ratio; the foreground result aligns with the displayed frame.
    submitInferJob(rgb);
}

void RMBGLiveWidget::onDisplayFrame(QImage& display, int frameIndex) {
    Q_UNUSED(frameIndex);
    // Cache the pre-overlay frame (implicit sharing; the QPainter work in
    // applyLiveComposite detaches `display`, leaving the cache untouched)
    // so onInferComplete can repaint immediately with a fresh mask.
    m_lastDisplayFrame = display;
    applyLiveComposite(display);
}

void RMBGLiveWidget::submitInferJob(const QImage& rgb) {
    if (!m_inferWorker || m_inferBusy) return;
    m_inferBusy = true;
    m_inferSubmitTime.restart();
    // First inference after a stream start can take tens of seconds on CPU
    // (model load + graph warmup); without an in-progress hint that silence
    // reads as "inference never runs".
    if (m_statusLabel) {
        m_statusLabel->setText(tr("inferring…"));
    }

    RMBGLiveInferWorker::Job job;
    job.rgb = rgb;
    job.modelPath = m_config.modelPath;
    job.device = m_config.device;
    job.threads = m_config.threads;
    job.alphaThreshold = m_config.alphaThreshold;
    job.generation = m_streamGeneration;
    QMetaObject::invokeMethod(m_inferWorker, "runJob", Qt::QueuedConnection,
                              Q_ARG(RMBGLiveInferWorker::Job, job));
}

void RMBGLiveWidget::onInferComplete(RMBGLiveInferWorker::Result result) {
    m_inferBusy = false;
    m_lastInferLatencyMs =
            m_inferSubmitTime.isValid() ? m_inferSubmitTime.elapsed() : -1;

    if (result.generation != m_streamGeneration) {
        // The stream looped / re-seeked while inference was running: the
        // mask belongs to a stale frame position and must not be composited.
        // Report it instead of dropping silently — slow CPU inference plus
        // a short looping clip makes every result stale, which previously
        // looked like "inference never runs".
        if (result.ok && m_statusLabel) {
            m_statusLabel->setText(
                    tr("mask dropped — stream advanced (infer %1)")
                            .arg(formatLatency(m_lastInferLatencyMs)));
        }
        return;
    }
    if (!isActive()) {
        return;
    }

    if (!result.ok) {
        // Full message goes to the log (appendLog); the live status line
        // mirrors the failure so the preview itself explains why no mask
        // is being composited.
        if (m_statusLabel) {
            m_statusLabel->setText(
                    tr("inference failed — %1").arg(result.error));
        }
        emit logMessage(
                tr("[RMBG] Live inference failed: %1").arg(result.error));
        return;
    }

    m_lastSnapshot = result.snapshot;
    m_hasSnapshot = true;
    // New mask: re-extract at display resolution on the next composite.
    m_lastResultImage = result.snapshot.resultImage;
    m_liveMask = QImage();
    // Surface the post-threshold foreground ratio (computed by the worker
    // after applyAlphaThreshold, i.e. exactly what the preview shows). With
    // the 0.5 default an uncooperative scene (e.g. aerial traffic footage)
    // can have its whole low-confidence mask cut away, and a full
    // checkerboard preview then looks exactly like "inference produced
    // nothing" — the percentage tells the two cases apart and points at
    // the remedy.
    const double fgPct = result.snapshot.foregroundRatio * 100.0;
    m_statusLabel->setText(
            fgPct < 1.0 ? tr("no foreground (%1%) — lower Alpha Threshold | "
                             "infer %2")
                                  .arg(fgPct, 0, 'f', 1)
                                  .arg(formatLatency(m_lastInferLatencyMs))
                        : tr("bg removed | fg %1% | infer %2")
                                  .arg(fgPct, 0, 'f', 1)
                                  .arg(formatLatency(m_lastInferLatencyMs)));
    emit snapshotUpdated(result.snapshot);
    repaintLivePreview();
}

void RMBGLiveWidget::applyLiveComposite(QImage& display) {
    if (m_lastResultImage.isNull() || display.isNull()) return;

    // Lazily scale the result to the display size and extract its alpha
    // channel. Thresholding was already applied by the worker on the
    // full-res alpha, so the mask carries the final cut-out.
    // Order matters: scaling an existing Format_Alpha8 image drops its
    // alpha data (verified on Qt 5.15 — every mask pixel becomes 0, so the
    // preview showed nothing but checkerboard). Scale in the ARGB domain
    // first, convert to Alpha8 afterwards.
    if (m_liveMask.isNull() || m_liveMask.size() != display.size()) {
        m_liveMask = m_lastResultImage
                             .scaled(display.size(), Qt::IgnoreAspectRatio,
                                     Qt::SmoothTransformation)
                             .convertToFormat(QImage::Format_Alpha8);
        if (m_liveMask.isNull()) return;
    }

    // Checkerboard backdrop at preview resolution (cached pattern), with
    // the mask-stamped frame blitted on top — a few tenths of a millisecond
    // at preview size, versus a full-res composite + smooth downscale per
    // inference frame in the old pipeline.
    QImage out = RMBGHelpers::makeCheckerboard(display.size());
    QImage fg = display.convertToFormat(QImage::Format_ARGB32);
    for (int y = 0; y < fg.height(); ++y) {
        QRgb* frow = reinterpret_cast<QRgb*>(fg.scanLine(y));
        const uchar* mrow = m_liveMask.constScanLine(y);
        for (int x = 0; x < fg.width(); ++x) {
            frow[x] = (frow[x] & 0x00ffffffu) |
                      (static_cast<quint32>(mrow[x]) << 24);
        }
    }
    QPainter p(&out);
    p.drawImage(0, 0, fg);
    p.end();
    display = out;
}

void RMBGLiveWidget::repaintLivePreview() {
    if (m_lastDisplayFrame.isNull() || !previewLabel()) return;
    // m_lastDisplayFrame is already scaled to the preview label by the
    // base-class pipeline — composite and swap directly.
    QImage frame = m_lastDisplayFrame;
    applyLiveComposite(frame);
    previewLabel()->setPixmap(QPixmap::fromImage(frame));
}

void RMBGLiveWidget::clearLiveOverlay() {
    m_lastResultImage = QImage();
    m_liveMask = QImage();
}

void RMBGLiveWidget::onVideoLooped() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RMBGLiveWidget::onStreamReset() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RMBGLiveWidget::onStreamResumed() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RMBGLiveWidget::onStreamStopping() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RMBGLiveWidget::onSourceChanged(InputSource source) {
    Q_UNUSED(source);
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RMBGLiveWidget::captureSnapshotToDb() {
    if (!m_hasSnapshot || m_lastSnapshot.resultImage.isNull()) return;
    emit captureToDbRequested(m_lastSnapshot);
}

void RMBGLiveWidget::shutdownInferThread() {
    if (!m_inferWorker || !m_inferThread) return;
    // QThread::finished is emitted from the worker thread itself during its
    // teardown. Because m_inferWorker lives on that thread, the queued
    // deleteLater connection is delivered as a DIRECT call before the event
    // loop stops draining — the worker is deleted before wait() returns, and
    // the explicit delete below then dereferences freed memory (segfault on
    // app exit). Drop the connection first so this function is the sole
    // owner of the worker's lifetime.
    disconnect(m_inferThread, &QThread::finished, m_inferWorker,
               &QObject::deleteLater);
    // releaseModel runs synchronously on the worker thread, so quit() below is
    // guaranteed to end the event loop; wait() can therefore never time out
    // (its upper bound is the single in-flight inference, which cannot be
    // interrupted). A bounded wait here would instead risk destroying a
    // still-running QThread from the widget destructor.
    QMetaObject::invokeMethod(m_inferWorker, "releaseModel",
                              Qt::BlockingQueuedConnection);
    m_inferThread->quit();
    m_inferThread->wait();
    delete m_inferWorker;
    m_inferWorker = nullptr;
}
