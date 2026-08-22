// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RFDetrLiveWidget.h"

#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QFont>
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

#include "RFDetrLiveInferWorker.h"
#include "RFDetrModelCatalog.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/rfdetr_capi.h"
#endif

namespace {

QString formatLatency(qint64 ms) {
    return ms >= 0 ? QStringLiteral("%1 ms").arg(ms) : QStringLiteral("--");
}

}  // namespace

RFDetrLiveWidget::RFDetrLiveWidget(QWidget* parent)
    : VideoPlaybackWidget(parent) {
    // ClockDriven (default): the decode clock advances the video and the
    // display tick paints the newest frame with the latest cached detections.
    // Inference runs as an async side branch — it must not pace the display
    // (the old ConsumerDriven handshake capped playback at the inference
    // rate and broke playback speed control).
    setupUi();
    setPreviewFixedHeight(300);

    m_inferThread = new QThread(this);
    m_inferWorker = new RFDetrLiveInferWorker;
    m_inferWorker->moveToThread(m_inferThread);
    connect(m_inferThread, &QThread::finished, m_inferWorker,
            &QObject::deleteLater);
    connect(m_inferWorker, &RFDetrLiveInferWorker::inferComplete, this,
            &RFDetrLiveWidget::onInferComplete, Qt::QueuedConnection);
    connect(m_inferWorker, &RFDetrLiveInferWorker::modelInfoReady, this,
            &RFDetrLiveWidget::modelInfoReady, Qt::QueuedConnection);
    m_inferThread->start();
}

RFDetrLiveWidget::~RFDetrLiveWidget() {
    stopStream();
    shutdownInferThread();
}

bool RFDetrLiveWidget::isAvailable() {
    return VideoPlaybackWidget::isAvailable();
}

void RFDetrLiveWidget::setConfig(const Config& config) { m_config = config; }

void RFDetrLiveWidget::setVideoFilePath(const QString& path, bool userChosen) {
    VideoPlaybackWidget::setVideoFilePath(path);
    m_videoPathUserChosen = userChosen;
}

void RFDetrLiveWidget::setupUi() {
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

    controls->addWidget(new QLabel(tr("Threshold:"), this));
    m_thresholdSpin = new QDoubleSpinBox(this);
    m_thresholdSpin->setRange(0.01, 1.0);
    m_thresholdSpin->setSingleStep(0.05);
    m_thresholdSpin->setValue(0.5);
    controls->addWidget(m_thresholdSpin);

    controls->addStretch();
    mainLayout()->insertLayout(0, controls);

    // Model selection mirrors the batch tab.
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { updateModelPathFromCombo(); });
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { emit deviceSelectionChanged(deviceId()); });
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int v) { emit threadCountChanged(v); });
    connect(m_thresholdSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double threshold) {
                m_config.threshold = static_cast<float>(threshold);
            });
}

QString RFDetrLiveWidget::modelFilename() const {
    return m_modelCombo ? m_modelCombo->currentData().toString() : QString();
}

QString RFDetrLiveWidget::deviceId() const {
    return m_deviceCombo ? m_deviceCombo->currentData().toString()
                         : QStringLiteral("auto");
}

int RFDetrLiveWidget::threadCount() const {
    return m_threadsSpin ? m_threadsSpin->value() : 0;
}

QString RFDetrLiveWidget::resolveModelPath() const {
    const QString selection = modelFilename();
    if (selection.isEmpty()) return QString();
    const QFileInfo selectedFile(selection);
    if (selectedFile.isAbsolute()) return selectedFile.absoluteFilePath();
    const QString dir = RFDetrHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return QDir(dir).filePath(selection);
}

void RFDetrLiveWidget::setModelPath(const QString& path) {
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
                            : QDir(RFDetrHelpers::modelCacheDir())
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

void RFDetrLiveWidget::setDevice(const QString& device) {
    m_config.device = device;
    if (m_deviceCombo) {
        const int idx = m_deviceCombo->findData(device);
        if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    }
}

void RFDetrLiveWidget::setThreads(int threads) {
    m_config.threads = threads;
    if (m_threadsSpin) m_threadsSpin->setValue(threads);
}

void RFDetrLiveWidget::setClassFilter(const QVector<uint32_t>& classFilter) {
    m_config.classFilter = classFilter;
}

void RFDetrLiveWidget::rebuildModelCombo(const QStringList& labels,
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

void RFDetrLiveWidget::rebuildDeviceCombo(const QComboBox* sourceDeviceCombo) {
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

void RFDetrLiveWidget::syncModelControlsFrom(const QComboBox* modelCombo,
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

void RFDetrLiveWidget::updateModelPathFromCombo() {
    if (m_syncingModelControls) return;
    m_config.modelPath = resolveModelPath();
    emit modelSelectionChanged(modelFilename());
}

void RFDetrLiveWidget::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRFDetr/live"));
    m_thresholdSpin->setValue(
            settings.value(QStringLiteral("threshold"), 0.5).toDouble());
    m_threadsSpin->setValue(
            settings.value(QStringLiteral("threads"), 0).toInt());
    settings.endGroup();
}

void RFDetrLiveWidget::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRFDetr/live"));
    settings.setValue(QStringLiteral("threshold"), m_thresholdSpin->value());
    settings.setValue(QStringLiteral("threads"), m_threadsSpin->value());
    settings.endGroup();
}

// ---- video_base hooks -----------------------------------------------------

bool RFDetrLiveWidget::onPrepareStream() {
    // Model must exist before the stream starts.
    if (m_config.modelPath.isEmpty() ||
        !QFileInfo::exists(m_config.modelPath)) {
        emit logMessage(
                tr("[RF-DETR] Model not available — download it in "
                   "the Image tab first."));
        return false;
    }
#ifdef AICore_ENABLED
    if (aicore_rfdetr_warmup_backend(m_config.device.toUtf8().constData()) !=
        0) {
        emit logMessage(
                tr("[RF-DETR] Backend unavailable, falling back to "
                   "CPU for this stream."));
        m_config.device = QStringLiteral("cpu");
    }
#endif
    return true;
}

void RFDetrLiveWidget::onFrameDecoded(cv::Mat& frame, int frameIndex) {
    Q_UNUSED(frameIndex);
    // Inference paces itself: frames decoded while the worker is busy are
    // skipped (the overlay lags 1-2 frames behind, imperceptible at preview
    // size). The RGB conversion only runs when a job is actually submitted —
    // it is a full-frame copy.
    if (m_inferBusy) return;

#ifdef HAS_OPENCV_FACE_CAPTURE
    const QImage rgb =
            VideoPlaybackWidget::cvMatToQImage(frame).convertToFormat(
                    QImage::Format_RGB888);
#else
    QImage rgb(frame.cols, frame.rows, QImage::Format_RGB888);
#endif
    if (rgb.isNull()) return;

    // AICore owns model-size preprocessing. Keeping the decoded resolution
    // here preserves one coordinate space for pixels, boxes, masks and DB
    // metadata, and avoids an extra resampling pass for small objects.
    // Implicit-shared copy — annotated rendering at capture time reuses it.
    m_lastSourceFrame = rgb;
    submitInferJob(rgb);
}

void RFDetrLiveWidget::onDisplayFrame(QImage& display, int frameIndex) {
    Q_UNUSED(frameIndex);
    // Cache the pre-overlay frame (implicit sharing; the QPainter blit in
    // drawLiveOverlay detaches `display`, leaving the cache untouched) so
    // onInferComplete can repaint immediately with fresh detections.
    m_lastDisplayFrame = display;
    drawLiveOverlay(display);
}

void RFDetrLiveWidget::submitInferJob(const QImage& rgb) {
    if (!m_inferWorker || m_inferBusy) return;
    m_inferBusy = true;
    m_inferSubmitTime.restart();

    RFDetrLiveInferWorker::Job job;
    job.rgb = rgb;
    job.generation = m_streamGeneration;
    job.modelPath = m_config.modelPath;
    job.device = m_config.device;
    job.threads = m_config.threads;
    job.threshold = m_config.threshold;
    job.topK = m_config.topK;
    job.classFilter = m_config.classFilter;
    QMetaObject::invokeMethod(m_inferWorker, "runJob", Qt::QueuedConnection,
                              Q_ARG(RFDetrLiveInferWorker::Job, job));
}

void RFDetrLiveWidget::onInferComplete(RFDetrLiveInferWorker::Result result) {
    m_inferBusy = false;
    // Wall-clock submit→complete: displayed as the e2e number so the user
    // can compare it against the model latency (infer). It includes
    // queued-connection hops and GUI-thread congestion, so a large gap
    // signals pipeline stalls, not model slowness.
    m_lastInferLatencyMs =
            m_inferSubmitTime.isValid() ? m_inferSubmitTime.elapsed() : -1;

    if (result.generation != m_streamGeneration || !isActive()) {
        return;
    }

    if (!result.ok) {
        emit logMessage(
                tr("[RF-DETR] Live inference failed: %1").arg(result.error));
        return;
    }

    // A device switch (e.g. the requested GPU lease failed and rfdetr fell
    // back to CPU) is worth a log line — it is the number one cause of
    // "latency is way higher than the benchmark" reports.
    if (result.snapshot.resolvedDevice != m_lastResolvedDevice) {
        emit logMessage(tr("[RF-DETR] Inference device: %1")
                                .arg(result.snapshot.resolvedDevice));
        m_lastResolvedDevice = result.snapshot.resolvedDevice;
    }

    m_lastSnapshot = result.snapshot;
    m_hasSnapshot = true;
    // Show BOTH latencies so the user can distinguish a slow model from a
    // congested pipeline:
    //   infer = MODEL latency (preprocess + forward + postprocess inside
    //          aicore_rfdetr_detect_rgb_json) — the same scope the upstream
    //          rf-detr.cpp benchmark measures.
    //   e2e  = submit→complete wall clock — includes queued-connection hops
    //          (GUI→worker→GUI) and GUI-thread congestion. A large gap
    //          between infer and e2e signals pipeline stalls, not model
    //          slowness; a high infer signals a slow backend (CPU fallback,
    //          SMT thread oversubscription, etc.).
    const qint64 modelMs =
            result.snapshot.runtimeMs >= 0.0
                    ? static_cast<qint64>(result.snapshot.runtimeMs)
                    : m_lastInferLatencyMs;
    m_statusLabel->setText(tr("Objects: %1 | infer %2 / e2e %3 (%4)")
                                   .arg(result.snapshot.detections.size())
                                   .arg(formatLatency(modelMs))
                                   .arg(formatLatency(m_lastInferLatencyMs))
                                   .arg(result.snapshot.resolvedDevice));
    emit snapshotUpdated(result.snapshot);
    // New detections: update overlay data and invalidate the layer cache;
    // the immediate repaint below rebuilds it at preview resolution.
    m_overlayDetections = result.snapshot.detections;
    m_overlaySourceSize = m_lastSourceFrame.size();
    ++m_overlayGeneration;
    repaintLivePreview();
}

/* 3-tap separable Gaussian blur [1,2,1]/4 on Grayscale8.
 * Converts hard binary edges to a soft gradient for smooth bilinear upscale. */
static void gaussianBlurMask3(QImage& img) {
    if (img.format() != QImage::Format_Grayscale8) return;
    const int w = img.width(), h = img.height();
    if (w <= 2 || h <= 2) return;
    QImage tmp(w, h, QImage::Format_Grayscale8);
    for (int y = 0; y < h; ++y) {
        const uchar* s = img.constScanLine(y);
        uchar* d = tmp.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const int l = (x > 0) ? s[x - 1] : 0;
            const int m = s[x];
            const int r = (x < w - 1) ? s[x + 1] : 0;
            d[x] = (uint8_t)((l + m * 2 + r) / 4);
        }
    }
    for (int y = 0; y < h; ++y) {
        uchar* d = img.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const int t = (y > 0) ? tmp.constScanLine(y - 1)[x] : 0;
            const int m = tmp.constScanLine(y)[x];
            const int b = (y < h - 1) ? tmp.constScanLine(y + 1)[x] : 0;
            d[x] = (uint8_t)((t + m * 2 + b) / 4);
        }
    }
}

void RFDetrLiveWidget::rebuildOverlayLayer(const QSize& displaySize) {
    m_overlayLayer = QImage();
    if (m_overlayDetections.isEmpty() || m_overlaySourceSize.isEmpty() ||
        displaySize.isEmpty()) {
        return;
    }

    // Same rendering semantics as RFDetrHelpers::drawDetections, but on the
    // small preview image with coordinates scaled from the source pixel
    // space; masks stretch over the full rect in both spaces.
    QImage layer(displaySize, QImage::Format_ARGB32_Premultiplied);
    layer.fill(Qt::transparent);
    const qreal sx = static_cast<qreal>(displaySize.width()) /
                     static_cast<qreal>(m_overlaySourceSize.width());
    const qreal sy = static_cast<qreal>(displaySize.height()) /
                     static_cast<qreal>(m_overlaySourceSize.height());

    QPainter p(&layer);
    p.setRenderHint(QPainter::Antialiasing, false);
    // Smooth interpolation for the mask stretch (same as drawDetections).
    p.setRenderHint(QPainter::SmoothPixmapTransform, true);

    // Pass 1: mask tints — all detections accumulated into one composite
    // at mask resolution, then stretched with a single blit (replaces N
    // separate drawImage calls).  Proportional alpha from the Gaussian-
    // blurred mask preserves the soft gradient through bilinear upscale.
    QImage composite;
    for (const RFDetrDetection& d : m_overlayDetections) {
        if (d.maskRaw.isEmpty() || d.maskWidth <= 0 || d.maskHeight <= 0)
            continue;
        QImage mask(d.maskWidth, d.maskHeight, QImage::Format_Grayscale8);
        std::memcpy(mask.bits(), d.maskRaw.constData(),
                    static_cast<size_t>(d.maskWidth) * d.maskHeight);
        if (mask.isNull()) continue;
        gaussianBlurMask3(mask);

        if (composite.isNull()) {
            composite =
                    QImage(mask.size(), QImage::Format_ARGB32_Premultiplied);
            composite.fill(Qt::transparent);
        }
        const QRgb tintPre = QColor(RFDetrHelpers::classColor(d.classId)).rgb();
        for (int y = 0; y < mask.height(); ++y) {
            const uchar* mrow = mask.constScanLine(y);
            QRgb* crow = reinterpret_cast<QRgb*>(composite.scanLine(y));
            for (int x = 0; x < mask.width(); ++x) {
                const int mv = mrow[x];
                if (mv <= 1) continue;
                if (mv > qAlpha(crow[x])) {
                    crow[x] = qRgba(qRed(tintPre) * mv / 255,
                                    qGreen(tintPre) * mv / 255,
                                    qBlue(tintPre) * mv / 255, mv);
                }
            }
        }
    }
    if (!composite.isNull()) {
        p.setOpacity(0.3f);
        p.drawImage(layer.rect(), composite);
        p.setOpacity(1.0);
    }

    // Pass 2: boxes + labels, scaled from the source pixel space.
    QFont font = p.font();
    font.setPixelSize(std::max(12, displaySize.height() / 60));
    p.setFont(font);
    for (const RFDetrDetection& d : m_overlayDetections) {
        const QColor color(RFDetrHelpers::classColor(d.classId));
        QPen pen(color);
        pen.setWidth(2);
        p.setPen(pen);
        p.drawRect(QRectF(d.x1 * sx, d.y1 * sy, (d.x2 - d.x1) * sx,
                          (d.y2 - d.y1) * sy));

        const QString label = QStringLiteral("%1 %2")
                                      .arg(d.className)
                                      .arg(d.score, 0, 'f', 2);
        // Keep the banner fully inside the preview (same rule as
        // RFDetrHelpers::drawDetections): clamp horizontally, flip below
        // the box top when the box hugs the top edge.
        QRect labelRect(static_cast<int>(d.x1 * sx),
                        static_cast<int>(d.y1 * sy) - font.pixelSize() - 6,
                        std::max(20, label.size() * font.pixelSize()),
                        font.pixelSize() + 6);
        labelRect.setWidth(std::min(labelRect.width(),
                                    std::max(20, displaySize.width() - 4)));
        labelRect.moveLeft(std::clamp(
                labelRect.left(), 2,
                std::max(2, displaySize.width() - labelRect.width() - 2)));
        if (labelRect.top() < 2) {
            labelRect.moveTop(static_cast<int>(d.y1 * sy) + 2);
        }
        labelRect.moveTop(std::min(
                labelRect.top(),
                std::max(2, displaySize.height() - labelRect.height() - 2)));
        p.fillRect(labelRect.adjusted(0, 0, 4, 2), color);
        p.setPen(Qt::white);
        p.drawText(labelRect.adjusted(2, 3, -2, -2), label);
        p.setPen(pen);
    }
    p.end();

    m_overlayLayer = layer;
    m_overlayLayerSize = displaySize;
    m_overlayRenderedGeneration = m_overlayGeneration;
}

void RFDetrLiveWidget::drawLiveOverlay(QImage& frame) {
    if (frame.isNull() || m_overlayDetections.isEmpty()) return;
    // Rebuild only when the detections changed or the preview was resized;
    // every display tick then pays just one premultiplied blit.
    if (m_overlayLayer.isNull() || m_overlayLayerSize != frame.size() ||
        m_overlayRenderedGeneration != m_overlayGeneration) {
        rebuildOverlayLayer(frame.size());
        if (m_overlayLayer.isNull()) return;
    }
    QPainter p(&frame);
    p.drawImage(0, 0, m_overlayLayer);
    p.end();
}

void RFDetrLiveWidget::repaintLivePreview() {
    if (m_lastDisplayFrame.isNull() || !previewLabel()) return;
    // m_lastDisplayFrame is already scaled to the preview label by the
    // base-class pipeline — overlay and swap directly.
    QImage frame = m_lastDisplayFrame;
    drawLiveOverlay(frame);
    previewLabel()->setPixmap(QPixmap::fromImage(frame));
}

void RFDetrLiveWidget::clearLiveOverlay() {
    m_lastSourceFrame = QImage();
    m_overlayDetections.clear();
    m_overlaySourceSize = QSize();
    m_overlayLayer = QImage();
}

void RFDetrLiveWidget::onVideoLooped() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RFDetrLiveWidget::onStreamReset() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RFDetrLiveWidget::onStreamResumed() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RFDetrLiveWidget::onStreamStopping() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RFDetrLiveWidget::onSourceChanged(InputSource source) {
    Q_UNUSED(source);
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void RFDetrLiveWidget::captureSnapshotToDb() {
    if (!m_hasSnapshot || m_lastSnapshot.detections.isEmpty()) return;
    // Annotated rendering is deferred to capture time (the live preview only
    // needs the downscaled overlay layer). qRFDetr's DB export requires
    // annotatedImage — render it once here from the cached source frame.
    if (m_lastSnapshot.annotatedImage.isNull() && !m_lastSourceFrame.isNull()) {
        QImage annotated = m_lastSourceFrame;
        RFDetrHelpers::drawDetections(&annotated, m_lastSnapshot.detections,
                                      0.3f, 2);
        m_lastSnapshot.annotatedImage = annotated;
    }
    emit captureToDbRequested(m_lastSnapshot);
}

void RFDetrLiveWidget::shutdownInferThread() {
    if (!m_inferWorker || !m_inferThread) {
        return;
    }
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
