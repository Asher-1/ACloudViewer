// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOLiveWidget.h"

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

#include "YOLOLiveInferWorker.h"
#include "YOLOModelCatalog.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/yolo_capi.h"
#endif

namespace {

QString formatLatency(qint64 ms) {
    return ms >= 0 ? QStringLiteral("%1 ms").arg(ms) : QStringLiteral("--");
}

// Blend weight of the colorized depth layer over the camera frame. Below
// ~0.5 the depth signal gets hard to read; above ~0.8 the underlying scene
// (needed to judge alignment) disappears.
constexpr qreal kDepthOverlayOpacity = 0.65;

}  // namespace

YOLOLiveWidget::YOLOLiveWidget(QWidget* parent) : VideoPlaybackWidget(parent) {
    // ClockDriven (default): the decode clock advances the video and the
    // display tick paints the newest frame with the latest cached results.
    // Inference runs as an async side branch — it must not pace the display.
    setupUi();
    setPreviewFixedHeight(300);

    m_inferThread = new QThread(this);
    m_inferWorker = new YOLOLiveInferWorker;
    m_inferWorker->moveToThread(m_inferThread);
    connect(m_inferThread, &QThread::finished, m_inferWorker,
            &QObject::deleteLater);
    connect(m_inferWorker, &YOLOLiveInferWorker::inferComplete, this,
            &YOLOLiveWidget::onInferComplete, Qt::QueuedConnection);
    m_inferThread->start();
}

YOLOLiveWidget::~YOLOLiveWidget() {
    stopStream();
    shutdownInferThread();
}

bool YOLOLiveWidget::isAvailable() {
    return VideoPlaybackWidget::isAvailable();
}

void YOLOLiveWidget::setConfig(const Config& config) { m_config = config; }

void YOLOLiveWidget::setVideoFilePath(const QString& path, bool userChosen) {
    VideoPlaybackWidget::setVideoFilePath(path);
    m_videoPathUserChosen = userChosen;
}

void YOLOLiveWidget::setupUi() {
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

    controls->addWidget(new QLabel(tr("Conf:"), this));
    m_confSpin = new QDoubleSpinBox(this);
    m_confSpin->setRange(0.01, 1.0);
    m_confSpin->setSingleStep(0.05);
    m_confSpin->setValue(0.25);
    m_confSpin->setToolTip(tr("Confidence threshold (detect models)"));
    controls->addWidget(m_confSpin);

    controls->addWidget(new QLabel(tr("IoU:"), this));
    m_iouSpin = new QDoubleSpinBox(this);
    m_iouSpin->setRange(0.1, 1.0);
    m_iouSpin->setSingleStep(0.05);
    m_iouSpin->setValue(0.7);
    m_iouSpin->setToolTip(tr("NMS IoU threshold (detect models)"));
    controls->addWidget(m_iouSpin);

    controls->addStretch();
    mainLayout()->insertLayout(0, controls);

    // Model selection mirrors the batch tab.
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { updateModelPathFromCombo(); });
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { emit deviceSelectionChanged(deviceId()); });
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int v) { emit threadCountChanged(v); });
    connect(m_confSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double conf) {
                m_config.confThres = static_cast<float>(conf);
            });
    connect(m_iouSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double iou) {
                m_config.iouThres = static_cast<float>(iou);
            });
}

QString YOLOLiveWidget::modelFilename() const {
    return m_modelCombo ? m_modelCombo->currentData().toString() : QString();
}

QString YOLOLiveWidget::deviceId() const {
    return m_deviceCombo ? m_deviceCombo->currentData().toString()
                         : QStringLiteral("auto");
}

int YOLOLiveWidget::threadCount() const {
    return m_threadsSpin ? m_threadsSpin->value() : 0;
}

QString YOLOLiveWidget::resolveModelPath() const {
    const QString selection = modelFilename();
    if (selection.isEmpty()) return QString();
    const QFileInfo selectedFile(selection);
    if (selectedFile.isAbsolute()) return selectedFile.absoluteFilePath();
    const QString dir = YOLOHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return QDir(dir).filePath(selection);
}

void YOLOLiveWidget::setModelPath(const QString& path) {
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
                            : QDir(YOLOHelpers::modelCacheDir())
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

void YOLOLiveWidget::setDevice(const QString& device) {
    m_config.device = device;
    if (m_deviceCombo) {
        const int idx = m_deviceCombo->findData(device);
        if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    }
}

void YOLOLiveWidget::setThreads(int threads) {
    m_config.threads = threads;
    if (m_threadsSpin) m_threadsSpin->setValue(threads);
}

void YOLOLiveWidget::rebuildModelCombo(const QStringList& labels,
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

void YOLOLiveWidget::rebuildDeviceCombo(const QComboBox* sourceDeviceCombo) {
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

void YOLOLiveWidget::syncModelControlsFrom(const QComboBox* modelCombo,
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

void YOLOLiveWidget::updateModelPathFromCombo() {
    if (m_syncingModelControls) return;
    m_config.modelPath = resolveModelPath();
    emit modelSelectionChanged(modelFilename());
}

void YOLOLiveWidget::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qYOLO/live"));
    m_confSpin->setValue(
            settings.value(QStringLiteral("conf"), 0.25).toDouble());
    m_iouSpin->setValue(settings.value(QStringLiteral("iou"), 0.7).toDouble());
    m_threadsSpin->setValue(
            settings.value(QStringLiteral("threads"), 0).toInt());
    settings.endGroup();
}

void YOLOLiveWidget::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qYOLO/live"));
    settings.setValue(QStringLiteral("conf"), m_confSpin->value());
    settings.setValue(QStringLiteral("iou"), m_iouSpin->value());
    settings.setValue(QStringLiteral("threads"), m_threadsSpin->value());
    settings.endGroup();
}

// ---- video_base hooks -----------------------------------------------------

bool YOLOLiveWidget::onPrepareStream() {
    // Model must exist before the stream starts.
    if (m_config.modelPath.isEmpty() ||
        !QFileInfo::exists(m_config.modelPath)) {
        emit logMessage(
                tr("[YOLO] Model not available — download it in "
                   "the Image tab first."));
        return false;
    }
#ifdef AICore_ENABLED
    if (aicore_yolo_warmup_backend(m_config.device.toUtf8().constData()) != 0) {
        emit logMessage(
                tr("[YOLO] Backend unavailable, falling back to "
                   "CPU for this stream."));
        m_config.device = QStringLiteral("cpu");
    }
#endif
    return true;
}

void YOLOLiveWidget::onFrameDecoded(cv::Mat& frame, int frameIndex) {
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
    // here preserves one coordinate space for pixels, boxes, depth and DB
    // metadata, and avoids an extra resampling pass for small objects.
    // Implicit-shared copy — annotated rendering at capture time reuses it.
    m_lastSourceFrame = rgb;
    submitInferJob(rgb);
}

void YOLOLiveWidget::onDisplayFrame(QImage& display, int frameIndex) {
    Q_UNUSED(frameIndex);
    // Cache the pre-overlay frame (implicit sharing; the QPainter blit in
    // drawLiveOverlay detaches `display`, leaving the cache untouched) so
    // onInferComplete can repaint immediately with fresh results.
    m_lastDisplayFrame = display;
    drawLiveOverlay(display);
}

void YOLOLiveWidget::submitInferJob(const QImage& rgb) {
    if (!m_inferWorker || m_inferBusy) return;
    m_inferBusy = true;
    m_inferSubmitTime.restart();

    YOLOLiveInferWorker::Job job;
    job.rgb = rgb;
    job.generation = m_streamGeneration;
    job.modelPath = m_config.modelPath;
    job.device = m_config.device;
    job.threads = m_config.threads;
    job.confThres = m_config.confThres;
    job.iouThres = m_config.iouThres;
    job.topK = m_config.topK;
    QMetaObject::invokeMethod(m_inferWorker, "runJob", Qt::QueuedConnection,
                              Q_ARG(YOLOLiveInferWorker::Job, job));
}

void YOLOLiveWidget::onInferComplete(YOLOLiveInferWorker::Result result) {
    m_inferBusy = false;
    // Wall-clock submit→complete, kept for diagnostics only: it includes
    // queued-connection hops and GUI-thread congestion, so it can read far
    // above the model's own latency.
    m_lastInferLatencyMs =
            m_inferSubmitTime.isValid() ? m_inferSubmitTime.elapsed() : -1;

    if (result.generation != m_streamGeneration || !isActive()) {
        return;
    }

    if (!result.ok) {
        emit logMessage(
                tr("[YOLO] Live inference failed: %1").arg(result.error));
        return;
    }

    // A device switch (e.g. the requested GPU lease failed and yolo fell
    // back to CPU) is worth a log line — it is the number one cause of
    // "latency is way higher than the benchmark" reports.
    const QString resolvedDevice = (result.task == QStringLiteral("depth"))
                                           ? result.depth.resolvedDevice
                                           : result.detect.resolvedDevice;
    if (resolvedDevice != m_lastResolvedDevice) {
        emit logMessage(tr("[YOLO] Inference device: %1").arg(resolvedDevice));
        m_lastResolvedDevice = resolvedDevice;
    }

    m_lastTask = result.task;
    m_hasSnapshot = true;

    if (result.task == QStringLiteral("depth")) {
        // Colorize once at source resolution; the overlay layer below only
        // scales it to preview size on rebuild.
        m_lastDepth = result.depth;
        m_overlayDetections.clear();
        m_overlayDepthImage = YOLOHelpers::depthColorImage(
                result.depth.depthMap.constData(), result.depth.width,
                result.depth.height, result.depth.stats.minDepth,
                result.depth.stats.p95Depth);
        // Show the MODEL latency (preprocess + forward + postprocess inside
        // aicore_yolo_depth_rgb) — same scope as the static-image benchmark.
        const qint64 modelMs =
                result.depth.runtimeMs >= 0.0
                        ? static_cast<qint64>(result.depth.runtimeMs)
                        : m_lastInferLatencyMs;
        m_statusLabel->setText(
                tr("Depth %1x%2 | %3-%4 m | infer %5 (%6)")
                        .arg(result.depth.width)
                        .arg(result.depth.height)
                        .arg(result.depth.stats.minDepth, 0, 'f', 1)
                        .arg(result.depth.stats.p95Depth, 0, 'f', 1)
                        .arg(formatLatency(modelMs))
                        .arg(result.depth.resolvedDevice));
        emit depthSnapshotUpdated(result.depth);
        ++m_overlayGeneration;
        repaintLivePreview();
        return;
    }

    m_lastSnapshot = result.detect;
    const qint64 modelMs =
            result.detect.runtimeMs >= 0.0
                    ? static_cast<qint64>(result.detect.runtimeMs)
                    : m_lastInferLatencyMs;
    m_statusLabel->setText(tr("Objects: %1 | infer %2 (%3)")
                                   .arg(result.detect.detections.size())
                                   .arg(formatLatency(modelMs))
                                   .arg(result.detect.resolvedDevice));
    emit snapshotUpdated(result.detect);
    // New detections: update overlay data and invalidate the layer cache;
    // the immediate repaint below rebuilds it at preview resolution.
    m_overlayDepthImage = QImage();
    m_overlayDetections = result.detect.detections;
    m_overlaySourceSize = m_lastSourceFrame.size();
    ++m_overlayGeneration;
    repaintLivePreview();
}

void YOLOLiveWidget::rebuildOverlayLayer(const QSize& displaySize) {
    m_overlayLayer = QImage();
    if (displaySize.isEmpty()) {
        return;
    }

    // Depth layer: the colorized map blended over the camera frame — one
    // premultiplied blit per display tick; no per-pixel work on the GUI
    // thread.
    if (!m_overlayDepthImage.isNull()) {
        QImage layer(displaySize, QImage::Format_ARGB32_Premultiplied);
        layer.fill(Qt::transparent);
        QPainter p(&layer);
        p.setOpacity(kDepthOverlayOpacity);
        p.drawImage(layer.rect(), m_overlayDepthImage);
        p.setOpacity(1.0);
        p.end();
        m_overlayLayer = layer;
        m_overlayLayerSize = displaySize;
        m_overlayRenderedGeneration = m_overlayGeneration;
        return;
    }

    if (m_overlayDetections.isEmpty() || m_overlaySourceSize.isEmpty()) {
        return;
    }

    // Same rendering semantics as YOLOHelpers::drawDetections, but on the
    // small preview image with coordinates scaled from the source pixel
    // space.
    QImage layer(displaySize, QImage::Format_ARGB32_Premultiplied);
    layer.fill(Qt::transparent);
    const qreal sx = static_cast<qreal>(displaySize.width()) /
                     static_cast<qreal>(m_overlaySourceSize.width());
    const qreal sy = static_cast<qreal>(displaySize.height()) /
                     static_cast<qreal>(m_overlaySourceSize.height());

    QPainter p(&layer);
    p.setRenderHint(QPainter::Antialiasing, false);

    // Boxes + labels, scaled from the source pixel space.
    QFont font = p.font();
    font.setPixelSize(std::max(12, displaySize.height() / 60));
    p.setFont(font);
    for (const YOLODetection& d : m_overlayDetections) {
        const QColor color(YOLOHelpers::classColor(d.classId));
        QPen pen(color);
        pen.setWidth(2);
        p.setPen(pen);
        p.drawRect(QRectF(d.x1 * sx, d.y1 * sy, (d.x2 - d.x1) * sx,
                          (d.y2 - d.y1) * sy));

        const QString label = QStringLiteral("%1 %2")
                                      .arg(d.className)
                                      .arg(d.score, 0, 'f', 2);
        const QRectF labelRect(d.x1 * sx, d.y1 * sy - font.pixelSize() - 6,
                               std::max(20, label.size() * font.pixelSize()),
                               font.pixelSize() + 6);
        const QRectF bg = labelRect.adjusted(0, 0, 4, 2);
        p.fillRect(bg.intersected(QRectF(0, 0, displaySize.width(),
                                         displaySize.height())),
                   color);
        p.setPen(Qt::white);
        p.drawText(labelRect.adjusted(2, 3, -2, -2), label);
        p.setPen(pen);
    }
    p.end();

    m_overlayLayer = layer;
    m_overlayLayerSize = displaySize;
    m_overlayRenderedGeneration = m_overlayGeneration;
}

void YOLOLiveWidget::drawLiveOverlay(QImage& frame) {
    if (frame.isNull() ||
        (m_overlayDetections.isEmpty() && m_overlayDepthImage.isNull())) {
        return;
    }
    // Rebuild only when the results changed or the preview was resized;
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

void YOLOLiveWidget::repaintLivePreview() {
    if (m_lastDisplayFrame.isNull() || !previewLabel()) return;
    // m_lastDisplayFrame is already scaled to the preview label by the
    // base-class pipeline — overlay and swap directly.
    QImage frame = m_lastDisplayFrame;
    drawLiveOverlay(frame);
    previewLabel()->setPixmap(QPixmap::fromImage(frame));
}

void YOLOLiveWidget::clearLiveOverlay() {
    m_lastSourceFrame = QImage();
    m_overlayDetections.clear();
    m_overlaySourceSize = QSize();
    m_overlayDepthImage = QImage();
    m_overlayLayer = QImage();
}

void YOLOLiveWidget::onVideoLooped() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void YOLOLiveWidget::onStreamReset() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void YOLOLiveWidget::onStreamResumed() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void YOLOLiveWidget::onStreamStopping() {
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void YOLOLiveWidget::onSourceChanged(InputSource source) {
    Q_UNUSED(source);
    ++m_streamGeneration;
    m_hasSnapshot = false;
    clearLiveOverlay();
}

void YOLOLiveWidget::captureSnapshotToDb() {
    if (!m_hasSnapshot || m_lastSourceFrame.isNull()) return;
    // Annotated rendering is deferred to capture time (the live preview only
    // needs the downscaled overlay layer). The DB export requires
    // annotatedImage — render it once here from the cached source frame.

    if (m_lastTask == QStringLiteral("depth")) {
        if (m_lastDepth.annotatedImage.isNull() &&
            !m_lastDepth.depthMap.isEmpty()) {
            m_lastDepth.annotatedImage = YOLOHelpers::depthColorImage(
                    m_lastDepth.depthMap.constData(), m_lastDepth.width,
                    m_lastDepth.height, m_lastDepth.stats.minDepth,
                    m_lastDepth.stats.p95Depth);
            if (!m_lastDepth.annotatedImage.isNull()) {
                YOLOHelpers::drawDepthLegend(&m_lastDepth.annotatedImage,
                                             m_lastDepth.stats.minDepth,
                                             m_lastDepth.stats.p95Depth);
            }
        }
        if (m_lastDepth.annotatedImage.isNull()) return;
        emit depthCaptureToDbRequested(m_lastDepth);
        return;
    }

    if (m_lastSnapshot.detections.isEmpty()) return;
    if (m_lastSnapshot.annotatedImage.isNull()) {
        QImage annotated = m_lastSourceFrame;
        YOLOHelpers::drawDetections(&annotated, m_lastSnapshot.detections, 2);
        m_lastSnapshot.annotatedImage = annotated;
    }
    emit captureToDbRequested(m_lastSnapshot);
}

void YOLOLiveWidget::shutdownInferThread() {
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
