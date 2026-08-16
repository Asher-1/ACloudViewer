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
#include <QFormLayout>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QThread>
#include <QtConcurrent/QtConcurrentRun>

#include "RFDetrModelCatalog.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/runtime_capi.h"
#endif

namespace {

constexpr int kInferEveryNthFrame = 5;   // video throttle
constexpr int kMaxInferWidth = 1280;     // downscale very large frames

QString formatLatency(qint64 ms) {
    return ms >= 0 ? QStringLiteral("%1 ms").arg(ms) : QStringLiteral("--");
}

}  // namespace

struct RFDetrLiveWidget::InferJob {
    QImage rgb;        // RGB888 copy at inference resolution
    float scale = 1.0f;  // original->rgb scale
    quint64 generation = 0;
    QString modelPath;
    QString device;
    int threads = 0;
    float threshold = 0.5f;
    uint32_t topK = 300;

    RFDetrRunResult run() const;
};

RFDetrRunResult RFDetrLiveWidget::InferJob::run() const {
    RFDetrRunResult result;
    result.resolvedDevice = device;
#ifdef AICore_ENABLED
    aicore_rfdetr_options* opts = aicore_rfdetr_options_new();
    if (!opts) return result;
    aicore_rfdetr_options_set_device(opts, device.toUtf8().constData());
    aicore_rfdetr_options_set_threads(opts, threads);
    aicore_rfdetr_ctx* ctx = aicore_rfdetr_load_opts(
            modelPath.toUtf8().constData(), opts);
    aicore_rfdetr_options_free(opts);
    if (!ctx || !aicore_rfdetr_is_ready(ctx)) {
        if (ctx) aicore_rfdetr_free(ctx);
        return result;
    }
    QElapsedTimer timer;
    timer.start();
    char* json = aicore_rfdetr_detect_rgb_json(
            ctx, rgb.constBits(), rgb.width(), rgb.height(), threshold, topK);
    const double ms = static_cast<double>(timer.elapsed());
    if (json) {
        RFDetrHelpers::parseDetectionsJson(QByteArray(json), &result);
        aicore_rfdetr_free_string(json);
    }
    if (aicore_rfdetr_context_has_segmentation(ctx)) {
        const int n = aicore_rfdetr_detection_count(ctx);
        for (int i = 0; i < n; ++i) {
            const int len = aicore_rfdetr_detection_mask_png(ctx, i, nullptr, 0);
            if (len <= 0) continue;
            QByteArray png;
            png.resize(len);
            if (aicore_rfdetr_detection_mask_png(
                        ctx, i, reinterpret_cast<unsigned char*>(png.data()),
                        len) == len) {
                result.detections[i].maskPng = png;
            }
        }
    }
    aicore_rfdetr_free(ctx);
    result.runtimeMs = ms;
    result.modelPath = modelPath;
    result.resolvedDevice = device;
    result.imageName = QStringLiteral("live");
#endif
    return result;
}

RFDetrLiveWidget::RFDetrLiveWidget(QWidget* parent)
    : VideoPlaybackWidget(parent) {
    setupUi();
    setPreviewFixedHeight(300);
}

RFDetrLiveWidget::~RFDetrLiveWidget() {
    shutdownInferThread();
}

bool RFDetrLiveWidget::isAvailable() {
    return VideoPlaybackWidget::isAvailable();
}

void RFDetrLiveWidget::setConfig(const Config& config) {
    m_config = config;
}

void RFDetrLiveWidget::setVideoFilePath(const QString& path, bool userChosen) {
    VideoPlaybackWidget::setVideoFilePath(path);
    m_videoPathUserChosen = userChosen;
}

void RFDetrLiveWidget::setupUi() {
    m_previewLabel = previewLabel();
    m_statusLabel = statusLabel();

    auto* controls = new QHBoxLayout;
    controls->setContentsMargins(0, 4, 0, 0);

    controls->addWidget(new QLabel(tr("Model:"), this));
    m_modelCombo = new QComboBox(this);
    m_modelCombo->setMinimumWidth(200);
    controls->addWidget(m_modelCombo);

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
            this, [this](int) {
                emit deviceSelectionChanged(deviceId());
            });
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int v) { emit threadCountChanged(v); });
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
    const QString filename = modelFilename();
    if (filename.isEmpty()) return QString();
    const QString dir = RFDetrHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

void RFDetrLiveWidget::setModelPath(const QString& path) {
    m_config.modelPath = path;
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
    rebuildModelCombo(QStringList(), QStringList(), QString());
    m_syncingModelControls = true;
    m_modelCombo->clear();
    for (int i = 0; i < modelCombo->count(); ++i) {
        m_modelCombo->addItem(modelCombo->itemText(i),
                              modelCombo->itemData(i));
    }
    m_deviceCombo->clear();
    for (int i = 0; i < deviceCombo->count(); ++i) {
        m_deviceCombo->addItem(deviceCombo->itemText(i),
                               deviceCombo->itemData(i));
    }
    m_threadsSpin->setRange(threadsSpin->minimum(), threadsSpin->maximum());
    m_threadsSpin->setValue(threadsSpin->value());
    m_syncingModelControls = false;
}

void RFDetrLiveWidget::updateModelPathFromCombo() {
    if (m_syncingModelControls) return;
    m_config.modelPath = resolveModelPath();
    emit modelSelectionChanged(modelFilename());
}

void RFDetrLiveWidget::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRFDetr/live"));
    m_thresholdSpin->setValue(settings.value(
            QStringLiteral("threshold"), 0.5).toDouble());
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
        emit logMessage(tr("[RF-DETR] Model not available — download it in "
                           "the Image tab first."));
        return false;
    }
#ifdef AICore_ENABLED
    if (aicore_rfdetr_warmup_backend(
                m_config.device.toUtf8().constData()) != 0) {
        emit logMessage(tr("[RF-DETR] Backend unavailable, falling back to "
                           "CPU for this stream."));
        m_config.device = QStringLiteral("cpu");
    }
#endif
    return true;
}

void RFDetrLiveWidget::onFrameDecoded(cv::Mat& frame, int frameIndex) {
    // Throttle: run inference every kInferEveryNthFrame video frames (and on
    // every camera frame — camera fps is already low enough).
    const bool isVideo = inputSource() == InputSource::VideoFile;
    if (isVideo && (frameIndex % kInferEveryNthFrame != 0)) return;
    if (m_inferBusy) return;

#ifdef HAS_OPENCV_FACE_CAPTURE
    const QImage rgb = VideoPlaybackWidget::cvMatToQImage(frame)
                               .convertToFormat(QImage::Format_RGB888);
#else
    QImage rgb(frame.cols, frame.rows, QImage::Format_RGB888);
#endif
    if (rgb.isNull()) return;

    // Downscale very large frames to bound inference time.
    QImage inferRgb = rgb;
    float scale = 1.0f;
    if (inferRgb.width() > kMaxInferWidth) {
        scale = static_cast<float>(kMaxInferWidth) / inferRgb.width();
        inferRgb = inferRgb.scaledToWidth(kMaxInferWidth, Qt::SmoothTransformation);
    }

    m_inferBusy = true;
    m_lastSubmitFrameNum = frameIndex;
    m_lastFrameSize = QSize(frame.cols, frame.rows);
    m_inferSubmitTime.restart();

    auto* job = new InferJob;
    job->rgb = inferRgb;
    job->scale = scale;
    job->generation = m_streamGeneration;
    job->modelPath = m_config.modelPath;
    job->device = m_config.device;
    job->threads = m_config.threads;
    job->threshold = m_config.threshold;
    job->topK = m_config.topK;

    if (m_inferWatcher) {
        m_inferWatcher->disconnect(this);
        m_inferWatcher->deleteLater();
        m_inferWatcher = nullptr;
    }
    m_inferWatcher = new QFutureWatcher<RFDetrRunResult>(this);
    connect(m_inferWatcher, &QFutureWatcher<RFDetrRunResult>::finished, this,
            [this]() {
                if (!m_inferWatcher) return;
                onInferComplete(m_inferWatcher->result());
            });
    m_inferWatcher->setFuture(QtConcurrent::run([job]() {
        const RFDetrRunResult result = job->run();
        delete job;
        return result;
    }));
}

void RFDetrLiveWidget::onInferComplete(const RFDetrRunResult& result) {
    m_inferBusy = false;
    m_lastInferLatencyMs =
            m_inferSubmitTime.isValid() ? m_inferSubmitTime.elapsed() : -1;

    if (result.detections.isEmpty() && result.modelVariant.isEmpty()) {
        emit logMessage(tr("[RF-DETR] Live inference failed (model load or "
                           "backend error)."));
        return;
    }

    // Cache overlay + snapshot at the ORIGINAL frame resolution so
    // drawLiveOverlay can scale coordinates exactly.
    m_overlayDetections = result.detections;
    m_overlayFrameNum = m_lastSubmitFrameNum;
    m_lastSnapshot = result;
    m_hasSnapshot = true;
    m_statusLabel->setText(tr("Objects: %1 | infer %2")
                                   .arg(result.detections.size())
                                   .arg(formatLatency(m_lastInferLatencyMs)));
    emit snapshotUpdated(result);
}

void RFDetrLiveWidget::onDisplayFrame(QImage& display, int frameIndex) {
    (void)frameIndex;
    drawLiveOverlay(display);
}

void RFDetrLiveWidget::drawLiveOverlay(QImage& frame) {
    if (m_overlayDetections.isEmpty() || m_lastFrameSize.isEmpty()) return;
    // Coordinates from AICore are in inference-RGB pixels; scale them to the
    // display frame (which is the original decode resolution).
    const float sx = static_cast<float>(frame.width()) / m_lastFrameSize.width();
    const float sy =
            static_cast<float>(frame.height()) / m_lastFrameSize.height();
    QVector<RFDetrDetection> scaled;
    scaled.reserve(m_overlayDetections.size());
    for (const RFDetrDetection& d : m_overlayDetections) {
        RFDetrDetection s = d;
        s.x1 *= sx;
        s.y1 *= sy;
        s.x2 *= sx;
        s.y2 *= sy;
        scaled.append(s);
    }
    RFDetrHelpers::drawDetections(&frame, scaled, 0.3f, 2);
}

void RFDetrLiveWidget::onVideoLooped() {
    // Overlays from the previous loop iteration may be stale; keep them —
    // they still describe the same video content.
}

void RFDetrLiveWidget::onStreamReset() {
    m_overlayDetections.clear();
    m_hasSnapshot = false;
}

void RFDetrLiveWidget::onStreamResumed() {
    m_overlayDetections.clear();
}

void RFDetrLiveWidget::onStreamStopping() {
    m_overlayDetections.clear();
    m_hasSnapshot = false;
}

void RFDetrLiveWidget::onSourceChanged(InputSource source) {
    (void)source;
    m_overlayDetections.clear();
}

void RFDetrLiveWidget::captureSnapshotToDb() {
    if (!m_hasSnapshot || m_lastSnapshot.detections.isEmpty()) return;
    emit captureToDbRequested(m_lastSnapshot);
}

void RFDetrLiveWidget::shutdownInferThread() {
    if (m_inferWatcher) {
        m_inferWatcher->disconnect(this);
        m_inferWatcher->waitForFinished();
        m_inferWatcher->deleteLater();
        m_inferWatcher = nullptr;
    }
    if (m_inferThread) {
        m_inferThread->quit();
        m_inferThread->wait(2000);
        delete m_inferThread;
        m_inferThread = nullptr;
    }
}
