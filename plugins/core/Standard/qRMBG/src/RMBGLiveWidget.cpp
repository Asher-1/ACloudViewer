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
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QThread>
#include <QtConcurrent/QtConcurrentRun>

#include "RMBGModelCatalog.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/rmbg_capi.h"
#include "aicore/runtime_capi.h"
#endif

namespace {

constexpr int kInferEveryNthFrame = 5;   // video throttle
constexpr int kMaxInferWidth = 1280;     // downscale very large frames

QString formatLatency(qint64 ms) {
    return ms >= 0 ? QStringLiteral("%1 ms").arg(ms) : QStringLiteral("--");
}

}  // namespace

struct RMBGLiveWidget::InferJob {
    QImage rgb;        // RGB888 copy at inference resolution
    quint64 generation = 0;
    QString modelPath;
    QString device;
    int threads = 0;

    RMBGRunResult run() const;
};

RMBGRunResult RMBGLiveWidget::InferJob::run() const {
    RMBGRunResult result;
    result.resolvedDevice = device;
#ifdef AICore_ENABLED
    aicore_rmbg_options* opts = aicore_rmbg_options_new();
    if (!opts) return result;
    aicore_rmbg_options_set_device(opts, device.toUtf8().constData());
    aicore_rmbg_options_set_threads(opts, threads);
    aicore_rmbg_ctx* ctx = aicore_rmbg_load_opts(
            modelPath.toUtf8().constData(), opts);
    aicore_rmbg_options_free(opts);
    if (!ctx || !aicore_rmbg_is_ready(ctx)) {
        if (ctx) aicore_rmbg_free(ctx);
        return result;
    }
    char* info = aicore_rmbg_info_json(ctx);
    if (info) {
        RMBGHelpers::parseInfoJson(QByteArray(info), &result);
        aicore_rmbg_free_string(info);
    }

    QElapsedTimer timer;
    timer.start();
    uint8_t* png = nullptr;
    int pngLen = 0;
    const int rc = aicore_rmbg_remove_background_rgb(
            ctx, rgb.constBits(), rgb.width(), rgb.height(), &png, &pngLen);
    const double ms = static_cast<double>(timer.elapsed());
    if (rc == 0 && png && pngLen > 0) {
        result.resultImage = QImage::fromData(
                QByteArray(reinterpret_cast<const char*>(png), pngLen),
                "PNG");
        aicore_rmbg_free_buffer(png);
        RMBGHelpers::computeAlphaStats(result.resultImage, &result.alphaMean,
                                       &result.foregroundRatio);
    }
    aicore_rmbg_free(ctx);
    result.runtimeMs = ms;
    result.modelPath = modelPath;
    result.imageName = QStringLiteral("live");
    if (result.resolvedDevice.isEmpty()) result.resolvedDevice = device;
#endif
    return result;
}

RMBGLiveWidget::RMBGLiveWidget(QWidget* parent)
    : VideoPlaybackWidget(parent) {
    setupUi();
    setPreviewFixedHeight(300);
}

RMBGLiveWidget::~RMBGLiveWidget() {
    shutdownInferThread();
}

bool RMBGLiveWidget::isAvailable() {
    return VideoPlaybackWidget::isAvailable();
}

void RMBGLiveWidget::setConfig(const Config& config) {
    m_config = config;
}

void RMBGLiveWidget::setVideoFilePath(const QString& path, bool userChosen) {
    VideoPlaybackWidget::setVideoFilePath(path);
    m_videoPathUserChosen = userChosen;
}

void RMBGLiveWidget::setupUi() {
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
    const QString filename = modelFilename();
    if (filename.isEmpty()) return QString();
    const QString dir = RMBGHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

void RMBGLiveWidget::setModelPath(const QString& path) {
    m_config.modelPath = path;
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
        emit logMessage(tr("[RMBG] Model not available — download it in "
                           "the Image tab first."));
        return false;
    }
#ifdef AICore_ENABLED
    if (aicore_rmbg_warmup_backend(
                m_config.device.toUtf8().constData()) != 0) {
        emit logMessage(tr("[RMBG] Backend unavailable, falling back to "
                           "CPU for this stream."));
        m_config.device = QStringLiteral("cpu");
    }
#endif
    return true;
}

void RMBGLiveWidget::onFrameDecoded(cv::Mat& frame, int frameIndex) {
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
    if (inferRgb.width() > kMaxInferWidth) {
        inferRgb = inferRgb.scaledToWidth(kMaxInferWidth,
                                          Qt::SmoothTransformation);
    }

    m_inferBusy = true;
    m_lastSubmitFrameNum = frameIndex;
    m_inferSubmitTime.restart();

    auto* job = new InferJob;
    job->rgb = inferRgb;
    job->generation = m_streamGeneration;
    job->modelPath = m_config.modelPath;
    job->device = m_config.device;
    job->threads = m_config.threads;

    if (m_inferWatcher) {
        m_inferWatcher->disconnect(this);
        m_inferWatcher->deleteLater();
        m_inferWatcher = nullptr;
    }
    m_inferWatcher = new QFutureWatcher<RMBGRunResult>(this);
    connect(m_inferWatcher, &QFutureWatcher<RMBGRunResult>::finished, this,
            [this]() {
                if (!m_inferWatcher) return;
                onInferComplete(m_inferWatcher->result());
            });
    m_inferWatcher->setFuture(QtConcurrent::run([job]() {
        const RMBGRunResult result = job->run();
        delete job;
        return result;
    }));
}

void RMBGLiveWidget::onInferComplete(const RMBGRunResult& result) {
    m_inferBusy = false;
    m_lastInferLatencyMs =
            m_inferSubmitTime.isValid() ? m_inferSubmitTime.elapsed() : -1;

    if (result.resultImage.isNull()) {
        emit logMessage(tr("[RMBG] Live inference failed (model load or "
                           "backend error)."));
        return;
    }

    // Cache the RGBA result; onDisplayFrame composites it over the current
    // frame at the ORIGINAL frame resolution.
    m_overlayRgba = result.resultImage;
    m_overlayFrameNum = m_lastSubmitFrameNum;
    m_lastSnapshot = result;
    m_hasSnapshot = true;
    m_statusLabel->setText(tr("bg removed | infer %1")
                                   .arg(formatLatency(m_lastInferLatencyMs)));
    emit snapshotUpdated(result);
}

void RMBGLiveWidget::onDisplayFrame(QImage& display, int frameIndex) {
    (void)frameIndex;
    if (m_overlayRgba.isNull()) return;

    // Scale the RGBA result (original decode resolution) to the display
    // frame, then composite it over a checkerboard background.
    QImage scaled = m_overlayRgba;
    if (scaled.size() != display.size()) {
        scaled = scaled.scaled(display.size(), Qt::IgnoreAspectRatio,
                               Qt::SmoothTransformation);
    }
    const QImage composite = RMBGHelpers::compositeOnCheckerboard(scaled);
    if (composite.size() == display.size()) {
        display = composite.copy();
    }
}

void RMBGLiveWidget::onVideoLooped() {
    // The cached result still describes the same video content — keep it.
}

void RMBGLiveWidget::onStreamReset() {
    m_overlayRgba = QImage();
    m_hasSnapshot = false;
}

void RMBGLiveWidget::onStreamResumed() {
    m_overlayRgba = QImage();
}

void RMBGLiveWidget::onStreamStopping() {
    m_overlayRgba = QImage();
    m_hasSnapshot = false;
}

void RMBGLiveWidget::onSourceChanged(InputSource source) {
    (void)source;
    m_overlayRgba = QImage();
}

void RMBGLiveWidget::captureSnapshotToDb() {
    if (!m_hasSnapshot || m_lastSnapshot.resultImage.isNull()) return;
    emit captureToDbRequested(m_lastSnapshot);
}

void RMBGLiveWidget::shutdownInferThread() {
    if (m_inferWatcher) {
        m_inferWatcher->disconnect(this);
        m_inferWatcher->waitForFinished();
        m_inferWatcher->deleteLater();
        m_inferWatcher = nullptr;
    }
}
