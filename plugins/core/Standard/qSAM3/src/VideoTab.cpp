// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoTab.h"

#ifdef HAS_OPENCV_FACE_CAPTURE

#include <aicore/sam3_capi.h>

#include "VideoFrameReader.h"
#include "VideoPlaybackWidget.h"  // cvMatToQImage

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QSlider>
#include <QVBoxLayout>

#include <algorithm>

namespace {
// Per-instance display colors, mirroring upstream examples/main_video.cpp.
constexpr const QColor kInstanceColors[] = {
        QColor(255, 51, 51),   QColor(51, 153, 255), QColor(51, 230, 76),
        QColor(255, 204, 26),  QColor(204, 76, 230), QColor(255, 128, 26),
        QColor(26, 230, 230),  QColor(230, 102, 153), QColor(128, 204, 51),
        QColor(76, 76, 255),   QColor(255, 153, 179), QColor(153, 255, 128),
};
constexpr int kNumColors = sizeof(kInstanceColors) / sizeof(kInstanceColors[0]);
}  // namespace

VideoTab::VideoTab(QWidget* parent) : QWidget(parent) {
    setupUi();
    populateModelCombo();
}

VideoTab::~VideoTab() {
    m_playTimer.stop();
    if (m_worker) {
        m_worker->requestCancel();
        m_worker->wait(5000);
        delete m_worker;
        m_worker = nullptr;
    }
    if (m_reader) {
        m_reader->release();
        delete m_reader;
        m_reader = nullptr;
    }
}

void VideoTab::setDevice(const QString& device) {
    const int idx = m_deviceCombo->findText(device, Qt::MatchStartsWith);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
}

void VideoTab::setupUi() {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 6, 4, 4);
    layout->setSpacing(6);

    // ── Row 1: mode + text prompt + playback controls ────────────────────
    auto* row1 = new QHBoxLayout();
    auto* modeLabel = new QLabel(tr("Mode:"));
    m_modeText = new QRadioButton(tr("Text"));
    m_modeBox = new QRadioButton(tr("Box"));
    m_modePoints = new QRadioButton(tr("Points"));
    m_modePoints->setChecked(true);
    auto* modeGroup = new QButtonGroup(this);
    modeGroup->addButton(m_modeText, 0);
    modeGroup->addButton(m_modeBox, 1);
    modeGroup->addButton(m_modePoints, 2);
    connect(modeGroup, QOverload<int>::of(&QButtonGroup::buttonClicked),
            this, &VideoTab::onModeChanged);

    m_textPrompt = new QLineEdit();
    m_textPrompt->setPlaceholderText(tr("Text prompt (SAM3 only)..."));
    m_textPrompt->setMinimumWidth(140);
    m_textPrompt->setEnabled(false);

    m_openBtn = new QPushButton(tr("Open video..."));
    m_playBtn = new QPushButton(tr("Play"));
    m_stepBtn = new QPushButton(tr("Step >>"));
    m_stepBtn->setEnabled(false);
    m_resetBtn = new QPushButton(tr("Reset"));
    m_resetBtn->setEnabled(false);

    row1->addWidget(modeLabel);
    row1->addWidget(m_modeText);
    row1->addWidget(m_modeBox);
    row1->addWidget(m_modePoints);
    row1->addSpacing(10);
    row1->addWidget(m_textPrompt, 1);
    row1->addSpacing(10);
    row1->addWidget(m_openBtn);
    row1->addWidget(m_playBtn);
    row1->addWidget(m_stepBtn);
    row1->addWidget(m_resetBtn);
    layout->addLayout(row1);

    // ── Row 2: model + device ────────────────────────────────────────────
    auto* row2 = new QHBoxLayout();
    auto* modelLabel = new QLabel(tr("Model:"));
    m_modelCombo = new QComboBox();
    m_modelCombo->setMinimumWidth(240);
    m_loadBtn = new QPushButton(tr("Load"));
    m_loadBtn->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: bold;"
            "  border: none; border-radius: 4px; padding: 5px 14px; }"
            "QPushButton:hover { background: #00796b; }");

    auto* deviceLabel = new QLabel(tr("Device:"));
    m_deviceCombo = new QComboBox();
    m_deviceCombo->addItems({"Auto", "CPU", "CUDA", "Vulkan"});
    m_backendLabel = new QLabel(tr("Backend: none"));
    m_backendLabel->setStyleSheet("color: #99b4d1;");

    row2->addWidget(modelLabel);
    row2->addWidget(m_modelCombo, 1);
    row2->addWidget(m_loadBtn);
    row2->addSpacing(16);
    row2->addWidget(deviceLabel);
    row2->addWidget(m_deviceCombo);
    row2->addSpacing(12);
    row2->addWidget(m_backendLabel);
    layout->addLayout(row2);

    // ── Canvas ────────────────────────────────────────────────────────────
    m_canvas = new VideoCanvas();
    m_canvas->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    layout->addWidget(m_canvas, 1);

    // ── Timeline ──────────────────────────────────────────────────────────
    m_timeline = new VideoTimeline();
    layout->addWidget(m_timeline);

    // ── Bottom row ────────────────────────────────────────────────────────
    auto* bottom = new QHBoxLayout();
    m_showMasks = new QCheckBox(tr("Show masks"));
    m_showMasks->setChecked(true);

    auto* speedLabel = new QLabel(tr("Speed:"));
    m_speedSlider = new QSlider(Qt::Horizontal);
    m_speedSlider->setRange(1, 40);  // 0.1x..4.0x, 0.1 steps
    m_speedSlider->setValue(10);
    m_speedSlider->setMaximumWidth(120);
    m_speedLabel = new QLabel(tr("1.0x"));
    m_speedLabel->setMinimumWidth(40);

    m_exportBtn = new QPushButton(tr("Export frame masks"));
    m_exportBtn->setEnabled(false);

    auto* instLabel = new QLabel(tr("Tracked instances:"));
    m_instanceLabel = new QLabel(tr("(none)"));
    m_instanceLabel->setStyleSheet("color: #999;");

    m_statusLabel = new QLabel(tr("Open a video and load a model to start."));
    m_statusLabel->setStyleSheet("color: #99ccff;");

    bottom->addWidget(m_showMasks);
    bottom->addSpacing(12);
    bottom->addWidget(speedLabel);
    bottom->addWidget(m_speedSlider);
    bottom->addWidget(m_speedLabel);
    bottom->addSpacing(12);
    bottom->addWidget(m_exportBtn);
    bottom->addSpacing(12);
    bottom->addWidget(instLabel);
    bottom->addWidget(m_instanceLabel, 1);
    bottom->addSpacing(12);
    bottom->addWidget(m_statusLabel, 1);
    layout->addLayout(bottom);

    // ── Connections ───────────────────────────────────────────────────────
    connect(m_openBtn, &QPushButton::clicked, this, &VideoTab::onOpenVideo);
    connect(m_loadBtn, &QPushButton::clicked, this, &VideoTab::onLoadModel);
    connect(m_playBtn, &QPushButton::clicked, this, &VideoTab::onPlayPause);
    connect(m_stepBtn, &QPushButton::clicked, this, &VideoTab::onStep);
    connect(m_resetBtn, &QPushButton::clicked, this, &VideoTab::onReset);
    connect(m_speedSlider, &QSlider::valueChanged, this, [this](int v) {
        m_speedLabel->setText(QString("%1x").arg(v / 10.0, 0, 'f', 1));
    });
    connect(m_showMasks, &QCheckBox::toggled, this,
            [this](bool) { updateCanvasInstances(); });
    connect(m_canvas, &VideoCanvas::boxDrawn, this, &VideoTab::onCanvasBox);
    connect(m_canvas, &VideoCanvas::instanceClicked, this,
            &VideoTab::onCanvasInstanceClicked);
    connect(m_canvas, &VideoCanvas::posPointAdded, this,
            &VideoTab::onCanvasPosPoint);
    connect(m_canvas, &VideoCanvas::negPointAdded, this,
            &VideoTab::onCanvasNegPoint);
    connect(m_timeline, &VideoTimeline::seekRequested, this,
            &VideoTab::onSeek);
    connect(&m_playTimer, &QTimer::timeout, this, &VideoTab::trackNextFrame);
}

void VideoTab::populateModelCombo() {
    const int n = aicore_sam3_model_count();
    for (int i = 0; i < n; ++i) {
        const auto* entry = aicore_sam3_model_at(i);
        if (!entry || !entry->filename) continue;
        m_modelCombo->addItem(
                QString("%1 (%2)").arg(entry->display_name)
                        .arg(entry->quant_note),
                entry->filename);
    }
    m_modelCombo->addItem(tr("Browse..."), QString("__browse__"));
}

QString VideoTab::modelPath() const {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty() || filename == "__browse__") return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QStringList searchDirs = {
            QDir::homePath() + "/.cache/cloudViewer/models/sam",
            QDir::homePath() + "/develop/code/github/dl/sam3-ggml/models",
    };
    for (const auto& dir : searchDirs) {
        const QString full = dir + "/" + filename;
        if (QFileInfo::exists(full)) return full;
    }
    return filename;
}

// ── Slots ──────────────────────────────────────────────────────────────────

void VideoTab::onOpenVideo() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Open video file"), QDir::homePath(),
            tr("Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)"));
    if (path.isEmpty()) return;
    openVideoFile(path);
}

void VideoTab::openVideoFile(const QString& path) {
    if (!m_reader) {
        m_reader = new VideoFrameReader(this);
        connect(m_reader, &VideoFrameReader::frameReady, this,
                &VideoTab::onFrameReady);
        connect(m_reader, &VideoFrameReader::frameReadFailed, this,
                [this]() {
                    m_playing = false;
                    m_playBtn->setText(tr("Play"));
                    setStatus(tr("Video read failed / end of stream."));
                });
    }
    m_reader->setConsumerDriven(true);
    m_reader->setPaused(true);
    if (!m_reader->openVideo(path.toStdString())) {
        QMessageBox::warning(this, tr("qSAM3"),
                             tr("Failed to open video:\n%1").arg(path));
        return;
    }
    m_videoPath = path;
    m_totalFrames = static_cast<int>(m_reader->getFrameCount());
    m_fps = m_reader->getFps();
    m_currentFrame = 0;
    m_processedMax = -1;
    m_timelineEntries.clear();
    m_timelineInstanceIds.clear();
    m_lastResult = SAM3WorkerResult{};
    m_timeline->setFrameCount(m_totalFrames);
    m_timeline->setCurrentFrame(0);
    m_timeline->setProcessedMax(-1);
    m_timeline->setTimeline(m_timelineEntries);
    m_timeline->setInstanceColors({});
    m_canvas->clearAll();
    m_canvas->setInteractive(m_worker && m_worker->hasModel());
    m_stepBtn->setEnabled(m_worker && m_worker->hasModel());
    m_resetBtn->setEnabled(true);
    m_playing = false;
    m_playBtn->setText(tr("Play"));

    appendLog(tr("Video: %1 | %2x%3 | %4 frames | %.1f fps")
                      .arg(QFileInfo(path).fileName())
                      .arg(m_reader->getFrameWidth())
                      .arg(m_reader->getFrameHeight())
                      .arg(m_totalFrames)
                      .arg(m_fps));
    setStatus(tr("Pause and annotate to add instances."));
}

void VideoTab::onLoadModel() {
    if (m_busy) {
        appendLog(tr("Worker is busy; wait for the current task."));
        return;
    }
    QString path = modelPath();
    if (path.isEmpty() ||
        m_modelCombo->currentData().toString() == "__browse__") {
        path = QFileDialog::getOpenFileName(
                this, tr("Select SAM3 GGUF model"), QDir::homePath(),
                tr("GGUF files (*.gguf);;All files (*)"));
        if (path.isEmpty()) return;
    }
    if (!m_worker) {
        m_worker = new VideoWorker(this);
        connect(m_worker, &VideoWorker::logMessage, this, &VideoTab::onLog);
        connect(m_worker, &VideoWorker::modelReady, this,
                &VideoTab::onModelReady);
        connect(m_worker, &VideoWorker::frameResultReady, this,
                &VideoTab::onFrameResult);
        connect(m_worker, &VideoWorker::instanceAdded, this,
                &VideoTab::onInstanceAdded);
        connect(m_worker, &VideoWorker::instanceRefined, this,
                &VideoTab::onInstanceRefined);
        connect(m_worker, &VideoWorker::busyChanged, this,
                &VideoTab::onBusyChanged);
    }

    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::LoadModel;
    req.modelPath = path;
    req.device = m_deviceCombo->currentText().toLower();
    if (m_modeText->isChecked() && m_modeText->isVisible()) {
        req.textPrompt = m_textPrompt->text();
    }
    setStatus(tr("Loading model..."));
    m_worker->post(req);
}

void VideoTab::onModelReady(const QString& backend, bool visualOnly) {
    m_visualOnly = visualOnly;
    m_backendLabel->setText(QString("Backend: %1").arg(backend));
    // Text mode is only available on full SAM3 models.
    m_modeText->setVisible(!visualOnly);
    m_textPrompt->setEnabled(!visualOnly && m_modeText->isChecked());
    if (visualOnly && m_modeText->isChecked()) {
        m_modeBox->setChecked(true);
    }
    m_canvas->setInteractive(true);
    m_stepBtn->setEnabled(true);
    m_trackerActive = true;
    setStatus(tr("Tracker created. Press Play or add instances."));
    // If a video is already open, process the current frame right away.
    if (!m_videoPath.isEmpty() && !m_busy) {
        trackNextFrame();
    }
}

void VideoTab::trackNextFrame() {
    if (m_busy || !m_worker || !m_worker->hasModel() || !m_reader) return;
    if (m_currentFrame >= m_totalFrames) {
        m_playing = false;
        m_playBtn->setText(tr("Play"));
        setStatus(tr("End of video."));
        return;
    }
    m_reader->seekToFrame(m_currentFrame);
    m_reader->readFrame();  // async: frameReady() delivers the decoded frame
}

void VideoTab::onFrameReady(const cv::Mat& rgbFrame, int frameIndex) {
    QImage img = VideoPlaybackWidget::cvMatToQImage(rgbFrame);
    if (img.isNull()) {
        appendLog(tr("Frame %1 has no data.").arg(frameIndex));
        return;
    }
    m_currentFrame = frameIndex;
    m_canvas->setFrame(img);
    m_timeline->setCurrentFrame(frameIndex);

    if (m_worker && m_worker->hasModel()) {
        VideoWorker::TrackRequest req;
        req.action = VideoWorker::Action::TrackFrame;
        req.frameIndex = frameIndex;
        req.frame = img;
        m_worker->post(req);
    } else {
        setStatus(tr("Frame %1/%2 — no tracker active")
                          .arg(frameIndex)
                          .arg(m_totalFrames));
        schedulePlayback();
    }
}

void VideoTab::onFrameResult(const SAM3WorkerResult& result, int frameIndex) {
    m_lastResult = result;
    m_currentFrame = frameIndex;
    if (frameIndex > m_processedMax) m_processedMax = frameIndex;
    updateCanvasInstances();
    updateTimeline(frameIndex, result);

    if (result.valid) {
        setStatus(tr("Frame %1/%2 — %3 objects tracked")
                          .arg(frameIndex)
                          .arg(m_totalFrames)
                          .arg(result.detCount));
    }
    schedulePlayback();
}

void VideoTab::schedulePlayback() {
    if (!m_playing || m_totalFrames <= 0) return;
    if (m_currentFrame + 1 >= m_totalFrames) {
        m_playing = false;
        m_playBtn->setText(tr("Play"));
        setStatus(tr("End of video."));
        return;
    }
    const double speed = m_speedSlider->value() / 10.0;
    const int intervalMs =
            m_fps > 0.0 ? static_cast<int>(1000.0 / (m_fps * speed)) : 40;
    m_playTimer.start(std::max(1, intervalMs));
}

void VideoTab::onPlayPause() {
    if (!m_worker || !m_worker->hasModel()) {
        appendLog(tr("Load a model first."));
        return;
    }
    if (m_videoPath.isEmpty()) {
        appendLog(tr("Open a video first."));
        return;
    }
    m_playing = !m_playing;
    m_playBtn->setText(m_playing ? tr("Pause") : tr("Play"));
    if (m_playing) {
        m_playTimer.stop();
        // If we are at the end, restart from the beginning.
        if (m_currentFrame + 1 >= m_totalFrames) {
            m_currentFrame = 0;
        }
        trackNextFrame();
    } else {
        m_playTimer.stop();
    }
}

void VideoTab::onStep() {
    if (!m_worker || !m_worker->hasModel()) return;
    if (m_currentFrame + 1 < m_totalFrames) {
        m_playing = false;
        m_playBtn->setText(tr("Play"));
        m_playTimer.stop();
        ++m_currentFrame;
        trackNextFrame();
    }
}

void VideoTab::onReset() {
    m_playing = false;
    m_playBtn->setText(tr("Play"));
    m_playTimer.stop();
    resetPrompts();
    m_lastResult = SAM3WorkerResult{};
    m_timelineEntries.clear();
    m_timelineInstanceIds.clear();
    m_timeline->setTimeline(m_timelineEntries);
    m_timeline->setInstanceColors({});
    m_canvas->clearAll();
    m_currentFrame = 0;
    m_processedMax = -1;
    m_timeline->setCurrentFrame(0);
    m_timeline->setProcessedMax(-1);

    if (m_worker && m_worker->hasModel() && !m_videoPath.isEmpty()) {
        VideoWorker::TrackRequest req;
        req.action = VideoWorker::Action::ResetTracker;
        if (m_modeText->isChecked() && m_modeText->isVisible()) {
            req.textPrompt = m_textPrompt->text();
        }
        m_worker->post(req);
        m_trackerActive = true;
        m_canvas->setInteractive(true);
        // Re-track frame 0 once the tracker is reset.
        m_currentFrame = 0;
        trackNextFrame();
    }
    setStatus(tr("Reset. Ready to annotate."));
}

void VideoTab::onCanvasBox() {
    addInstanceFromPrompts();
}

void VideoTab::onCanvasInstanceClicked(int id) {
    if (!m_trackerActive || !m_worker) return;
    // Refine with a positive point at the instance center (upstream behavior:
    // clicking a mask refines it with a positive point at the click).
    refineInstance(id, {}, {});
    setStatus(tr("Refined instance #%1").arg(id));
}

void VideoTab::onCanvasPosPoint(const QPointF& p) {
    if (!m_trackerActive || !m_worker) return;
    QVector<QPointF> pos{p};
    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::RefineInstance;
    req.instanceId = -2;  // sentinel: add new instance from points
    for (const auto& pt : pos) {
        req.prompt.posPoints.append({static_cast<float>(pt.x()),
                                     static_cast<float>(pt.y())});
    }
    m_worker->post(req);
}

void VideoTab::onCanvasNegPoint(const QPointF& p) {
    // Negative point: refine the instance under the cursor, or queue it for
    // the next AddInstance.
    const int hit = m_canvas->hitTestInstance(p);
    if (hit >= 0 && m_trackerActive && m_worker) {
        refineInstance(hit, {}, {p});
        return;
    }
    appendLog(tr("Negative point on empty area: add a positive prompt first."));
}

void VideoTab::addInstanceFromPrompts() {
    if (!m_trackerActive || !m_worker || !m_canvas) return;
    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::AddInstance;
    if (m_canvas->hasBox()) {
        const QRectF b = m_canvas->box();
        req.prompt.usePvsBox = true;
        req.prompt.pvsBox = {static_cast<float>(b.left()),
                             static_cast<float>(b.top()),
                             static_cast<float>(b.right()),
                             static_cast<float>(b.bottom())};
    }
    for (const auto& pt : m_canvas->posPoints()) {
        req.prompt.posPoints.append({static_cast<float>(pt.x()),
                                     static_cast<float>(pt.y())});
    }
    for (const auto& pt : m_canvas->negPoints()) {
        req.prompt.negPoints.append({static_cast<float>(pt.x()),
                                     static_cast<float>(pt.y())});
    }
    if (req.prompt.posPoints.isEmpty() && !req.prompt.usePvsBox) {
        appendLog(tr("Click a positive point or drag a box first."));
        return;
    }
    m_worker->post(req);
    resetPrompts();
}

void VideoTab::refineInstance(int id, const QVector<QPointF>& pos,
                              const QVector<QPointF>& neg) {
    if (!m_trackerActive || !m_worker) return;
    VideoWorker::TrackRequest req;
    req.action = VideoWorker::Action::RefineInstance;
    req.instanceId = id;
    for (const auto& pt : pos) {
        req.prompt.posPoints.append({static_cast<float>(pt.x()),
                                     static_cast<float>(pt.y())});
    }
    for (const auto& pt : neg) {
        req.prompt.negPoints.append({static_cast<float>(pt.x()),
                                     static_cast<float>(pt.y())});
    }
    m_worker->post(req);
}

void VideoTab::onInstanceAdded(int instanceId) {
    appendLog(tr("Added instance #%1").arg(instanceId));
    setStatus(tr("Added instance #%1").arg(instanceId));
}

void VideoTab::onInstanceRefined(int instanceId, bool ok) {
    setStatus(ok ? tr("Refined instance #%1").arg(instanceId)
                 : tr("Failed to refine instance #%1").arg(instanceId));
}

void VideoTab::onBusyChanged(bool busy) {
    m_busy = busy;
    m_loadBtn->setEnabled(!busy);
    m_modelCombo->setEnabled(!busy);
    m_deviceCombo->setEnabled(!busy);
    m_canvas->setInteractive(!busy);
    m_stepBtn->setEnabled(!busy && m_worker && m_worker->hasModel());
    if (busy) m_playTimer.stop();
}

void VideoTab::onLog(const QString& msg) {
    setStatus(msg);
}

void VideoTab::onSeek(int frame) {
    if (m_videoPath.isEmpty()) return;
    m_playing = false;
    m_playBtn->setText(tr("Play"));
    m_playTimer.stop();
    m_currentFrame = qBound(0, frame, m_totalFrames - 1);
    trackNextFrame();
}

void VideoTab::onModeChanged() {
    const bool textMode = m_modeText->isChecked() && !m_visualOnly;
    m_textPrompt->setEnabled(textMode);
    resetPrompts();
    if (m_modeText->isChecked() && m_visualOnly) {
        // Visual-only model: fall back to box mode.
        m_modeBox->setChecked(true);
    }
    appendLog(tr("Mode: %1").arg(textMode ? tr("Text") : tr("Box / Points")));
}

void VideoTab::onExportMasks() {
    if (m_lastResult.valid && !m_lastResult.instanceMasks.isEmpty()) {
        const QString dir = QFileDialog::getExistingDirectory(
                this, tr("Export masks to directory"), QDir::homePath());
        if (dir.isEmpty()) return;
        int exported = 0;
        for (int i = 0; i < m_lastResult.instanceMasks.size(); ++i) {
            const QString path = QString("%1/frame%2_mask%3.png")
                                         .arg(dir)
                                         .arg(m_currentFrame, 4, 10, QLatin1Char('0'))
                                         .arg(i);
            if (m_lastResult.instanceMasks[i].save(path)) ++exported;
        }
        appendLog(tr("Exported %1 mask(s) to %2").arg(exported).arg(dir));
    } else {
        appendLog(tr("No masks on the current frame."));
    }
}

// ── Private helpers ────────────────────────────────────────────────────────

void VideoTab::updateCanvasInstances() {
    QVector<VideoInstanceBox> boxes;
    if (m_showMasks->isChecked() && m_lastResult.valid) {
        for (int i = 0; i < m_lastResult.detCount; ++i) {
            VideoInstanceBox b;
            b.id = m_lastResult.instanceIds.value(i);
            const aicore_sam3_box box = m_lastResult.boxes.value(i);
            b.box = QRectF(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
            b.color = instanceColor(b.id);
            b.score = m_lastResult.scores.value(i);
            boxes.append(b);
        }
    }
    m_canvas->setInstances(boxes, m_showMasks->isChecked()
                                          ? m_lastResult.instanceMasks
                                          : QVector<QImage>());

    // Instance list label.
    if (!m_lastResult.valid || m_lastResult.detCount <= 0) {
        m_instanceLabel->setText(tr("(none)"));
        m_instanceLabel->setStyleSheet("color: #999;");
    } else {
        QString html;
        for (int i = 0; i < m_lastResult.detCount; ++i) {
            const QColor c = instanceColor(m_lastResult.instanceIds.value(i));
            html += QString("<span style='color:%1; font-weight:bold;'>"
                            "#%2: %3</span> ")
                            .arg(c.name())
                            .arg(m_lastResult.instanceIds.value(i))
                            .arg(m_lastResult.scores.value(i), 0, 'f', 2);
        }
        m_instanceLabel->setText(html);
        m_instanceLabel->setStyleSheet(QString());
    }
    m_exportBtn->setEnabled(m_lastResult.valid &&
                            !m_lastResult.instanceMasks.isEmpty());
}

void VideoTab::updateTimeline(int frameIndex, const SAM3WorkerResult& result) {
    if (frameIndex >= m_timelineEntries.size()) {
        m_timelineEntries.resize(frameIndex + 1);
    }
    m_timelineEntries[frameIndex].instances.clear();
    for (int i = 0; i < result.detCount; ++i) {
        m_timelineEntries[frameIndex].instances.append(
                {result.instanceIds.value(i), result.scores.value(i)});
        const int id = result.instanceIds.value(i);
        if (!m_timelineInstanceIds.contains(id)) {
            m_timelineInstanceIds.append(id);
        }
    }
    std::sort(m_timelineInstanceIds.begin(), m_timelineInstanceIds.end());

    QVector<QColor> colors;
    for (const int id : m_timelineInstanceIds) {
        colors.append(instanceColor(id));
    }
    m_timeline->setCurrentFrame(frameIndex);
    m_timeline->setProcessedMax(m_processedMax);
    m_timeline->setTimeline(m_timelineEntries);
    m_timeline->setInstanceColors(colors);
}

QColor VideoTab::instanceColor(int id) const {
    const int ci = id > 0 ? (id - 1) % kNumColors : 0;
    return kInstanceColors[ci];
}

void VideoTab::resetPrompts() {
    m_canvas->setPromptPoints({}, {});
    m_canvas->clearBox();
}

void VideoTab::appendLog(const QString& msg) {
    setStatus(msg);
}

void VideoTab::setStatus(const QString& msg) {
    m_statusLabel->setText(msg);
}

#endif  // HAS_OPENCV_FACE_CAPTURE
