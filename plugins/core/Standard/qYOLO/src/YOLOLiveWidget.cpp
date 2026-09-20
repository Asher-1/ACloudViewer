// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOLiveWidget.h"

#include <QtCompat.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QFont>
#include <QFormLayout>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QMetaObject>
#include <QPainter>
#include <QSettings>
#include <QSizePolicy>
#include <QSpinBox>
#include <QThread>
#include <QtMath>
#include <algorithm>
#include <cstring>

#include "YOLOLiveInferWorker.h"
#include "YOLOModelCatalog.h"
#include "ecvAICoreUiHelper.h"
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
    // Live preview is the main content of the dialog: size its baseline to
    // the screen (~60% of the available height; availableGeometry is in
    // device-independent pixels, so this adapts to any resolution and
    // platform) instead of a fixed 300 px that squeezed the video under
    // the surrounding controls. The label still grows with the window
    // (fixed height = minimum here, no maximum is imposed).
    const QRect screenAvail =
            QGuiApplication::primaryScreen()
                    ? QGuiApplication::primaryScreen()->availableGeometry()
                    : QRect(0, 0, 1280, 800);
    setPreviewFixedHeight(std::clamp(screenAvail.height() * 3 / 5, 480, 900));

    ensureInferThread();
}

YOLOLiveWidget::~YOLOLiveWidget() {
    // Destruction guard (see ~VideoPlaybackWidget): this body runs before
    // the base destructor, so drop outgoing connections before stopStream()
    // emits streamStopped at ancestor-context slots.
    disconnect(this, nullptr, nullptr, nullptr);
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

    // Row 1: model selection (takes all spare width).
    auto* controls = new QHBoxLayout;
    controls->setContentsMargins(0, 2, 0, 0);
    controls->setSpacing(4);

    controls->addWidget(new QLabel(tr("Model:"), this));
    m_modelCombo = new QComboBox(this);
    m_modelCombo->setMinimumContentsLength(16);
    m_modelCombo->setSizeAdjustPolicy(
            QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_modelCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    controls->addWidget(m_modelCombo, 1);
    mainLayout()->insertLayout(0, controls);

    // Row 2: runtime parameters (device / threads).
    auto* paramsRow = new QHBoxLayout;
    paramsRow->setSpacing(4);
    paramsRow->addWidget(new QLabel(tr("Device:"), this));
    m_deviceCombo = new QComboBox(this);
    paramsRow->addWidget(m_deviceCombo);
    paramsRow->addWidget(new QLabel(tr("Threads:"), this));
    m_threadsSpin = new QSpinBox(this);
    m_threadsSpin->setRange(0, 64);
    m_threadsSpin->setValue(0);
    m_threadsSpin->setToolTip(tr("0 = auto"));
    paramsRow->addWidget(m_threadsSpin);
    paramsRow->addStretch();
    mainLayout()->insertLayout(1, paramsRow);

    // Row 3: detection thresholds (hidden for depth models).
    auto* thresholdRow = new QHBoxLayout;
    thresholdRow->setSpacing(4);
    auto* confLabel = new QLabel(tr("Conf:"), this);
    thresholdRow->addWidget(confLabel);
    m_confSpin = new QDoubleSpinBox(this);
    m_confSpin->setRange(0.01, 1.0);
    m_confSpin->setSingleStep(0.05);
    m_confSpin->setValue(0.25);
    m_confSpin->setToolTip(tr("Confidence threshold (detect/segment models)"));
    thresholdRow->addWidget(m_confSpin);
    auto* iouLabel = new QLabel(tr("IoU:"), this);
    thresholdRow->addWidget(iouLabel);
    m_iouSpin = new QDoubleSpinBox(this);
    m_iouSpin->setRange(0.1, 1.0);
    m_iouSpin->setSingleStep(0.05);
    m_iouSpin->setValue(0.7);
    m_iouSpin->setToolTip(tr("NMS IoU threshold (detect/segment models)"));
    thresholdRow->addWidget(m_iouSpin);
    auto* topKLabel = new QLabel(tr("Top-K:"), this);
    thresholdRow->addWidget(topKLabel);
    m_topKSpin = new QSpinBox(this);
    m_topKSpin->setRange(1, 1000);
    m_topKSpin->setValue(300);
    m_topKSpin->setToolTip(tr("Max detections per frame"));
    thresholdRow->addWidget(m_topKSpin);
    thresholdRow->addStretch();
    mainLayout()->insertLayout(2, thresholdRow);

    // Model selection mirrors the batch tab.
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                updateModelPathFromCombo();
                updateThresholdVisibility();
            });
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
    connect(m_topKSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int k) { m_config.topK = static_cast<uint32_t>(k); });

    // Threshold controls + labels share visibility: depth models have no
    // detection thresholds, detect/segment models do.
    m_thresholdWidgets = {
            confLabel, m_confSpin, iouLabel, m_iouSpin, topKLabel, m_topKSpin,
    };

    // Row 4: multi-object tracking (detect/segment/pose/obb models). The
    // tracker types mirror the six official ultralytics/cfg/trackers YAMLs;
    // GMC methods mirror utils/gmc.py (the extension methods need the
    // plugin's OpenCV build — same gate as the upstream runtime).
    auto* trackRow = new QHBoxLayout;
    trackRow->setSpacing(4);
    m_trackCheck = new QCheckBox(tr("Track"), this);
    m_trackCheck->setToolTip(
            tr("Multi-object tracking with stable ids (six official "
               "tracker modes)"));
    trackRow->addWidget(m_trackCheck);
    // Official with_reid (model="auto"): the tracker consumes the
    // detector's per-detection feature rows for appearance matching; the
    // official path downgrades to a separate ReID encoder on end2end
    // heads, so there the term degrades to motion-only association.
    m_reidCheck = new QCheckBox(tr("ReID"), this);
    m_reidCheck->setToolTip(
            tr("Appearance re-identification from the detector features "
               "(official model=\"auto\" path; end2end heads degrade to "
               "motion-only association)"));
    trackRow->addWidget(m_reidCheck);
    // Trails overlay (default off — boxes+ids only, like the official
    // model.track preview; a pinned target always shows its trail).
    m_trailsCheck = new QCheckBox(tr("Trails"), this);
    m_trailsCheck->setToolTip(
            tr("Draw per-identity motion trails (official solutions style; "
               "a pinned target always shows its trail)"));
    connect(m_trailsCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_showTrails = on;
        repaintLivePreview();
    });
    trackRow->addWidget(m_trailsCheck);
    trackRow->addWidget(new QLabel(tr("Type:"), this));
    m_trackerCombo = new QComboBox(this);
    m_trackerCombo->addItem(QStringLiteral("tracktrack"),
                            QStringLiteral("tracktrack"));
    m_trackerCombo->addItem(QStringLiteral("bytetrack"),
                            QStringLiteral("bytetrack"));
    m_trackerCombo->addItem(QStringLiteral("botsort"),
                            QStringLiteral("botsort"));
    m_trackerCombo->addItem(QStringLiteral("ocsort"), QStringLiteral("ocsort"));
    m_trackerCombo->addItem(QStringLiteral("deepocsort"),
                            QStringLiteral("deepocsort"));
    m_trackerCombo->addItem(QStringLiteral("fasttrack"),
                            QStringLiteral("fasttrack"));
    m_trackerCombo->setToolTip(
            tr("Tracker backend; switching types resets the tracking "
               "parameters and GMC method to the official defaults of "
               "the selected ultralytics/cfg/trackers YAML"));
    trackRow->addWidget(m_trackerCombo);
    trackRow->addWidget(new QLabel(tr("GMC:"), this));
    m_gmcCombo = new QComboBox(this);
    m_gmcCombo->addItem(QStringLiteral("sparseOptFlow"),
                        QStringLiteral("sparseOptFlow"));
    m_gmcCombo->addItem(QStringLiteral("none"), QStringLiteral("none"));
#ifdef QYOLO_WITH_OPENCV
    m_gmcCombo->insertItem(1, QStringLiteral("orb"), QStringLiteral("orb"));
    m_gmcCombo->insertItem(2, QStringLiteral("sift"), QStringLiteral("sift"));
    m_gmcCombo->insertItem(3, QStringLiteral("ecc"), QStringLiteral("ecc"));
#endif
    m_gmcCombo->setToolTip(
            tr("Global motion compensation (camera-motion estimate); "
               "used by botsort/deepocsort/tracktrack"));
    trackRow->addWidget(m_gmcCombo);
    trackRow->addStretch();
    mainLayout()->insertLayout(3, trackRow);

    // Row 5: tracking parameter area (visible while tracking is enabled).
    auto* trackParamsRow = new QHBoxLayout;
    trackParamsRow->setSpacing(4);
    trackParamsRow->addWidget(new QLabel(tr("High:"), this));
    m_trackHighSpin = new QDoubleSpinBox(this);
    m_trackHighSpin->setRange(0.05, 1.0);
    m_trackHighSpin->setSingleStep(0.05);
    m_trackHighSpin->setValue(0.25);
    m_trackHighSpin->setToolTip(
            tr("First association threshold (track_high_thresh)"));
    trackParamsRow->addWidget(m_trackHighSpin);
    trackParamsRow->addWidget(new QLabel(tr("Low:"), this));
    m_trackLowSpin = new QDoubleSpinBox(this);
    m_trackLowSpin->setRange(0.01, 1.0);
    m_trackLowSpin->setSingleStep(0.05);
    m_trackLowSpin->setValue(0.1);
    m_trackLowSpin->setToolTip(
            tr("Second association threshold (track_low_thresh)"));
    trackParamsRow->addWidget(m_trackLowSpin);
    trackParamsRow->addWidget(new QLabel(tr("New:"), this));
    m_newTrackSpin = new QDoubleSpinBox(this);
    m_newTrackSpin->setRange(0.05, 1.0);
    m_newTrackSpin->setSingleStep(0.05);
    m_newTrackSpin->setValue(0.25);
    m_newTrackSpin->setToolTip(tr("New-track threshold (new_track_thresh)"));
    trackParamsRow->addWidget(m_newTrackSpin);
    trackParamsRow->addWidget(new QLabel(tr("Buffer:"), this));
    m_trackBufferSpin = new QSpinBox(this);
    m_trackBufferSpin->setRange(1, 300);
    m_trackBufferSpin->setValue(30);
    m_trackBufferSpin->setToolTip(
            tr("Lost-track buffer in frames (track_buffer)"));
    trackParamsRow->addWidget(m_trackBufferSpin);
    trackParamsRow->addWidget(new QLabel(tr("Match:"), this));
    m_matchSpin = new QDoubleSpinBox(this);
    m_matchSpin->setRange(0.1, 1.0);
    m_matchSpin->setSingleStep(0.05);
    m_matchSpin->setValue(0.8);
    m_matchSpin->setToolTip(tr("Association match threshold (match_thresh)"));
    trackParamsRow->addWidget(m_matchSpin);
    trackParamsRow->addStretch();
    mainLayout()->insertLayout(4, trackParamsRow);

    m_trackWidgets = {
            m_trackCheck,
            m_reidCheck,
            m_trackerCombo,
            m_gmcCombo,
    };
    m_trackParamWidgets = {
            m_trackHighSpin,   m_trackLowSpin, m_newTrackSpin,
            m_trackBufferSpin, m_matchSpin,
    };

    connect(m_trackCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_config.trackerType = on ? trackerTypeId() : QString();
        updateTrackVisibility();
        updateGmcEnabled();
        if (on) {
            // Upstream track-mode interaction: Model.track() and the CLI
            // track subcommand default conf to 0.1 (the ByteTrack-family
            // two-stage association needs low-confidence rows). Move the
            // spin only while it still carries the detect default, so a
            // user-tuned value is never overwritten.
            if (m_confSpin->value() == 0.25) m_confSpin->setValue(0.1);
        }
    });
    connect(m_reidCheck, &QCheckBox::toggled, this,
            [this](bool on) { m_config.withReid = on; });
    connect(m_trackerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                // Upstream tracker=<yaml> semantics: selecting a tracker
                // loads that YAML's defaults into the exposed controls
                // (no-op while setupUi is still building the widgets).
                applyOfficialTrackDefaults(trackerTypeId());
                updateGmcEnabled();
                if (m_trackCheck->isChecked())
                    m_config.trackerType = trackerTypeId();
            });
    connect(m_gmcCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { m_config.gmcMethod = gmcMethodId(); });
    connect(m_trackHighSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double v) {
                m_config.trackHighThresh = static_cast<float>(v);
            });
    connect(m_trackLowSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double v) {
                m_config.trackLowThresh = static_cast<float>(v);
            });
    connect(m_newTrackSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double v) {
                m_config.newTrackThresh = static_cast<float>(v);
            });
    connect(m_trackBufferSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int v) { m_config.trackBuffer = v; });
    connect(m_matchSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this,
            [this](double v) { m_config.matchThresh = static_cast<float>(v); });

    updateThresholdVisibility();
    updateTrackVisibility();
    updateGmcEnabled();
    // Install the official per-type defaults for the initial tracker
    // selection (tracktrack): the spin initializers above carry the shared
    // ByteTrack-era values, and the combo's first-addItem signal fired
    // before the spins existed.
    applyOfficialTrackDefaults(trackerTypeId());
}

void YOLOLiveWidget::updateGmcEnabled() {
    // GMC (camera-motion compensation) is consumed by botsort /
    // deepocsort / tracktrack only; keep the combo visible but disabled
    // for the other three so an irrelevant control is never editable.
    if (!m_trackerCombo || !m_gmcCombo) return;
    const QString type = trackerTypeId();
    const bool usesGmc = type == QStringLiteral("botsort") ||
                         type == QStringLiteral("deepocsort") ||
                         type == QStringLiteral("tracktrack");
    m_gmcCombo->setEnabled(usesGmc);
}

void YOLOLiveWidget::applyOfficialTrackDefaults(const QString& type) {
    if (!m_trackHighSpin || !m_trackLowSpin || !m_newTrackSpin ||
        !m_trackBufferSpin || !m_matchSpin) {
        return;  // setupUi not finished yet (combo fires while being built)
    }
    // Official per-type defaults of ultralytics/cfg/trackers/<type>.yaml
    // for the five exposed thresholds — the same values tracker.cpp's
    // default_tracker_config installs; the remaining YAML keys are not
    // surfaced in the UI and keep those defaults in the worker.
    double high = 0.25;
    double low = 0.1;
    double newTrack = 0.25;
    double match = 0.8;
    if (type == QStringLiteral("tracktrack")) {
        high = 0.6;
        low = 0.25;
        newTrack = 0.7;
        match = 0.7;
    } else if (type == QStringLiteral("deepocsort")) {
        high = 0.3;
        newTrack = 0.3;
    }
    m_trackHighSpin->setValue(high);
    m_trackLowSpin->setValue(low);
    m_newTrackSpin->setValue(newTrack);
    m_trackBufferSpin->setValue(30);
    m_matchSpin->setValue(match);
    // GMC consumers carry a per-type official default (botsort/tracktrack
    // use sparseOptFlow, deepocsort ships none); the other types never
    // read it, so their combo selection is left untouched.
    if (!m_gmcCombo) return;
    QString gmc;
    if (type == QStringLiteral("botsort") ||
        type == QStringLiteral("tracktrack"))
        gmc = QStringLiteral("sparseOptFlow");
    else if (type == QStringLiteral("deepocsort"))
        gmc = QStringLiteral("none");
    if (!gmc.isEmpty()) {
        const int gmcIdx = m_gmcCombo->findData(gmc);
        if (gmcIdx >= 0) m_gmcCombo->setCurrentIndex(gmcIdx);
    }
}

QString YOLOLiveWidget::trackerTypeId() const {
    return m_trackerCombo ? m_trackerCombo->currentData().toString()
                          : QStringLiteral("tracktrack");
}

QString YOLOLiveWidget::gmcMethodId() const {
    return m_gmcCombo ? m_gmcCombo->currentData().toString()
                      : QStringLiteral("sparseOptFlow");
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
    // Still syncing: refresh the config path without emitting
    // modelSelectionChanged (the state was mirrored from the batch tab).
    updateModelPathFromCombo();
    m_syncingModelControls = false;
    updateThresholdVisibility();
}

void YOLOLiveWidget::populateAllModels(const QString& keepFilename) {
    // The live pipeline covers the real-time families (detect / segment /
    // depth plus the trackable pose / obb). The batch-only families
    // (classify / semantic) and the text-conditioned world/yoloe models
    // (which need a class list + text tower per context) are offered by
    // their dedicated task tabs instead.
    QVector<YOLOModelEntry> all;
    for (const YOLOModelEntry& e : YOLOHelpers::catalogModels()) {
        const bool closedSet = !e.textInput;
        const bool liveTask = e.task == QStringLiteral("detect") ||
                              e.task == QStringLiteral("segment") ||
                              e.task == QStringLiteral("depth") ||
                              e.task == QStringLiteral("pose") ||
                              e.task == QStringLiteral("obb");
        if (closedSet && liveTask) all.append(e);
    }
    m_syncingModelControls = true;
    m_modelCombo->clear();
    for (const YOLOModelEntry& e : all) {
        m_modelCombo->addItem(YOLOHelpers::modelDisplayLabel(e), e.filename);
    }
    // Single selection policy for every AICore dialog (the all-model Live
    // view has no single catalog-declared default: defaultIndex -1 falls
    // through to the "(recommended)" guard row).
    ecvAICoreUi::selectModelRow(m_modelCombo, keepFilename, -1);
    // Still syncing: refresh the config path without emitting
    // modelSelectionChanged — the dialog construction used to mirror this
    // auto-picked row back into the matching batch tab and record it as an
    // explicit user choice, pinning F32 across restarts.
    updateModelPathFromCombo();
    m_syncingModelControls = false;
    updateThresholdVisibility();
}

void YOLOLiveWidget::updateThresholdVisibility() {
    // Depth models produce a depth map, not detections — hide the
    // detection-threshold row. The task follows the selected model entry.
    YOLOModelEntry entry;
    const bool isDepth =
            YOLOHelpers::findModelByFilename(modelFilename(), &entry) &&
            entry.task == QStringLiteral("depth");
    for (QWidget* w : m_thresholdWidgets) {
        if (w) w->setVisible(!isDepth);
    }
    // A hidden depth spin keeps its value; restoring the row shows the last
    // detection thresholds, which matches the batch tab behavior.
    m_config.confThres = static_cast<float>(m_confSpin->value());
    m_config.iouThres = static_cast<float>(m_iouSpin->value());
    m_config.topK = static_cast<uint32_t>(m_topKSpin->value());
    updateTrackVisibility();
}

void YOLOLiveWidget::updateTrackVisibility() {
    // Depth models have no detections to track; everything else (detect /
    // segment / pose / obb) supports the six tracker modes.
    YOLOModelEntry entry;
    const bool isDepth =
            YOLOHelpers::findModelByFilename(modelFilename(), &entry) &&
            entry.task == QStringLiteral("depth");
    const bool tracking = m_trackCheck && m_trackCheck->isChecked() && !isDepth;
    for (QWidget* w : m_trackWidgets) {
        if (w) w->setVisible(!isDepth);
    }
    for (QWidget* w : m_trackParamWidgets) {
        if (w) w->setVisible(tracking);
    }
    if (m_trackCheck) {
        m_config.trackerType = tracking ? trackerTypeId() : QString();
    }
    m_config.gmcMethod = gmcMethodId();
    m_config.trackHighThresh = static_cast<float>(m_trackHighSpin->value());
    m_config.trackLowThresh = static_cast<float>(m_trackLowSpin->value());
    m_config.newTrackThresh = static_cast<float>(m_newTrackSpin->value());
    m_config.trackBuffer = m_trackBufferSpin->value();
    m_config.matchThresh = static_cast<float>(m_matchSpin->value());
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
    // Still syncing: refresh the config path without emitting
    // modelSelectionChanged (mirrored state, not a user pick).
    updateModelPathFromCombo();
    m_syncingModelControls = false;
    m_config.device = deviceId();
    m_config.threads = threadCount();
}

void YOLOLiveWidget::updateModelPathFromCombo() {
    // The config path always tracks the combo; the signal is emitted only
    // for user-driven changes — the rebuild/populate/sync paths call this
    // while m_syncingModelControls is still set so that the mirrored state
    // is never written back into the batch tabs (and never recorded as an
    // explicit model choice there).
    m_config.modelPath = resolveModelPath();
    if (m_syncingModelControls) return;
    emit modelSelectionChanged(modelFilename());
}

void YOLOLiveWidget::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qYOLO/live"));
    m_confSpin->setValue(
            settings.value(QStringLiteral("conf"), 0.25).toDouble());
    m_iouSpin->setValue(settings.value(QStringLiteral("iou"), 0.7).toDouble());
    m_topKSpin->setValue(settings.value(QStringLiteral("topK"), 300).toInt());
    m_threadsSpin->setValue(
            settings.value(QStringLiteral("threads"), 0).toInt());
    // Tracking (six official tracker modes; disabled by default).
    m_trackCheck->setChecked(
            settings.value(QStringLiteral("track"), false).toBool());
    m_reidCheck->setChecked(
            settings.value(QStringLiteral("reid"), false).toBool());
    m_config.withReid = m_reidCheck->isChecked();
    const QString type =
            settings.value(QStringLiteral("trackerType"), "tracktrack")
                    .toString();
    const int typeIdx = m_trackerCombo->findData(type);
    if (typeIdx >= 0) m_trackerCombo->setCurrentIndex(typeIdx);
    const QString gmc =
            settings.value(QStringLiteral("gmc"), "sparseOptFlow").toString();
    const int gmcIdx = m_gmcCombo->findData(gmc);
    if (gmcIdx >= 0) m_gmcCombo->setCurrentIndex(gmcIdx);
    // The tracker-type sync above already installed the official per-type
    // defaults; only persisted user values may override them (the fallback
    // is the current spin value, not a hardcoded ByteTrack-era default).
    m_trackHighSpin->setValue(settings.value(QStringLiteral("trackHigh"),
                                             m_trackHighSpin->value())
                                      .toDouble());
    m_trackLowSpin->setValue(
            settings.value(QStringLiteral("trackLow"), m_trackLowSpin->value())
                    .toDouble());
    m_newTrackSpin->setValue(
            settings.value(QStringLiteral("newTrack"), m_newTrackSpin->value())
                    .toDouble());
    m_trackBufferSpin->setValue(settings.value(QStringLiteral("trackBuffer"),
                                               m_trackBufferSpin->value())
                                        .toInt());
    m_matchSpin->setValue(
            settings.value(QStringLiteral("match"), m_matchSpin->value())
                    .toDouble());
    settings.endGroup();
    updateTrackVisibility();
}

void YOLOLiveWidget::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qYOLO/live"));
    settings.setValue(QStringLiteral("conf"), m_confSpin->value());
    settings.setValue(QStringLiteral("iou"), m_iouSpin->value());
    settings.setValue(QStringLiteral("topK"), m_topKSpin->value());
    settings.setValue(QStringLiteral("threads"), m_threadsSpin->value());
    settings.setValue(QStringLiteral("track"), m_trackCheck->isChecked());
    settings.setValue(QStringLiteral("reid"), m_reidCheck->isChecked());
    settings.setValue(QStringLiteral("trackerType"), trackerTypeId());
    settings.setValue(QStringLiteral("gmc"), gmcMethodId());
    settings.setValue(QStringLiteral("trackHigh"), m_trackHighSpin->value());
    settings.setValue(QStringLiteral("trackLow"), m_trackLowSpin->value());
    settings.setValue(QStringLiteral("newTrack"), m_newTrackSpin->value());
    settings.setValue(QStringLiteral("trackBuffer"),
                      m_trackBufferSpin->value());
    settings.setValue(QStringLiteral("match"), m_matchSpin->value());
    settings.endGroup();
}

// ---- video_base hooks -----------------------------------------------------

bool YOLOLiveWidget::onPrepareStream() {
    // Model must exist before the stream starts.
    if (m_config.modelPath.isEmpty() ||
        !QFileInfo::exists(m_config.modelPath)) {
        emit logMessage(
                tr("[YOLO] Model file not found — download it from a task "
                   "tab first."));
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
    ensureInferThread();  // recreate after a dialog-close shutdown
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
    job.classes = m_config.classes;
    job.textModelPath = m_config.textModelPath;
    job.trackerType = m_config.trackerType;
    job.withReid = m_config.withReid;
    job.gmcMethod = m_config.gmcMethod;
    job.trackHighThresh = m_config.trackHighThresh;
    job.trackLowThresh = m_config.trackLowThresh;
    job.newTrackThresh = m_config.newTrackThresh;
    job.trackBuffer = m_config.trackBuffer;
    job.matchThresh = m_config.matchThresh;
    job.trackZone = m_trackZone;
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
    // Tracking warnings (tracker rejected for this build/config) repeat on
    // every frame while parked — log each distinct reason once.
    if (result.warning != m_lastTrackWarning) {
        if (!result.warning.isEmpty()) {
            emit logMessage(
                    tr("[YOLO] Tracking disabled: %1").arg(result.warning));
        }
        m_lastTrackWarning = result.warning;
    }

    m_lastTask = result.task;
    m_hasSnapshot = true;

    if (result.task == QStringLiteral("depth")) {
        // Colorize once at source resolution; the overlay layer below only
        // scales it to preview size on rebuild.
        m_lastDepth = result.depth;
        m_overlayDetections.clear();
        m_overlayMasks.clear();
        m_overlayObbs.clear();
        m_overlayKeypointSets.clear();
        m_overlayTrackIds.clear();
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
    m_overlayMasks = result.detect.masks;    // empty for non-segment models
    m_overlayObbs = result.detect.obbBoxes;  // obb only, else empty
    m_overlayKeypointSets = result.detect.keypointSets;  // pose only
    m_overlayTrackIds = result.trackIds;  // empty while tracking is off
    // Trails (official solutions track_history): per-id center path in
    // source pixels. Ids absent this frame keep their history so a
    // re-found target resumes its line; turning tracking off or switching
    // the source clears everything (empty/size-mismatched ids below).
    if (!m_overlayTrackIds.isEmpty() &&
        m_overlayTrackIds.size() == m_overlayDetections.size()) {
        for (qsizetype i = 0; i < m_overlayTrackIds.size(); ++i) {
            const int id = m_overlayTrackIds[i];
            if (id < 0) continue;
            const YOLODetection& d = m_overlayDetections[i];
            QVector<QPointF>& pts = m_trails[id];
            pts.append(QPointF((d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0));
            while (pts.size() > 30) pts.removeFirst();
        }
    } else {
        m_trails.clear();
    }
    m_overlaySourceSize = m_lastSourceFrame.size();
    ++m_overlayGeneration;
    repaintLivePreview();
}

// ---- Official trackzone interaction (Ctrl+drag draw / Ctrl+click clear) ---

bool YOLOLiveWidget::onPreviewMousePress(QMouseEvent* event) {
    if (!(event->modifiers() & Qt::ControlModifier) ||
        event->button() != Qt::LeftButton) {
        return false;  // everything else keeps the label's own behavior
    }
    m_zoneDragging = true;
    m_zoneDragStartLabel = qtCompatMouseEventPos(event);
    m_trackZoneDraft = QRectF();
    return true;  // consume: no click-to-enlarge during selection
}

bool YOLOLiveWidget::onPreviewMouseMove(QMouseEvent* event) {
    if (!m_zoneDragging) return false;
    const QPointF startSrc = mapPreviewToSource(m_zoneDragStartLabel);
    const QPointF curSrc = mapPreviewToSource(qtCompatMouseEventPos(event));
    if (startSrc.x() < 0 || curSrc.x() < 0) return true;
    m_trackZoneDraft = QRectF(startSrc, curSrc).normalized();
    repaintLivePreview();  // live rubber band at preview resolution
    return true;
}

bool YOLOLiveWidget::onPreviewMouseRelease(QMouseEvent* event) {
    if (!m_zoneDragging || event->button() != Qt::LeftButton) return false;
    m_zoneDragging = false;
    const QPointF startSrc = mapPreviewToSource(m_zoneDragStartLabel);
    const QPointF curSrc = mapPreviewToSource(qtCompatMouseEventPos(event));
    const QRectF rect(startSrc, curSrc);
    if (startSrc.x() < 0 || curSrc.x() < 0 || rect.width() < 8.0 ||
        rect.height() < 8.0) {
        // Ctrl+click without a real drag: pin the clicked tracked identity
        // ("track this one" — only it keeps its color, banner and trail),
        // or release the pin when clicking empty space / the same target.
        int hit = -1;
        for (int i = 0; i < m_overlayDetections.size() && hit < 0; ++i) {
            if (i >= m_overlayTrackIds.size()) break;
            const int tid = m_overlayTrackIds[static_cast<qsizetype>(i)];
            if (tid <= 0) continue;
            const YOLODetection& d =
                    m_overlayDetections[static_cast<qsizetype>(i)];
            if (curSrc.x() >= d.x1 && curSrc.x() <= d.x2 &&
                curSrc.y() >= d.y1 && curSrc.y() <= d.y2) {
                hit = tid;
            }
        }
        m_pinnedTrackId = (hit >= 0 && hit != m_pinnedTrackId) ? hit : -1;
        m_trackZone = QRectF();
        m_trackZoneDraft = QRectF();
        repaintLivePreview();
        return true;
    }
    m_trackZone = rect.normalized();
    m_pinnedTrackId = -1;  // zone mode replaces the single-target pin
    m_trackZoneDraft = QRectF();
    repaintLivePreview();
    return true;
}

/* 3-tap separable Gaussian blur [1,2,1]/4 on Grayscale8. */
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

    if (!m_overlayDetections.isEmpty() || !m_overlayMasks.isEmpty() ||
        !m_overlayObbs.isEmpty() || !m_overlayKeypointSets.isEmpty()) {
        if (m_overlaySourceSize.isEmpty()) {
            return;
        }
    } else {
        return;
    }

    // Same rendering semantics as YOLOHelpers::drawDetections /
    // drawSegmentation, but on the small preview image with coordinates
    // scaled from the source pixel space. Box strokes and label font use
    // the official Annotator defaults for the PREVIEW size (the full-res
    // capture path applies the same formula at source resolution).
    const int lw = YOLOHelpers::officialAnnotatorLineWidth(
            displaySize.width(), displaySize.height());
    QImage layer(displaySize, QImage::Format_ARGB32_Premultiplied);
    layer.fill(Qt::transparent);
    const qreal sx = static_cast<qreal>(displaySize.width()) /
                     static_cast<qreal>(m_overlaySourceSize.width());
    const qreal sy = static_cast<qreal>(displaySize.height()) /
                     static_cast<qreal>(m_overlaySourceSize.height());

    QPainter p(&layer);
    p.setRenderHint(QPainter::Antialiasing, false);

    // Segment masks: translucent per-class tint, scaled to the display size
    // (same mapping as YOLOHelpers::drawSegmentation).
    if (!m_overlayMasks.isEmpty()) {
        for (int i = 0; i < m_overlayMasks.size(); ++i) {
            const YOLOSegMask& mask = m_overlayMasks[static_cast<size_t>(i)];
            if (mask.w <= 0 || mask.h <= 0 ||
                mask.bits.size() < static_cast<qint64>(mask.w) * mask.h) {
                continue;
            }
            QImage maskImage(mask.w, mask.h, QImage::Format_Grayscale8);
            /* Row-by-row copy: QImage scanlines are 32-bit aligned, so for
             * a width that is not a multiple of 4 bytesPerLine > width and
             * one contiguous memcpy shears the mask. */
            for (int y = 0; y < mask.h; ++y) {
                std::memcpy(
                        maskImage.scanLine(y),
                        mask.bits.constData() + static_cast<qint64>(y) * mask.w,
                        static_cast<size_t>(mask.w));
            }
            // Scale the binary {0,1} mask to preview size FIRST with
            // nearest-neighbour (lossless for binary data), then convert
            // to {0,255}, blur and blend at the MUCH smaller display
            // resolution — avoids the expensive full-resolution Gaussian
            // blur + full-resolution SmoothTransformation bilinear scale
            // that made live video unusable.
            maskImage = maskImage.scaled(
                    displaySize.width(), displaySize.height(),
                    Qt::IgnoreAspectRatio, Qt::FastTransformation);
            for (int b = 0; b < maskImage.sizeInBytes(); ++b) {
                if (maskImage.bits()[b]) maskImage.bits()[b] = 255;
            }
            gaussianBlurMask3(maskImage);
            const QColor tint =
                    i < m_overlayDetections.size()
                            ? QColor(YOLOHelpers::classColor(
                                      m_overlayDetections[i].classId))
                            : QColor(220, 220, 220);
            for (int y = 0; y < displaySize.height(); ++y) {
                const uchar* src = maskImage.constScanLine(y);
                QRgb* dst = reinterpret_cast<QRgb*>(layer.scanLine(y));
                for (int x = 0; x < displaySize.width(); ++x) {
                    if (src[x] == 0) continue;
                    // Proportional alpha: the SmoothTransformation-scaling
                    // produces fractional coverage values at mask boundaries,
                    // so map 0..255 → 0..170 to keep anti-aliased edges
                    // proportionally translucent.
                    const int a = src[x] * 170 / 255;
                    if (a == 0) continue;
                    dst[x] = qRgba(tint.red(), tint.green(), tint.blue(), a);
                }
            }
        }
    }

    // Boxes + labels, scaled from the source pixel space. Tracked runs
    // prefix the banner with the stable id in the official results.plot()
    // format (id:<n>), same as the capture rendering in
    // YOLOHelpers::drawDetections.
    QFont font = p.font();
    font.setPixelSize(YOLOHelpers::officialAnnotatorFontPixelSize(lw));
    p.setFont(font);
    // Official trackzone border: white, double-width, around the tracking
    // region (live rubber band while dragging).
    const QRectF& zoneRect = m_zoneDragging ? m_trackZoneDraft : m_trackZone;
    if (!zoneRect.isNull()) {
        QPen zonePen(Qt::white);
        zonePen.setWidth(lw * 2);
        p.setPen(zonePen);
        p.drawRect(QRectF(zoneRect.x() * sx, zoneRect.y() * sy,
                          zoneRect.width() * sx, zoneRect.height() * sy));
    }
    // Track trails (official solutions annotator): only when the Trails
    // toggle is on, or always for the pinned identity. Per-id color,
    // beneath the boxes; the pinned trail is drawn heavier.
    if ((!m_showTrails && m_pinnedTrackId < 0) || m_trails.isEmpty()) {
        // no trails requested
    } else {
        p.setRenderHint(QPainter::Antialiasing, true);
        QPen trailPen;
        for (auto it = m_trails.constBegin(); it != m_trails.constEnd(); ++it) {
            const bool pinned = it.key() == m_pinnedTrackId;
            if (m_pinnedTrackId >= 0 && !pinned) continue;  // pinned view
            const QVector<QPointF>& pts = it.value();
            if (pts.size() < 2) continue;
            trailPen.setWidthF(std::max(
                    1.0, static_cast<double>(lw) * (pinned ? 1.25 : 0.66)));
            trailPen.setColor(QColor(YOLOHelpers::trackIdColor(it.key())));
            p.setPen(trailPen);
            QPolygonF line;
            line.reserve(pts.size());
            for (const QPointF& pt : pts) {
                line.append(QPointF(pt.x() * sx, pt.y() * sy));
            }
            p.drawPolyline(line);
        }
        p.setRenderHint(QPainter::Antialiasing, false);
    }
    for (int detIdx = 0; detIdx < m_overlayDetections.size(); ++detIdx) {
        const YOLODetection& d =
                m_overlayDetections[static_cast<qsizetype>(detIdx)];
        // Official solutions identity coloring: a tracked row takes the
        // stable per-id color; untracked rows keep the per-class palette.
        // With a pinned target, every other row is demoted to a thin gray
        // box without a banner so the tracked one stands out.
        const int tid =
                detIdx < m_overlayTrackIds.size()
                        ? m_overlayTrackIds[static_cast<qsizetype>(detIdx)]
                        : -1;
        const bool demoted = m_pinnedTrackId >= 0 && tid != m_pinnedTrackId;
        const QColor color(
                demoted ? QColor(150, 150, 150)
                        : (tid >= 0 ? QColor(YOLOHelpers::trackIdColor(tid))
                                    : QColor(YOLOHelpers::classColor(
                                              d.classId))));
        QPen pen(color);
        pen.setWidth(
                demoted ? std::max(1, lw / 2)
                        : (tid >= 0 && tid == m_pinnedTrackId ? lw * 2 : lw));
        p.setPen(pen);
        p.drawRect(QRectF(d.x1 * sx, d.y1 * sy, (d.x2 - d.x1) * sx,
                          (d.y2 - d.y1) * sy));

        if (demoted) continue;  // thin gray box only, no banner

        QString label;
        if (detIdx < m_overlayTrackIds.size() &&
            m_overlayTrackIds[static_cast<qsizetype>(detIdx)] > 0) {
            label = QStringLiteral("id:%1 ").arg(
                    m_overlayTrackIds[static_cast<qsizetype>(detIdx)]);
        }
        label += QStringLiteral("%1 %2")
                         .arg(d.className)
                         .arg(d.score, 0, 'f', 2);
        // Keep the banner fully inside the preview (same rule as
        // YOLOHelpers::drawDetections): clamp horizontally, flip below the
        // box top when the box hugs the top edge.
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

    // Pose skeletons: COCO-17 lines + keypoint dots (same edge table as
    // YOLOHelpers::drawPose), scaled to the preview. The det boxes + labels
    // above already carry the pose rows (index-aligned with keypointSets).
    if (!m_overlayKeypointSets.isEmpty()) {
        const QVector<QPair<int, int>> skeleton =
                YOLOHelpers::cocoSkeletonEdges();
        const qreal kptRadius = std::max(2.0, displaySize.height() / 300.0);
        QPen linePen(Qt::white);
        linePen.setWidth(1);
        for (const YOLOKeypointSet& set : m_overlayKeypointSets) {
            const QColor color(YOLOHelpers::classColor(set.det.classId));
            linePen.setColor(color);
            p.setPen(linePen);
            for (const auto& edge : skeleton) {
                const int a = edge.first, b = edge.second;
                if (a >= set.kpts.size() || b >= set.kpts.size()) continue;
                const YOLOKeypoint& ka = set.kpts[a];
                const YOLOKeypoint& kb = set.kpts[b];
                if (ka.visibility < 0.5f || kb.visibility < 0.5f) continue;
                p.drawLine(QPointF(ka.x * sx, ka.y * sy),
                           QPointF(kb.x * sx, kb.y * sy));
            }
            p.setPen(Qt::NoPen);
            for (const YOLOKeypoint& k : set.kpts) {
                if (k.visibility < 0.5f) continue;
                p.setBrush(QColor(Qt::white));
                p.drawEllipse(QPointF(k.x * sx, k.y * sy), kptRadius,
                              kptRadius);
                p.setBrush(color);
                p.drawEllipse(QPointF(k.x * sx, k.y * sy), kptRadius * 0.6,
                              kptRadius * 0.6);
            }
        }
    }

    // Oriented boxes: rotated rectangles + center mark, scaled from the
    // source pixel space (same shape as YOLOHelpers::drawObb).
    for (int obbIdx = 0; obbIdx < m_overlayObbs.size(); ++obbIdx) {
        const YOLOObbBox& b = m_overlayObbs[static_cast<qsizetype>(obbIdx)];
        const QColor color(YOLOHelpers::classColor(b.classId));
        QPen pen(color);
        pen.setWidth(lw);
        p.setPen(pen);
        p.setBrush(Qt::NoBrush);
        const QPointF center(b.cx * sx, b.cy * sy);
        p.save();
        p.translate(center);
        p.rotate(qRadiansToDegrees(b.angle));
        p.drawRect(
                QRectF(-b.w * sx / 2.0, -b.h * sy / 2.0, b.w * sx, b.h * sy));
        p.restore();
        p.drawLine(center + QPointF(-4, 0), center + QPointF(4, 0));
        p.drawLine(center + QPointF(0, -4), center + QPointF(0, 4));

        const int deg = static_cast<int>(qRadiansToDegrees(b.angle) + 0.5);
        QString label;
        if (obbIdx < m_overlayTrackIds.size() &&
            m_overlayTrackIds[static_cast<qsizetype>(obbIdx)] > 0) {
            label = QStringLiteral("id:%1 ").arg(
                    m_overlayTrackIds[static_cast<qsizetype>(obbIdx)]);
        }
        label += QStringLiteral("%1 %2%3")
                         .arg(b.className)
                         .arg(b.score, 0, 'f', 2)
                         .arg(deg)
                         .arg(QChar(0x00B0));
        QRect labelRect(static_cast<int>(b.cx * sx - b.w * sx / 2.0),
                        static_cast<int>(b.cy * sy - b.h * sy / 2.0) -
                                font.pixelSize() - 6,
                        std::max(20, label.size() * font.pixelSize()),
                        font.pixelSize() + 6);
        labelRect.setWidth(std::min(labelRect.width(),
                                    std::max(20, displaySize.width() - 4)));
        labelRect.moveLeft(std::clamp(
                labelRect.left(), 2,
                std::max(2, displaySize.width() - labelRect.width() - 2)));
        if (labelRect.top() < 2) {
            labelRect.moveTop(static_cast<int>(b.cy * sy - b.h * sy / 2.0) + 2);
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

void YOLOLiveWidget::drawLiveOverlay(QImage& frame) {
    if (frame.isNull() ||
        (m_overlayDetections.isEmpty() && m_overlayDepthImage.isNull() &&
         m_overlayObbs.isEmpty() && m_overlayKeypointSets.isEmpty())) {
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
    m_overlayMasks.clear();
    m_overlayObbs.clear();
    m_overlayKeypointSets.clear();
    m_overlayTrackIds.clear();
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

    if (m_lastSnapshot.detections.isEmpty() && m_lastSnapshot.masks.isEmpty() &&
        m_lastSnapshot.obbBoxes.isEmpty() &&
        m_lastSnapshot.keypointSets.isEmpty()) {
        return;
    }
    if (m_lastSnapshot.annotatedImage.isNull()) {
        QImage annotated = m_lastSourceFrame;
        if (!m_lastSnapshot.masks.isEmpty()) {
            YOLOHelpers::drawSegmentation(&annotated, m_lastSnapshot.masks,
                                          m_lastSnapshot.detections, 0,
                                          m_overlayTrackIds);
        } else if (!m_lastSnapshot.obbBoxes.isEmpty()) {
            YOLOHelpers::drawObb(&annotated, m_lastSnapshot.obbBoxes, 0);
        } else if (!m_lastSnapshot.keypointSets.isEmpty()) {
            YOLOHelpers::drawPose(&annotated, m_lastSnapshot.keypointSets, 0);
        } else {
            YOLOHelpers::drawDetections(&annotated, m_lastSnapshot.detections,
                                        0, m_overlayTrackIds);
        }
        m_lastSnapshot.annotatedImage = annotated;
    }
    emit captureToDbRequested(m_lastSnapshot);
}

void YOLOLiveWidget::ensureInferThread() {
    if (m_inferWorker) return;
    // Rebuild the async side branch after releaseGpuResources() tore it
    // down on dialog close. A finished QThread object is reusable; only
    // the worker must be recreated.
    if (!m_inferThread) m_inferThread = new QThread(this);
    m_inferWorker = new YOLOLiveInferWorker;
    m_inferWorker->moveToThread(m_inferThread);
    connect(m_inferThread, &QThread::finished, m_inferWorker,
            &QObject::deleteLater);
    connect(m_inferWorker, &YOLOLiveInferWorker::inferComplete, this,
            &YOLOLiveWidget::onInferComplete, Qt::QueuedConnection);
    if (!m_inferThread->isRunning()) m_inferThread->start();
}

void YOLOLiveWidget::releaseGpuResources() {
    // Shut the infer thread down (releasing the resident model) when the
    // owning dialog closes for good; ensureInferThread() rebuilds it on
    // the next live start.
    shutdownInferThread();
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
