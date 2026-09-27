// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qLingbotMap.h"

#include <CVLog.h>
#include <ecvCameraSensor.h>
#include <ecvCameraSensorDisplayUtils.h>
#include <ecvColorTypes.h>
#include <ecvDrawContext.h>
#include <ecvHObjectCaster.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvRedrawScope.h>
#include <ecvScalarField.h>
#include <ecvViewManager.h>
#include <ecvViewportParameters.h>

#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QSettings>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <vector>

#include "ecvPersistentSettings.h"

namespace {

constexpr double kMaxWorldCoordinate = 1.0e6;  // sanity bound per point

// COLMAP-style camera frustum colors (same values as qFreeSplatter).
constexpr ecvColor::Rgb kColmapCameraPlaneColor(255, 25, 0);
constexpr ecvColor::Rgb kColmapCameraFrameColor(204, 25, 0);

// Official StreamingViewer frustum sizing: 35% of the median camera
// baseline, with an absolute fallback for the very first frames.
constexpr float kCameraBaselineFraction = 0.35f;
constexpr float kFallbackCameraSize = 0.05f;

// Live-preview DB refresh throttle (ms) — VTK redraws stay cheap while
// frames stream in.
constexpr qint64 kOnlineRefreshMinMs = 33;  /**< 30 fps refresh floor */
constexpr qint64 kOnlineRefreshMaxMs = 250; /**< burst-coalesce ceiling */

bool isLingbotOutputEntity(const ccHObject* obj) {
    if (!obj) return false;
    const QString name = obj->getName();
    if (name.startsWith(QStringLiteral("LingbotMap_"))) return true;
    return obj->getMetaData(QStringLiteral("LingbotMap")).isValid();
}

float medianBaseline(const std::vector<float>& distances) {
    if (distances.empty()) return 0.f;
    std::vector<float> sorted(distances);
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    return n % 2 == 1 ? sorted[n / 2]
                      : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
}

/** Camera center (row-major 4x4 c2w, translation column). */
CCVector3f cameraCenter(const float* c2w) {
    return CCVector3f(c2w[3], c2w[7], c2w[11]);
}

/** COLMAP-style camera sensor from a row-major cam2world pose (same
 *  VtkColmap convention as ModelViewerWidget / qFreeSplatter). The frustum
 *  plane width in world units is \p imageDisplaySize. */
ccCameraSensor* buildCameraSensor(const float* rowMajorCam2world,
                                  float focalPx,
                                  int imageWidth,
                                  int imageHeight,
                                  float imageDisplaySize) {
    if (!rowMajorCam2world || imageWidth < 1 || imageHeight < 1 ||
        imageDisplaySize <= 0.f) {
        return nullptr;
    }
    const float displayFocalMm =
            ecvCameraSensorDisplay::ComputeFrustumDisplayFocalMm(
                    imageDisplaySize, imageWidth, imageHeight, focalPx);
    const float viewportVFovRad =
            ecvCameraSensorDisplay::ComputeVerticalFovRad(focalPx, imageHeight);

    auto* sensor = new ccCameraSensor();
    sensor->setPoseFrame(ccCameraSensor::PoseFrame::VtkColmap);
    sensor->setPlaneColor(kColmapCameraPlaneColor);
    sensor->setFrameColor(kColmapCameraFrameColor);

    int retinaScale = 1;
    if (auto* view = ecvViewManager::instance().getEffectiveView()) {
        retinaScale = std::max(view->getDevicePixelRatio(), 1);
    }

    ccCameraSensor::IntrinsicParameters iParams;
    iParams.zNear_mm = 1e-3f;
    const float estImagePlaneDepth = std::abs(displayFocalMm);
    iParams.zFar_mm =
            std::max(std::max(estImagePlaneDepth * 4.0f, 1e-3f), 1e-3f);
    iParams.pixelSize_mm[0] = imageDisplaySize;
    iParams.pixelSize_mm[1] = imageDisplaySize;
    iParams.vertFocal_pix = ccCameraSensor::ConvertFocalMMToPix(
            displayFocalMm, imageDisplaySize);
    iParams.vFOV_rad = viewportVFovRad;
    iParams.arrayWidth = imageWidth;
    iParams.arrayHeight = imageHeight;
    iParams.principal_point[0] = static_cast<float>(imageWidth) * 0.5f;
    iParams.principal_point[1] = static_cast<float>(imageHeight) * 0.5f;
    sensor->setIntrinsicParameters(iParams);
    sensor->setApplyViewportVFov_rad(viewportVFovRad);
    sensor->setGraphicScale(PC_ONE /
                            static_cast<PointCoordinateType>(retinaScale));
    sensor->setRigidTransformation(
            ecvCameraSensorDisplay::RowMajorCam2worldToVtkCameraSensorMatrix(
                    rowMajorCam2world));
    sensor->setEnabled(true);
    sensor->setVisible(true);
    sensor->setLocked(true);
    return sensor;
}

}  // namespace

qLingbotMap::qLingbotMap(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qLingbotMap/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qLingbotMap"));
    qRegisterMetaType<LingbotRunResult>("LingbotRunResult");
    qRegisterMetaType<LingbotFrameResult>("LingbotFrameResult");
    qRegisterMetaType<LingbotFramePreview>("LingbotFramePreview");
    qRegisterMetaType<LingbotAlignedPreview>("LingbotAlignedPreview");
    qRegisterMetaType<LingbotMapWorker::Settings>("LingbotMapWorker::Settings");
    m_action = new QAction(tr("LingBot-Map Reconstruction"), this);
    m_action->setToolTip(
            tr("LingBot-Map (Geometric Context Transformer) streaming RGB-D "
               "3D reconstruction from an image sequence via GGML"));
    m_action->setIcon(QIcon(":/CC/plugin/qLingbotMap/images/qLingbotMap.svg"));
    connect(m_action, &QAction::triggered, this, &qLingbotMap::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(10000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 10;
        m_dialog->appendLog(tr("[LingbotMap] Task is running (%1 s elapsed)...")
                                    .arg(m_inferenceElapsedSeconds));
    });

    // Loop playback (official viewer Playing/FPS semantics).
    m_playbackTimer = new QTimer(this);
    connect(m_playbackTimer, &QTimer::timeout, this,
            &qLingbotMap::onPlaybackTick);
}

QList<QAction*> qLingbotMap::getActions() { return {m_action}; }

void qLingbotMap::showDialog() {
    if (!m_dialog) {
        m_dialog = new LingbotMapDialog(
                m_app != nullptr ? m_app->getMainWindow() : nullptr);
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &LingbotMapDialog::runRequested, this,
                &qLingbotMap::executeTask);
        connect(m_dialog, &LingbotMapDialog::cancelRequested, this,
                &qLingbotMap::cancelTask);
        // Esc/reject bypasses closeEvent: route it through the same
        // cooperative cancel so closing for good always stops the task.
        connect(m_dialog, &LingbotMapDialog::rejected, this,
                &qLingbotMap::cancelTask);
        connect(m_dialog, &LingbotMapDialog::playbackSettingsChanged, this,
                &qLingbotMap::onPlaybackSettingsChanged);
        connect(m_dialog, &LingbotMapDialog::playbackReset, this,
                &qLingbotMap::onPlaybackReset);
        connect(m_dialog, &LingbotMapDialog::replayPlayRequested, this,
                &qLingbotMap::onReplayPlay);
        connect(m_dialog, &LingbotMapDialog::replayPauseRequested, this,
                &qLingbotMap::onReplayPause);
        connect(m_dialog, &LingbotMapDialog::replayStopRequested, this,
                &qLingbotMap::onReplayStop);
        connect(m_dialog, &LingbotMapDialog::replaySeekRequested, this,
                &qLingbotMap::onReplaySeek);
        connect(m_dialog, &LingbotMapDialog::liveViewModeChanged, this,
                &qLingbotMap::onLiveViewModeChanged);
        connect(m_dialog, &LingbotMapDialog::resultDisplaySettingsChanged, this,
                &qLingbotMap::onResultDisplayChanged);
        connect(m_dialog, &LingbotMapDialog::resultViewRequested, this,
                &qLingbotMap::onResultViewRequested);
        connect(m_dialog, &LingbotMapDialog::pauseRequested, this,
                [this](bool paused) {
                    if (!m_worker) return;
                    if (paused) {
                        m_worker->requestPause();
                        m_dialog->appendLog(
                                tr("[LingbotMap] Pause "
                                   "requested..."));
                    } else {
                        m_worker->requestResume();
                        m_dialog->appendLog(tr("[LingbotMap] Resumed."));
                    }
                });
    }
    // Publish the persisted FPS/mode/auto-play values that were loaded
    // while nothing was connected yet (the dialog constructor emits into
    // the void).
    m_dialog->publishPlaybackSettings();
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qLingbotMap::executeTask(const LingbotMapWorker::Settings& settings) {
    if (m_worker && m_worker->isRunning()) {
        QMessageBox::warning(m_dialog, tr("LingBot-Map"),
                             tr("A reconstruction task is already running."));
        return;
    }
    m_worker = new LingbotMapWorker(settings, this);
    // Dual log sink: the dialog panel is the interactive view; CVLog (the
    // app console) persists the SAME lines into the session log file. Without
    // the file sink, a failed run leaves NO trace in logs/ACloudViewer.*.log
    // and parameter questions cannot be answered after the fact.
    auto forwardLog = [this](const QString& msg) {
        m_dialog->appendLog(msg);
        CVLog::Print(msg);
    };
    connect(m_worker, &LingbotMapWorker::logMessage, this, forwardLog);
    // Stage dedup: the worker re-emits the SAME stage text every frame
    // ("Streaming reconstruction (286 frames)") with a new percentage —
    // only a CHANGED stage is logged (the progress bar always updates).
    // mutable value capture: the functor outlives this stack frame.
    QString lastStage;
    connect(m_worker, &LingbotMapWorker::taskStage, this,
            [this, lastStage, forwardLog](const QString& stage,
                                          int percent) mutable {
                if (percent >= 0) m_dialog->setProgress(percent);
                if (stage != lastStage) {
                    lastStage = stage;
                    forwardLog(stage);
                }
            });
    connect(m_worker, &LingbotMapWorker::framePreviewReady, this,
            &qLingbotMap::onFramePreview);
    connect(m_worker, &LingbotMapWorker::alignedPreviewReady, this,
            &qLingbotMap::onAlignedPreview);
    connect(m_worker, &LingbotMapWorker::previewImage, m_dialog,
            &LingbotMapDialog::updatePreviewImage);
    connect(m_worker, &LingbotMapWorker::pauseStateChanged, m_dialog,
            &LingbotMapDialog::setPaused);
    connect(m_worker, &LingbotMapWorker::pauseStateChanged, this,
            [this](bool paused) {
                m_dialog->appendLog(paused ? tr("[LingbotMap] Stream "
                                                "paused (no engine "
                                                "frame is consumed).")
                                           : tr("[LingbotMap] Stream "
                                                "resumed."));
            });
    connect(m_worker, &LingbotMapWorker::resultReady, this,
            &qLingbotMap::onResultReady);
    connect(m_worker, &LingbotMapWorker::taskFinished, this,
            &qLingbotMap::onTaskFinished);
    m_inferenceElapsedSeconds = 0;
    m_dialog->setTaskRunning(true);
    m_inferenceHeartbeat->start();
    m_pendingResult = LingbotRunResult{};
    m_lastSettings = settings;

    // Fresh online-preview state for this run (official StreamingViewer:
    // the scene grows frame by frame while the engine streams).
    disposeOnlineGroup();
    m_lastOnlineSensor = nullptr;  // died with the online group
    // The previous ride may have left the view's apply-viewport preview
    // state armed (stale sensor id + overlay); clear it before the new run.
    if (m_app) {
        clearAppliedViewportCameraPreview(m_app->getActiveGLDisplay());
    }
    m_onlineBaselines.clear();
    m_lastPreviewC2w.clear();
    m_onlineRefreshThrottle.start();
    // A previous result's replay must not keep running into the new run.
    stopPlaybackAndRestore();
    // A fresh run starts from a CLEAN SLATE: drop the previous run's
    // materialized result group (its cameras/trajectory/point clouds would
    // otherwise pile up next to the new run's output — the user sees the
    // old camera trail at full density while the new reconstruction is
    // still on its first frames and reads it as "replay not syncing").
    // The online preview group is disposed above; this targets the
    // materialized result group(s) only.
    if (ccHObject* rootDb = m_app ? m_app->dbRootObject() : nullptr) {
        std::vector<ccHObject*> doomed;
        for (unsigned i = 0; i < rootDb->getChildrenNumber(); ++i) {
            ccHObject* child = rootDb->getChild(i);
            if (child && isLingbotOutputEntity(child) &&
                !child->getName().startsWith(
                        QStringLiteral("LingbotMap_Online"))) {
                doomed.push_back(child);
            }
        }
        for (ccHObject* child : doomed) {
            m_app->removeFromDB(child, /*autoDelete=*/true);
        }
        if (!doomed.empty()) {
            m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/false);
        }
    }
    // Refresh the replay UI after the reset (a stale "1 / 286" from the
    // previous run would otherwise linger until the next materialization).
    syncReplayStateToDialog();
    m_onlineCloudPoints = 0;
    m_alignedBatchCount = 0;
    m_refreshCostSamples.clear();
    m_refreshCostTotalMs = 0;
    m_lastFrameHasPoints = false;
    m_lastFrameRadius = 0.0;
    std::memset(m_lastFrameC2w, 0, sizeof(m_lastFrameC2w));
    m_frameCameras.clear();
    m_frameClouds.clear();
    m_replayResultIndex.clear();
    m_resultGroupId = 0;
    m_resultFramesFolder = nullptr;
    m_resultFramesFolderId = 0;
    m_resultConfApplied = m_resultConfMin;
    m_resultShowCameras = true;
    m_resultCamStride = 1;
    m_resultCamSizePercent = 100;
    if (m_dialog) {
        m_dialog->setResultDisplayReady(false, m_resultConfMin);
        // No result yet: disable the replay transport.
        syncReplayStateToDialog();
    }

    m_worker->start();
    m_dialog->appendLog(tr("[LingbotMap] Task started."));
}

void qLingbotMap::cancelTask() {
    if (m_worker && m_worker->isRunning()) {
        m_worker->requestTaskCancel();
        m_dialog->appendLog(tr("[LingbotMap] Cancellation requested…"));
    }
}

void qLingbotMap::onResultReady(const LingbotRunResult& result) {
    m_pendingResult = result;
}

void qLingbotMap::onTaskFinished(bool success) {
    m_inferenceHeartbeat->stop();
    if (m_dialog) m_dialog->setTaskRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    // End-of-task stage probes: the crash log's last stage brackets the
    // throwing call. Print level — no dialogs.
    auto taskStageProbe = [this](const char* stage) {
        CVLog::Print(QStringLiteral("[LingbotMap] stage: %1").arg(stage));
    };
    taskStageProbe("entry");
    if (success) {
        // The transient online preview is replaced by the final (full-res,
        // aligned) result.
        disposeOnlineGroup();
        m_lastOnlineSensor = nullptr;  // died with the online group
        // The disposed sensor's apply-viewport preview state (and its
        // screen-space overlay) must not leak into the result view.
        if (m_app) {
            clearAppliedViewportCameraPreview(m_app->getActiveGLDisplay());
        }
        taskStageProbe("online group disposed");
    } else {
        // A failed run's online preview (the aligned cross-window map
        // included) is the ONLY reconstruction output it produced — keep
        // it in the DB tree for inspection instead of deleting the user's
        // evidence; the next executeTask() disposes it on restart.
        if (m_dialog && m_onlineGroup) {
            m_dialog->appendLog(
                    tr("[LingbotMap] Partial online preview (aligned map) "
                       "kept in the DB tree for inspection."));
        }
        // No full-res result exists for a failed run: never fall through
        // to the materialization below (m_pendingResult may still hold the
        // PREVIOUS run's frames).
        // End the ride WITHOUT disposing: un-hides the last ridden sensor
        // and clears the armed apply-viewport preview so the kept preview's
        // last frustum renders again.
        endLiveCameraRide();
        return;
    }
    if (m_dialog) {
        m_dialog->appendLog(
                tr("[LingbotMap] Task finished (%1 frames, %2 s).")
                        .arg(m_pendingResult.frames.size())
                        .arg(m_pendingResult.elapsedMs / 1000.0, 0, 'f', 1));
        // Live-preview refresh latency report (acceptance budget: the
        // main-thread refresh path must stay well under 50 ms; the
        // adaptive interval back-off keeps the UI at up to 30 fps).
        if (!m_refreshCostSamples.empty()) {
            std::vector<qint64> sorted = m_refreshCostSamples;
            std::sort(sorted.begin(), sorted.end());
            const qint64 p95 =
                    sorted[static_cast<size_t>(0.95 * (sorted.size() - 1))];
            const double avg = static_cast<double>(m_refreshCostTotalMs) /
                               static_cast<double>(sorted.size());
            m_dialog->appendLog(
                    tr("[LingbotMap] Live preview refresh: avg %1 ms, "
                       "p95 %2 ms over %3 refreshes (%4 preview points, "
                       "%5 window batches).")
                            .arg(avg, 0, 'f', 1)
                            .arg(p95)
                            .arg(sorted.size())
                            .arg(m_onlineCloudPoints)
                            .arg(m_alignedBatchCount));
        }
    }
    taskStageProbe("post-finish bookkeeping");
    if (m_lastSettings.addResultToDb && !m_pendingResult.frames.isEmpty()) {
        taskStageProbe("materialization begin");
        if (addResultToDb(m_pendingResult, m_lastSettings) && m_dialog) {
            taskStageProbe("materialization done");
            // Enable the official-viewer display sliders over the fresh
            // result (threshold slider bottoms at the materialization
            // threshold).
            m_dialog->setResultDisplayReady(true, m_resultConfMin);
        }
    }
}

void qLingbotMap::onFramePreview(const LingbotFramePreview& preview) {
    if (!m_app) return;
    // Streaming runs only (the worker skips per-frame previews for windowed
    // runs): the single stream IS the global coordinate frame, so the
    // per-frame points accumulate exactly like the official StreamingViewer.
    // The user removed the online group mid-run: drop the preview instead
    // of rebuilding it behind their back.
    if (m_onlineGroup && m_app->dbRootObject() &&
        !m_app->dbRootObject()->find(m_onlineGroup->getUniqueID())) {
        m_onlineGroup = nullptr;
    }

    ccHObject* parent = ensureOnlineGroup();
    if (!parent) return;

    // Official fused-viewer structure: ONE IMMUTABLE cloud per frame (the
    // same "LingbotMap_frames/LingbotMap_frame_%06d" shape as the final
    // result). Immutability is what keeps the live refresh O(new points):
    // the engine's per-entity draw gate (VtkDisplayTools::drawPointCloud →
    // checkEntityNeedUpdate) only rebuilds a cloud's vtkPolyData when its
    // point count or bounds change, so a stable per-frame cloud costs
    // nothing after its creation pass, while a single growing cloud forced
    // a full O(total) conversion plus a full vertex-buffer re-upload on
    // EVERY refresh — the stall that grew with each streamed frame.
    if (!preview.points.isEmpty()) {
        if (ccPointCloud* cloud =
                    onlineFrameCloud(parent, preview.globalIndex)) {
            const unsigned n = static_cast<unsigned>(preview.points.size() / 3);
            const unsigned cur = cloud->size();
            // Exact reserve: the cloud is written exactly once, then never
            // mutated again — no slack growth needed. The points buffer
            // MUST be allocated before the RGB table: ccPointCloud forbids
            // properties allocation ahead of points (the log-spam
            // "reserveTheRGBTable ... before points allocation is
            // forbidden" bug that silently stripped every preview color).
            if (cloud->reserve(cur + n)) {
                bool hasColors = !preview.colors.isEmpty();
                if (hasColors && !cloud->hasColors() &&
                    !cloud->reserveTheRGBTable()) {
                    hasColors = false;
                }
                const float* xyz = preview.points.constData();
                for (unsigned i = 0; i < n; ++i) {
                    cloud->addPoint(CCVector3(xyz[i * 3], xyz[i * 3 + 1],
                                              xyz[i * 3 + 2]));
                    if (hasColors) {
                        cloud->addRGBColor(preview.colors[i * 3],
                                           preview.colors[i * 3 + 1],
                                           preview.colors[i * 3 + 2]);
                    }
                }
                cloud->showColors(hasColors);
                m_onlineCloudPoints += n;
            }
        }
    }
    addOnlineCamera(preview.c2w.constData(), preview.intrinsics, preview.width,
                    preview.height, preview.globalIndex);

    // Live camera FIRST (geometry only — it rewrites the VTK camera), so
    // the redraw below renders the new entities AND the new pose in one
    // pass. Live modes re-render every streamed frame (the engine's frame
    // pace is the real pacemaker); Global keeps the adaptive throttle.
    updateLiveViewCamera(preview.c2w.constData(), preview.points.constData(),
                         static_cast<unsigned>(preview.points.size() / 3),
                         preview.globalIndex + 1);
    const bool cameraLive = m_lastSettings.liveViewMode > 0;
    if (!cameraLive &&
        m_onlineRefreshThrottle.elapsed() < onlineRefreshIntervalMs()) {
        return;
    }
    m_onlineRefreshThrottle.restart();
    QElapsedTimer refreshTimer;
    refreshTimer.start();
    { ecvRedrawScope scope(/*only2D=*/false, /*forceRedraw=*/false); }
    const qint64 cost = refreshTimer.elapsed();
    m_refreshCostSamples.push_back(cost);
    m_refreshCostTotalMs += cost;
    if (m_refreshCostSamples.size() > 128) {
        m_refreshCostTotalMs -= m_refreshCostSamples.front();
        m_refreshCostSamples.erase(m_refreshCostSamples.begin());
    }
}

qint64 qLingbotMap::onlineRefreshIntervalMs() const {
    if (m_refreshCostSamples.empty()) return kOnlineRefreshMinMs;
    qint64 worst = 0;
    for (qint64 c : m_refreshCostSamples) worst = std::max(worst, c);
    // Back off to ~2x the worst measured refresh cost, clamped to
    // [33ms (30 fps), 250ms] — the frame-arrival pace of the engine is
    // the real pacemaker; this only coalesces bursts.
    const qint64 adapted = 2 * worst;
    return std::min(kOnlineRefreshMaxMs,
                    std::max(kOnlineRefreshMinMs, adapted));
}

void qLingbotMap::updateLiveViewCamera(const float* c2w,
                                       const float* xyz,
                                       unsigned n,
                                       int frameCount1Based) {
    ecvGenericGLDisplay* display =
            m_app ? m_app->getActiveGLDisplay() : nullptr;
    if (!display) return;
    const int mode = m_lastSettings.liveViewMode;

    if (!xyz || n == 0) {
        // Degenerate frame (frustum only): nothing to cache; the Follow
        // ride still tracks the newest camera pose.
        if (mode == 2 && m_lastOnlineSensor &&
            m_lastOnlineSensor->getDisplay()) {
            m_lastOnlineSensor->applyViewport();
        }
        return;
    }

    // Latest-frame extent (preview points are already world-space).
    // Computed and cached for EVERY mode — Global included: a mid-run
    // switch to Fit/Follow applies this cached state immediately
    // (onLiveViewModeChanged), so it must stay fresh even while the active
    // policy never consumes it per-frame. O(n) over the decimated preview
    // points only — negligible against the per-frame refresh budget.
    CCVector3d center(0, 0, 0);
    for (unsigned i = 0; i < n; ++i) {
        center += CCVector3d(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]);
    }
    center /= static_cast<double>(n);
    double radius = 0.0;
    for (unsigned i = 0; i < n; ++i) {
        radius = std::max(radius, (CCVector3d(xyz[i * 3], xyz[i * 3 + 1],
                                              xyz[i * 3 + 2]) -
                                   center)
                                          .norm());
    }
    radius = std::max(radius, 1e-3);

    // Cache the extent/pose so a mid-run Live camera mode switch can be
    // applied immediately (onLiveViewModeChanged) without waiting for the
    // next streamed frame.
    m_lastFrameCenter = center;
    m_lastFrameRadius = radius;
    if (c2w) std::memcpy(m_lastFrameC2w, c2w, 16 * sizeof(float));
    m_lastFrameHasPoints = true;

    if (mode <= 0) {
        // Global view: the user owns the camera, never touch it.
        return;
    }

    // mode == 1 ("Fit latest frame", user-verified official StreamingViewer
    // behavior): the view tracks the NEWEST frame EVERY streamed frame — the
    // fixed diagonal overhead eye over the growing extent makes the visible
    // motion a smooth follow, not a static vantage. The extent/pose are
    // cached above for every mode, so this is O(1) per frame and falls
    // through to the applyLiveCameraPose call below.

    if (mode == 2) {
        // Follow = the sensor's OWN viewport (official apply-view parity):
        // ccCameraSensor::applyViewport routes the sensor pose + the
        // aspect-aware FOV through the engine's projective-viewport setup —
        // the exact same chain as "Apply view" on the camera entity, so the
        // render window matches the input-image viewpoint exactly (the
        // hand-rolled FromViewDirAndUpDir variant left the map sitting
        // low/off-center in the frame).
        if (m_lastOnlineSensor && m_lastOnlineSensor->getDisplay()) {
            m_lastOnlineSensor->applyViewport();
        }
        return;
    }
    applyLiveCameraPose(mode, c2w, center, radius);
}

void qLingbotMap::applyLiveCameraPose(int mode,
                                      const float* c2w,
                                      const CCVector3d& center,
                                      double radius,
                                      double pullback) {
    ecvGenericGLDisplay* display =
            m_app ? m_app->getActiveGLDisplay() : nullptr;
    if (!display) return;

    ecvViewportParameters vp = display->getViewportParameters();
    vp.perspectiveView = true;
    vp.objectCenteredView = false;  // eye-centered: focal = pos + dir*dist

    CCVector3d eye;
    CCVector3d forward;
    CCVector3d up(0, 0, 1);
    if (mode == 1) {
        // Official viewer "Fit latest frame": a TOP-DOWN FOLLOW eye — the
        // camera hovers ABOVE the newest frame's center looking straight
        // down, so the reconstruction grows underneath a bird's-eye view
        // that tracks the travel. Height scales with the frame radius so
        // the newest frame fills the view; world +Y is the screen-up.
        eye = center + CCVector3d(0, 0, std::max(2.2 * radius, 1e-3));
        forward = center - eye;  // straight down
        up = CCVector3d(0, 1, 0);
    } else {
        // Follow latest camera: sit at the frustum's camera center and
        // look along its optical axis (OpenCV convention: +Z forward,
        // image -Y up — same as the worker's unprojection).
        if (!c2w) return;
        eye = CCVector3d(c2w[3], c2w[7], c2w[11]);
        forward = CCVector3d(c2w[2], c2w[6], c2w[10]);
        up = CCVector3d(-static_cast<double>(c2w[1]),
                        -static_cast<double>(c2w[5]),
                        -static_cast<double>(c2w[9]));
    }
    if (forward.norm() < 1e-12) return;
    forward.normalize();
    // "Go to camera" pullback: the eye otherwise sits EXACTLY on the
    // camera center, with the frustum's 50%-alpha near plane a frustum
    // scale away — it fills the whole screen. Backing off along the view
    // direction keeps the first frame's viewpoint readable.
    if (pullback > 0.0) {
        eye -= forward * pullback;
    }

    // FromViewDirAndUpDir builds the world→camera rotation (view dir =
    // -Z) and re-orthogonalizes the up vector internally. The camera
    // position lives in the (protected) cameraCenter field — set via
    // setCameraCenter; setViewportParameters derives the VTK camera
    // position from it (inv(computeViewMatrix).translation). The caller
    // must trigger a Render AFTER this for the pose to be visible.
    vp.viewMat = ccGLMatrixd::FromViewDirAndUpDir(forward, up);
    vp.setCameraCenter(eye, /*autoUpdateFocal=*/false);
    // Keep the newest frame comfortably inside the view.
    vp.setFocalDistance(std::max(2.0 * radius, 1e-3));
    display->setViewportParameters(vp);
}

void qLingbotMap::endLiveCameraRide() {
    if (m_lastOnlineSensor) {
        m_lastOnlineSensor->setVisible(true);
        m_lastOnlineSensor = nullptr;
    }
    // The Follow ride routes every streamed frame through the sensor's
    // applyViewport, which arms the view's apply-viewport preview state:
    // the ridden sensor's gizmo draw is suppressed (drawMeOnly early-out)
    // and its actors are removed until the state is cleared. Left armed,
    // the last ridden frustum never renders again — its first-draw
    // forceRedraw latch is long gone, so the plain-redraw escape path
    // cannot recreate the actors. The engine's
    // clearAppliedViewportCameraPreview un-suppresses, removes the
    // preview overlay and force-redraws; it no-ops when no preview is
    // armed.
    if (m_app) {
        clearAppliedViewportCameraPreview(m_app->getActiveGLDisplay());
    }
}

void qLingbotMap::onLiveViewModeChanged(int mode) {
    if (m_lastSettings.liveViewMode == mode) return;
    m_lastSettings.liveViewMode = mode;
    // Leaving Follow: end the ride — the ridden online sensor becomes a
    // normal frustum again (it was hidden exactly because the eye sat on
    // it) and the view's apply-viewport preview state is cleared so its
    // frustum is drawn again. This also supplies the redraw for the
    // Global-view case below (which never touches the camera).
    if (mode != 2) {
        endLiveCameraRide();
    }
    if (m_dialog) {
        m_dialog->appendLog(tr("[LingbotMap] Live camera mode: %1 (applies "
                               "to the running reconstruction).")
                                    .arg(mode == 2 ? tr("Follow latest camera")
                                         : mode == 1 ? tr("Fit latest frame")
                                                     : tr("Global view")));
    }
    if (mode > 0 && m_lastFrameHasPoints) {
        // Immediate feedback: re-apply from the cached latest frame and
        // render once (no next-frame wait). Follow goes through the
        // sensor's applyViewport (same chain as the per-frame path).
        if (mode == 2 && m_lastOnlineSensor &&
            m_lastOnlineSensor->getDisplay()) {
            m_lastOnlineSensor->applyViewport();
        } else {
            applyLiveCameraPose(mode, m_lastFrameC2w, m_lastFrameCenter,
                                m_lastFrameRadius);
        }
        if (m_app) m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/false);
    }
}

void qLingbotMap::onPlaybackReset() {
    // Reset replay == Stop (restore every frame + rewind). The result-
    // display camera state (Show Camera / stride) stays authoritative on
    // top of the all-visible baseline inside stopPlaybackAndRestore.
    stopPlaybackAndRestore();
}

void qLingbotMap::onResultDisplayChanged(double threshold,
                                         int pointSize,
                                         int camSizePercent,
                                         bool showCameras,
                                         int camStride) {
    if (!m_app) return;
    // Result-display state cache (also consumed by onPlaybackReset).
    m_resultShowCameras = showCameras;
    m_resultCamStride = std::max(1, camStride);
    m_resultCamSizePercent = std::max(1, camSizePercent);

    // 1. Visibility threshold re-filter (official Visibility Threshold
    //    slider). The re-filter can only TIGHTEN the display: points below
    //    the materialization threshold were never materialized. When it
    //    fires it RE-MATERIALIZES the whole result group, so the entity
    //    caches are rebuilt and the camera/point-size state must be
    //    (re-)applied afterwards — hence the ordering below.
    bool rematerialized = false;
    if (m_resultFramesFolder && m_resultFramesFolderId != 0 &&
        std::abs(threshold - m_resultConfApplied) > 1e-6) {
        refilterResultClouds(std::max(threshold, m_resultConfMin));
        rematerialized = true;
    }

    // 2. Camera display state (official Show Camera / Camera Downsample /
    //    Camera Size sliders): pure visibility + graphic-scale writes, no
    //    per-entity redraw. During playback the replay formula below owns
    //    the visibility (density is applied live); when stopped,
    //    applyCameraDisplay is the writer.
    applyCameraDisplay(showCameras, camStride, camSizePercent);
    if (m_replayState != ReplayState::Stopped) {
        applyReplayVisibility();
    }

    // 3. Point size (official Point Size slider): the VTK property path —
    //    actor-level SetPointSize, no geometry rebuild.
    if (ecvGenericGLDisplay* display = m_app->getActiveGLDisplay()) {
        const unsigned char size = static_cast<unsigned char>(
                std::max(1, std::min(16, pointSize)));
        for (ccPointCloud* cloud : m_frameClouds) {
            if (!cloud) continue;
            cloud->setPointSize(size);
            PROPERTY_PARAM param(cloud, size);
            param.viewId = cloud->getViewId();
            display->changeEntityProperties(param);
        }
    }

    // One forced render for the whole batch (a rematerialization already
    // rendered through addToDB; property/visibility edits render nothing
    // on their own).
    if (!rematerialized) {
        m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/true);
    }
}

void qLingbotMap::refilterResultClouds(double threshold) {
    if (!m_app) return;
    // Official _regenerate_point_clouds equivalent: the visibility-threshold
    // change re-materializes the whole result group from the engine result
    // (m_pendingResult holds the raw per-frame depth/conf within the
    // session) with the new threshold. A GROUP-level removal is mandatory
    // here: per-entity folder->removeChild()+delete bypasses the DB tree's
    // QAbstractItemModel row bookkeeping (ccDBRoot::removeElement does
    // beginRemoveRows/endRemoveRows) and would leave dangling model rows.
    // This is also strictly closer to the official semantics than
    // re-filtering the already-downsampled points: the official viewer
    // re-runs conf filtering AND the arange downsample from the full data.
    if (m_pendingResult.frames.isEmpty() || m_resultGroupId == 0) {
        if (m_dialog) {
            m_dialog->appendLog(
                    tr("[LingbotMap] Visibility threshold needs the last "
                       "reconstruction result (session-scoped); re-run to "
                       "use it."));
        }
        return;
    }
    ccHObject* root = m_app->dbRootObject();
    ccHObject* group = root ? root->find(m_resultGroupId) : nullptr;
    if (!group) {
        if (m_dialog) {
            m_dialog->appendLog(
                    tr("[LingbotMap] The result group was deleted; nothing "
                       "to re-filter."));
        }
        return;
    }

    LingbotMapWorker::Settings settings = m_lastSettings;
    settings.confThreshold = static_cast<float>(threshold);
    // Hard-stop the replay BEFORE the group dies: the cached sensors and
    // clouds are about to be deleted.
    if (m_playbackTimer) m_playbackTimer->stop();
    m_replayState = ReplayState::Stopped;
    m_app->removeFromDB(group, /*autoDelete=*/true);
    m_frameClouds.clear();
    m_frameCameras.clear();
    m_replayResultIndex.clear();
    m_resultGroupId = 0;
    m_resultFramesFolder = nullptr;
    m_resultFramesFolderId = 0;
    if (addResultToDb(m_pendingResult, settings)) {
        // Re-base the dialog's threshold slider on the new materialization
        // threshold (addResultToDb updated m_resultConfMin).
        if (m_dialog) {
            m_dialog->setResultDisplayReady(true, m_resultConfMin);
            m_dialog->appendLog(
                    tr("[LingbotMap] Visibility threshold %1: result "
                       "re-materialized.")
                            .arg(threshold, 0, 'f', 2));
        }
    }
}

void qLingbotMap::applyCameraDisplay(bool showCameras,
                                     int camStride,
                                     int camSizePercent) {
    const int stride = std::max(1, camStride);
    const PointCoordinateType scale = static_cast<PointCoordinateType>(
            m_cameraBaseScale * std::max(1, camSizePercent) / 100.0);
    // Replay owns camera visibility while active: a slider tick (Camera
    // size % / stride / Show cameras fire on valueChanged) would otherwise
    // stomp the incremental per-frame visibility set by onPlaybackTick —
    // the "playback shows all cameras" symptom. Graphic scale stays live;
    // the replay baseline is re-applied by stopPlaybackAndRestore on Stop.
    const bool replayOwnsVisibility = m_replayState != ReplayState::Stopped;
    int frameIndex = 0;
    for (ccHObject* cam : m_frameCameras) {
        if (!cam) continue;
        // Official Camera Downsample Factor semantics: show one frustum
        // every N frames (visibility, NOT the DB-tree enable state).
        if (!replayOwnsVisibility) {
            cam->setVisible(showCameras && (frameIndex % stride == 0));
        }
        if (auto* sensor = ccHObjectCaster::ToSensor(cam)) {
            sensor->setGraphicScale(scale);  // marks the geometry dirty
        }
        ++frameIndex;
    }
}

void qLingbotMap::onResultViewRequested(int which) {
    ecvGenericGLDisplay* display =
            m_app ? m_app->getActiveGLDisplay() : nullptr;
    if (!display) return;
    if (which == 1) {
        // Fit all (official Overview button): refit on the result bbox.
        ccHObject* root = m_app->dbRootObject();
        ccHObject* group = root && m_resultGroupId != 0
                                   ? root->find(m_resultGroupId)
                                   : nullptr;
        if (!group) return;
        ccBBox box = group->getBB_recursive(/*onlyVisibleEntities=*/true,
                                            /*autoCompute=*/false);
        if (!box.isValid()) {
            box = group->getBB_recursive(false, true);
        }
        if (box.isValid()) {
            display->updateConstellationCenterAndZoom(&box);
        }
        return;
    }
    // Move to the first camera (official "First Camera" button): the
    // Follow-camera pose of frame 0, focal distance from the result radius.
    if (m_pendingResult.frames.isEmpty()) return;
    const LingbotFrameResult& frame = m_pendingResult.frames.first();
    if (frame.c2w.size() < 16) return;
    double radius = m_lastFrameRadius;
    if (ccHObject* root = m_app->dbRootObject()) {
        if (ccHObject* group = m_resultGroupId != 0
                                       ? root->find(m_resultGroupId)
                                       : nullptr) {
            const ccBBox box = group->getBB_recursive(true, false);
            if (box.isValid()) {
                radius = box.getDiagNorm() / 2.0;
            }
        }
    }
    applyLiveCameraPose(/*mode=*/2, frame.c2w.constData(),
                        CCVector3d(frame.c2w[3], frame.c2w[7], frame.c2w[11]),
                        radius,
                        /*pullback=*/radius * 0.1);
    m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/true);
}

ccHObject* qLingbotMap::ensureOnlineGroup() {
    if (!m_app) return nullptr;
    if (m_onlineGroup) return m_onlineGroup;
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(m_lastSettings.modelPath);
    m_onlineGroup =
            new ccHObject(QStringLiteral("LingbotMap_Online_%1").arg(modelTag));
    m_onlineGroup->setMetaData(QStringLiteral("LingbotMap"), true);
    m_onlineGroup->setDisplay(m_app->getActiveGLDisplay());
    m_onlineGroup->setEnabled(true);
    m_onlineGroup->setVisible(true);
    m_app->addToDB(m_onlineGroup, /*updateZoom=*/false,
                   /*autoExpandDBTree=*/false, /*checkDimensions=*/false,
                   /*autoRedraw=*/false);
    return m_onlineGroup;
}

ccHObject* qLingbotMap::ensureAlignedFramesFolder() {
    ccHObject* group = ensureOnlineGroup();
    if (!group) return nullptr;
    for (unsigned i = 0; i < group->getChildrenNumber(); ++i) {
        ccHObject* child = group->getChild(i);
        if (child &&
            child->getName() == QStringLiteral("LingbotMap_aligned_frames")) {
            return child;
        }
    }
    auto* folder = new ccHObject(QStringLiteral("LingbotMap_aligned_frames"));
    folder->setMetaData(QStringLiteral("LingbotMap"), true);
    folder->setDisplay(m_app->getActiveGLDisplay());
    folder->setVisible(true);
    group->addChild(folder);
    return folder;
}

ccPointCloud* qLingbotMap::onlineAlignedBatchCloud(int windowIndex) {
    ccHObject* folder = ensureAlignedFramesFolder();
    if (!folder) return nullptr;
    const QString name = QStringLiteral("LingbotMap_aligned_w%1")
                                 .arg(windowIndex, 4, 10, QLatin1Char('0'));
    for (unsigned i = 0; i < folder->getChildrenNumber(); ++i) {
        ccHObject* child = folder->getChild(i);
        if (child && child->getName() == name) {
            return ccHObjectCaster::ToPointCloud(child);
        }
    }
    auto* cloud = new ccPointCloud(name);
    cloud->setMetaData(QStringLiteral("LingbotMap"), true);
    cloud->setDisplay(m_app->getActiveGLDisplay());
    cloud->setVisible(true);
    // First-draw latch (same escape-path mechanism as onlineFrameCloud).
    cloud->setForceRedraw(true);
    folder->addChild(cloud);
    return cloud;
}

void qLingbotMap::onAlignedPreview(const LingbotAlignedPreview& preview) {
    if (!m_app) return;
    // The user removed the online group mid-run: rebuild it for the
    // aligned map (same policy as onFramePreview).
    if (m_onlineGroup && m_app->dbRootObject() &&
        !m_app->dbRootObject()->find(m_onlineGroup->getUniqueID())) {
        m_onlineGroup = nullptr;
    }
    if (!preview.points.isEmpty()) {
        if (ccPointCloud* cloud =
                    onlineAlignedBatchCloud(preview.windowIndex)) {
            const unsigned n = static_cast<unsigned>(preview.points.size() / 3);
            const unsigned cur = cloud->size();
            // Exact reserve: a batch cloud is written once, never mutated.
            // Points buffer FIRST, then the RGB table (ccPointCloud forbids
            // properties allocation ahead of points — see onFramePreview).
            if (cloud->reserve(cur + n)) {
                bool hasColors = !preview.colors.isEmpty();
                if (hasColors && !cloud->hasColors() &&
                    !cloud->reserveTheRGBTable()) {
                    hasColors = false;
                }
                const float* xyz = preview.points.constData();
                for (unsigned i = 0; i < n; ++i) {
                    cloud->addPoint(CCVector3(xyz[i * 3], xyz[i * 3 + 1],
                                              xyz[i * 3 + 2]));
                    if (hasColors) {
                        cloud->addRGBColor(preview.colors[i * 3],
                                           preview.colors[i * 3 + 1],
                                           preview.colors[i * 3 + 2]);
                    }
                }
                cloud->showColors(hasColors);
                m_onlineCloudPoints += n;
                ++m_alignedBatchCount;
            }
        }
    }
    // One aligned camera frustum per contributed frame (the c2w values are
    // already in the first window's coordinate frame, like the points).
    const int camCount = static_cast<int>(preview.camFrames.size());
    for (int i = 0; i < camCount; ++i) {
        const int base16 = i * 16;
        const int base4 = i * 4;
        addOnlineCamera(preview.camC2w.constData() + base16,
                        preview.camIntr.mid(base4, 4), preview.width,
                        preview.height, preview.camFrames[i]);
    }
    // Live camera FIRST (same rationale as onFramePreview), then one
    // redraw per window batch: the plain (false,false) scope drives the
    // tree pass whose escape path first-draws exactly the new entities
    // (m_forceRedraw latched at creation, self-cleared after the first
    // draw) and renders with the updated camera. Fires immediately (no
    // throttle: one batch per window) so the accumulated map is visible
    // before the next window's aligned batch lands.
    if (!preview.camFrames.isEmpty()) {
        const int base16 = (preview.camFrames.size() - 1) * 16;
        updateLiveViewCamera(preview.camC2w.constData() + base16,
                             preview.points.constData(),
                             static_cast<unsigned>(preview.points.size() / 3),
                             preview.camFrames.back() + 1);
    }
    {
        QElapsedTimer refreshTimer;
        refreshTimer.start();
        { ecvRedrawScope scope(/*only2D=*/false, /*forceRedraw=*/false); }
        const qint64 cost = refreshTimer.elapsed();
        m_refreshCostSamples.push_back(cost);
        m_refreshCostTotalMs += cost;
        if (m_refreshCostSamples.size() > 128) {
            m_refreshCostTotalMs -= m_refreshCostSamples.front();
            m_refreshCostSamples.erase(m_refreshCostSamples.begin());
        }
    }
}

ccHObject* qLingbotMap::ensureCamerasGroup() {
    ccHObject* group = ensureOnlineGroup();
    if (!group) return nullptr;
    for (unsigned i = 0; i < group->getChildrenNumber(); ++i) {
        ccHObject* child = group->getChild(i);
        if (child && child->getName() == QStringLiteral("LingbotMap_cameras")) {
            return child;
        }
    }
    // Dedicated subfolder: every online camera frustum lives here, never
    // mixed with the point clouds, so the DB tree stays manageable.
    auto* cameras = new ccHObject(QStringLiteral("LingbotMap_cameras"));
    cameras->setMetaData(QStringLiteral("LingbotMap"), true);
    cameras->setDisplay(m_app->getActiveGLDisplay());
    cameras->setEnabled(true);
    cameras->setVisible(true);
    group->addChild(cameras);
    return cameras;
}

void qLingbotMap::addOnlineCamera(const float* c2w,
                                  const QVector<float>& intrinsics,
                                  int width,
                                  int height,
                                  int globalIndex) {
    if (!c2w || intrinsics.isEmpty()) return;
    if (m_lastPreviewC2w.size() >= 16) {
        const CCVector3f prev = cameraCenter(m_lastPreviewC2w.constData());
        const CCVector3f curr = cameraCenter(c2w);
        m_onlineBaselines.push_back((curr - prev).norm());
    }
    m_lastPreviewC2w = QVector<float>(c2w, c2w + 16);
    const float med = medianBaseline(m_onlineBaselines);
    const float displaySize =
            med > 0.f ? med * kCameraBaselineFraction : kFallbackCameraSize;
    const float fx = intrinsics.value(0, 0.f);
    ccCameraSensor* sensor =
            buildCameraSensor(c2w, fx, width, height, displaySize);
    if (!sensor) return;
    sensor->setName(QStringLiteral("LingbotMap_online_cam_%1")
                            .arg(globalIndex, 6, 10, QLatin1Char('0')));
    sensor->setMetaData(QStringLiteral("LingbotMap"), true);
    sensor->setDisplay(m_app->getActiveGLDisplay());
    // First-draw latch (same escape-path mechanism as onlineFrameCloud).
    sensor->setForceRedraw(true);
    // Live-camera Follow mode puts the EYE at this camera's pose — the
    // frustum sitting there wraps the whole view (its red near-plane
    // wireframe rides the screen edge and the side lines sweep past the
    // camera). A camera cannot see its own frustum anyway: hide the ridden
    // sensor while Follow is active (restored on every mode change).
    if (m_lastSettings.liveViewMode == 2) {
        sensor->setVisible(false);
        // Hand the ride over: the previous frame's sensor becomes a normal
        // (visible) historical frustum again.
        if (m_lastOnlineSensor && m_lastOnlineSensor != sensor) {
            m_lastOnlineSensor->setVisible(true);
        }
        m_lastOnlineSensor = sensor;
    } else {
        m_lastOnlineSensor = nullptr;
    }
    if (ccHObject* cameras = ensureCamerasGroup()) {
        cameras->addChild(sensor);
    } else {
        delete sensor;
    }
}

ccPointCloud* qLingbotMap::onlineFrameCloud(ccHObject* parent,
                                            int globalIndex) {
    if (!parent) return nullptr;
    // "LingbotMap_frames" subfolder — the same home the final result uses,
    // so the online preview and the materialized result share one shape.
    ccHObject* framesFolder = nullptr;
    for (unsigned i = 0; i < parent->getChildrenNumber(); ++i) {
        ccHObject* child = parent->getChild(i);
        if (child && child->getName() == QStringLiteral("LingbotMap_frames")) {
            framesFolder = child;
            break;
        }
    }
    if (!framesFolder) {
        framesFolder = new ccHObject(QStringLiteral("LingbotMap_frames"));
        framesFolder->setMetaData(QStringLiteral("LingbotMap"), true);
        framesFolder->setDisplay(m_app->getActiveGLDisplay());
        framesFolder->setVisible(true);
        parent->addChild(framesFolder);
    }
    const QString name = QStringLiteral("LingbotMap_frame_%1")
                                 .arg(globalIndex, 6, 10, QLatin1Char('0'));
    for (unsigned i = 0; i < framesFolder->getChildrenNumber(); ++i) {
        ccHObject* child = framesFolder->getChild(i);
        if (child && child->getName() == name) {
            return ccHObjectCaster::ToPointCloud(child);
        }
    }
    auto* cloud = new ccPointCloud(name);
    cloud->setMetaData(QStringLiteral("LingbotMap"), true);
    cloud->setDisplay(m_app->getActiveGLDisplay());
    cloud->setVisible(true);
    // First-draw latch: the plain (false,false) redraw scope's tree pass
    // re-draws entities with an unset VTK actor through the hide/show
    // escape path only when their m_forceRedraw latch is set; the escape
    // clears the latch itself after the first draw.
    cloud->setForceRedraw(true);
    framesFolder->addChild(cloud);
    return cloud;
}

void qLingbotMap::disposeOnlineGroup() {
    if (!m_app || !m_onlineGroup) return;
    ccHObject* root = m_app->dbRootObject();
    if (root && root->find(m_onlineGroup->getUniqueID())) {
        m_app->removeFromDB(m_onlineGroup, /*autoDelete=*/true);
        m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/false);
    }
    m_onlineGroup = nullptr;
}

bool qLingbotMap::addResultToDb(const LingbotRunResult& result,
                                const LingbotMapWorker::Settings& settings) {
    if (!m_app) return false;
    ccHObject* root = m_app->dbRootObject();
    if (!root) return false;

    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(result.modelFile);
    const QString deviceTag =
            ecvPluginDbNaming::deviceTagFromName(result.device);
    const QString groupBase =
            QStringLiteral("LingbotMap_%1_%2").arg(modelTag, deviceTag);

    ccHObject* group = new ccHObject(groupBase);
    group->setMetaData(QStringLiteral("LingbotMap"), true);
    group->setEnabled(true);
    group->setDisplay(m_app->getActiveGLDisplay());
    group->setVisible(true);

    QElapsedTimer timer;
    timer.start();

    // Camera frustum size from the running median baseline (official
    // viewer semantics) — needs all centers first.
    std::vector<CCVector3f> centers;
    centers.reserve(result.frames.size());
    for (const LingbotFrameResult& frame : result.frames) {
        if (frame.c2w.size() >= 12) {
            centers.push_back(cameraCenter(frame.c2w.constData()));
        }
    }
    std::vector<float> baselineDists;
    for (size_t i = 1; i < centers.size(); ++i) {
        baselineDists.push_back((centers[i] - centers[i - 1]).norm());
    }
    const float medBaseline = medianBaseline(baselineDists);
    const float cameraDisplaySize =
            medBaseline > 0.f ? medBaseline * kCameraBaselineFraction
                              : kFallbackCameraSize;

    // Official fused-viewer output structure (PointCloudViewer): ONE point
    // cloud per frame — there is no merged/fused entity upstream — each
    // filtered by parse_pc_data semantics: finite points with confidence
    // STRICTLY above the threshold (sky pixels arrive with zeroed
    // confidence, the engine's --mask_sky semantics), then the official
    // sequential downsample (indices arange(0, n, --downsample_factor)).
    // parse_pc_data keeps huge finite outliers (no world-bounds filter), so
    // none is applied here either.
    const int downsample = std::max(1, settings.downsampleFactor);
    auto* framesFolder = new ccHObject(QStringLiteral("LingbotMap_frames"));
    framesFolder->setMetaData(QStringLiteral("LingbotMap"), true);
    framesFolder->setDisplay(m_app->getActiveGLDisplay());
    framesFolder->setVisible(true);
    group->addChild(framesFolder);

    unsigned long long totalPoints = 0;
    const float confThreshold = settings.confThreshold;
    m_frameClouds.clear();
    m_frameClouds.reserve(result.frames.size());
    for (int f = 0; f < result.frames.size(); ++f) {
        // Aligned slot: m_frameClouds[f] must track the RESULT-FRAME index
        // (same order as m_frameCameras / m_replayResultIndex) even when a
        // frame produces no cloud — the replay pairs camera i with cloud i.
        m_frameClouds.push_back(nullptr);
        const LingbotFrameResult& frame = result.frames[f];
        const int w = frame.width;
        const int h = frame.height;
        if (frame.depth.isEmpty() || w <= 0 || h <= 0) continue;
        const float fx = frame.intrinsics.value(0, 0.f);
        const float fy = frame.intrinsics.value(1, 0.f);
        const float cx = frame.intrinsics.value(2, 0.f);
        const float cy = frame.intrinsics.value(3, 0.f);
        if (fx <= 0.f || fy <= 0.f) continue;
        if (frame.c2w.size() < 16) continue;

        const size_t plane = static_cast<size_t>(w) * h;
        // pass 1: parse_pc_data's finite + confidence predicate
        std::vector<uint32_t> kept;
        kept.reserve(plane / 2);
        for (size_t i = 0; i < plane; ++i) {
            const float d = frame.depth[i];
            if (!std::isfinite(d)) continue;
            if (frame.depthConf[i] <= confThreshold) continue;
            if (!frame.skyKeep.isEmpty() && frame.skyKeep[i] == 0) continue;
            kept.push_back(static_cast<uint32_t>(i));
        }
        // pass 2: the official sequential downsample stride
        const unsigned count = static_cast<unsigned>(
                kept.size() / static_cast<size_t>(downsample));
        if (count == 0) continue;

        const bool hasRgb = !frame.frameRgb.isNull() &&
                            frame.frameRgb.size() == QSize(w, h);
        const bool fastRgb =
                hasRgb && frame.frameRgb.format() == QImage::Format_RGB888;
        const uchar* rgbBits = fastRgb ? frame.frameRgb.constBits() : nullptr;
        const auto rgbStride = frame.frameRgb.bytesPerLine();
        const float* R = frame.c2w.data();
        const int globalIndex = frame.globalIndex >= 0 ? frame.globalIndex : f;
        auto* cloud = new ccPointCloud(
                QStringLiteral("LingbotMap_frame_%1")
                        .arg(globalIndex, 6, 10, QLatin1Char('0')));
        cloud->setMetaData(QStringLiteral("LingbotMap"), true);
        if (!cloud->reserve(count)) {
            delete cloud;
            continue;
        }
        bool cloudHasRgb = hasRgb;
        if (cloudHasRgb && !cloud->reserveTheRGBTable()) cloudHasRgb = false;
        // Per-point confidence scalar field: the data source of the live
        // Visibility-Threshold re-filter (official viewer slider parity).
        // 4 B per displayed point — negligible against the geometry.
        ccScalarField* confSf = nullptr;
        {
            const int sfIdx = cloud->addScalarField("LingbotMap_conf");
            if (sfIdx >= 0) {
                confSf = static_cast<ccScalarField*>(
                        cloud->getScalarField(sfIdx));
                // CRITICAL: a freshly added SF has ZERO elements — writing
                // setValue(0, ...) on it threw vector::at out_of_range
                // (the end-of-task crash). Size it to the reserved point
                // count BEFORE any per-point write.
                if (confSf && !confSf->resizeSafe(count)) {
                    cloud->deleteScalarField(sfIdx);
                    confSf = nullptr;
                }
            }
        }

        const uchar* rgbRow = nullptr;
        for (unsigned k = 0; k < count; ++k) {
            const size_t i = kept[static_cast<size_t>(k) *
                                  static_cast<size_t>(downsample)];
            const float d = frame.depth[i];
            const int x = static_cast<int>(i % w);
            const int y = static_cast<int>(i / w);
            // Official unprojection (parse chain): raw OpenCV pixel coords,
            // then the decoded camera-to-world.
            const float X = (x - cx) / fx * d;
            const float Y = (y - cy) / fy * d;
            const float Z = d;
            const float wx = R[0] * X + R[1] * Y + R[2] * Z + R[3];
            const float wy = R[4] * X + R[5] * Y + R[6] * Z + R[7];
            const float wz = R[8] * X + R[9] * Y + R[10] * Z + R[11];
            if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz))
                continue;
            cloud->addPoint(CCVector3(wx, wy, wz));
            if (confSf) confSf->setValue(cloud->size() - 1, frame.depthConf[i]);
            if (cloudHasRgb) {
                const QRgb rgb =
                        rgbBits ? qRgb(rgbBits[y * rgbStride + x * 3 + 0],
                                       rgbBits[y * rgbStride + x * 3 + 1],
                                       rgbBits[y * rgbStride + x * 3 + 2])
                                : frame.frameRgb.pixel(x, y);
                cloud->addRGBColor(static_cast<ColorCompType>(qRed(rgb)),
                                   static_cast<ColorCompType>(qGreen(rgb)),
                                   static_cast<ColorCompType>(qBlue(rgb)));
            }
        }
        const unsigned added = cloud->size();
        if (added == 0) {
            delete cloud;
            continue;
        }
        if (confSf) confSf->computeMinAndMax();
        cloud->showColors(cloudHasRgb);
        cloud->setDisplay(m_app->getActiveGLDisplay());
        cloud->setVisible(true);
        framesFolder->addChild(cloud);
        m_frameClouds.back() = cloud;
        totalPoints += added;
    }
    if (totalPoints == 0) {
        // No frame produced points (all degenerate): drop the empty folder
        // rather than shipping an empty node.
        group->removeChild(framesFolder);
        delete framesFolder;
    }

    // COLMAP-style camera frustum per frame (the same ccCameraSensor
    // objects the Reconstruction module renders). All of them live in the
    // dedicated "LingbotMap_cameras" subfolder — the same policy as the
    // online preview — so the DB tree keeps the map cloud and the camera
    // set independently manageable.
    auto* cameraGroup = new ccHObject(QStringLiteral("LingbotMap_cameras"));
    cameraGroup->setMetaData(QStringLiteral("LingbotMap"), true);
    cameraGroup->setDisplay(m_app->getActiveGLDisplay());
    cameraGroup->setEnabled(true);
    cameraGroup->setVisible(true);
    group->addChild(cameraGroup);
    // Same aligned-slot contract as the clouds above: camera i, cloud i and
    // replayResultIndex[i] all belong to result frame i.
    m_frameCameras.clear();
    m_replayResultIndex.clear();
    m_frameCameras.reserve(result.frames.size());
    m_replayResultIndex.reserve(result.frames.size());
    for (int f = 0; f < result.frames.size(); ++f) {
        m_frameCameras.push_back(nullptr);
        m_replayResultIndex.push_back(-1);
        const LingbotFrameResult& frame = result.frames[f];
        if (frame.c2w.size() < 16) continue;
        const float fx = frame.intrinsics.value(0, 0.f);
        if (fx <= 0.f) continue;
        ccCameraSensor* sensor =
                buildCameraSensor(frame.c2w.constData(), fx, frame.width,
                                  frame.height, cameraDisplaySize);
        if (!sensor) continue;
        sensor->setName(
                QStringLiteral("LingbotMap_cam_%1")
                        .arg(frame.globalIndex >= 0 ? frame.globalIndex : f, 6,
                             10, QLatin1Char('0')));
        sensor->setMetaData(QStringLiteral("LingbotMap"), true);
        sensor->setDisplay(m_app->getActiveGLDisplay());
        cameraGroup->addChild(sensor);
        m_frameCameras.back() = sensor;
        // Replay index → frames index (for the input-image preview sync).
        m_replayResultIndex.back() = f;
    }

    // Camera trajectory as a continuous polyline through the per-frame
    // c2w translations (real camera-path semantics; rendered in the active
    // VTK window next to the point clouds).
    ccPointCloud* trajVertices =
            new ccPointCloud(QStringLiteral("LingbotMap_traj_vertices"));
    const unsigned trajCount = static_cast<unsigned>(result.frames.size());
    ccPolyline* trajectory = nullptr;
    if (trajCount >= 2 && trajVertices->reserve(trajCount)) {
        for (const CCVector3f& center : centers) {
            trajVertices->addPoint(CCVector3(center.x, center.y, center.z));
        }
        trajVertices->resize(trajVertices->size());
        trajVertices->setRGBColor(static_cast<ColorCompType>(255),
                                  static_cast<ColorCompType>(140),
                                  static_cast<ColorCompType>(0));
        trajVertices->showColors(true);
        trajectory = new ccPolyline(trajVertices);
        trajectory->addChild(trajVertices);
        trajVertices->setEnabled(false);
        if (trajectory->addPointIndex(0, trajVertices->size())) {
            trajectory->setName(QStringLiteral("LingbotMap_trajectory"));
            trajectory->setMetaData(QStringLiteral("LingbotMap"), true);
            trajectory->setClosed(false);
            trajectory->showColors(true);
            trajectory->setVisible(true);
            trajectory->setDisplay(m_app->getActiveGLDisplay());
            group->addChild(trajectory);
        } else {
            delete trajectory;
            trajectory = nullptr;
        }
    }
    if (!trajectory) {
        delete trajVertices;
    }

    // The cameras subfolder is always present; "empty" means no map cloud
    // AND no camera survived the filter (i.e. nothing to show at all).
    const bool emptyResult = totalPoints == 0 && m_frameCameras.empty();
    if (emptyResult) {
        delete group;  // cascades into the cameras subfolder
        m_resultFramesFolder = nullptr;
        m_resultFramesFolderId = 0;
        m_dialog->appendLog(tr("[LingbotMap] No valid points after filtering "
                               "(threshold %1).")
                                    .arg(settings.confThreshold));
        return false;
    }

    // Result-display state (official viewer sliders parity): the live
    // threshold slider bottoms at the materialization threshold (points
    // below it were never materialized) and the camera-size slider scales
    // the base graphic scale captured here.
    m_resultFramesFolder = framesFolder;
    m_resultFramesFolderId = framesFolder->getUniqueID();
    m_resultConfMin = settings.confThreshold;
    m_resultConfApplied = settings.confThreshold;
    {
        int retinaScale = 1;
        if (auto* view = ecvViewManager::instance().getEffectiveView()) {
            retinaScale = std::max(view->getDevicePixelRatio(), 1);
        }
        m_cameraBaseScale =
                PC_ONE / static_cast<PointCoordinateType>(retinaScale);
    }

    // Playback state over this result (official viewer playback).
    m_resultGroupId = group->getUniqueID();
    m_playbackFrame = 0;
    const unsigned cameraCount = static_cast<unsigned>(m_frameCameras.size());

    // addToDB (not a raw addChild): the app facade handles the DB tree
    // insertion AND refits the active VTK window on the new reconstruction,
    // so the result is immediately visible without manual navigation.
    m_app->addToDB(group, /*updateZoom=*/true, /*autoExpandDBTree=*/true,
                   /*checkDimensions=*/false, /*autoRedraw=*/true);
    m_dialog->appendLog(
            tr("[LingbotMap] Added merged map cloud (%1 points) + %2 cameras "
               "+ trajectory in %3 s.")
                    .arg(totalPoints)
                    .arg(cameraCount)
                    .arg(timer.elapsed() / 1000.0, 0, 'f', 1));
    syncReplayStateToDialog();
    if (m_autoPlayAfterRun) {
        startPlayback();
    }
    return true;
}

void qLingbotMap::onPlaybackSettingsChanged(bool enabled,
                                            int fps,
                                            bool currentFrameOnly) {
    // "enabled" is the AUTO-PLAY preset (start the replay as soon as a
    // result is materialized); live control goes through the explicit
    // Play/Pause/Stop buttons. FPS changes apply to a running replay
    // immediately.
    m_autoPlayAfterRun = enabled;
    m_playbackFps = std::max(1, fps);
    m_playbackCurrentFrameOnly = currentFrameOnly;
    if (m_replayState == ReplayState::Playing && m_playbackTimer) {
        m_playbackTimer->start(1000 / m_playbackFps);
    }
}

void qLingbotMap::onReplayPlay() {
    if (m_replayState == ReplayState::Playing) return;
    startPlayback();
}

void qLingbotMap::onReplayPause() {
    if (m_replayState != ReplayState::Playing) return;
    pausePlayback();
}

void qLingbotMap::onReplayStop() {
    if (m_replayState == ReplayState::Stopped) return;
    stopPlaybackAndRestore();
}

void qLingbotMap::onReplaySeek(int frame) {
    if (m_replayState == ReplayState::Playing) return;  // official: disabled
    seekPlayback(frame);
}

void qLingbotMap::startPlayback() {
    if (!m_app || m_frameCameras.empty()) return;
    if (m_replayState == ReplayState::Stopped) {
        // Entering Playing from Stopped restarts from frame 0; resuming
        // from Paused continues from where it froze.
        m_playbackFrame = 0;
        // Official replay keeps a GLOBAL view: the incremental growth is
        // only legible from a fixed vantage point, while a Follow/Fit live
        // camera leaves the ride-along pose from the streaming phase (the
        // replay then looks like a single point blob flying past the
        // frustum trail).
        if (m_lastSettings.liveViewMode > 0) {
            if (ccHObject* group =
                        m_app->dbRootObject()->find(m_resultGroupId)) {
                ccBBox box = group->getBB_recursive(true, false);
                if (box.isValid()) {
                    if (ecvGenericGLDisplay* display =
                                m_app->getActiveGLDisplay()) {
                        display->updateConstellationCenterAndZoom(&box);
                    }
                }
            }
            // Frame 0 visibility immediately (no first-tick wait).
            applyReplayVisibility();
        }
    }
    m_replayState = ReplayState::Playing;
    syncReplayStateToDialog();
    m_playbackTimer->start(1000 / std::max(1, m_playbackFps));
}

void qLingbotMap::pausePlayback() {
    // Official "Playing unchecked": freeze on the current frame — the
    // per-frame visibility stays exactly as the last tick left it, only
    // the clock stops (the dialog's timeline slider unlocks for scrubbing).
    if (m_playbackTimer) m_playbackTimer->stop();
    if (m_replayState != ReplayState::Playing) return;
    m_replayState = ReplayState::Paused;
    syncReplayStateToDialog();
}

void qLingbotMap::stopPlaybackAndRestore() {
    if (m_playbackTimer) m_playbackTimer->stop();
    m_replayState = ReplayState::Stopped;
    m_playbackFrame = 0;
    // Restore EVERY replay-controlled entity (the frustums AND the
    // per-frame clouds — the replay hides both) and force one real redraw.
    // A bare refreshAll(false,false) only clears redraw flags and renders
    // nothing, which left the window frozen on whatever the last played
    // frame showed — the "playback cannot be stopped" symptom.
    if (m_app && m_resultGroupId != 0) {
        if (ccHObject* group = m_app->dbRootObject()->find(m_resultGroupId)) {
            for (ccHObject* cam : m_frameCameras) {
                if (cam) cam->setVisible(true);
            }
            for (ccPointCloud* cloud : m_frameClouds) {
                if (cloud) cloud->setVisible(true);
            }
            // The user's result-display camera state (Show cameras /
            // stride) stays authoritative on top of the all-visible
            // baseline.
            applyCameraDisplay(m_resultShowCameras, m_resultCamStride,
                               m_resultCamSizePercent);
            m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/true);
        }
    }
    syncReplayStateToDialog();
}

void qLingbotMap::seekPlayback(int frame) {
    if (!m_app || m_frameCameras.empty()) return;
    const int count = static_cast<int>(m_frameCameras.size());
    if (count == 0) return;
    m_playbackFrame =
            ((frame % count) + count) % count;  // official modulo wrap
    // Same visibility rule as a tick, without advancing the clock.
    applyReplayVisibility();
    m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/false);
    syncReplayStateToDialog();
    syncReplayPreviewToDialog();
}

void qLingbotMap::syncReplayStateToDialog() {
    if (!m_dialog) return;
    const int count = static_cast<int>(m_frameCameras.size());
    m_dialog->setReplayState(static_cast<int>(m_replayState), m_playbackFrame,
                             count);
}

void qLingbotMap::syncReplayPreviewToDialog() {
    if (!m_dialog) return;
    const int i = m_playbackFrame;
    if (i < 0 || i >= static_cast<int>(m_replayResultIndex.size())) return;
    const int resultIdx = m_replayResultIndex[static_cast<size_t>(i)];
    if (resultIdx < 0 || resultIdx >= m_pendingResult.frames.size()) return;
    const LingbotFrameResult& fr = m_pendingResult.frames.at(resultIdx);
    if (!fr.frameRgb.isNull()) {
        m_dialog->updatePreviewImage(fr.frameRgb, i);
    }
}

void qLingbotMap::applyReplayVisibility() {
    // Replay-owned visibility, ONE formula for tick/seek/slider changes:
    //   camera  = Show cameras && (ordinal % stride == 0) && frameShown
    //   cloud   = frameShown
    // where frameShown is the official replay rule (3D: frames UP TO the
    // current one — incremental; 4D: current only). The stride stays live
    // DURING playback: dragging the slider re-densifies/–thins the
    // frustums on the very next call instead of being a no-op.
    const int count = static_cast<int>(m_frameCameras.size());
    const int stride = std::max(1, m_resultCamStride);
    int visibleCams = 0;
    ecvGenericGLDisplay* reconcileDisplay =
            m_app ? m_app->getActiveGLDisplay() : nullptr;
    for (int i = 0; i < count; ++i) {
        const bool frameShown = m_playbackCurrentFrameOnly
                                        ? i == m_playbackFrame
                                        : i <= m_playbackFrame;
        const bool camVisible =
                m_resultShowCameras && (i % stride == 0) && frameShown;
        if (ccHObject* cam = m_frameCameras[static_cast<size_t>(i)]) {
            cam->setVisible(camVisible);
            if (camVisible) ++visibleCams;
            // Deterministic reconcile: drive the engine's hide/show DIRECTLY
            // for aggregated frustums (composite per-block attributes)
            // instead of relying solely on the next redraw's tree pass —
            // the replay cannot be hostage to pass-ordering subtleties.
            // Hides/shows the frustum block AND its "#plane" face actor
            // (the engine's hideShowEntities probes the suffix).
            if (reconcileDisplay) {
                CC_DRAW_CONTEXT ctx;
                ecvGenericGLDisplay* camDisplay = cam->getDisplay();
                ctx.display = camDisplay ? camDisplay : reconcileDisplay;
                ctx.viewID = cam->getViewId();
                ctx.hideShowEntityType = ENTITY_TYPE::ECV_SENSOR;
                ctx.visible = camVisible;
                ctx.defaultViewPort = 0;
                reconcileDisplay->hideShowEntities(ctx);
            }
        }
        // Clouds have no stride/visibility density concept — they follow
        // the frame rule only. Aligned slots: cloud i belongs to frame i.
        if (i < static_cast<int>(m_frameClouds.size())) {
            if (ccPointCloud* cloud = m_frameClouds[static_cast<size_t>(i)]) {
                cloud->setVisible(frameShown);
            }
        }
    }
    if (CVLog::diagnosticsEnabled()) {
        CVLog::Print("[replay] frame=%d/%d stride=%d visibleCams=%d",
                     m_playbackFrame, count, stride, visibleCams);
    }
}

void qLingbotMap::onPlaybackTick() {
    if (!m_app || m_frameCameras.empty()) {
        // No result to replay over: hard-stop (nothing to restore).
        if (m_playbackTimer) m_playbackTimer->stop();
        m_replayState = ReplayState::Stopped;
        syncReplayStateToDialog();
        return;
    }
    // The result group was deleted by the user: stop the loop (the cached
    // sensor pointers die together with it).
    ccHObject* root = m_app->dbRootObject();
    if (!root || !root->find(m_resultGroupId)) {
        if (m_playbackTimer) m_playbackTimer->stop();
        m_replayState = ReplayState::Stopped;
        m_frameCameras.clear();
        m_frameClouds.clear();
        m_replayResultIndex.clear();
        m_resultGroupId = 0;
        m_resultFramesFolder = nullptr;
        m_resultFramesFolderId = 0;
        syncReplayStateToDialog();
        return;
    }
    // Official viewer replay semantics (point_cloud_viewer.py
    // update_frame_visibility): 3D mode shows every frame UP TO the
    // current one (incremental replay — the reconstruction visibly grows
    // back frame by frame), 4D shows only the current frame — combined
    // with the user's camera-density (stride) choice. Visibility goes
    // through setVisible (m_visible → the draw pass's hideShowEntities):
    // setEnabled is the DB-tree semantics and its hideObject_recursive
    // detour left the frustum actors on screen (the "playback shows
    // everything at once" symptom). The sensors are cached pointers — a
    // per-tick group->find per camera used to be O(cameras x tree) at
    // playback FPS.
    applyReplayVisibility();
    m_playbackFrame =
            (m_playbackFrame + 1) % static_cast<int>(m_frameCameras.size());
    // Reconstruction ↔ input-stream lockstep: the dialog preview shows the
    // frame just became current (official frame-panel behavior).
    syncReplayPreviewToDialog();
    m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/false);
}
