// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qLingbotMap.h"

#include <ecvCameraSensor.h>
#include <ecvCameraSensorDisplayUtils.h>
#include <ecvColorTypes.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvViewManager.h>

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
constexpr qint64 kOnlineRefreshIntervalMs = 250;

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
    }
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
    connect(m_worker, &LingbotMapWorker::logMessage, m_dialog,
            &LingbotMapDialog::appendLog);
    connect(m_worker, &LingbotMapWorker::taskStage, m_dialog,
            &LingbotMapDialog::appendLog);
    connect(m_worker, &LingbotMapWorker::taskStage, m_dialog,
            [this](const QString& stage, int percent) {
                if (percent >= 0) m_dialog->setProgress(percent);
            });
    connect(m_worker, &LingbotMapWorker::framePreviewReady, this,
            &qLingbotMap::onFramePreview);
    connect(m_worker, &LingbotMapWorker::resultReady, this,
            &qLingbotMap::onResultReady);
    connect(m_worker, &LingbotMapWorker::taskFinished, this,
            &qLingbotMap::onTaskFinished);
    m_inferenceElapsedSeconds = 0;
    m_dialog->setTaskRunning(true);
    m_inferenceHeartbeat->start();
    m_lastSettings = settings;

    // Fresh online-preview state for this run (official StreamingViewer:
    // the scene grows frame by frame while the engine streams).
    disposeOnlineGroup();
    m_onlineWindowCount = 0;
    m_onlineCurrentWindow = nullptr;
    m_onlineBaselines.clear();
    m_lastPreviewC2w.clear();
    m_onlineRefreshThrottle.start();
    stopPlayback();
    m_frameCloudIds.clear();
    m_resultGroupId = 0;

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
    // The transient online preview is replaced by the final (full-res,
    // aligned) result in every case.
    disposeOnlineGroup();
    if (!success) {
        return;
    }
    if (m_dialog) {
        m_dialog->appendLog(
                tr("[LingbotMap] Task finished (%1 frames, %2 s).")
                        .arg(m_pendingResult.frames.size())
                        .arg(m_pendingResult.elapsedMs / 1000.0, 0, 'f', 1));
    }
    if (m_lastSettings.addResultToDb && !m_pendingResult.frames.isEmpty()) {
        addResultToDb(m_pendingResult, m_lastSettings);
    }
}

void qLingbotMap::onFramePreview(const LingbotFramePreview& preview) {
    if (!m_app) return;
    // The user removed the online group mid-run: drop the preview instead
    // of rebuilding it behind their back.
    if (m_onlineGroup && m_app->dbRootObject() &&
        !m_app->dbRootObject()->find(m_onlineGroup->getUniqueID())) {
        m_onlineGroup = nullptr;
        m_onlineCurrentWindow = nullptr;
    }
    if (m_onlineWindowCount == 0 && preview.windowCount > 0) {
        m_onlineWindowCount = preview.windowCount;
    }

    ccHObject* parent =
            onlineWindowGroup(preview.windowIndex, preview.windowCount);
    if (!parent) return;

    if (!preview.points.isEmpty()) {
        const unsigned n = static_cast<unsigned>(preview.points.size() / 3);
        ccPointCloud* cloud = new ccPointCloud(
                QStringLiteral("LingbotMap_online_%1")
                        .arg(preview.globalIndex, 6, 10, QLatin1Char('0')));
        cloud->setMetaData(QStringLiteral("LingbotMap"), true);
        if (cloud->reserve(n)) {
            const float* xyz = preview.points.constData();
            const bool hasColors =
                    !preview.colors.isEmpty() && cloud->reserveTheRGBTable();
            for (unsigned i = 0; i < n; ++i) {
                cloud->addPoint(
                        CCVector3(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]));
                if (hasColors) {
                    cloud->addRGBColor(preview.colors[i * 3],
                                       preview.colors[i * 3 + 1],
                                       preview.colors[i * 3 + 2]);
                }
            }
            if (hasColors) cloud->showColors(true);
            cloud->setDisplay(m_app->getActiveGLDisplay());
            cloud->setVisible(true);
            parent->addChild(cloud);
        } else {
            delete cloud;
        }
    }
    addOnlineCamera(parent, preview);

    // Windowed runs re-anchor nothing until the stitch: opening a new
    // window hides the previous one (window-local coordinates; mirrors the
    // official window-tagged scene keys).
    if (m_onlineWindowCount > 1 && m_onlineCurrentWindow &&
        m_onlineCurrentWindow != parent) {
        m_onlineCurrentWindow->setEnabled(false);
    }
    m_onlineCurrentWindow = parent;

    if (m_onlineRefreshThrottle.elapsed() >= kOnlineRefreshIntervalMs) {
        m_onlineRefreshThrottle.restart();
        m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/false);
    }
}

ccHObject* qLingbotMap::onlineWindowGroup(int windowIndex, int windowCount) {
    if (!m_app) return nullptr;
    if (!m_onlineGroup) {
        const QString modelTag = ecvPluginDbNaming::modelTagFromFilename(
                m_lastSettings.modelPath);
        m_onlineGroup = new ccHObject(
                QStringLiteral("LingbotMap_Online_%1").arg(modelTag));
        m_onlineGroup->setMetaData(QStringLiteral("LingbotMap"), true);
        m_onlineGroup->setDisplay(m_app->getActiveGLDisplay());
        m_onlineGroup->setEnabled(true);
        m_onlineGroup->setVisible(true);
        m_app->addToDB(m_onlineGroup, /*updateZoom=*/false,
                       /*autoExpandDBTree=*/false, /*checkDimensions=*/false,
                       /*autoRedraw=*/false);
    }
    if (windowCount <= 1) {
        return m_onlineGroup;
    }
    // Per-window subgroups (window-local coordinates before the stitch).
    const QString name = QStringLiteral("window_%1")
                                 .arg(windowIndex, 2, 10, QLatin1Char('0'));
    for (unsigned i = 0; i < m_onlineGroup->getChildrenNumber(); ++i) {
        ccHObject* child = m_onlineGroup->getChild(i);
        if (child->getName() == name) return child;
    }
    auto* windowGroup = new ccHObject(name);
    windowGroup->setMetaData(QStringLiteral("LingbotMap"), true);
    windowGroup->setDisplay(m_app->getActiveGLDisplay());
    windowGroup->setEnabled(true);
    windowGroup->setVisible(true);
    m_onlineGroup->addChild(windowGroup);
    return windowGroup;
}

void qLingbotMap::addOnlineCamera(ccHObject* parent,
                                  const LingbotFramePreview& p) {
    if (!parent || p.c2w.isEmpty()) return;
    if (m_lastPreviewC2w.size() >= 16) {
        const CCVector3f prev = cameraCenter(m_lastPreviewC2w.constData());
        const CCVector3f curr = cameraCenter(p.c2w.constData());
        m_onlineBaselines.push_back((curr - prev).norm());
    }
    m_lastPreviewC2w = p.c2w;
    const float med = medianBaseline(m_onlineBaselines);
    const float displaySize =
            med > 0.f ? med * kCameraBaselineFraction : kFallbackCameraSize;
    const float fx = p.intrinsics.value(0, 0.f);
    ccCameraSensor* sensor = buildCameraSensor(p.c2w.constData(), fx, p.width,
                                               p.height, displaySize);
    if (!sensor) return;
    sensor->setName(QStringLiteral("LingbotMap_online_cam_%1")
                            .arg(p.globalIndex, 6, 10, QLatin1Char('0')));
    sensor->setMetaData(QStringLiteral("LingbotMap"), true);
    sensor->setDisplay(m_app->getActiveGLDisplay());
    parent->addChild(sensor);
}

void qLingbotMap::disposeOnlineGroup() {
    if (!m_app || !m_onlineGroup) return;
    ccHObject* root = m_app->dbRootObject();
    if (root && root->find(m_onlineGroup->getUniqueID())) {
        m_app->removeFromDB(m_onlineGroup, /*autoDelete=*/true);
        m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/false);
    }
    m_onlineGroup = nullptr;
    m_onlineCurrentWindow = nullptr;
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

    unsigned long long totalPoints = 0;
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

    for (int f = 0; f < result.frames.size(); ++f) {
        const LingbotFrameResult& frame = result.frames[f];
        const int w = frame.width;
        const int h = frame.height;
        if (frame.depth.isEmpty() || w <= 0 || h <= 0) continue;
        const float fx = frame.intrinsics.value(0, 0.f);
        const float fy = frame.intrinsics.value(1, 0.f);
        const float cx = frame.intrinsics.value(2, 0.f);
        const float cy = frame.intrinsics.value(3, 0.f);
        if (fx <= 0.f || fy <= 0.f) continue;

        const float* R = frame.c2w.data();  // row-major 4x4
        bool hasRgb = !frame.frameRgb.isNull() &&
                      frame.frameRgb.size() == QSize(w, h);

        // First pass: count valid pixels.
        unsigned count = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = static_cast<size_t>(y) * w + x;
                const float d = frame.depth[i];
                if (d <= 0.f || !std::isfinite(d)) continue;
                if (frame.depthConf[i] < settings.confThreshold) continue;
                if (!frame.skyKeep.isEmpty() && frame.skyKeep[i] == 0) continue;
                ++count;
            }
        }
        if (count == 0) continue;

        ccPointCloud* cloud = new ccPointCloud(
                QStringLiteral("LingbotMap_frame_%1")
                        .arg(frame.globalIndex >= 0 ? frame.globalIndex : f, 6,
                             10, QLatin1Char('0')));
        cloud->setMetaData(QStringLiteral("LingbotMap"), true);
        cloud->setMetaData(QStringLiteral("source"),
                           QFileInfo(frame.sourceFile).fileName());
        if (!cloud->reserve(count)) {
            delete cloud;
            continue;
        }
        if (hasRgb && !cloud->reserveTheRGBTable()) {
            hasRgb = false;
        }

        unsigned added = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = static_cast<size_t>(y) * w + x;
                const float d = frame.depth[i];
                if (d <= 0.f || !std::isfinite(d)) continue;
                if (frame.depthConf[i] < settings.confThreshold) continue;
                if (!frame.skyKeep.isEmpty() && frame.skyKeep[i] == 0) continue;
                // OpenCV camera frame (x right, y down, z forward), matching
                // the official unprojection; then c2w into the world frame.
                const float X = (x - cx) / fx * d;
                const float Y = -(y - cy) / fy * d;
                const float Z = d;
                const float wx = R[0] * X + R[1] * Y + R[2] * Z + R[3];
                const float wy = R[4] * X + R[5] * Y + R[6] * Z + R[7];
                const float wz = R[8] * X + R[9] * Y + R[10] * Z + R[11];
                if (!std::isfinite(wx) || !std::isfinite(wy) ||
                    !std::isfinite(wz) || std::abs(wx) > kMaxWorldCoordinate ||
                    std::abs(wy) > kMaxWorldCoordinate ||
                    std::abs(wz) > kMaxWorldCoordinate) {
                    continue;
                }
                cloud->addPoint(CCVector3(wx, wy, wz));
                if (hasRgb) {
                    const QRgb rgb = frame.frameRgb.pixel(x, y);
                    cloud->addRGBColor(static_cast<ColorCompType>(qRed(rgb)),
                                       static_cast<ColorCompType>(qGreen(rgb)),
                                       static_cast<ColorCompType>(qBlue(rgb)));
                }
                ++added;
            }
        }
        if (added == 0) {
            delete cloud;
            continue;
        }
        cloud->resize(added);
        if (hasRgb) cloud->showColors(true);
        cloud->setDisplay(m_app->getActiveGLDisplay());
        cloud->setVisible(true);
        group->addChild(cloud);
        m_frameCloudIds.push_back(cloud->getUniqueID());
        totalPoints += added;

        // COLMAP-style camera frustum per frame.
        if (frame.c2w.size() >= 16) {
            ccCameraSensor* sensor = buildCameraSensor(
                    frame.c2w.constData(), fx, w, h, cameraDisplaySize);
            if (sensor) {
                sensor->setName(QStringLiteral("LingbotMap_cam_%1")
                                        .arg(frame.globalIndex >= 0
                                                     ? frame.globalIndex
                                                     : f,
                                             6, 10, QLatin1Char('0')));
                sensor->setMetaData(QStringLiteral("LingbotMap"), true);
                sensor->setDisplay(m_app->getActiveGLDisplay());
                group->addChild(sensor);
            }
        }
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

    if (group->getChildrenNumber() == 0) {
        delete group;
        m_dialog->appendLog(tr("[LingbotMap] No valid points after filtering "
                               "(threshold %1).")
                                    .arg(settings.confThreshold));
        return false;
    }

    // Playback state over this result (official viewer playback).
    m_resultGroupId = group->getUniqueID();
    m_playbackFrame = 0;
    const unsigned cameraCount = static_cast<unsigned>(
            group->getChildrenNumber() - m_frameCloudIds.size() -
            (trajectory ? 1u : 0u));

    // addToDB (not a raw addChild): the app facade handles the DB tree
    // insertion AND refits the active VTK window on the new reconstruction,
    // so the result is immediately visible without manual navigation.
    m_app->addToDB(group, /*updateZoom=*/true, /*autoExpandDBTree=*/true,
                   /*checkDimensions=*/false, /*autoRedraw=*/true);
    m_dialog->appendLog(
            tr("[LingbotMap] Added %1 frame clouds (%2 points) + %3 cameras + "
               "trajectory in %4 s.")
                    .arg(m_frameCloudIds.size())
                    .arg(totalPoints)
                    .arg(cameraCount)
                    .arg(timer.elapsed() / 1000.0, 0, 'f', 1));
    if (m_playbackEnabled) {
        startPlayback();
    }
    return true;
}

void qLingbotMap::onPlaybackSettingsChanged(bool enabled,
                                            int fps,
                                            bool currentFrameOnly) {
    m_playbackEnabled = enabled;
    m_playbackFps = std::max(1, fps);
    m_playbackCurrentFrameOnly = currentFrameOnly;
    if (enabled) {
        startPlayback();
    } else {
        stopPlayback();
        // 3D fallback: everything visible again.
        if (m_app && m_resultGroupId != 0) {
            if (ccHObject* group =
                        m_app->dbRootObject()->find(m_resultGroupId)) {
                for (unsigned id : m_frameCloudIds) {
                    if (ccHObject* cloud = group->find(id)) {
                        cloud->setEnabled(true);
                    }
                }
                m_app->refreshAll(false, false);
            }
        }
    }
}

void qLingbotMap::startPlayback() {
    if (!m_app || m_frameCloudIds.empty()) return;
    m_playbackFrame = 0;
    m_playbackTimer->start(1000 / std::max(1, m_playbackFps));
}

void qLingbotMap::stopPlayback() {
    if (m_playbackTimer) m_playbackTimer->stop();
}

void qLingbotMap::onPlaybackTick() {
    if (!m_app || m_frameCloudIds.empty()) {
        stopPlayback();
        return;
    }
    ccHObject* root = m_app->dbRootObject();
    ccHObject* group = root ? root->find(m_resultGroupId) : nullptr;
    if (!group) {  // result deleted by the user: stop the loop
        stopPlayback();
        m_frameCloudIds.clear();
        m_resultGroupId = 0;
        return;
    }
    const int count = static_cast<int>(m_frameCloudIds.size());
    for (int i = 0; i < count; ++i) {
        ccHObject* cloud = group->find(m_frameCloudIds[static_cast<size_t>(i)]);
        if (!cloud) continue;
        cloud->setEnabled(!m_playbackCurrentFrameOnly || i == m_playbackFrame);
    }
    m_playbackFrame = (m_playbackFrame + 1) % count;
    m_app->refreshAll(/*only2D=*/false, /*forceRedraw=*/false);
}
