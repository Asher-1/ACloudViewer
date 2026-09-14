// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qLingbotMap.h"

#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QSettings>
#include <QTimer>

#include "ecvPersistentSettings.h"

namespace {

constexpr double kMaxWorldCoordinate = 1.0e6;  // sanity bound per point

bool isLingbotOutputEntity(const ccHObject* obj) {
    if (!obj) return false;
    const QString name = obj->getName();
    if (name.startsWith(QStringLiteral("LingbotMap_"))) return true;
    return obj->getMetaData(QStringLiteral("LingbotMap")).isValid();
}

}  // namespace

qLingbotMap::qLingbotMap(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qLingbotMap/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qLingbotMap"));
    qRegisterMetaType<LingbotRunResult>("LingbotRunResult");
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
    connect(m_worker, &LingbotMapWorker::resultReady, this,
            &qLingbotMap::onResultReady);
    connect(m_worker, &LingbotMapWorker::taskFinished, this,
            &qLingbotMap::onTaskFinished);
    m_inferenceElapsedSeconds = 0;
    m_dialog->setTaskRunning(true);
    m_inferenceHeartbeat->start();
    m_lastSettings = settings;
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

        ccPointCloud* cloud = new ccPointCloud();
        cloud->setName(QStringLiteral("LingbotMap_frame_%1")
                               .arg(f, 6, 10, QLatin1Char('0')));
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
        totalPoints += added;
    }

    // Camera trajectory as a continuous polyline through the per-frame
    // c2w translations (real camera-path semantics; rendered in the active
    // VTK window next to the point clouds).
    ccPointCloud* trajVertices =
            new ccPointCloud(QStringLiteral("LingbotMap_traj_vertices"));
    const unsigned trajCount = static_cast<unsigned>(result.frames.size());
    ccPolyline* trajectory = nullptr;
    if (trajCount >= 2 && trajVertices->reserve(trajCount)) {
        for (const LingbotFrameResult& frame : result.frames) {
            if (frame.c2w.size() >= 12) {
                trajVertices->addPoint(
                        CCVector3(frame.c2w[3], frame.c2w[7], frame.c2w[11]));
            }
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

    // addToDB (not a raw addChild): the app facade handles the DB tree
    // insertion AND refits the active VTK window on the new reconstruction,
    // so the result is immediately visible without manual navigation.
    m_app->addToDB(group, /*updateZoom=*/true, /*autoExpandDBTree=*/true,
                   /*checkDimensions=*/false, /*autoRedraw=*/true);
    m_dialog->appendLog(
            tr("[LingbotMap] Added %1 frame clouds (%2 points) + trajectory "
               "in %3 s.")
                    .arg(group->getChildrenNumber() - 1)
                    .arg(totalPoints)
                    .arg(timer.elapsed() / 1000.0, 0, 'f', 1));
    return true;
}
