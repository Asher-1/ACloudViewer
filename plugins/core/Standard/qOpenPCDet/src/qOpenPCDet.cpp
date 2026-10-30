// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qOpenPCDet.h"

#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

#include <QMainWindow>
#include <QTimer>

#include "OpenPCDetBatchCommand.h"
#include "OpenPCDetDbExport.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#endif

qOpenPCDet::qOpenPCDet(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qOpenPCDet/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qOpenPCDet"));
    qRegisterMetaType<OpenPCDetRunResult>("OpenPCDetRunResult");
    qRegisterMetaType<OpenPCDetDialog::Settings>("OpenPCDetDialog::Settings");
    m_action = new QAction(tr("OpenPCDet 3D Detection"), this);
    m_action->setToolTip(
            tr("LiDAR point cloud 3D object detection (PointPillars / "
               "SECOND, GGML)"));
    m_action->setIcon(QIcon(":/CC/plugin/qOpenPCDet/images/qOpenPCDet.svg"));
    connect(m_action, &QAction::triggered, this, &qOpenPCDet::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(10000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 10;
        m_dialog->appendLog(tr("[OpenPCDet] Task is running (%1 s "
                               "elapsed)...")
                                    .arg(m_inferenceElapsedSeconds));
    });
}

QList<QAction*> qOpenPCDet::getActions() { return {m_action}; }

void qOpenPCDet::registerCommands(ccCommandLineInterface* cmd) {
    if (cmd == nullptr) {
        return;
    }
    cmd->registerCommand(ccCommandLineInterface::Command::Shared(
            new CommandOpenPCDetDetect));
}

ccPointCloud* qOpenPCDet::firstSelectedCloud() const {
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj) continue;
        if (obj->isA(CV_TYPES::POINT_CLOUD)) {
            ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(obj);
            if (cloud && cloud->size() > 0) return cloud;
        }
        // Direct hit on a group: accept its first non-empty point cloud.
        ccHObject::Container clouds;
        obj->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD, false);
        for (ccHObject* child : clouds) {
            ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(child);
            if (cloud && cloud->size() > 0) return cloud;
        }
    }
    return nullptr;
}

void qOpenPCDet::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    // The multi-modal scene checkpoints (CaDDN / BEVFusion from a scene
    // directory) run without a DB point cloud, so the action stays enabled
    // unconditionally; the dialog validates the per-panel input on Run.
    m_action->setEnabled(true);
    if (m_dialog && m_dialog->isVisible()) {
        ccPointCloud* cloud = firstSelectedCloud();
        m_dialog->setActiveCloud(cloud, cloud ? cloud->getName() : QString());
    }
}

void qOpenPCDet::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new OpenPCDetDialog(
                static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &OpenPCDetDialog::runRequested, this,
                &qOpenPCDet::executeTask);
        connect(m_dialog, &OpenPCDetDialog::cancelRequested, this,
                &qOpenPCDet::cancelTask);
    }
    m_dialog->refreshModelList();
    if (ccPointCloud* cloud = firstSelectedCloud()) {
        m_dialog->setActiveCloud(cloud, cloud->getName());
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qOpenPCDet::executeTask(const OpenPCDetDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) return;

    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    // Input resolution: an explicit point file wins; else the active DB
    // cloud (scene runs without a LiDAR branch, e.g. CaDDN, may have
    // neither — the worker validates against the model's requirements).
    const bool wantsCloud = settings.pointsFile.isEmpty();
    ccPointCloud* cloud = wantsCloud ? firstSelectedCloud() : nullptr;
    if (wantsCloud && !cloud && settings.sceneDir.isEmpty()) {
        m_dialog->appendLog(
                tr("[Error] Select a non-empty point cloud in the DB tree "
                   "first."));
        return;
    }
    if (settings.modelPath.isEmpty()) {
        m_dialog->appendLog(tr("[Error] Model required."));
        return;
    }

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (aicore_openpcdet_warmup_backend(workerDevice.toUtf8().constData()) !=
        0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[OpenPCDet] GPU backend unavailable — using CPU for "
                       "this run."));
        }
    }
#endif

    OpenPCDetWorker::Settings ws;
    ws.modelPath = settings.modelPath;
    ws.device = workerDevice;
    ws.threads = settings.threads;
    ws.scoreThresh = settings.scoreThresh;
    ws.nmsThresh = settings.nmsThresh;
    ws.maxBoxes = settings.maxBoxes;
    ws.pointsFile = settings.pointsFile;
    ws.sceneDir = settings.sceneDir;

    m_currentSettings = settings;
    m_runSourceCloud = cloud;  // annotated-cloud export source
    m_worker = new OpenPCDetWorker(ws, cloud, this);
    connect(m_worker, &OpenPCDetWorker::logMessage, m_dialog,
            &OpenPCDetDialog::appendLog, Qt::QueuedConnection);
    connect(m_worker, &OpenPCDetWorker::progressUpdate, this,
            &qOpenPCDet::onWorkerProgress, Qt::QueuedConnection);
    connect(m_worker, &OpenPCDetWorker::resultReady, this,
            &qOpenPCDet::onResultReady, Qt::QueuedConnection);
    connect(m_worker, &OpenPCDetWorker::taskFinished, this,
            &qOpenPCDet::onTaskFinished, Qt::QueuedConnection);
    m_dialog->setRunning(true);
    m_inferenceElapsedSeconds = 0;
    m_inferenceHeartbeat->start();
    m_worker->start();
}

void qOpenPCDet::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qOpenPCDet::onResultReady(const OpenPCDetRunResult& result) {
    m_dialog->showDetectionSummary(result);
    if (m_currentSettings.addBoxesToDb) {
        addBoxesToDb(result);
    }
    if (m_currentSettings.annotatedCloudToDb) {
        exportAnnotatedCloud(result);
    }
}

void qOpenPCDet::addBoxesToDb(const OpenPCDetRunResult& result) {
    if (!m_app || result.detections.isEmpty()) return;

    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(
                    m_currentSettings.modelPath);
    const QString groupName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("OpenPCDet detections (%1)").arg(modelTag),
            m_app);
    ccHObject* group = BuildDetectionGroup(result, modelTag);
    if (!group) return;
    // makeUnique renamed the top group; keep the tree name consistent.
    group->setName(groupName);
    group->setMetaData(QStringLiteral("OpenPCDet/Model"),
                       QFileInfo(m_currentSettings.modelPath).fileName());

    m_app->addToDB(group, true, true, false, true);
    m_dialog->appendLog(tr("[OpenPCDet] Added %1 wireframe box(es) to the DB "
                           "tree as '%2'.")
                                .arg(result.detections.size())
                                .arg(groupName));
}

void qOpenPCDet::exportAnnotatedCloud(const OpenPCDetRunResult& result) {
    if (!m_app || result.detections.isEmpty()) return;
    ccPointCloud* source = m_runSourceCloud;
    if (!source) {
        m_dialog->appendLog(
                tr("[OpenPCDet] Annotated-cloud export skipped: the run had "
                   "no point-cloud source (file-only or pointless scene)."));
        return;
    }
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(
                    m_currentSettings.modelPath);
    unsigned tinted = 0;
    QString error;
    ccPointCloud* annotated =
            BuildAnnotatedCloud(source, result, modelTag, &tinted, &error);
    if (!annotated) {
        m_dialog->appendLog(tr("[Error] Annotated-cloud export failed: %1.")
                                    .arg(error));
        return;
    }
    const QString cloudName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("OpenPCDet annotated (%1)").arg(modelTag), m_app);
    annotated->setName(cloudName);

    m_app->addToDB(annotated, true, true, false, true);
    m_dialog->appendLog(
            tr("[OpenPCDet] Exported annotated cloud '%1' (%2 of %3 points "
               "tinted).")
                    .arg(cloudName)
                    .arg(tinted)
                    .arg(annotated->size()));
}

void qOpenPCDet::onTaskFinished(bool success) {
    m_inferenceHeartbeat->stop();
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}

void qOpenPCDet::onWorkerProgress(int current, int total) {
    if (!m_dialog) return;
    m_dialog->setProgress(current, total);
}
