// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "qQuerySplat.h"

#include <CVLog.h>
#include <ecvCameraSensor.h>
#include <ecvDisplayTools.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPointCloud.h>

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QMessageBox>

#include "QuerySplatBatchCommand.h"

qQuerySplat::qQuerySplat(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qQuerySplat/info.json") {}

qQuerySplat::~qQuerySplat() {
    if (m_worker) {
        m_worker->wait();
        m_worker->deleteLater();
    }
}

void qQuerySplat::onNewSelection(const ccHObject::Container& selected) {
    m_selectedEntities = selected;
}

QList<QAction*> qQuerySplat::getActions() {
    if (!m_action) {
        m_action = new QAction(QObject::tr("QuerySplat Gaussian Prediction"),
                               this);
        m_action->setToolTip(QObject::tr(
                "Multi-view image-to-3DGS prediction (VGGT-Omega encoder + "
                "QuerySplat decoder) with rendering, TTO and PLY export"));
        connect(m_action, &QAction::triggered, this, &qQuerySplat::showDialog);
    }
    return {m_action};
}

void qQuerySplat::showDialog() {
    if (!m_dialog) {
        m_dialog =
                new QuerySplatDialog(m_app ? m_app->getMainWindow() : nullptr);
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &QuerySplatDialog::runRequested, this,
                &qQuerySplat::executeTask);
        connect(m_dialog, &QuerySplatDialog::cancelRequested, this,
                &qQuerySplat::cancelTask);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qQuerySplat::executeTask(const QuerySplatDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) {
        QMessageBox::warning(m_dialog ? m_dialog : nullptr, tr("Busy"),
                             tr("A QuerySplat task is already running."));
        return;
    }
    if (m_worker) {
        m_worker->deleteLater();
    }
    m_currentSettings = settings;
    QuerySplatWorker::Settings workerSettings;
    workerSettings.decoderPath = settings.decoderPath;
    workerSettings.encoderPath = settings.encoderPath;
    workerSettings.imagePaths = settings.imagePaths;
    workerSettings.threads = settings.threads;
    workerSettings.device = settings.device;
    workerSettings.imageSize = settings.imageSize;
    workerSettings.ttoEnabled = settings.ttoEnabled;
    workerSettings.ttoSteps = settings.ttoSteps;
    workerSettings.ttoLr = settings.ttoLr;
    workerSettings.ttoLpipsWeight = settings.ttoLpipsWeight;
    workerSettings.opacityThreshold = settings.opacityThreshold;
    workerSettings.addDepthViews = settings.addDepthViews;
    workerSettings.addRenderViews = settings.addRenderViews;
    workerSettings.addDepthPointCloud = settings.addDepthPointCloud;
    workerSettings.exportPly = settings.exportPly;
    workerSettings.outputDir = settings.outputDir;
    m_worker = new QuerySplatWorker(this);
    m_worker->setSettings(workerSettings);
    connect(m_worker, &QuerySplatWorker::progressUpdate, m_dialog,
            &QuerySplatDialog::setProgress);
    connect(m_worker, &QuerySplatWorker::stageUpdate, m_dialog,
            [this](const QString& stage) {
                if (m_dialog) m_dialog->setTaskStage(stage);
            });
    connect(m_worker, &QuerySplatWorker::logMessage, m_dialog,
            &QuerySplatDialog::appendLog);
    connect(m_worker, &QuerySplatWorker::resultReady, this,
            &qQuerySplat::onResultReady);
    connect(m_worker, &QuerySplatWorker::finished, this,
            &qQuerySplat::onTaskFinished);
    if (m_dialog) {
        m_dialog->setRunning(true);
        m_dialog->setTaskStage(tr("Running..."), 0);
    }
    m_worker->start();
}

void qQuerySplat::cancelTask() {
    if (m_worker && m_worker->isRunning()) {
        m_worker->requestInterruption();
        if (m_dialog) m_dialog->appendLog(tr("[QS] Cancellation requested."));
    }
}

void qQuerySplat::onTaskFinished() {
    if (m_dialog) m_dialog->setRunning(false);
}

void qQuerySplat::onResultReady(const QuerySplatResult& result) {
    m_lastResult = result;
    addResultToDb(result);
    if (m_dialog) {
        m_dialog->setTaskStage(tr("Done: %1 gaussians, %2 views, device %3")
                                       .arg(result.gaussianCount)
                                       .arg(result.viewCount)
                                       .arg(result.resolvedDevice),
                               100);
    }
}

void qQuerySplat::addResultToDb(const QuerySplatResult& result) {
    if (!m_app) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) return;

    const QString groupName =
            QStringLiteral("QuerySplat (%1)").arg(result.sourceLabel);
    ccHObject* group = new ccHObject(groupName);
    root->addChild(group);

    // typed gaussian path: centers + sh0 colors, opacity-filtered
    if (result.gaussians && result.gaussianCount > 0) {
        const int stride = result.gaussianStride;
        const float thr = m_currentSettings.opacityThreshold;
        std::vector<CCVector3> pts;
        std::vector<ColorCompType> cols;
        cols.reserve(size_t(result.gaussianCount) * 3);
        pts.reserve(size_t(result.gaussianCount));
        cols.reserve(size_t(result.gaussianCount));
        const float* g = result.gaussians.get();
        for (int64_t i = 0; i < result.gaussianCount; ++i) {
            const float* r = g + size_t(i) * stride;
            if (r[3] <= thr) continue;
            pts.emplace_back(r[0], r[1], r[2]);
            cols.push_back(ColorCompType(
                    std::min(std::max(r[stride - 3], 0.0f), 1.0f) * 255.0f));
            cols.push_back(ColorCompType(
                    std::min(std::max(r[stride - 2], 0.0f), 1.0f) * 255.0f));
            cols.push_back(ColorCompType(
                    std::min(std::max(r[stride - 1], 0.0f), 1.0f) * 255.0f));
        }
        if (!pts.empty()) {
            ccPointCloud* cloud = new ccPointCloud(
                    QStringLiteral("gaussians (opacity >= %1)").arg(thr));
            const bool hasColors = cloud->reserveTheRGBTable();
            if (cloud->reserve(unsigned(pts.size()))) {
                for (size_t i = 0; i < pts.size(); ++i) {
                    cloud->addPoint(pts[i]);
                    if (hasColors) {
                        cloud->addRGBColor(cols[i * 3], cols[i * 3 + 1],
                                           cols[i * 3 + 2]);
                    }
                }
                cloud->resize(unsigned(pts.size()));
                cloud->showColors(hasColors);
                cloud->setVisible(true);
                group->addChild(cloud);
            } else {
                delete cloud;
            }
        }
    }

    // predicted cameras
    if (result.camPoses16.size() == size_t(result.viewCount) * 16) {
        for (int v = 0; v < result.viewCount; ++v) {
            const float* c2w = result.camPoses16.constData() + v * 16;
            const float fx = result.camIntrinsics[v * 4 + 0];
            auto* sensor = new ccCameraSensor();
            sensor->setName(QStringLiteral("view %1").arg(v));
            ccCameraSensor::IntrinsicParameters p;
            p.arrayWidth = result.width;
            p.arrayHeight = result.height;
            p.vertFocal_pix = fx;
            p.pixelSize_mm[0] = 1.0f;
            p.pixelSize_mm[1] = 1.0f;
            p.principal_point[0] = result.width * 0.5f;
            p.principal_point[1] = result.height * 0.5f;
            p.vFOV_rad = 2.0f * std::atan(result.height * 0.5f / fx);
            p.zNear_mm = 1e-3f;
            p.zFar_mm = 1000.0f;
            sensor->setIntrinsicParameters(p);
            // c2w row-major -> the sensor's rigid transform
            ccGLMatrix mat;
            float* m = mat.data();
            // ccGLMatrix is column-major-ish 4x4 stored as 16 floats
            // (translation in the last 3 components of the last row of the
            // 4x4 matrix); map the row-major c2w accordingly.
            const float* R = c2w;
            m[0] = R[0];
            m[1] = R[4];
            m[2] = R[8];
            m[3] = R[12];
            m[4] = R[1];
            m[5] = R[5];
            m[6] = R[9];
            m[7] = R[13];
            m[8] = R[2];
            m[9] = R[6];
            m[10] = R[10];
            m[11] = R[14];
            m[12] = R[3];
            m[13] = R[7];
            m[14] = R[11];
            m[15] = R[15];
            sensor->setRigidTransformation(mat);
            sensor->setEnabled(true);
            sensor->setVisible(true);
            group->addChild(sensor);
        }
    }

    // depth + render rasters
    int rasterIdx = 0;
    auto addRaster = [&](const QImage& img, const QString& name) {
        if (img.isNull()) return;
        QImage rgb = img.convertToFormat(QImage::Format_RGB888);
        ccImage* image = new ccImage(rgb, name);
        image->setEnabled(true);
        image->setVisible(true);
        group->addChild(image);
        ++rasterIdx;
    };
    for (int v = 0; v < result.depthViews.size(); ++v) {
        addRaster(result.depthViews[v], QStringLiteral("depth view %1").arg(v));
    }
    for (int v = 0; v < result.renderViews.size(); ++v) {
        addRaster(result.renderViews[v],
                  QStringLiteral("render view %1").arg(v));
    }

    // depth point cloud
    if (result.pointCount > 0 &&
        result.points.size() == size_t(result.pointCount) * 3) {
        ccPointCloud* cloud =
                new ccPointCloud(QStringLiteral("depth point cloud"));
        const bool hasColors = cloud->reserveTheRGBTable();
        if (cloud->reserve(unsigned(result.pointCount))) {
            for (int i = 0; i < result.pointCount; ++i) {
                cloud->addPoint(CCVector3(result.points[size_t(i) * 3],
                                          result.points[size_t(i) * 3 + 1],
                                          result.points[size_t(i) * 3 + 2]));
                if (hasColors) {
                    cloud->addRGBColor(result.colors[size_t(i) * 3],
                                       result.colors[size_t(i) * 3 + 1],
                                       result.colors[size_t(i) * 3 + 2]);
                }
            }
            cloud->resize(unsigned(result.pointCount));
            cloud->showColors(hasColors);
            cloud->setVisible(true);
            group->addChild(cloud);
        } else {
            delete cloud;
        }
    }

    m_app->addToDB(group, true, true, false, true);

    // PLY file is a sidecar artifact: report its path through the console.
    if (!result.plyPath.isEmpty()) {
        CVLog::Print(QStringLiteral("[QS] 3DGS PLY exported: %1")
                             .arg(result.plyPath));
    }
}

void qQuerySplat::registerCommands(ccCommandLineInterface* cmd) {
    if (cmd == nullptr) return;
    cmd->registerCommand(ccCommandLineInterface::Command::Shared(
            new CommandQuerySplatPredict));
}
