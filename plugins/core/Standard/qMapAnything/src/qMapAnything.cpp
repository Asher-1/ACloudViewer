// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qMapAnything.h"

#include <ecvCameraSensor.h>
#include <ecvCameraSensorDisplayUtils.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvViewManager.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QTimer>
#include <QUuid>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "MapAnythingReconstructCommand.h"
#include "MapAnythingSceneMeta.h"
#include "aicore/inference_log.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#endif

namespace {

constexpr ecvColor::Rgb kMapCameraPlaneColor(255, 25, 0);
constexpr ecvColor::Rgb kMapCameraFrameColor(204, 25, 0);
constexpr float kCameraBaselineFraction = 0.12f;
constexpr float kFallbackCameraDisplaySize = 0.01f;

float medianCameraBaseline(std::vector<float> baselines) {
    if (baselines.empty()) return 0.0f;
    std::sort(baselines.begin(), baselines.end());
    const size_t middle = baselines.size() / 2;
    return baselines.size() % 2 == 0
                   ? 0.5f * (baselines[middle - 1] + baselines[middle])
                   : baselines[middle];
}

// Mirrors qLingbotMap's proven sensor builder (shared
// ecvCameraSensorDisplay helpers; VtkColmap pose frame).
ccCameraSensor* buildMapCameraSensor(const float* rowMajorCam2world,
                                     float focalPx,
                                     int imageWidth,
                                     int imageHeight,
                                     float imageDisplaySize) {
    if (!rowMajorCam2world || imageWidth < 1 || imageHeight < 1 ||
        imageDisplaySize <= 0.f || focalPx <= 0.f) {
        return nullptr;
    }
    const float displayFocalMm =
            ecvCameraSensorDisplay::ComputeFrustumDisplayFocalMm(
                    imageDisplaySize, imageWidth, imageHeight, focalPx);
    const float viewportVFovRad =
            ecvCameraSensorDisplay::ComputeVerticalFovRad(focalPx, imageHeight);

    auto* sensor = new ccCameraSensor();
    sensor->setPoseFrame(ccCameraSensor::PoseFrame::VtkColmap);
    sensor->setPlaneColor(kMapCameraPlaneColor);
    sensor->setFrameColor(kMapCameraFrameColor);

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

qMapAnything::qMapAnything(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qMapAnything/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qMapAnything"));
    qRegisterMetaType<MapAnythingResult>("MapAnythingResult");
    qRegisterMetaType<MapAnythingDialog::Settings>(
            "MapAnythingDialog::Settings");
    m_action = new QAction(tr("MapAnything Multi-View Reconstruction"), this);
    m_action->setToolTip(
            tr("Universal feed-forward metric 3D reconstruction "
               "(VGGT-Omega / Pi3 / MapAnything / DUSt3R, "
               "GGML)"));
    m_action->setIcon(
            QIcon(":/CC/plugin/qMapAnything/images/qMapAnything.svg"));
    connect(m_action, &QAction::triggered, this, &qMapAnything::showDialog);
}

QList<QAction*> qMapAnything::getActions() { return {m_action}; }

void qMapAnything::registerCommands(ccCommandLineInterface* cmd) {
    if (cmd == nullptr) return;
    cmd->registerCommand(ccCommandLineInterface::Command::Shared(
            new CommandMapAnythingReconstruct));
}

ccImage* qMapAnything::findDbImage(const QString& name) const {
    if (!m_app) return nullptr;
    ccHObject* root = m_app->dbRootObject();
    if (!root) return nullptr;
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    for (ccHObject* obj : images) {
        if (obj && obj->getName() == name) {
            return dynamic_cast<ccImage*>(obj);
        }
    }
    return nullptr;
}

QStringList qMapAnything::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (!obj) continue;
        if (obj->isA(CV_TYPES::IMAGE)) {
            ccImage* img = dynamic_cast<ccImage*>(obj);
            if (!img || img->data().isNull()) continue;
            names.append(obj->getName());
        } else if (obj->isGroup()) {
            ccHObject::Container images;
            obj->filterChildren(images, true, CV_TYPES::IMAGE, false);
            for (ccHObject* child : images) {
                if (!child) continue;
                ccImage* img = dynamic_cast<ccImage*>(child);
                if (!img || img->data().isNull()) continue;
                names.append(child->getName());
            }
        }
    }
    return names;
}

void qMapAnything::onNewSelection(
        const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;
    const QStringList names = selectedDbImageNames();
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

void qMapAnything::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    // The shared input list keeps db:// entries; selection changes append
    // the newly selected DB images (dedup happens in the dialog).
    const QStringList names = selectedDbImageNames();
    if (!names.isEmpty()) {
        m_dialog->applyDbTreeSelection(names);
    }
}

void qMapAnything::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new MapAnythingDialog(
                static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &MapAnythingDialog::runRequested, this,
                &qMapAnything::executeTask);
        connect(m_dialog, &MapAnythingDialog::cancelRequested, this,
                &qMapAnything::cancelTask);
        connect(m_dialog, &MapAnythingDialog::refreshDbImagesRequested, this,
                [this]() { refreshDbImages(); });
    }
    m_dialog->refreshModelList();
    refreshDbImages();
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

bool qMapAnything::resolveInputPaths(MapAnythingDialog::Settings& settings) {
    QStringList resolved;
    for (const QString& raw : settings.imagePaths) {
        if (!raw.startsWith(QStringLiteral("db://"))) {
            if (QFile::exists(raw)) resolved.append(raw);
            continue;
        }
        const QString name = raw.mid(5);
        ccImage* img = findDbImage(name);
        if (!img || img->data().isNull()) {
            m_dialog->appendLog(
                    tr("[Error] DB image not found or empty: %1").arg(name));
            clearStagedInputFiles();
            return false;
        }
        const QString tmpDir = MapAnythingDialog::modelCacheDir() + "/../tmp";
        QDir().mkpath(tmpDir);
        const QString outPath =
                tmpDir + "/mapanything-" +
                QUuid::createUuid().toString(QUuid::WithoutBraces) + ".png";
        if (!img->data().save(outPath)) {
            m_dialog->appendLog(
                    tr("[Error] Failed to export DB image: %1").arg(name));
            clearStagedInputFiles();
            return false;
        }
        m_stagedInputFiles << outPath;
        resolved.append(outPath);
    }
    settings.imagePaths = resolved;
    return !resolved.isEmpty();
}

void qMapAnything::clearStagedInputFiles() {
    for (const QString& path : m_stagedInputFiles) {
        QFile::remove(path);
    }
    m_stagedInputFiles.clear();
}

void qMapAnything::executeTask(const MapAnythingDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) return;

    // Materialized-result cleanup (qLingbotMap lesson): remove the previous
    // run's outputs before starting a new one, otherwise the old cloud /
    // depth rasters alias the new run in the DB tree and read as
    // "reconstruction out of sync".
    if (m_app != nullptr) {
        ccHObject* root = m_app->dbRootObject();
        if (root != nullptr) {
            std::vector<ccHObject*> stale;
            ccHObject::Container children;
            root->filterChildren(children, true, CV_TYPES::POINT_CLOUD, false);
            root->filterChildren(children, true, CV_TYPES::IMAGE, false);
            root->filterChildren(children, true, CV_TYPES::HIERARCHY_OBJECT,
                                 false);
            for (ccHObject* child : children) {
                if (child == nullptr ||
                    !(child->getName().startsWith(
                              QStringLiteral("MapAnything_")) ||
                      child->getName().startsWith(
                              QStringLiteral("MapAnythingDepth_")))) {
                    continue;
                }
                // Keep only TOP-LEVEL stale items: removeFromDB deletes the
                // whole subtree (autoDelete=true), so a nested descendant
                // collected here would be a dangling pointer by the time
                // its turn came (SIGSEGV on the second run of a session
                // that already holds MapAnything results).
                bool ancestorStale = false;
                for (ccHObject* p = child->getParent();
                     p != nullptr && p != root; p = p->getParent()) {
                    const QString& pn = p->getName();
                    if (pn.startsWith(QStringLiteral("MapAnything_")) ||
                        pn.startsWith(QStringLiteral("MapAnythingDepth_"))) {
                        ancestorStale = true;
                        break;
                    }
                }
                if (!ancestorStale) stale.push_back(child);
            }
            for (ccHObject* child : stale) {
                m_app->removeFromDB(child);  // DB tree ownership
            }
        }
    }

    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    clearStagedInputFiles();

    if (settings.modelPath.isEmpty()) {
        m_dialog->appendLog(tr("[Error] Model required."));
        return;
    }
    if (settings.imagePaths.isEmpty()) {
        m_dialog->appendLog(tr("[Error] No input images."));
        return;
    }

    MapAnythingDialog::Settings workerSettings = settings;
    if (!resolveInputPaths(workerSettings)) {
        m_dialog->appendLog(tr("[Error] No resolvable input images."));
        return;
    }

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[MapAnything] GPU backend unavailable — using CPU "
                       "for this run."));
        }
    }
#endif
    workerSettings.device = workerDevice;

    m_currentSettings = settings;
    MapAnythingWorker::Settings ws;
    ws.modelPath = workerSettings.modelPath;
    ws.imagePaths = workerSettings.imagePaths;
    ws.threads = workerSettings.threads;
    ws.device = workerDevice;
    ws.imageSize = workerSettings.imageSize;
    ws.resizeMode = workerSettings.resizeMode;
    ws.addPointCloud = workerSettings.addPointCloud;
    ws.addDepthRasters = workerSettings.addDepthRasters;
    ws.addPerViewClouds = workerSettings.addPerViewClouds;
    ws.responsiveStreaming = workerSettings.responsiveStreaming;
    ws.minConf = workerSettings.minConf;
    if (!workerSettings.refImagesPath.isEmpty()) {
        // The user's own camera positions (model_aligner text). Names are
        // matched against the resolved paths, so DB-staged copies (random
        // file names) are not matched and fail the run loudly.
        ws.refImagesPath = workerSettings.refImagesPath;
        ws.refIsGps = workerSettings.refIsGps;
    } else if (workerSettings.anchorToReferenceFrame) {
        // Resolved paths: DB-staged copies never sit in a benchmark scene.
        ws.gtMetaPath = MapAnythingSceneMeta::discover(ws.imagePaths);
        if (ws.gtMetaPath.isEmpty()) {
            m_dialog->appendLog(
                    tr("[MapAnything] Not a benchmark scene (no "
                       "scene_meta.json beside images/) and no reference "
                       "cameras file: the result keeps the model's official "
                       "output frame and unit."));
        }
    }

    m_worker = new MapAnythingWorker(ws, this);
    connect(m_worker, &MapAnythingWorker::logMessage, m_dialog,
            &MapAnythingDialog::appendLog, Qt::QueuedConnection);
    connect(m_worker, &MapAnythingWorker::progressUpdate, this,
            &qMapAnything::onWorkerProgress, Qt::QueuedConnection);
    connect(m_worker, &MapAnythingWorker::resultReady, this,
            &qMapAnything::onResultReady, Qt::QueuedConnection);
    connect(m_worker, &MapAnythingWorker::taskFinished, this,
            &qMapAnything::onTaskFinished, Qt::QueuedConnection);
    connect(m_worker, &MapAnythingWorker::modelInfoReady, m_dialog,
            &MapAnythingDialog::appendLog, Qt::QueuedConnection);
    m_dialog->setRunning(true);
    m_worker->start();
}

void qMapAnything::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qMapAnything::onWorkerProgress(int current, int total) {
    if (!m_dialog) return;
    m_dialog->setProgress(current, total);
    const int pct = total > 0 ? (current * 100 / total) : 0;
    // Step layout (see MapAnythingWorker::runInference): 1 model load,
    // 1 decode, N per-view assembly (one per view, first-cover chunk),
    // 1 scene filter. Map stages at those boundaries so the label never
    // disagrees with the bar.
    QString stage;
    if (current <= 1) {
        stage = tr("Loading GGUF model...");
    } else if (current == 2) {
        stage = tr("Decoding views & running multi-view inference...");
    } else if (current < total) {
        stage = tr("Inference & per-view assembly... (%1%)").arg(pct);
    } else {
        stage = tr("Scene-wide confidence filter... (%1%)").arg(pct);
    }
    m_dialog->setTaskStage(stage, pct);
}

void qMapAnything::onResultReady(const MapAnythingResult& result) {
    if (!m_app) return;
    addResultToDb(result);
}

void qMapAnything::addResultToDb(const MapAnythingResult& result) {
    QString dbStage = QStringLiteral("start");
    try {
        // Stage-tracked exception guard: libstdc++ range-check exceptions
        // thrown from deep inside the DB/display layer carry no usable
        // throw-site stack (the __cxa_throw interposition cannot see them), so
        // the exported entities are tagged with the stage they were produced
        // in - a failure names the exact producer instead of a bare
        // "_M_range_check" line.
        if (!m_app) return;

        const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
                result.resolvedDevice.isEmpty() ? m_currentSettings.device
                                                : result.resolvedDevice);
        const QString modelTag = ecvPluginDbNaming::modelTagFromFilename(
                result.modelPath.isEmpty() ? m_currentSettings.modelPath
                                           : result.modelPath);
        // Grouped layout: one root folder per run holding the cloud, the
        // depth folder, the confidence folder and the camera folder — flat
        // 16-raster dumps made the DB tree unmanageable.
        auto* rootGroup = new ccHObject(
                QStringLiteral("MapAnything_%1_%2").arg(modelTag, deviceTag));
        rootGroup->setMetaData(QStringLiteral("MapAnything"), true);
        dbStage = QStringLiteral("depth/conf/cam folders");
        auto* depthFolder = new ccHObject(
                QStringLiteral("MapAnythingDepth_%1").arg(modelTag));
        auto* confFolder = new ccHObject(
                QStringLiteral("MapAnythingConf_%1").arg(modelTag));
        auto* camFolder = new ccHObject(
                QStringLiteral("MapAnythingCameras_%1").arg(modelTag));
        auto* maskFolder = new ccHObject(
                QStringLiteral("MapAnythingMasks_%1").arg(modelTag));
        auto* perViewFolder = new ccHObject(
                QStringLiteral("MapAnythingViews_%1").arg(modelTag));
        depthFolder->setMetaData(QStringLiteral("MapAnything"), true);
        confFolder->setMetaData(QStringLiteral("MapAnything"), true);
        camFolder->setMetaData(QStringLiteral("MapAnything"), true);
        maskFolder->setMetaData(QStringLiteral("MapAnything"), true);
        perViewFolder->setMetaData(QStringLiteral("MapAnything"), true);
        rootGroup->addChild(depthFolder);
        rootGroup->addChild(confFolder);
        rootGroup->addChild(camFolder);
        rootGroup->addChild(maskFolder);
        rootGroup->addChild(perViewFolder);

        dbStage = QStringLiteral("merged cloud");
        // Reference frames with large coordinates (UTM, ECEF...): the
        // entities hold local floats, the clouds carry the offset as their
        // global shift (local = global + shift), so picking, measurement
        // and export see the true coordinates.
        const CCVector3d globalShift(-result.referenceOffset[0],
                                     -result.referenceOffset[1],
                                     -result.referenceOffset[2]);
        if (result.pointCount > 0) {
            const QString name = ecvPluginDbNaming::makeUnique(
                    QStringLiteral("MapAnything_%1_%2")
                            .arg(modelTag, deviceTag),
                    m_app);
            auto* cloud = new ccPointCloud(name);
            if (cloud->reserve(result.pointCount)) {
                for (int i = 0; i < result.pointCount; ++i) {
                    cloud->addPoint(CCVector3(result.points[3 * i + 0],
                                              result.points[3 * i + 1],
                                              result.points[3 * i + 2]));
                }
                if (cloud->reserveTheRGBTable()) {
                    for (int i = 0; i < result.pointCount; ++i) {
                        const ecvColor::Rgb col(result.colors[3 * i + 0],
                                                result.colors[3 * i + 1],
                                                result.colors[3 * i + 2]);
                        cloud->addRGBColor(col);
                    }
                    cloud->showColors(true);
                }
                cloud->setMetaData(QStringLiteral("MapAnything"), true);
                cloud->setMetaData(QStringLiteral("MapAnything/Architecture"),
                                   result.architecture);
                cloud->setMetaData(QStringLiteral("MapAnything/Views"),
                                   static_cast<qlonglong>(result.viewCount));
                cloud->setMetaData(
                        QStringLiteral("MapAnything/MinConf"),
                        static_cast<double>(m_currentSettings.minConf));
                cloud->setMetaData(QStringLiteral("Runtime (ms)"),
                                   result.e2eMs);
                cloud->setGlobalShift(globalShift);
                if (!result.referenceFrame.isEmpty()) {
                    cloud->setMetaData(
                            QStringLiteral("MapAnything/ReferenceFrame"),
                            result.referenceFrame);
                    cloud->setMetaData(
                            QStringLiteral("MapAnything/ReferenceScale"),
                            static_cast<double>(result.referenceScale));
                }
                if (result.hasMetricScale) {
                    // mapanything only: multiply the local geometry by this to
                    // obtain metric units.
                    cloud->setMetaData(
                            QStringLiteral("MapAnything/MetricScale"),
                            static_cast<double>(result.metricScale));
                }
                if (!result.resolvedDevice.isEmpty()) {
                    cloud->setMetaData(QStringLiteral("Device"),
                                       result.resolvedDevice);
                }
                rootGroup->addChild(cloud);
                m_dialog->appendLog(
                        tr("[MapAnything] Added point cloud '%1' (%2 points).")
                                .arg(name)
                                .arg(result.pointCount));
            } else {
                delete cloud;
                m_dialog->appendLog(
                        tr("[Error] Failed to allocate the output "
                           "point cloud."));
            }
        }

        for (int i = 0; i < result.views.size(); ++i) {
            const MapAnythingViewResult& view = result.views[i];
            if (view.depthImage.isNull()) continue;
            const QString name = ecvPluginDbNaming::makeUnique(
                    QStringLiteral("MapAnythingDepth_%1_%2")
                            .arg(i + 1)
                            .arg(modelTag),
                    m_app);
            auto* img = new ccImage(view.depthImage, name);
            img->setMetaData(QStringLiteral("MapAnything"), true);
            img->setMetaData(QStringLiteral("MapAnything/View"),
                             static_cast<qlonglong>(i + 1));
            img->setMetaData(QStringLiteral("MapAnything/Architecture"),
                             result.architecture);
            if (!view.sourcePath.isEmpty()) {
                img->setMetaData(QStringLiteral("Source"), view.sourcePath);
            }
            // NOT hidden per-entity: the DB-tree checkbox drives the ENABLED
            // state, and the raster folders ship UNCHECKED below (one checkbox
            // hides the whole folder; ticking it shows every raster).
            depthFolder->addChild(img);
            m_dialog->appendLog(
                    tr("[MapAnything] Added depth raster '%1'.").arg(name));
        }

        // Normalized-confidence rasters (same crop as the depth rasters).
        for (int i = 0; i < result.views.size(); ++i) {
            const MapAnythingViewResult& view = result.views[i];
            if (view.confImage.isNull()) continue;
            const QString name = ecvPluginDbNaming::makeUnique(
                    QStringLiteral("MapAnythingConf_%1_%2")
                            .arg(i + 1)
                            .arg(modelTag),
                    m_app);
            auto* img = new ccImage(view.confImage, name);
            img->setMetaData(QStringLiteral("MapAnything"), true);
            img->setMetaData(QStringLiteral("MapAnything/View"),
                             static_cast<qlonglong>(i + 1));
            if (!view.sourcePath.isEmpty()) {
                img->setMetaData(QStringLiteral("Source"), view.sourcePath);
            }
            confFolder->addChild(img);
        }

        dbStage = QStringLiteral("mask rasters");
        // Validity-mask rasters (mapanything architecture only; same crop as
        // the depth/conf rasters; 255 = in-mask).
        for (int i = 0; i < result.views.size(); ++i) {
            const MapAnythingViewResult& view = result.views[i];
            if (view.maskImage.isNull()) continue;
            const QString name = ecvPluginDbNaming::makeUnique(
                    QStringLiteral("MapAnythingMask_%1_%2")
                            .arg(i + 1)
                            .arg(modelTag),
                    m_app);
            auto* img = new ccImage(view.maskImage, name);
            img->setMetaData(QStringLiteral("MapAnything"), true);
            img->setMetaData(QStringLiteral("MapAnything/View"),
                             static_cast<qlonglong>(i + 1));
            if (!view.sourcePath.isEmpty()) {
                img->setMetaData(QStringLiteral("Source"), view.sourcePath);
            }
            maskFolder->addChild(img);
        }

        // Cameras (vggt family: FoV intrinsics are published by pose_enc).
        if (result.hasCameras) {
            // A frustum is renderer-only auxiliary geometry.  Its scale derives
            // from the actual c2w trajectory, never from the
            // reconstructed-cloud extent: learned/non-metric geometry and
            // outliers must not enlarge a camera.  camPoses16 remains
            // byte-for-byte untouched.
            std::vector<float> baselines;
            baselines.reserve(std::max(0, result.viewCount - 1));
            for (int i = 1; i < result.viewCount; ++i) {
                const float* previous =
                        result.camPoses16.constData() + (size_t)(i - 1) * 16;
                const float* current =
                        result.camPoses16.constData() + (size_t)i * 16;
                const float dx = current[3] - previous[3];
                const float dy = current[7] - previous[7];
                const float dz = current[11] - previous[11];
                const float baseline = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (std::isfinite(baseline) && baseline > 1e-6f) {
                    baselines.push_back(baseline);
                }
            }
            const float medianBaseline =
                    medianCameraBaseline(std::move(baselines));
            const float planeWidth =
                    medianBaseline > 0.0f
                            ? medianBaseline * kCameraBaselineFraction
                            : kFallbackCameraDisplaySize;
            const float displaySize = planeWidth *
                                      ecvCameraSensorDisplay::kBaseCameraWidth /
                                      std::max(1, result.camWidth);
            int added = 0;
            for (int i = 0; i < result.viewCount; ++i) {
                const float focal = result.camFocalPx.value(i, 0.0f);
                auto* sensor = buildMapCameraSensor(
                        result.camPoses16.constData() + (size_t)i * 16, focal,
                        result.camWidth, result.camHeight, displaySize);
                if (sensor == nullptr) continue;
                sensor->setName(QStringLiteral("MapAnythingCam_%1_%2")
                                        .arg(i + 1)
                                        .arg(modelTag));
                sensor->setMetaData(QStringLiteral("MapAnything"), true);
                sensor->setMetaData(QStringLiteral("MapAnything/View"),
                                    static_cast<qlonglong>(i + 1));
                camFolder->addChild(sensor);
                ++added;
            }
            if (added > 0) {
                m_dialog->appendLog(
                        tr("[MapAnything] Added %1 camera sensors to "
                           "'MapAnythingCameras_%2'.")
                                .arg(added)
                                .arg(modelTag));
            }
        }

        // Opt-in per-view world-frame clouds (Settings::addPerViewClouds,
        // large).
        for (int i = 0; i < result.perViewPoints.size(); ++i) {
            const QVector<float>& pts = result.perViewPoints[i];
            const int n = pts.size() / 3;
            if (n == 0) {
                // Fully confidence-filtered view: an EMPTY ccPointCloud in the
                // DB tree makes the display/pick path throw
                // std::vector::_M_range_check on its zero-length internals.
                continue;
            }
            const QString name = ecvPluginDbNaming::makeUnique(
                    QStringLiteral("MapAnythingView_%1_%2")
                            .arg(i + 1)
                            .arg(modelTag),
                    m_app);
            auto* cloud = new ccPointCloud(name);
            if (!cloud->reserve(n)) {
                delete cloud;
                continue;
            }
            for (int p = 0; p < n; ++p) {
                cloud->addPoint(CCVector3(pts[3 * p + 0], pts[3 * p + 1],
                                          pts[3 * p + 2]));
            }
            const QVector<quint8>& cols = result.perViewColors[i];
            if (cloud->reserveTheRGBTable()) {
                // reserveTheRGBTable() reserves CAPACITY only (size stays 0):
                // indexed setPointColor writes throw vector::_M_range_check on
                // the empty table - the exact GUI-crash fingerprint. Append
                // instead, same as the merged-cloud section above.
                for (int p = 0; p < n; ++p) {
                    const ecvColor::Rgb col(cols[3 * p + 0], cols[3 * p + 1],
                                            cols[3 * p + 2]);
                    cloud->addRGBColor(col);
                }
                cloud->showColors(true);
            }
            cloud->setMetaData(QStringLiteral("MapAnything"), true);
            cloud->setMetaData(QStringLiteral("MapAnything/View"),
                               static_cast<qlonglong>(i + 1));
            cloud->setGlobalShift(globalShift);
            perViewFolder->addChild(cloud);
        }

        // vggt-omega-256-text: the embedding has no geometric meaning, so it is
        // only summarized here (the headless command writes the raw sidecar).
        if (result.textEmbeddingDim > 0) {
            const float* e = reinterpret_cast<const float*>(
                    result.textEmbedding.constData());
            double l2 = 0.0;
            for (int i = 0; i < result.textEmbeddingDim; ++i) {
                l2 += double(e[i]) * double(e[i]);
            }
            m_dialog->appendLog(tr("[MapAnything] Text embedding: dim=%1, "
                                   "L2=%.4f (headless "
                                   "runs export the raw sidecar file).")
                                        .arg(result.textEmbeddingDim)
                                        .arg(std::sqrt(l2)));
        }

        // Raster folders ship UNCHECKED — the DB-tree checkbox IS the enabled
        // state, and the folder ALONE carries it: the children keep
        // enabled=true (their checkboxes render checked, matching the
        // expectation that only the group is switched off). At birth the
        // folder's disabled state suppresses the whole subtree through
        // ccHObject::draw's entry gate
        // (!isEnabled -> hideObject_recursive), so nothing renders; ticking the
        // folder re-enables it and the tree's enable branch (forceRedraw +
        // redraw) reaches the children, whose draw pass creates their
        // representations. Disabling the CHILDREN here would permanently wedge
        // them: the tree's enable path never restores child enabled flags.
        // NOTE: also MUST run BEFORE the finalize section below —
        // rootGroup->removeChild() DELETES an emptied folder (DP_DELETE_OTHER
        // dependency), so any folder touched afterwards would be a dangling
        // pointer (SIGSEGV in the disable recursion, crash log 092256).
        depthFolder->setEnabled(false);
        confFolder->setEnabled(false);
        maskFolder->setEnabled(false);
        dbStage = QStringLiteral("finalize folders");
        if (depthFolder->getChildrenNumber() == 0)
            rootGroup->removeChild(depthFolder);
        if (confFolder->getChildrenNumber() == 0)
            rootGroup->removeChild(confFolder);
        if (camFolder->getChildrenNumber() == 0)
            rootGroup->removeChild(camFolder);
        if (maskFolder->getChildrenNumber() == 0)
            rootGroup->removeChild(maskFolder);
        if (perViewFolder->getChildrenNumber() == 0)
            rootGroup->removeChild(perViewFolder);
        if (rootGroup->getChildrenNumber() == 0) {
            delete rootGroup;
            return;
        }
        m_app->addToDB(rootGroup, true, true, false, true);
        if (result.pointCount > 0) {
            // Child order: [depth?, conf?, cameras?, cloud] — the cloud is the
            // LAST child whenever it exists.
            ccHObject* cloudNode =
                    rootGroup->getChild(rootGroup->getChildrenNumber() - 1);
            m_app->setSelectedInDB(
                    cloudNode && cloudNode->isA(CV_TYPES::POINT_CLOUD)
                            ? cloudNode
                            : rootGroup,
                    true);
        }
        m_dialog->appendLog(tr("[MapAnything] Results grouped under '%1'.")
                                    .arg(rootGroup->getName()));
    } catch (const std::exception& e) {
        aicore_inference_log::log(
                QStringLiteral("[Error] [MapAnything] DB export failed during "
                               "stage '%1': %2")
                        .arg(dbStage, QString::fromUtf8(e.what())));
    } catch (...) {
        aicore_inference_log::log(
                QStringLiteral("[Error] [MapAnything] DB export failed during "
                               "stage '%1': unknown exception")
                        .arg(dbStage));
    }
}

void qMapAnything::onTaskFinished(bool success) {
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    clearStagedInputFiles();
    if (!success) m_dialog->appendLog(tr("[Error] Task failed."));
}
