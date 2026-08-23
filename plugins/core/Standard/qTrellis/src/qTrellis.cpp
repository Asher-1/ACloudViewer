// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qTrellis.h"

#include <ecvGenericMesh.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvMesh.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QSettings>

#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/trellis_capi.h"
#endif

qTrellis::qTrellis(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qTrellis/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qTrellis"));
    qRegisterMetaType<TrellisRunResult>("TrellisRunResult");
    qRegisterMetaType<TrellisDialog::Settings>("TrellisDialog::Settings");
    m_action = new QAction(tr("TRELLIS.2 Image to 3D"), this);
    m_action->setToolTip(
            tr("Generate a textured 3D mesh from a single image "
               "(TRELLIS.2, GGML)"));
    m_action->setIcon(QIcon(":/CC/plugin/qTrellis/images/qTrellis.svg"));
    connect(m_action, &QAction::triggered, this, &qTrellis::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(30000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 30;
        m_dialog->appendLog(tr("[TRELLIS] Task is running (%1 s elapsed)...")
                                    .arg(m_inferenceElapsedSeconds));
    });
}

QList<QAction*> qTrellis::getActions() { return {m_action}; }

void qTrellis::onNewSelection(const ccHObject::Container& selectedEntities) {
    // qTrellis works from a file path; nothing to sync here for now.
    Q_UNUSED(selectedEntities);
}

void qTrellis::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new TrellisDialog(
                static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &TrellisDialog::runRequested, this,
                &qTrellis::executeTask);
        connect(m_dialog, &TrellisDialog::cancelRequested, this,
                &qTrellis::cancelTask);
    }
    m_dialog->refreshModelState();
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

bool qTrellis::resolveInputPath(const QString& rawPath,
                                QString& outPath,
                                QString* errorMsg) {
    outPath.clear();
    if (rawPath.isEmpty()) {
        if (errorMsg) *errorMsg = tr("Input path is empty.");
        return false;
    }
    if (QFile::exists(rawPath)) {
        outPath = rawPath;
        return true;
    }
    if (errorMsg) *errorMsg = tr("Input file not found: %1").arg(rawPath);
    return false;
}

void qTrellis::executeTask(const TrellisDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) return;

    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    QString resolvedPath;
    QString err;
    if (!resolveInputPath(settings.inputPath, resolvedPath, &err)) {
        m_dialog->appendLog(err);
        return;
    }

    TrellisWorker::Settings workerSettings;
    workerSettings.modelPaths = settings.modelPaths;
    workerSettings.inputPath = resolvedPath;
    workerSettings.presetName = settings.presetName;
    workerSettings.pipelineType = settings.pipelineType;
    workerSettings.backgroundMode = settings.backgroundMode;
    workerSettings.steps = settings.steps;
    workerSettings.guidance = settings.guidance;
    workerSettings.textureSteps = settings.textureSteps;
    workerSettings.seed = settings.seed;
    workerSettings.threads = settings.threads;
    workerSettings.device = settings.device;
    workerSettings.shapeDecPlacement = settings.shapeDecPlacement;
    workerSettings.useRmbg = settings.useRmbg;
    workerSettings.textureEnabled = settings.textureEnabled;

    if (workerSettings.modelPaths.size() < 3) {
        m_dialog->appendLog(
                tr("[TRELLIS] Model paths unresolved — select a "
                   "preset and download the models first."));
        return;
    }

    m_currentSettings = settings;
    m_worker = new TrellisWorker(workerSettings, this);
    connect(m_worker, &TrellisWorker::logMessage, m_dialog,
            &TrellisDialog::appendLog, Qt::QueuedConnection);
    connect(m_worker, &TrellisWorker::progressUpdate, this,
            &qTrellis::onWorkerProgress, Qt::QueuedConnection);
    connect(m_worker, &TrellisWorker::resultReady, this,
            &qTrellis::onResultReady, Qt::QueuedConnection);
    connect(m_worker, &TrellisWorker::taskFinished, this,
            &qTrellis::onTaskFinished, Qt::QueuedConnection);
    m_dialog->setRunning(true);
    m_inferenceElapsedSeconds = 0;
    m_inferenceHeartbeat->start();
    m_worker->start();
}

void qTrellis::cancelTask() {
    if (m_worker && m_worker->isRunning()) {
        m_worker->requestInterruption();
    }
}

void qTrellis::onWorkerProgress(int stage, int step, int total) {
    if (!m_dialog) return;
    QString stageName;
    switch (stage) {
        case AICORE_TRELLIS_STAGE_PREPROCESS:
            stageName = tr("preprocess");
            break;
        case AICORE_TRELLIS_STAGE_DINO:
            stageName = tr("dino");
            break;
        case AICORE_TRELLIS_STAGE_SS_FLOW:
            stageName = tr("sparse structure");
            break;
        case AICORE_TRELLIS_STAGE_SS_DEC:
            stageName = tr("occupancy decode");
            break;
        case AICORE_TRELLIS_STAGE_SLAT_FLOW:
            stageName = tr("shape flow");
            break;
        case AICORE_TRELLIS_STAGE_SHAPE_DEC:
            stageName = tr("shape decode");
            break;
        case AICORE_TRELLIS_STAGE_MESH:
            stageName = tr("mesh extraction");
            break;
        case AICORE_TRELLIS_STAGE_UPSAMPLE:
            stageName = tr("upsample");
            break;
        case AICORE_TRELLIS_STAGE_SLAT_FLOW_HR:
            stageName = tr("shape flow 1024");
            break;
        case AICORE_TRELLIS_STAGE_SHAPE_DEC_HR:
            stageName = tr("shape decode 1024");
            break;
        case AICORE_TRELLIS_STAGE_TEXTURE:
            stageName = tr("PBR texture");
            break;
        default:
            stageName = tr("generation");
            break;
    }
    m_dialog->setProgressStage(stage, stageName, step, total);
}

void qTrellis::onResultReady(const TrellisRunResult& result) {
    if (!m_app) return;
    if (m_currentSettings.addResultToDb) {
        addResultToDb(result, m_currentSettings);
    }
    if (m_currentSettings.addRmbgImageToDb) {
        addRmbgImageToDb(result, m_currentSettings);
    }
    if (!m_currentSettings.saveGlbDir.isEmpty()) {
        saveResultGlb(result, m_currentSettings, result.sourceImage);
    }
}

void qTrellis::onTaskFinished(bool success) {
    m_inferenceHeartbeat->stop();
    if (m_dialog) {
        m_dialog->setRunning(false);
        if (!success) {
            m_dialog->appendLog(tr("[TRELLIS] Task failed."));
        }
    }
}

void qTrellis::addResultToDb(const TrellisRunResult& result,
                             const TrellisDialog::Settings& settings) {
    if (!m_app || result.verts.isEmpty() || result.tris.isEmpty()) return;

    const QString sourceName = QFileInfo(result.sourceImage).completeBaseName();
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("TRELLIS_%1").arg(sourceName), m_app);
    auto* cloud = new ccPointCloud(name);
    const int nv = result.verts.size() / 3;
    if (!cloud->reserve(nv)) {
        delete cloud;
        return;
    }
    for (int i = 0; i < nv; ++i) {
        cloud->addPoint(CCVector3(result.verts[i * 3], result.verts[i * 3 + 1],
                                  result.verts[i * 3 + 2]));
    }

    // PBR -> vertex colors (base_color rgb) + scalar fields.
    if (result.hasPbr && result.pbr.size() == nv * 6) {
        if (cloud->resizeTheRGBTable()) {
            for (int i = 0; i < nv; ++i) {
                const float* pbr = result.pbr.constData() + i * 6;
                cloud->setPointColor(
                        i,
                        ecvColor::Rgb(
                                static_cast<ColorCompType>(pbr[0] * 255.0f),
                                static_cast<ColorCompType>(pbr[1] * 255.0f),
                                static_cast<ColorCompType>(pbr[2] * 255.0f)));
            }
        }
        if (result.normals.size() == nv * 3 && cloud->resizeTheNormsTable()) {
            for (int i = 0; i < nv; ++i) {
                const float* n = result.normals.constData() + i * 3;
                cloud->setPointNormal(i, CCVector3(n[0], n[1], n[2]));
            }
        }
        const char* kMetallic = "PBR metallic";
        const char* kRoughness = "PBR roughness";
        const char* kAlpha = "PBR alpha";
        int idxM = cloud->addScalarField(kMetallic);
        int idxR = cloud->addScalarField(kRoughness);
        int idxA = cloud->addScalarField(kAlpha);
        if (idxM >= 0 && idxR >= 0 && idxA >= 0) {
            ccScalarField* sfM =
                    static_cast<ccScalarField*>(cloud->getScalarField(idxM));
            ccScalarField* sfR =
                    static_cast<ccScalarField*>(cloud->getScalarField(idxR));
            ccScalarField* sfA =
                    static_cast<ccScalarField*>(cloud->getScalarField(idxA));
            for (int i = 0; i < nv; ++i) {
                const float* pbr = result.pbr.constData() + i * 6;
                sfM->setValue(i, pbr[3]);
                sfR->setValue(i, pbr[4]);
                sfA->setValue(i, pbr[5]);
            }
            sfM->computeMinAndMax();
            sfR->computeMinAndMax();
            sfA->computeMinAndMax();
            // Keep the PBR channels queryable as scalar fields, but show the
            // base-color vertex colours by default: displaying the metallic
            // field here would paint the whole mesh grey (SF rendering
            // overrides vertex colours).
            cloud->showSF(false);
            cloud->showColors(true);
        }
    } else if (result.normals.size() == nv * 3) {
        if (cloud->resizeTheNormsTable()) {
            for (int i = 0; i < nv; ++i) {
                const float* n = result.normals.constData() + i * 3;
                cloud->setPointNormal(i, CCVector3(n[0], n[1], n[2]));
            }
        }
    }

    ccMesh* mesh = new ccMesh(cloud);
    mesh->addChild(cloud);
    const int nt = result.tris.size() / 3;
    if (!mesh->reserve(nt)) {
        delete mesh;
        return;
    }
    for (int i = 0; i < nt; ++i) {
        mesh->addTriangle(result.tris[i * 3], result.tris[i * 3 + 1],
                          result.tris[i * 3 + 2]);
    }
    // Keep the robust structure-tensor normals produced by fdg::vertex_normals
    // (aicore_trellis_capi): ccMesh::computePerVertexNormals would overwrite
    // them with a naive area-weighted average, which cancels out on the dual
    // grid's winding-mixed faces and leaves the mesh looking full of holes
    // under backface culling. Fall back to the classic computation only when
    // the pipeline delivered no normals at all.
    if (!cloud->hasNormals()) {
        mesh->computeNormals(true);
    }
    mesh->showNormals(true);

    mesh->setMetaData(QStringLiteral("Source"), result.sourceImage);
    mesh->setMetaData(QStringLiteral("Preset"), result.presetName);
    mesh->setMetaData(QStringLiteral("Runtime (ms)"), result.totalRuntimeMs);
    mesh->setMetaData(QStringLiteral("Backend"), result.backend);
    mesh->setMetaData(QStringLiteral("Model"),
                      QFileInfo(result.modelPath).fileName());

    m_app->addToDB(mesh);
    m_app->refreshAll();
    m_app->updateUI();
}

void qTrellis::addRmbgImageToDb(const TrellisRunResult& result,
                                const TrellisDialog::Settings& settings) {
    if (!m_app) return;
    if (result.rmbgImage.isNull()) {
        // Reached only when the AI matting did not run (no RMBG model loaded
        // despite the option): the solid-color fallback has no matted image.
        m_dialog->appendLog(tr("[TRELLIS] RMBG image unavailable: AI "
                               "background removal did not run."));
        return;
    }
    const QString sourceName = QFileInfo(result.sourceImage).completeBaseName();
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("TRELLIS_RMBG_%1").arg(sourceName), m_app);
    auto* img = new ccImage(result.rmbgImage, name);
    img->setMetaData(QStringLiteral("Source"), result.sourceImage);
    img->setMetaData(QStringLiteral("Preset"), result.presetName);
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.totalRuntimeMs);
    img->setMetaData(QStringLiteral("Backend"), result.backend);
    m_app->addToDB(img, /*updateZoom=*/false, /*autoExpandDBTree=*/true,
                   /*checkDimensions=*/false, /*autoRedraw=*/true);
    m_dialog->appendLog(tr("[TRELLIS] Added RMBG image '%1' to DB tree.")
                                .arg(name));
}

void qTrellis::saveResultGlb(const TrellisRunResult& result,
                             const TrellisDialog::Settings& settings,
                             const QString& sourceLabel) {
#ifdef AICore_ENABLED
    if (result.verts.isEmpty() || result.tris.isEmpty()) return;
    const QString dir = settings.saveGlbDir;
    QDir().mkpath(dir);
    const QString base = QFileInfo(sourceLabel).completeBaseName();
    const QString path = dir + QDir::separator() +
                         QStringLiteral("TRELLIS_%1_%2.glb")
                                 .arg(base)
                                 .arg(QDateTime::currentDateTime().toString(
                                         "yyyyMMdd_hhmmss"));
    char err[512] = {0};
    int outLen = 0;
    uint8_t* glb = aicore_trellis_bake_glb(
            result.verts.constData(), result.verts.size() / 3,
            result.tris.constData(), result.tris.size() / 3,
            result.hasPbr ? result.pbr.constData() : nullptr, 2048, 0, &outLen,
            err, sizeof(err));
    if (!glb) {
        m_dialog->appendLog(tr("[TRELLIS] GLB bake failed: %1")
                                    .arg(QString::fromUtf8(err)));
        return;
    }
    QFile f(path);
    if (f.open(QIODevice::WriteOnly)) {
        f.write(reinterpret_cast<const char*>(glb), outLen);
        f.close();
        m_dialog->appendLog(tr("[TRELLIS] GLB saved: %1").arg(path));
    }
    aicore_trellis_free_buffer(glb);
#else
    Q_UNUSED(result);
    Q_UNUSED(settings);
    Q_UNUSED(sourceLabel);
#endif
}
