// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qTrellis.h"

#include <FileIOFilter.h>
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
#include <QStandardPaths>
#include <QTemporaryFile>

#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/trellis_capi.h"
#endif

qTrellis::qTrellis(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qTrellis/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qTrellis"));
    qRegisterMetaType<TrellisRunResult>("TrellisRunResult");
    qRegisterMetaType<TrellisStagePreview>("TrellisStagePreview");
    qRegisterMetaType<TrellisDialog::Settings>("TrellisDialog::Settings");
    qRegisterMetaType<TrellisPrintResult>("TrellisPrintResult");
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
        connect(m_dialog, &TrellisDialog::exportRequested, this,
                &qTrellis::onExportRequested);
        connect(m_dialog, &TrellisDialog::printWrapRequested, this,
                &qTrellis::onPrintWrapRequested);
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
    workerSettings.runMode = settings.runMode;
    workerSettings.textureEnabled = settings.textureEnabled;
    workerSettings.quantization = settings.quantization;
    workerSettings.livePreview = true;

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
    connect(m_worker, &TrellisWorker::stagePreview, m_dialog,
            &TrellisDialog::setStagePreview, Qt::QueuedConnection);
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
        m_worker->cancelBake();
        m_worker->requestInterruption();
    }
    if (m_bakeWorker && m_bakeWorker->isRunning()) {
        m_bakeWorker->cancelBake();
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
    if (m_dialog) {
        // Step-strip completion + hand the result to the export page.
        m_dialog->setLastResult(result);
        m_dialog->applyResultToStrip(result);
        m_dialog->setStageState(3, TrellisDialog::kStageDone);
        m_dialog->setStageState(4, result.hasPbr
                                           ? TrellisDialog::kStageDone
                                           : TrellisDialog::kStagePending);
        m_dialog->updateExportInfo(result);
    }
    if (m_currentSettings.addResultToDb) {
        addResultToDb(result, m_currentSettings);
    }
    if (m_currentSettings.addRmbgImageToDb) {
        addRmbgImageToDb(result, m_currentSettings);
    }
    if (!m_currentSettings.saveGlbDir.isEmpty()) {
        saveResultGlb(result, m_currentSettings, result.sourceImage);
        if (m_dialog) {
            m_dialog->setStageState(5, TrellisDialog::kStageDone);
        }
    }
}

void qTrellis::onExportRequested() {
    // Export page: re-bake the textured GLB from the last generation with
    // the page's texture size / component-filter settings, then route the
    // same bytes to the persisted destination (DB tree by default, GLB
    // file, or both).
#ifdef AICore_ENABLED
    if (!m_dialog) return;
    const TrellisRunResult& result = m_dialog->lastResult();
    if (result.verts.isEmpty() || result.tris.isEmpty()) {
        m_dialog->appendLog(
                tr("[TRELLIS] Nothing to export yet — run a "
                   "generation first."));
        return;
    }
    if (m_worker && m_worker->isRunning()) {
        m_dialog->appendLog(
                tr("[TRELLIS] A generation is running - wait for it to "
                   "finish before re-baking."));
        return;
    }
    // The bake runs on its own worker (GUI stays responsive; minutes-long
    // bakes are cancellable) and routes on completion — see onBakeGlbReady.
    if (m_bakeWorker && m_bakeWorker->isRunning()) {
        m_dialog->appendLog(tr("[TRELLIS] A bake is already running."));
        return;
    }
    delete m_bakeWorker;
    TrellisWorker::Settings bs;
    bs.runMode = TrellisRunMode::BakeOnly;
    bs.bakeInput = result;
    bs.bakeTextureSize = m_dialog->exportTextureSize();
    bs.bakeComponentFilter = m_dialog->exportComponentFilter();
    m_bakeWorker = new TrellisWorker(bs, this);
    connect(m_bakeWorker, &TrellisWorker::bakeGlbReady, this,
            &qTrellis::onBakeGlbReady);
    connect(m_bakeWorker, &TrellisWorker::logMessage, m_dialog,
            &TrellisDialog::appendLog);
    connect(m_bakeWorker, &TrellisWorker::bakeProgress, m_dialog,
            &TrellisDialog::setBakeProgress, Qt::QueuedConnection);
    connect(m_bakeWorker, &TrellisWorker::taskFinished, this, [this](bool ok) {
        m_dialog->setRunning(false);
        m_dialog->setExportBusy(false,
                                ok ? tr("Bake finished.")
                                   : tr("Bake stopped (failed or cancelled)."));
        m_bakeWorker->deleteLater();
        m_bakeWorker = nullptr;
    });
    m_dialog->setRunning(true);
    m_dialog->setExportBusy(true);
    m_bakeWorker->start();
}

void qTrellis::onBakeGlbReady(const QByteArray& glb) {
    const TrellisRunResult& result = m_dialog->lastResult();
    const int destination = m_dialog->exportDestination();
    const bool wantDb = destination != TrellisDialog::kExportFile;
    const bool wantFile = destination != TrellisDialog::kExportDb;
    TrellisDialog::Settings settings = m_currentSettings;
    if (wantFile && settings.saveGlbDir.isEmpty()) {
        settings.saveGlbDir = QStandardPaths::writableLocation(
                                      QStandardPaths::DownloadLocation) +
                              QStringLiteral("/TRELLIS");
    }
    if (wantDb) {
        const QString sourceName =
                QFileInfo(result.sourceImage).completeBaseName();
        const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
                result.backend.isEmpty() ? settings.device : result.backend);
        const QString name = ecvPluginDbNaming::makeUnique(
                QStringLiteral("TRELLIS_%1_%2_%3px")
                        .arg(sourceName, deviceTag)
                        .arg(m_dialog->exportTextureSize()),
                m_app);
        if (ccHObject* imported = importGlbEntity(glb, result, name)) {
            m_app->addToDB(imported);
            m_app->refreshAll();
            m_app->updateUI();
            m_dialog->appendLog(
                    tr("[TRELLIS] Re-baked GLB added to the DB tree as "
                       "'%1' (full PBR material).")
                            .arg(name));
        }
    }
    if (wantFile) {
        QDir().mkpath(settings.saveGlbDir);
        writeGlbFile(glb, glbFilePath(settings.saveGlbDir, result.sourceImage));
    }
    if (m_dialog) m_dialog->setStageState(5, TrellisDialog::kStageDone);
#endif
}

void qTrellis::onPrintWrapRequested() {
    // Export page: watertight CGAL Alpha-Wrap print mesh of the last
    // generation, routed with the same destination combo as the re-bake
    // (DB tree by default, GLB file, or both). The wrap is pure CPU
    // geometry (no AICore context), so it runs on its own worker without
    // taking the inference device lock.
    if (!m_dialog) return;
    if (m_printWorker && m_printWorker->isRunning()) {
        m_dialog->appendLog(
                tr("[TRELLIS] Print wrap already running — wait for it to "
                   "finish."));
        return;
    }
    const TrellisRunResult result = m_dialog->lastResult();  // snapshot copy
    if (result.verts.isEmpty() || result.tris.isEmpty()) {
        m_dialog->appendLog(
                tr("[TRELLIS] Nothing to export yet — run a "
                   "generation first."));
        return;
    }
#ifdef AICore_ENABLED
    if (!aicore_trellis_print_remesh_available()) {
        m_dialog->appendLog(
                tr("[TRELLIS] Print wrap unavailable: rebuild ACloudViewer "
                   "with CGAL >= 5.5 to enable the Alpha Wrap."));
        return;
    }
    TrellisPrintRequest request;
    request.source = result;
    request.componentFilter = m_dialog->exportComponentFilter();
    request.textureSize = m_dialog->exportTextureSize();
    request.bakeGlb = result.hasPbr;  // projected bake requires source PBR
    m_printDestination = static_cast<TrellisDialog::ExportDestination>(
            m_dialog->exportDestination());

    m_printWorker = new TrellisPrintWorker(request, this);
    connect(m_printWorker, &TrellisPrintWorker::logMessage, m_dialog,
            &TrellisDialog::appendLog, Qt::QueuedConnection);
    connect(m_printWorker, &TrellisPrintWorker::printResultReady, this,
            &qTrellis::onPrintWrapReady, Qt::QueuedConnection);
    connect(m_printWorker, &TrellisPrintWorker::taskFinished, this,
            &qTrellis::onPrintWrapFinished, Qt::QueuedConnection);
    // Destruction must wait for run() to return: finished fires from the
    // worker thread after the queued results have landed on the GUI thread.
    connect(m_printWorker, &QThread::finished, m_printWorker,
            &QObject::deleteLater);
    m_dialog->setPrintWrapRunning(true);
    m_dialog->appendLog(
            tr("[TRELLIS] Print wrap started (CGAL Alpha Wrap) on the "
               "worker thread..."));
    m_printWorker->start();
#endif
}

void qTrellis::onPrintWrapReady(const TrellisPrintResult& print) {
    if (!m_app || !m_dialog) return;
    const TrellisRunResult& result = m_dialog->lastResult();
    const QString sourceName = QFileInfo(result.sourceImage).completeBaseName();
    const bool wantDb = m_printDestination != TrellisDialog::kExportFile;
    const bool wantFile = m_printDestination != TrellisDialog::kExportDb;
    TrellisDialog::Settings settings = m_currentSettings;
    if (wantFile && settings.saveGlbDir.isEmpty()) {
        settings.saveGlbDir = QStandardPaths::writableLocation(
                                      QStandardPaths::DownloadLocation) +
                              QStringLiteral("/TRELLIS");
    }
    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.backend.isEmpty() ? settings.device : result.backend);

    if (wantDb) {
        const QString name = ecvPluginDbNaming::makeUnique(
                QStringLiteral("TRELLIS_PRINT_%1_%2")
                        .arg(sourceName, deviceTag),
                m_app);
        // Preferred display path: the projected GLB carries the full PBR
        // material projected from the dense source onto the wrap geometry;
        // the vertex-colour fallback shows the projected per-vertex preview.
        ccHObject* entity = nullptr;
        if (!print.glb.isEmpty()) {
            entity = importGlbEntity(print.glb, result, name);
            if (entity) {
                entity->setMetaData(
                        QStringLiteral("Material"),
                        QStringLiteral("PBR projected GLB (print wrap)"));
            }
        }
        if (!entity) {
            auto* mesh =
                    buildVertexColorMesh(print.verts, print.normals, print.pbr,
                                         print.hasPbr, print.tris, name);
            if (mesh) {
                mesh->setMetaData(QStringLiteral("Source"), result.sourceImage);
                mesh->setMetaData(QStringLiteral("Preset"), result.presetName);
                mesh->setMetaData(QStringLiteral("Backend"), result.backend);
                mesh->setMetaData(QStringLiteral("Model"),
                                  QFileInfo(result.modelPath).fileName());
                mesh->setMetaData(
                        QStringLiteral("Material"),
                        QStringLiteral("vertex colours (print wrap)"));
                entity = mesh;
            }
        }
        if (entity) {
            entity->setMetaData(QStringLiteral("Print wrap (ms)"),
                                print.wrapMs);
            m_app->addToDB(entity);
            m_app->refreshAll();
            m_app->updateUI();
            m_dialog->appendLog(
                    tr("[TRELLIS] Watertight print mesh added to the DB "
                       "tree as '%1' (%2 verts / %3 tris, wrap %4 ms).")
                            .arg(name)
                            .arg(print.verts.size() / 3)
                            .arg(print.tris.size() / 3)
                            .arg(print.wrapMs, 0, 'f', 0));
        }
    }
    if (wantFile) {
        if (print.glb.isEmpty()) {
            // Untextured generation: the projected bake is impossible (the
            // C API requires source PBR) — say so instead of writing a stub.
            m_dialog->appendLog(
                    tr("[TRELLIS] GLB file export skipped: the projected bake "
                       "needs a textured generation."));
        } else {
            QDir().mkpath(settings.saveGlbDir);
            const QString path =
                    settings.saveGlbDir + QDir::separator() +
                    QStringLiteral("TRELLIS_PRINT_%1_%2_%3.glb")
                            .arg(sourceName, deviceTag)
                            .arg(QDateTime::currentDateTime().toString(
                                    "yyyyMMdd_hhmmss"));
            writeGlbFile(print.glb, path);
        }
    }
}

void qTrellis::onPrintWrapFinished(bool success) {
    if (m_dialog) {
        m_dialog->setPrintWrapRunning(false);
        if (!success) {
            m_dialog->appendLog(tr("[TRELLIS] Print wrap failed."));
        }
    }
    // The QThread::finished -> deleteLater connection owns destruction;
    // drop only our pointer (run() may still be finishing on the thread).
    m_printWorker = nullptr;
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
    // Device tag keeps CPU vs GPU (CuMesh) runs separable in the DB tree.
    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.backend.isEmpty() ? settings.device : result.backend);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("TRELLIS_%1_%2").arg(sourceName, deviceTag), m_app);

#ifdef AICore_ENABLED
    // Preferred display path: import the baked GLB through the shared file
    // filters so the entity carries the full PBR material (base-colour +
    // metallic-roughness atlases, alpha blend, double-sided). Vertex colours
    // cannot express metallic/roughness, so they remain only as the fallback
    // when the bake or the import fails (importGlbEntity logs it).
    if (ccHObject* imported = importGlbEntity(result.glb, result, name)) {
        m_app->addToDB(imported);
        m_app->refreshAll();
        m_app->updateUI();
        m_dialog->appendLog(
                tr("[TRELLIS] Added textured GLB mesh '%1' to the DB "
                   "tree (full PBR material).")
                        .arg(name));
        return;
    }
#endif

    ccMesh* mesh =
            buildVertexColorMesh(result.verts, result.normals, result.pbr,
                                 result.hasPbr, result.tris, name);
    if (!mesh) return;

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

ccMesh* qTrellis::buildVertexColorMesh(const QVector<float>& verts,
                                       const QVector<float>& normals,
                                       const QVector<float>& pbr,
                                       bool hasPbr,
                                       const QVector<int>& tris,
                                       const QString& name) {
    auto* cloud = new ccPointCloud(name);
    const int nv = verts.size() / 3;
    if (!cloud->reserve(nv)) {
        delete cloud;
        return nullptr;
    }
    for (int i = 0; i < nv; ++i) {
        cloud->addPoint(
                CCVector3(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]));
    }

    // PBR -> vertex colors (base_color rgb) + scalar fields.
    if (hasPbr && pbr.size() == nv * 6) {
        if (cloud->resizeTheRGBTable()) {
            for (int i = 0; i < nv; ++i) {
                const float* c = pbr.constData() + i * 6;
                cloud->setPointColor(
                        i, ecvColor::Rgb(
                                   static_cast<ColorCompType>(c[0] * 255.0f),
                                   static_cast<ColorCompType>(c[1] * 255.0f),
                                   static_cast<ColorCompType>(c[2] * 255.0f)));
            }
        }
        if (normals.size() == nv * 3 && cloud->resizeTheNormsTable()) {
            for (int i = 0; i < nv; ++i) {
                const float* n = normals.constData() + i * 3;
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
                const float* c = pbr.constData() + i * 6;
                sfM->setValue(i, c[3]);
                sfR->setValue(i, c[4]);
                sfA->setValue(i, c[5]);
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
    } else if (normals.size() == nv * 3) {
        if (cloud->resizeTheNormsTable()) {
            for (int i = 0; i < nv; ++i) {
                const float* n = normals.constData() + i * 3;
                cloud->setPointNormal(i, CCVector3(n[0], n[1], n[2]));
            }
        }
    }

    auto* mesh = new ccMesh(cloud);
    mesh->addChild(cloud);
    // The vertices cloud is a child container that carries the colours and
    // normals — drawing it as points over the mesh reads as a green stipple
    // overlay. Keep it in the DB tree for inspection, hidden by default.
    cloud->setVisible(false);
    const int nt = tris.size() / 3;
    if (!mesh->reserve(nt)) {
        // The cloud is a child of the mesh: freed with it.
        delete mesh;
        return nullptr;
    }
    for (int i = 0; i < nt; ++i) {
        mesh->addTriangle(tris[i * 3], tris[i * 3 + 1], tris[i * 3 + 2]);
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
    // Render the PBR vertex colours: without this the mesh falls back to
    // the entity's default display colour (green), regardless of the
    // colours set on the vertices cloud.
    mesh->showColors(true);
    return mesh;
}

void qTrellis::addRmbgImageToDb(const TrellisRunResult& result,
                                const TrellisDialog::Settings& settings) {
    if (!m_app) return;
    if (result.rmbgImage.isNull()) {
        // Reached only when the AI matting did not run (no RMBG model loaded
        // despite the option): the solid-color fallback has no matted image.
        m_dialog->appendLog(
                tr("[TRELLIS] RMBG image unavailable: AI "
                   "background removal did not run."));
        return;
    }
    const QString sourceName = QFileInfo(result.sourceImage).completeBaseName();
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("TRELLIS_RMBG_%1_%2")
                    .arg(sourceName, ecvPluginDbNaming::deviceTagFromName(
                                             result.backend.isEmpty()
                                                     ? QStringLiteral("auto")
                                                     : result.backend)),
            m_app);
    auto* img = new ccImage(result.rmbgImage, name);
    img->setMetaData(QStringLiteral("Source"), result.sourceImage);
    img->setMetaData(QStringLiteral("Preset"), result.presetName);
    img->setMetaData(QStringLiteral("Runtime (ms)"), result.totalRuntimeMs);
    img->setMetaData(QStringLiteral("Backend"), result.backend);
    m_app->addToDB(img, /*updateZoom=*/false, /*autoExpandDBTree=*/true,
                   /*checkDimensions=*/false, /*autoRedraw=*/true);
    m_dialog->appendLog(
            tr("[TRELLIS] Added RMBG image '%1' to DB tree.").arg(name));
}

void qTrellis::saveResultGlb(const TrellisRunResult& result,
                             const TrellisDialog::Settings& settings,
                             const QString& sourceLabel) {
#ifdef AICore_ENABLED
    // Reuse the GLB baked on the worker thread (identical 2048 / keep-tiny
    // parameters) instead of re-baking on the GUI thread.
    if (!result.glb.isEmpty()) {
        QDir().mkpath(settings.saveGlbDir);
        const QString base = QFileInfo(sourceLabel).completeBaseName();
        const QString path = settings.saveGlbDir + QDir::separator() +
                             QStringLiteral("TRELLIS_%1_%2.glb")
                                     .arg(base)
                                     .arg(QDateTime::currentDateTime().toString(
                                             "yyyyMMdd_hhmmss"));
        QFile f(path);
        if (f.open(QIODevice::WriteOnly)) {
            f.write(result.glb);
            f.close();
            m_dialog->appendLog(tr("[TRELLIS] GLB saved: %1").arg(path));
        }
        return;
    }
    saveResultGlbEx(result, settings, sourceLabel, 2048, 0);
#else
    Q_UNUSED(result);
    Q_UNUSED(settings);
    Q_UNUSED(sourceLabel);
#endif
}

QByteArray qTrellis::bakeResultGlb(const TrellisRunResult& result,
                                   int textureSize,
                                   int componentFilter) {
    QByteArray glb;
#ifdef AICore_ENABLED
    if (result.verts.isEmpty() || result.tris.isEmpty()) return glb;
    char err[512] = {0};
    int outLen = 0;
    uint8_t* bytes = aicore_trellis_bake_glb(
            result.verts.constData(), result.verts.size() / 3,
            result.tris.constData(), result.tris.size() / 3,
            result.hasPbr ? result.pbr.constData() : nullptr, textureSize,
            componentFilter, &outLen, err, sizeof(err));
    if (!bytes) {
        m_dialog->appendLog(tr("[TRELLIS] GLB bake failed: %1")
                                    .arg(QString::fromUtf8(err)));
        return glb;
    }
    glb = QByteArray(reinterpret_cast<const char*>(bytes), outLen);
    aicore_trellis_free_buffer(bytes);
#else
    Q_UNUSED(textureSize);
    Q_UNUSED(componentFilter);
#endif
    return glb;
}

QString qTrellis::glbFilePath(const QString& dir,
                              const QString& sourceLabel) const {
    const QString base = QFileInfo(sourceLabel).completeBaseName();
    return dir + QDir::separator() +
           QStringLiteral("TRELLIS_%1_%2.glb")
                   .arg(base)
                   .arg(QDateTime::currentDateTime().toString(
                           "yyyyMMdd_hhmmss"));
}

bool qTrellis::writeGlbFile(const QByteArray& glb, const QString& path) {
    if (glb.isEmpty()) return false;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) return false;
    f.write(glb);
    f.close();
    m_dialog->appendLog(tr("[TRELLIS] GLB saved: %1").arg(path));
    return true;
}

ccHObject* qTrellis::importGlbEntity(const QByteArray& glb,
                                     const TrellisRunResult& result,
                                     const QString& entityName) {
    if (glb.isEmpty() || !m_app) return nullptr;
    // Import through the shared file filters (qMeshIO's assimp glTF reader)
    // so the entity carries the full PBR material (base-colour +
    // metallic-roughness atlases, alpha blend, double-sided) — the same
    // rendering the upstream GLB gives when opened in the viewer.
    QTemporaryFile tmp(QDir::tempPath() +
                       QStringLiteral("/TRELLIS_XXXXXX.glb"));
    if (!tmp.open()) return nullptr;
    tmp.write(glb);
    tmp.flush();
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::NO_DIALOG_AUTO_SHIFT;
    CC_FILE_ERROR err = CC_FERR_NO_ERROR;
    ccHObject* imported =
            FileIOFilter::LoadFromFile(tmp.fileName(), params, err);
    tmp.close();
    if (!imported) {
        m_dialog->appendLog(
                tr("[TRELLIS] GLB import failed (error %1); falling back to "
                   "the vertex-colour mesh.")
                        .arg(static_cast<int>(err)));
        return nullptr;
    }
    imported->setName(entityName);
    imported->setMetaData(QStringLiteral("Source"), result.sourceImage);
    imported->setMetaData(QStringLiteral("Preset"), result.presetName);
    imported->setMetaData(QStringLiteral("Runtime (ms)"),
                          result.totalRuntimeMs);
    imported->setMetaData(QStringLiteral("Backend"), result.backend);
    imported->setMetaData(QStringLiteral("Model"),
                          QFileInfo(result.modelPath).fileName());
    imported->setMetaData(QStringLiteral("Material"),
                          QStringLiteral("PBR textured GLB"));
    return imported;
}

void qTrellis::saveResultGlbEx(const TrellisRunResult& result,
                               const TrellisDialog::Settings& settings,
                               const QString& sourceLabel,
                               int textureSize,
                               int componentFilter) {
    if (result.verts.isEmpty() || result.tris.isEmpty()) return;
    const QByteArray glb = bakeResultGlb(result, textureSize, componentFilter);
    if (glb.isEmpty()) return;
    QDir().mkpath(settings.saveGlbDir);
    writeGlbFile(glb, glbFilePath(settings.saveGlbDir, sourceLabel));
}
