// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qSAM3D.h"

#include <FileIOFilter.h>
#include <ecvGlobalShiftManager.h>
#include <ecvMainAppInterface.h>
#include <ecvMesh.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>
#include <QTemporaryFile>

#include "Sam3dGenerateCommand.h"
#include "Sam3dVertexColors.h"
#include "ecvPersistentSettings.h"

qSAM3D::qSAM3D(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qSAM3D/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qSAM3D"));
    qRegisterMetaType<Sam3dRunResult>("Sam3dRunResult");
    qRegisterMetaType<Sam3dSceneResult>("Sam3dSceneResult");
    m_action = new QAction(tr("SAM 3D Objects Image to 3D"), this);
    m_action->setToolTip(
            tr("Generate a Gaussian PLY and FlexiCubes mesh from a single "
               "image (SAM 3D Objects, GGML)"));
    m_action->setIcon(QIcon(":/CC/plugin/qSAM3D/images/qSAM3D.svg"));
    connect(m_action, &QAction::triggered, this, &qSAM3D::showDialog);
}

QList<QAction*> qSAM3D::getActions() { return {m_action}; }

void qSAM3D::registerCommands(ccCommandLineInterface* cmd) {
    if (cmd == nullptr) {
        assert(false);
        return;
    }
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandSam3dGenerate));
}

void qSAM3D::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog =
                new Sam3dDialog(static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &Sam3dDialog::runRequested, this,
                &qSAM3D::executeTask);
        connect(m_dialog, &Sam3dDialog::cancelRequested, this,
                &qSAM3D::cancelTask);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qSAM3D::executeTask(const Sam3dDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) {
        QMessageBox::warning(m_dialog, tr("SAM 3D"),
                             tr("A generation task is already running; wait "
                                "for it to finish."));
        return;
    }
    m_lastSettings = settings;
    m_dialog->setRunning(true);
    m_dialog->appendLog(tr("[SAM3D] Starting generation from '%1'...")
                                .arg(settings.imagePath));

    m_worker = new Sam3dWorker(this);
    m_worker->configure(settings);
    // 1-arg signal -> default-level console log (Qt 5 PMF connect requires
    // slot arity <= signal arity, so route through a lambda).
    connect(m_worker, &Sam3dWorker::logMessage, m_dialog,
            [this](const QString& message) { m_dialog->appendLog(message); });
    connect(m_worker, &Sam3dWorker::stageChanged, this, &qSAM3D::onWorkerStage);
    connect(m_worker, &Sam3dWorker::resultReady, this, &qSAM3D::onResultReady);
    connect(m_worker, &Sam3dWorker::sceneReady, this, &qSAM3D::onSceneReady);
    connect(m_worker, &Sam3dWorker::taskFinished, this,
            &qSAM3D::onTaskFinished);
    m_worker->start();
}

void qSAM3D::cancelTask() {
    if (m_worker && m_worker->isRunning()) {
        m_worker->requestCancel();
        m_dialog->appendLog(tr("[SAM3D] Cancellation requested."));
    }
}

void qSAM3D::onWorkerStage(int stage, int, int) {
    // Stage 0..7 maps onto a coarse progress bar; stage-level granularity is
    // what the AICore contract provides.
    m_dialog->setProgress(0);
    Q_UNUSED(stage);
}

void qSAM3D::onResultReady(const Sam3dRunResult& result) {
    addResultToDb(result);
}

void qSAM3D::onSceneReady(const Sam3dSceneResult& result) {
    addSceneResultToDb(result);
}

void qSAM3D::addSceneResultToDb(const Sam3dSceneResult& result) {
    if (!m_app) return;
    const bool importToDb = m_lastSettings.importToDb;
    const QString sourceName = QFileInfo(result.sourceImage).completeBaseName();
    const QString groupName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("SAM3D_%1_%2")
                    .arg(sourceName, ecvPluginDbNaming::deviceTagFromName(
                                             result.backend.isEmpty()
                                                     ? QStringLiteral("auto")
                                                     : result.backend)),
            m_app);

    ccHObject* group = nullptr;
    auto groupRef = [&]() -> ccHObject* {
        if (!importToDb) return nullptr;
        if (!group) {
            group = new ccHObject(groupName);
        }
        return group;
    };

    int importedObjects = 0;
    for (const Sam3dRunResult& object : result.objects) {
        const QString entityName =
                QStringLiteral("%1.%2").arg(groupName, object.objectLabel);
        ccHObject* glbEntity = nullptr;
        if (!object.glbPath.isEmpty() && QFileInfo::exists(object.glbPath)) {
            glbEntity =
                    importGlbFile(object.glbPath, result, object, entityName);
        } else if (!object.glb.isEmpty()) {
            // Defensive: a bake that kept its bytes (no output dir).
            glbEntity = importGlbEntity(object.glb, object, entityName);
        }
        if (glbEntity) {
            ++importedObjects;
            glbEntity->setDisplay(m_app->getActiveGLDisplay());
            glbEntity->setVisible(true);
            if (ccHObject* parent = groupRef()) {
                parent->addChild(glbEntity);
            } else {
                m_app->addToDB(glbEntity, false, true, false, true);
            }
        }
    }

    // Composed scene splat cloud: every object's pose-applied gaussians in
    // one world frame (the official scene-assemble product).
    const int n = result.sceneSplatCount;
    const bool haveScene =
            n > 0 &&
            result.sceneCenters.size() == static_cast<qsizetype>(n) * 3 &&
            result.sceneRgb.size() == result.sceneCenters.size();
    if (haveScene) {
        auto* cloud =
                new ccPointCloud(QStringLiteral("%1.scene").arg(groupName));
        if (cloud->reserve(n)) {
            const float* centers = result.sceneCenters.constData();
            const float* rgb = result.sceneRgb.constData();
            for (int i = 0; i < n; ++i) {
                cloud->addPoint(CCVector3(centers[i * 3 + 0],
                                          centers[i * 3 + 1],
                                          centers[i * 3 + 2]));
            }
            if (cloud->resizeTheRGBTable()) {
                for (int i = 0; i < n; ++i) {
                    cloud->setPointColor(
                            i, ecvColor::Rgb(static_cast<ColorCompType>(
                                                     rgb[i * 3 + 0] * 255.0f),
                                             static_cast<ColorCompType>(
                                                     rgb[i * 3 + 1] * 255.0f),
                                             static_cast<ColorCompType>(
                                                     rgb[i * 3 + 2] * 255.0f)));
                }
                cloud->showColors(true);
            }
            cloud->setMetaData(QStringLiteral("Source"), result.sourceImage);
            cloud->setMetaData(QStringLiteral("Backend"), result.backend);
            cloud->setMetaData(QStringLiteral("Quantization"), result.dtype);
            cloud->setMetaData(QStringLiteral("Objects"),
                               static_cast<int>(result.objects.size()));
            if (ccHObject* parent = groupRef()) {
                parent->addChild(cloud);
            } else {
                cloud->setDisplay(m_app->getActiveGLDisplay());
                cloud->setVisible(true);
                m_app->addToDB(cloud, false, true, false, true);
            }
        } else {
            delete cloud;
        }
    }

    if (group) {
        group->setDisplay(m_app->getActiveGLDisplay());
        group->setVisible(true);
        m_app->addToDB(group, false, true, false, true);
    } else if (!importToDb) {
        m_dialog->appendLog(tr("[SAM3D] Scene kept out of the DB (import "
                               "disabled); %1 GLB file(s) remain in the "
                               "output directory.")
                                    .arg(result.objects.size()));
    }
    m_dialog->appendLog(
            tr("[SAM3D] Scene '%1': %2 object mesh(es) imported, %3 composed "
               "splats.")
                    .arg(groupName)
                    .arg(importedObjects)
                    .arg(haveScene ? n : 0));
}

ccHObject* qSAM3D::importGlbFile(const QString& glbPath,
                                 const Sam3dSceneResult& scene,
                                 const Sam3dRunResult& object,
                                 const QString& entityName) {
    if (!m_app || glbPath.isEmpty()) return nullptr;
    FileIOFilter::LoadParameters params;
    params.alwaysDisplayLoadDialog = false;
    params.shiftHandlingMode = ecvGlobalShiftManager::NO_DIALOG_AUTO_SHIFT;
    CC_FILE_ERROR err = CC_FERR_NO_ERROR;
    ccHObject* imported = FileIOFilter::LoadFromFile(glbPath, params, err);
    if (!imported) {
        m_dialog->appendLog(tr("[SAM3D] GLB import failed for %1 (error %2).")
                                    .arg(glbPath)
                                    .arg(static_cast<int>(err)),
                            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return nullptr;
    }
    imported->setName(entityName);
    imported->setMetaData(QStringLiteral("Source"), object.sourceImage);
    imported->setMetaData(QStringLiteral("Backend"), object.backend);
    imported->setMetaData(QStringLiteral("Quantization"), object.dtype);
    imported->setMetaData(QStringLiteral("Material"),
                          QStringLiteral("Baked UV-atlas GLB"));
    imported->setMetaData(QStringLiteral("Object"), object.objectLabel);
    imported->setMetaData(QStringLiteral("Scene"), scene.sourceImage);
    return imported;
}

void qSAM3D::onTaskFinished(bool success) {
    if (m_worker) {
        if (!success) {
            const QString error = m_worker->error();
            m_dialog->appendLog(tr("[SAM3D] Task failed: %1").arg(error),
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            QMessageBox::critical(m_dialog, tr("SAM 3D"),
                                  tr("Generation failed:\n%1").arg(error));
        } else {
            m_dialog->appendLog(tr("[SAM3D] Task finished."));
        }
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    m_dialog->setRunning(false);
}

void qSAM3D::addResultToDb(const Sam3dRunResult& result) {
    if (!m_app) return;
    const bool importToDb = m_lastSettings.importToDb;
    const QString sourceName = QFileInfo(result.sourceImage).completeBaseName();
    const QString entityName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("SAM3D_%1_%2")
                    .arg(sourceName, ecvPluginDbNaming::deviceTagFromName(
                                             result.backend.isEmpty()
                                                     ? QStringLiteral("auto")
                                                     : result.backend)),
            m_app);

    // Preferred display path: the baked GLB carries the full textured
    // material (xatlas UV atlas), which is the look to align with upstream.
    // The vertex-color mesh below stays as the fallback when the bake was
    // disabled or the import fails.
    ccHObject* glbEntity = nullptr;
    if (!result.glb.isEmpty()) {
        glbEntity = importGlbEntity(result.glb, result, entityName);
    }

    // Gaussian splats (world-domain centers + display RGB) -> colored point
    // cloud: the artifact the upstream renderer draws. Gated by the output
    // artifact selection (the splat data itself is always generated — the
    // mesh coloring and the bake consume it in memory).
    const bool wantSplatCloud = m_lastSettings.outputPointCloud;
    const bool haveSplats =
            result.gaussianCount > 0 &&
            result.splatCenters.size() ==
                    static_cast<qsizetype>(result.gaussianCount) * 3 &&
            result.splatRgb.size() == result.splatCenters.size();
    ccPointCloud* splatCloud = nullptr;
    if (wantSplatCloud && haveSplats) {
        splatCloud = new ccPointCloud(
                QStringLiteral("%1.gaussians").arg(entityName));
        if (!splatCloud->reserve(result.gaussianCount)) {
            delete splatCloud;
            splatCloud = nullptr;
        } else {
            const float* centers = result.splatCenters.constData();
            const float* rgb = result.splatRgb.constData();
            for (int i = 0; i < result.gaussianCount; ++i) {
                splatCloud->addPoint(CCVector3(centers[i * 3 + 0],
                                               centers[i * 3 + 1],
                                               centers[i * 3 + 2]));
            }
            // resizeTheRGBTable sizes the table to the current point count,
            // so it must follow the addPoint loop (qTrellis pattern).
            if (splatCloud->resizeTheRGBTable()) {
                for (int i = 0; i < result.gaussianCount; ++i) {
                    splatCloud->setPointColor(
                            i, ecvColor::Rgb(static_cast<ColorCompType>(
                                                     rgb[i * 3 + 0] * 255.0f),
                                             static_cast<ColorCompType>(
                                                     rgb[i * 3 + 1] * 255.0f),
                                             static_cast<ColorCompType>(
                                                     rgb[i * 3 + 2] * 255.0f)));
                }
                splatCloud->showColors(true);
            }
        }
    }

    ccMesh* mesh = nullptr;
    if (glbEntity == nullptr && result.meshTriangleCount > 0) {
        auto* cloud =
                new ccPointCloud(QStringLiteral("%1.vertices").arg(entityName));
        const int nv = result.meshVertexCount;
        if (!cloud->reserve(nv)) {
            delete cloud;
            cloud = nullptr;
        } else {
            for (int i = 0; i < nv; ++i) {
                const float* p = result.vertices.constData() + i * 3;
                cloud->addPoint(CCVector3(p[0], p[1], p[2]));
            }
            // Vertex colors from the nearest gaussian splat: the FlexiCubes
            // mesh carries no colors of its own (the upstream mesh export is
            // geometry-only; its look comes from the splat rendering).
            bool colored = false;
            if (haveSplats && cloud->resizeTheRGBTable()) {
                sam3d_colors::SplatColorGrid grid(result.splatCenters,
                                                  result.splatRgb);
                const float* verts = result.vertices.constData();
                for (int i = 0; i < nv; ++i) {
                    float rgb[3];
                    grid.colorAt(verts[i * 3 + 0], verts[i * 3 + 1],
                                 verts[i * 3 + 2], rgb);
                    cloud->setPointColor(
                            i,
                            ecvColor::Rgb(
                                    static_cast<ColorCompType>(rgb[0] * 255.0f),
                                    static_cast<ColorCompType>(rgb[1] * 255.0f),
                                    static_cast<ColorCompType>(rgb[2] *
                                                               255.0f)));
                }
                colored = true;
            }
            auto* meshEntity = new ccMesh(cloud);
            meshEntity->addChild(cloud);
            cloud->setVisible(false);
            const int nt = result.meshTriangleCount;
            if (!meshEntity->reserve(nt)) {
                delete meshEntity;
                meshEntity = nullptr;
            } else {
                for (int i = 0; i < nt; ++i) {
                    meshEntity->addTriangle(result.triangles[i * 3],
                                            result.triangles[i * 3 + 1],
                                            result.triangles[i * 3 + 2]);
                }
                meshEntity->computeNormals(true);
                meshEntity->showNormals(true);
                // Render the per-vertex splat colors: without this the mesh
                // falls back to the entity's default display colour (green).
                meshEntity->showColors(colored);
                mesh = meshEntity;
            }
        }
    }

    if (importToDb) {
        if (glbEntity) {
            glbEntity->setMetaData(QStringLiteral("Pipeline (ms)"),
                                   result.e2eMs);
            glbEntity->setMetaData(QStringLiteral("Gaussians"),
                                   result.gaussianCount);
            glbEntity->setDisplay(m_app->getActiveGLDisplay());
            glbEntity->setVisible(true);
            m_app->addToDB(glbEntity, false, true, false, true);
            m_dialog->appendLog(
                    tr("[SAM3D] Added textured GLB entity '%1' (%2 vertices, "
                       "%3 faces, xatlas UV atlas).")
                            .arg(entityName)
                            .arg(result.meshVertexCount)
                            .arg(result.meshTriangleCount));
        }
        if (mesh) {
            mesh->setName(entityName);
            mesh->setMetaData(QStringLiteral("Source"), result.sourceImage);
            mesh->setMetaData(QStringLiteral("Backend"), result.backend);
            mesh->setMetaData(QStringLiteral("Quantization"), result.dtype);
            mesh->setMetaData(QStringLiteral("Pipeline (ms)"), result.e2eMs);
            mesh->setMetaData(QStringLiteral("Gaussians"),
                              result.gaussianCount);
            mesh->setDisplay(m_app->getActiveGLDisplay());
            mesh->setVisible(true);
            m_app->addToDB(mesh, false, true, false, true);
            m_dialog->appendLog(
                    tr("[SAM3D] Added mesh '%1' (%2 vertices, %3 faces, "
                       "vertex-colored).")
                            .arg(entityName)
                            .arg(result.meshVertexCount)
                            .arg(result.meshTriangleCount));
        } else if (!glbEntity) {
            m_dialog->appendLog(
                    tr("[SAM3D] Mesh unavailable or allocation failed."));
        }
        if (splatCloud) {
            splatCloud->setMetaData(QStringLiteral("Source"),
                                    result.sourceImage);
            splatCloud->setMetaData(QStringLiteral("Backend"), result.backend);
            splatCloud->setMetaData(QStringLiteral("Quantization"),
                                    result.dtype);
            splatCloud->setMetaData(QStringLiteral("Pipeline (ms)"),
                                    result.e2eMs);
            splatCloud->setDisplay(m_app->getActiveGLDisplay());
            splatCloud->setVisible(true);
            m_app->addToDB(splatCloud, false, true, false, true);
            m_dialog->appendLog(
                    tr("[SAM3D] Added gaussian splat cloud '%1.gaussians' "
                       "(%2 points, colored).")
                            .arg(entityName)
                            .arg(result.gaussianCount));
        } else if (!haveSplats && wantSplatCloud) {
            m_dialog->appendLog(
                    tr("[SAM3D] Gaussian splat artifacts unavailable."));
        }
        if (!glbEntity && !result.glb.isEmpty()) {
            m_dialog->appendLog(
                    tr("[SAM3D] GLB import failed; the vertex-color mesh is "
                       "the DB fallback."));
        }
    } else {
        // DB import declined by the user: nothing owns these entities —
        // delete to avoid leaks.
        delete mesh;
        delete splatCloud;
        delete glbEntity;
    }

    if (!result.glbPath.isEmpty()) {
        if (QFileInfo::exists(result.glbPath)) {
            m_dialog->appendLog(
                    tr("[SAM3D] Textured GLB written to %1 "
                       "(%2 MB).")
                            .arg(result.glbPath)
                            .arg(result.glb.size() / (1024.0 * 1024.0), 0, 'f',
                                 1));
        } else {
            m_dialog->appendLog(tr("[SAM3D] GLB file export failed: %1")
                                        .arg(result.glbPath),
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        }
    }
    if (!result.plyPath.isEmpty()) {
        if (QFileInfo::exists(result.plyPath)) {
            m_dialog->appendLog(tr("[SAM3D] Gaussian PLY written to %1 "
                                   "(%2 Gaussians).")
                                        .arg(result.plyPath)
                                        .arg(result.gaussianCount));
        } else {
            m_dialog->appendLog(tr("[SAM3D] Gaussian PLY export failed: %1")
                                        .arg(result.plyPath),
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        }
    }
    m_dialog->appendLog(tr("[SAM3D] Pipeline wall time: %1 s")
                                .arg(result.e2eMs / 1000.0, 0, 'f', 1));
}

ccHObject* qSAM3D::importGlbEntity(const QByteArray& glb,
                                   const Sam3dRunResult& result,
                                   const QString& entityName) {
    if (glb.isEmpty() || !m_app) return nullptr;
    // Import through the shared file filters (qMeshIO's assimp glTF reader)
    // so the entity carries the full textured material (base-color atlas,
    // alpha blend) — the same rendering the upstream GLB gives when opened
    // in a viewer (qTrellis uses the identical path).
    QTemporaryFile tmp(QDir::tempPath() + QStringLiteral("/SAM3D_XXXXXX.glb"));
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
        m_dialog->appendLog(tr("[SAM3D] GLB import failed (error %1).")
                                    .arg(static_cast<int>(err)),
                            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return nullptr;
    }
    imported->setName(entityName);
    imported->setMetaData(QStringLiteral("Source"), result.sourceImage);
    imported->setMetaData(QStringLiteral("Backend"), result.backend);
    imported->setMetaData(QStringLiteral("Quantization"), result.dtype);
    imported->setMetaData(QStringLiteral("Material"),
                          QStringLiteral("Baked UV-atlas GLB"));
    return imported;
}
