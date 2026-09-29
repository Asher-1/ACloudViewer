// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qSAM3D.h"

#include <FileIOFilter.h>
#include <ecvGLMatrix.h>
#include <ecvGlobalShiftManager.h>
#include <ecvImage.h>
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
#include <algorithm>

#include "Sam3dGenerateCommand.h"
#include "Sam3dVertexColors.h"
#include "ecvPersistentSettings.h"

namespace {

// Place an imported GLB entity into the DB's Z-up world:
//
// 1. Every GLB this plugin imports (the official PBR bake and the shared
//    AICore bake alike) stores its vertices rotated by the writers'
//    (x, y, z) -> (x, z, -y) glTF Y-up action, while the DB world and the
//    splat clouds are Z-up. Without the inverse rotation the mesh lands 90°
//    away from its own splats.
// 2. The official PBR GLB additionally carries the DECODER-frame mesh
//    (AICore bakes before the scene pose), so the official make_scene pose
//    is applied on top: p' = R^T @ diag(scale) @ p + t (scene_assemble.cpp
//    pytorch3d row-vector action — the naive R reading is wrong by exactly
//    R vs R^T). CPU-baked GLBs arrive with posed vertices and carry no pose.
void applyGlbPlacement(ccHObject* glbEntity, const Sam3dRunResult& object) {
    if (!glbEntity) return;
    // Column-major (x, y, z) -> (x, -z, y): the exact inverse of the GLB
    // writers' Y-up rotation.
    float mat16[16] = {1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1};
    if (object.hasPbr && object.pose.size() == 10) {
        const float* q = object.pose.constData();  // wxyz(4)+t(3)+s(3)
        const float norm2 =
                q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if (norm2 > 0.0f) {
            const float two_s = 2.0f / norm2;
            // Row-major quaternion_to_matrix (pytorch3d convention) — the
            // same matrix the worker and the official composer use.
            const float R[3][3] = {
                    {1.0f - two_s * (q[2] * q[2] + q[3] * q[3]),
                     two_s * (q[1] * q[2] - q[3] * q[0]),
                     two_s * (q[1] * q[3] + q[2] * q[0])},
                    {two_s * (q[1] * q[2] + q[3] * q[0]),
                     1.0f - two_s * (q[1] * q[1] + q[3] * q[3]),
                     two_s * (q[2] * q[3] - q[1] * q[0])},
                    {two_s * (q[1] * q[3] - q[2] * q[0]),
                     two_s * (q[2] * q[3] + q[1] * q[0]),
                     1.0f - two_s * (q[1] * q[1] + q[2] * q[2])}};
            const float* s = q + 7;  // uniform scale (official receipt)
            const float* t = q + 4;  // translation
            // Compose M = Pose · YupInv in place: the YupInv columns select
            // (x, -z, y), so the combined 3x3 is
            //   col0 = R[0][:]·s[0]
            //   col1 = R[2][:]·s[2]
            //   col2 = -R[1][:]·s[1]
            for (int row = 0; row < 3; ++row) {
                mat16[row] = R[0][row] * s[0];
                mat16[4 + row] = R[2][row] * s[2];
                mat16[8 + row] = -R[1][row] * s[1];
                mat16[12 + row] = t[row];
            }
        }
    }
    ccGLMatrix transform(mat16);
    glbEntity->setGLTransformation(transform);
    glbEntity->applyGLTransformation_recursive();
}

}  // namespace

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
        // Two-step scene workflow, step 1: the worker sees previewOnly=true
        // in the settings and stops after the segmentation preview.
        connect(m_dialog, &Sam3dDialog::previewMasksRequested, this,
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
    // Fresh incremental-delivery state for this run.
    m_sceneGroup = nullptr;
    m_meshesGroup = nullptr;
    m_splatsGroup = nullptr;
    m_sceneGroupName.clear();
    m_sceneMeshCount = 0;
    m_sceneCloudCount = 0;
    m_sceneObjectTotal = 0;
    if (settings.previewOnly) {
        // Two-step workflow, step 1: make the mode unmistakable in both the
        // status line and the console (a preview run must never read as the
        // full reconstruction pipeline).
        m_dialog->setRunning(
                true, tr("Running the SAM3 segmentation preview only..."));
        m_dialog->appendLog(tr("[SAM3D] Starting segmentation preview from "
                               "'%1' (no reconstruction)...")
                                    .arg(settings.imagePath));
    } else {
        m_dialog->setRunning(true);
        m_dialog->appendLog(tr("[SAM3D] Starting generation from '%1'...")
                                    .arg(settings.imagePath));
    }

    m_worker = new Sam3dWorker(this);
    m_worker->configure(settings);
    // 1-arg signal -> default-level console log (Qt 5 PMF connect requires
    // slot arity <= signal arity, so route through a lambda).
    connect(m_worker, &Sam3dWorker::logMessage, m_dialog,
            [this](const QString& message) { m_dialog->appendLog(message); });
    // Progress: the worker maps pipeline stages / diffusion steps / scene
    // object slices onto a single 0..100 percent.
    connect(m_worker, &Sam3dWorker::progressChanged, m_dialog,
            &Sam3dDialog::setProgress);
    connect(m_worker, &Sam3dWorker::resultReady, this, &qSAM3D::onResultReady);
    // Incremental scene delivery: each finished object lands in the DB
    // tree while the remaining objects are still generating.
    connect(m_worker, &Sam3dWorker::objectReady, this, &qSAM3D::onObjectReady);
    connect(m_worker, &Sam3dWorker::sceneObjectTotalChanged, this,
            [this](int total) { m_sceneObjectTotal = total; });
    // Segmentation diagnostic: the masks-over-image preview lands in the DB
    // tree before the first object finishes, so recall gaps are visible
    // while the run is still going.
    connect(m_worker, &Sam3dWorker::segmentationPreviewReady, this,
            &qSAM3D::onSegmentationPreview);
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

void qSAM3D::onResultReady(const Sam3dRunResult& result) {
    addResultToDb(result);
}

void qSAM3D::onSceneReady(const Sam3dSceneResult& result) {
    addSceneResultToDb(result);
}

void qSAM3D::onObjectReady(const Sam3dRunResult& object) {
    addObjectToDb(object);
}

ccHObject* qSAM3D::sceneGroupRef(const Sam3dRunResult& object) {
    if (!m_lastSettings.importToDb) return nullptr;
    if (m_sceneGroup) return m_sceneGroup;
    // The per-run group name is computed ONCE and the group entity is kept
    // for the whole run: repeated ecvPluginDbNaming::makeUnique calls would
    // suffix the name once the group is in the DB.
    const QString sourceName = QFileInfo(object.sourceImage).completeBaseName();
    m_sceneGroupName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("SAM3D_%1_%2")
                    .arg(sourceName, ecvPluginDbNaming::deviceTagFromName(
                                             object.backend.isEmpty()
                                                     ? QStringLiteral("auto")
                                                     : object.backend)),
            m_app);
    m_sceneGroup = new ccHObject(m_sceneGroupName);
    m_sceneGroup->setDisplay(m_app->getActiveGLDisplay());
    m_sceneGroup->setVisible(true);
    m_app->addToDB(m_sceneGroup, false, true, false, true);
    m_sceneSourceImage = object.sourceImage;
    m_sceneBackend = object.backend;
    m_sceneDtype = object.dtype;
    return m_sceneGroup;
}

void qSAM3D::addObjectToDb(const Sam3dRunResult& object) {
    if (!m_app) return;
    ccHObject* root = sceneGroupRef(object);
    const QString entityName =
            m_sceneGroupName.isEmpty()
                    ? object.objectLabel
                    : QStringLiteral("%1.%2").arg(m_sceneGroupName,
                                                  object.objectLabel);

    // Two dedicated sub-groups: the textured meshes and the splat point
    // clouds stay in separate folders inside the run group.
    auto subGroupRef = [this, root](const QString& suffix,
                                    ccHObject** group) -> ccHObject* {
        if (*group != nullptr) return *group;
        *group = new ccHObject(
                QStringLiteral("%1.%2").arg(m_sceneGroupName, suffix));
        (*group)->setDisplay(m_app->getActiveGLDisplay());
        (*group)->setVisible(true);
        if (root) {
            root->addChild(*group);
            // The DB tree model only learns about entities through addToDB
            // (ccDBRoot::addElement inserts the row under the recorded
            // parent); an addChild alone leaves the subtree invisible.
            m_app->addToDB(*group, false, true, false, false);
        }
        return *group;
    };

    // One reconstruction product per object: its own textured GLB mesh plus
    // its own colored splat point cloud. The splat centers and the CPU-bake
    // mesh vertices are already in the scene frame (the worker applied the
    // official make_scene position action); the official PBR GLB carries the
    // DECODER-frame mesh (AICore bakes before the pose), so the official
    // pose is applied here as one GL transformation instead (see
    // applyGlbPlacement for the full frame-contract, including the glTF
    // Y-up correction).
    if (!object.glbPath.isEmpty() && QFileInfo::exists(object.glbPath)) {
        if (ccHObject* glbEntity =
                    importGlbFile(object.glbPath, object, entityName)) {
            applyGlbPlacement(glbEntity, object);
            ++m_sceneMeshCount;
            glbEntity->setDisplay(m_app->getActiveGLDisplay());
            glbEntity->setVisible(true);
            if (ccHObject* parent =
                        subGroupRef(QStringLiteral("meshes"), &m_meshesGroup)) {
                parent->addChild(glbEntity);
                m_app->addToDB(glbEntity, false, true, false, true);
            } else {
                m_app->addToDB(glbEntity, false, true, false, true);
            }
        }
    } else if (!object.glb.isEmpty()) {
        // Defensive: a bake that kept its bytes (no output dir).
        if (ccHObject* glbEntity =
                    importGlbEntity(object.glb, object, entityName)) {
            applyGlbPlacement(glbEntity, object);
            ++m_sceneMeshCount;
            glbEntity->setDisplay(m_app->getActiveGLDisplay());
            glbEntity->setVisible(true);
            if (ccHObject* parent =
                        subGroupRef(QStringLiteral("meshes"), &m_meshesGroup)) {
                parent->addChild(glbEntity);
                m_app->addToDB(glbEntity, false, true, false, true);
            } else {
                m_app->addToDB(glbEntity, false, true, false, true);
            }
        }
    }

    const int n = object.gaussianCount;
    const bool haveCloud =
            n > 0 &&
            object.splatCenters.size() == static_cast<qsizetype>(n) * 3 &&
            object.splatSh0.size() == object.splatCenters.size();
    if (!haveCloud) return;
    auto* cloud = new ccPointCloud(QStringLiteral("%1.splats").arg(entityName));
    if (!cloud->reserve(n)) {
        delete cloud;
        return;
    }
    const float* centers = object.splatCenters.constData();
    const float* sh0 = object.splatSh0.constData();
    constexpr float kSh0 = 0.28209479177387814f;
    for (int i = 0; i < n; ++i) {
        cloud->addPoint(CCVector3(centers[i * 3 + 0], centers[i * 3 + 1],
                                  centers[i * 3 + 2]));
    }
    if (cloud->resizeTheRGBTable()) {
        for (int i = 0; i < n; ++i) {
            const float rgb[3] = {0.5f + kSh0 * sh0[i * 3 + 0],
                                  0.5f + kSh0 * sh0[i * 3 + 1],
                                  0.5f + kSh0 * sh0[i * 3 + 2]};
            auto clamp = [](float v) {
                return static_cast<ColorCompType>(
                        std::min(1.0f, std::max(0.0f, v)) * 255.0f);
            };
            cloud->setPointColor(i, ecvColor::Rgb(clamp(rgb[0]), clamp(rgb[1]),
                                                  clamp(rgb[2])));
        }
        cloud->showColors(true);
    }
    cloud->setMetaData(QStringLiteral("Source"), object.sourceImage);
    cloud->setMetaData(QStringLiteral("Backend"), object.backend);
    cloud->setMetaData(QStringLiteral("Quantization"), object.dtype);
    cloud->setMetaData(QStringLiteral("Object"), object.objectLabel);
    ++m_sceneCloudCount;
    cloud->setDisplay(m_app->getActiveGLDisplay());
    cloud->setVisible(true);
    if (ccHObject* parent =
                subGroupRef(QStringLiteral("splats"), &m_splatsGroup)) {
        parent->addChild(cloud);
        m_app->addToDB(cloud, false, true, false, true);
    } else {
        m_app->addToDB(cloud, false, true, false, true);
    }
    m_dialog->appendLog(tr("[SAM3D] object '%1' imported (%2/%3).")
                                .arg(object.objectLabel)
                                .arg(m_sceneMeshCount + m_sceneCloudCount)
                                .arg(m_sceneObjectTotal * 2));
}

void qSAM3D::addSceneResultToDb(const Sam3dSceneResult& result) {
    // Run-completion summary: the entities were already imported one by one
    // through onObjectReady — nothing to batch here anymore.
    if (!m_lastSettings.importToDb) {
        m_dialog->appendLog(tr("[SAM3D] Scene kept out of the DB (import "
                               "disabled); %1 GLB file(s) remain in the "
                               "output directory.")
                                    .arg(result.objects.size()));
        return;
    }
    const int expected = static_cast<int>(result.objects.size());
    if (m_sceneMeshCount < expected) {
        m_dialog->appendLog(
                tr("[SAM3D] %1 object GLB bake(s) did not import — those "
                   "objects keep their splat point cloud only.")
                        .arg(expected - m_sceneMeshCount),
                ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
    }
    m_dialog->appendLog(
            tr("[SAM3D] Scene '%1' complete: %2 object mesh(es) + %3 splat "
               "cloud(s) imported.")
                    .arg(m_sceneGroupName)
                    .arg(m_sceneMeshCount)
                    .arg(m_sceneCloudCount));
    // The next run starts a fresh group.
    m_sceneGroup = nullptr;
    m_sceneGroupName.clear();
}

ccHObject* qSAM3D::importGlbFile(const QString& glbPath,
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
    imported->setMetaData(QStringLiteral("Scene"), object.sourceImage);
    return imported;
}

void qSAM3D::onSegmentationPreview(const QImage& preview, int maskCount) {
    if (!m_app || preview.isNull()) return;
    if (!m_lastSettings.importToDb) {
        m_dialog->appendLog(
                tr("[SAM3D] Segmentation preview kept out of the "
                   "DB (import disabled)."));
        return;
    }
    const QString sourceName =
            QFileInfo(m_lastSettings.imagePath).completeBaseName();
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("SAM3D_seg_%1").arg(sourceName), m_app);
    auto* img = new ccImage(preview, name);
    img->setMetaData(QStringLiteral("Source"), m_lastSettings.imagePath);
    img->setMetaData(QStringLiteral("Role"),
                     QStringLiteral("scene segmentation preview"));
    img->setMetaData(QStringLiteral("MaskCount"),
                     static_cast<qlonglong>(maskCount));
    img->setDisplay(m_app->getActiveGLDisplay());
    img->setVisible(true);
    m_app->addToDB(img, false, true, false, true);
    m_dialog->appendLog(
            tr("[SAM3D] Segmentation preview '%1' imported (%2 object "
               "mask(s)).")
                    .arg(name)
                    .arg(maskCount));
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
            m_dialog->setProgress(100);
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
        // Single-object runs carry no scene pose, but the GLB still needs
        // its glTF Y-up vertices rotated back into the DB's Z-up world (the
        // same frame the splat cloud below uses) or the two never align.
        applyGlbPlacement(glbEntity, result);
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

    // GLB/PLY file-export results are reported once, by the worker, at the
    // moment of the write — no duplicate confirmation here.
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
