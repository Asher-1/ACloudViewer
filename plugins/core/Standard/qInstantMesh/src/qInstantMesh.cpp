// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qInstantMesh.h"

#include <ecvMainAppInterface.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>

#include <QAction>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMainWindow>
#include <algorithm>
#include <cmath>
#include <memory>

#include "InstantMeshBatchCommand.h"
#include "InstantMeshDialog.h"
#include "InstantMeshWorker.h"

namespace {
bool validResultEnvelope(const InstantMeshRunResult& result) {
    if (result.verts.isEmpty() || result.verts.size() % 3 ||
        result.tris.isEmpty() || result.tris.size() % 3 ||
        result.vcolors.size() != result.verts.size())
        return false;
    for (float value : result.verts)
        if (!std::isfinite(value)) return false;
    for (float value : result.vcolors)
        if (!std::isfinite(value) || value < 0 || value > 1) return false;
    for (int index : result.tris)
        if (index < 0 || index >= result.verts.size() / 3) return false;
    if (result.texRgba.isEmpty() && result.uvs.isEmpty() && result.texW == 0 &&
        result.texH == 0)
        return true;
    if (result.texW <= 0 || result.texW > 4096 || result.texH <= 0 ||
        result.texH > 4096 ||
        static_cast<uint64_t>(result.texW) * result.texH * 4 !=
                static_cast<uint64_t>(result.texRgba.size()) ||
        static_cast<uint64_t>(result.uvs.size()) !=
                static_cast<uint64_t>(result.tris.size()) * 2)
        return false;
    for (float value : result.uvs)
        if (!std::isfinite(value) || value < 0 || value > 1) return false;
    return true;
}
}  // namespace

qInstantMesh::qInstantMesh(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qInstantMesh/info.json") {}

qInstantMesh::~qInstantMesh() {
    m_acceptResults = false;
    ++m_generation;
    if (m_worker) {
        m_worker->cancel();
        m_worker->wait();
        delete m_worker;
        m_worker = nullptr;
    }
}

void qInstantMesh::onNewSelection(
        const ccHObject::Container& selectedEntities) {
    Q_UNUSED(selectedEntities);
}

void qInstantMesh::registerCommands(ccCommandLineInterface* cmd) {
    if (cmd == nullptr) {
        return;
    }
    cmd->registerCommand(ccCommandLineInterface::Command::Shared(
            new CommandInstantMeshGenerate));
}

QList<QAction*> qInstantMesh::getActions() {
    if (!m_action) {
        m_action = new QAction(
                QIcon(":/CC/plugin/qInstantMesh/images/qInstantMesh.svg"),
                QStringLiteral("InstantMesh Image to 3D"), this);
        m_action->setToolTip(
                tr("Generate a textured 3D mesh from one image (Zero123++ "
                   "multi-view diffusion + FlexiCubes, native GGML)"));
        connect(m_action, &QAction::triggered, this,
                &qInstantMesh::onNewAction);
    }
    return {m_action};
}

void qInstantMesh::onNewAction() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new InstantMeshDialog(
                static_cast<QWidget*>(m_app->getMainWindow()));
        m_dialog->setWindowIcon(
                QIcon(":/CC/plugin/qInstantMesh/images/qInstantMesh.svg"));
        m_dialog->setAppInterface(m_app);
        connect(m_dialog, &InstantMeshDialog::runRequested, this,
                &qInstantMesh::onRunRequested);
        connect(m_dialog, &InstantMeshDialog::cancelRequested, this,
                &qInstantMesh::onCancelRequested);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qInstantMesh::onRunRequested(const InstantMeshDialog::Settings& settings) {
    if (m_worker) {
        if (m_dialog)
            m_dialog->appendLog(
                    tr("[InstantMesh] A generation is already running."));
        return;
    }
    startWorker(settings);
}

void qInstantMesh::onCancelRequested() {
    m_acceptResults = false;
    ++m_generation;
    if (m_worker) m_worker->cancel();
    if (m_dialog && m_dialog->isVisible())
        m_dialog->appendLog(
                tr("[InstantMesh] Cancellation requested; waiting for backend "
                   "work to finish safely."));
}

void qInstantMesh::startWorker(const InstantMeshDialog::Settings& settings) {
    InstantMeshWorker::Settings ws;
    ws.modelPaths = settings.modelPaths;
    ws.inputPath = settings.inputPath;
    ws.generation = ++m_generation;
    m_source = settings.inputPath;
    m_acceptResults = true;
    m_taskSucceeded = false;
    ws.presetName = settings.presetName;
    ws.steps = settings.steps;
    ws.seed = settings.seed;
    ws.guidance = settings.guidance;
    ws.gridRes = settings.gridRes;
    ws.gridScale = settings.gridScale;
    ws.textureRes = settings.textureRes;
    ws.bakeTexture = settings.bakeTexture;
    ws.threads = settings.threads;
    ws.device = settings.device;
    ws.useRmbg = settings.useRmbg;
    ws.rmbgModelPath = settings.rmbgModelPath;
    ws.quantization = settings.quantization;
    ws.addResultToDb = settings.addResultToDb;
    m_worker = new InstantMeshWorker(ws, this);
    connect(m_worker, &InstantMeshWorker::logMessage, this,
            [this](const QString& message) {
                if (m_acceptResults && m_dialog && m_dialog->isVisible())
                    m_dialog->appendLog(message);
            });
    connect(m_worker, &InstantMeshWorker::progressUpdate, this,
            [this](int stage, int step, int total) {
                static const char* kNames[] = {
                        "preprocess", "rmbg",   "multiview",
                        "vae_decode", "encode", "triplane",
                        "synth",      "mesh",   "texture"};
                if (m_acceptResults && m_dialog && m_dialog->isVisible()) {
                    m_dialog->setProgressStage(
                            stage,
                            tr("Stage: %1")
                                    .arg(QLatin1String(
                                            kNames[stage >= 0 && stage < 9
                                                           ? stage
                                                           : 0])),
                            step, total);
                }
            });
    connect(m_worker, &InstantMeshWorker::resultReady, this,
            &qInstantMesh::onResultReady);
    connect(m_worker, &InstantMeshWorker::taskFinished, this,
            &qInstantMesh::onTaskFinished);
    auto* worker = m_worker;
    connect(worker, &QThread::finished, this, [this, worker] {
        if (m_worker == worker) {
            m_worker = nullptr;
            if (m_dialog) {
                m_dialog->setRunning(false);
                if (m_acceptResults && m_dialog->isVisible())
                    m_dialog->appendLog(
                            m_taskSucceeded
                                    ? tr("[InstantMesh] Done.")
                                    : tr("[InstantMesh] Generation failed — "
                                         "see the log above."));
            }
        }
        worker->deleteLater();
    });
    m_dialog->setRunning(true);
    m_dialog->appendLog(tr("[InstantMesh] Generation started (%1, preset "
                           "\"%2\")...")
                                .arg(settings.device, settings.presetName));
    m_worker->start();
}

void qInstantMesh::onResultReady(const InstantMeshRunResult& result) {
    if (!m_dialog || !m_worker || !m_acceptResults || !m_dialog->isVisible() ||
        result.generation != m_generation || result.sourceImage != m_source)
        return;
    // UI, export and DB share the typed contract; validate before any consumer
    // mutates state.
    if (!validResultEnvelope(result)) {
        m_dialog->appendLog(
                tr("[InstantMesh] Rejected invalid mesh, color or texture "
                   "results."));
        return;
    }
    m_dialog->setLastResult(result);
    m_dialog->applyResultToStrip(result);

    // The worker consumed the original settings; the DB-destination flag
    // travels on the result envelope (the worker has no DB access).
    InstantMeshDialog::Settings settings;
    settings.device = result.backend;
    if (result.addResultToDb) {
        addResultToDb(result, settings);
    }
}

void qInstantMesh::onTaskFinished(bool success) {
    // Logical completion records status only; thread objects and UI completion
    // follow QThread::finished.
    m_taskSucceeded = success;
}

void qInstantMesh::addResultToDb(const InstantMeshRunResult& result,
                                 const InstantMeshDialog::Settings& settings) {
    if (!m_app || result.verts.isEmpty() || result.tris.isEmpty()) return;

    const QString sourceName = QFileInfo(result.sourceImage).completeBaseName();
    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.backend.isEmpty() ? settings.device : result.backend);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("InstantMesh_%1_%2").arg(sourceName, deviceTag),
            m_app);

    // DB directly consumes typed buffers; GLB is export-only, without temporary
    // files or import filters.
    ccMesh* mesh = buildResultMesh(result, name);
    if (!mesh) return;

    mesh->setMetaData(QStringLiteral("Source"), result.sourceImage);
    mesh->setMetaData(QStringLiteral("Preset"), result.presetName);
    mesh->setMetaData(QStringLiteral("Runtime (ms)"), result.totalRuntimeMs);
    mesh->setMetaData(QStringLiteral("Backend"), result.backend);
    mesh->setMetaData(QStringLiteral("Views"),
                      result.multiview ? QStringLiteral("zero123pp-diffusion")
                                       : QStringLiteral("replicated"));

    m_app->addToDB(mesh);
    m_app->refreshAll();
    m_app->updateUI();
    if (m_dialog) {
        m_dialog->appendLog(
                mesh->hasTextures()
                        ? tr("[InstantMesh] Added UV-textured mesh '%1' to DB.")
                                  .arg(name)
                        : tr("[InstantMesh] Added vertex-colored mesh '%1' to "
                             "DB.")
                                  .arg(name));
    }
}

ccMesh* qInstantMesh::buildResultMesh(const InstantMeshRunResult& result,
                                      const QString& name) {
    if (!validResultEnvelope(result)) return nullptr;
    std::unique_ptr<ccMesh> mesh(buildVertexColorMesh(
            result.verts, result.vcolors, result.tris, name));
    if (!mesh) return nullptr;
    if (result.texRgba.isEmpty()) return mesh.release();

    // QImage must own a pixel copy, not borrow worker-envelope storage.
    const QImage image =
            QImage(result.texRgba.constData(), result.texW, result.texH,
                   result.texW * 4, QImage::Format_RGBA8888)
                    .copy();
    if (image.isNull() || !mesh->reservePerTriangleTexCoordIndexes() ||
        !mesh->reservePerTriangleMtlIndexes())
        return nullptr;
    auto coords = std::make_unique<TextureCoordsContainer>();
    if (!coords->reserveSafe(result.uvs.size() / 2)) return nullptr;
    for (int i = 0; i < result.uvs.size(); i += 2)
        coords->emplace_back(TexCoords2D(result.uvs[i], result.uvs[i + 1]));
    mesh->setTexCoordinatesTable(coords.release());
    for (unsigned i = 0; i < mesh->size(); ++i) {
        mesh->addTriangleTexCoordIndexes(3 * i, 3 * i + 1, 3 * i + 2);
        mesh->addTriangleMtlIndex(0);
    }
    auto materials = std::make_unique<ccMaterialSet>(
            name + QStringLiteral("_materials"));
    ccMaterial::Shared material(
            new ccMaterial(QStringLiteral("InstantMesh atlas")));
    material->setDiffuse(ecvColor::Rgbaf(1, 1, 1, 1));
    material->setTexture(image,
                         QStringLiteral("InstantMesh_%1.png")
                                 .arg(material->getUniqueIdentifier()),
                         false);
    // The cached owned image supplies the diffuse map without a file transport.
    if (!material->loadAndSetTextureMap(ccMaterial::TextureMapType::DIFFUSE,
                                        material->getTextureFilename()))
        return nullptr;
    materials->addMaterial(material);
    mesh->setMaterialSet(materials.release());
    mesh->showMaterials(true);
    mesh->showColors(false);
    mesh->setMetaData(QStringLiteral("Material"),
                      QStringLiteral("Baked UV texture"));
    return mesh.release();
}

ccMesh* qInstantMesh::buildVertexColorMesh(const QVector<float>& verts,
                                           const QVector<float>& vcolors,
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
    if (vcolors.size() == nv * 3 && cloud->resizeTheRGBTable()) {
        for (int i = 0; i < nv; ++i) {
            const float* c = vcolors.constData() + i * 3;
            cloud->setPointColor(
                    i,
                    ecvColor::Rgb(static_cast<ColorCompType>(
                                          std::min(1.0f, std::max(0.0f, c[0])) *
                                          255.0f),
                                  static_cast<ColorCompType>(
                                          std::min(1.0f, std::max(0.0f, c[1])) *
                                          255.0f),
                                  static_cast<ColorCompType>(
                                          std::min(1.0f, std::max(0.0f, c[2])) *
                                          255.0f)));
        }
        cloud->showColors(true);
    }

    auto* mesh = new ccMesh(cloud);
    const int nt = tris.size() / 3;
    if (!mesh->reserve(nt)) {
        delete mesh;
        delete cloud;
        return nullptr;
    }
    for (int i = 0; i < nt; ++i) {
        mesh->addTriangle(tris[i * 3], tris[i * 3 + 1], tris[i * 3 + 2]);
    }
    mesh->addChild(cloud);
    cloud->setEnabled(false);
    mesh->showColors(true);
    return mesh;
}
