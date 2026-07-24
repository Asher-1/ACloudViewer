// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qSIBR.h"

#include <ecvCommandLineInterface.h>
#include <ecvHObjectCaster.h>
#include <ecvMainAppInterface.h>

#include <QAction>
#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QMainWindow>
#include <QMenu>
#include <QMessageBox>
#include <QSettings>

#include "SIBROptionsDialog.h"
#include "SIBRViewerThread.h"
#include "qSIBRCommands.h"

qSIBR::qSIBR(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qSIBR/info.json") {}

qSIBR::~qSIBR() {
    for (auto* viewer : m_activeViewers) {
        viewer->requestStop();
        viewer->wait(3000);
        delete viewer;
    }
    m_activeViewers.clear();
}

void qSIBR::registerCommands(ccCommandLineInterface* cmd) {
    if (!cmd) {
        assert(false);
        return;
    }
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandSIBRTool));
    cmd->registerCommand(
            ccCommandLineInterface::Command::Shared(new CommandSIBRViewer));
}

void qSIBR::onNewSelection(const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;

    bool hasEntity = !selectedEntities.empty();
    if (m_actionQuickView) m_actionQuickView->setEnabled(hasEntity);
}

QList<QAction*> qSIBR::getActions() {
    QList<QAction*> group;

    auto loadIcon = [](const QString& name) {
        return QIcon(QString(":/CC/plugin/qSIBR/images/%1").arg(name));
    };

    if (!m_actionULR) {
        m_actionULR = new QAction(tr("ULR Viewer"), this);
        m_actionULR->setToolTip(
                tr("ULR (Unstructured Lumigraph Rendering) - "
                   "full IBR rendering pipeline"));
        m_actionULR->setIcon(loadIcon("icon_ulr.svg"));
        connect(m_actionULR, &QAction::triggered, this,
                &qSIBR::launchULRViewer);
    }
    group.append(m_actionULR);

    if (!m_actionULRv2) {
        m_actionULRv2 = new QAction(tr("ULR v2/v3 Viewer"), this);
        m_actionULRv2->setToolTip(
                tr("ULR v2/v3 with texture arrays, masks and Poisson"));
        m_actionULRv2->setIcon(loadIcon("icon_ulrv2.svg"));
        connect(m_actionULRv2, &QAction::triggered, this,
                &qSIBR::launchULRv2Viewer);
    }
    group.append(m_actionULRv2);

    if (!m_actionTexturedMesh) {
        m_actionTexturedMesh = new QAction(tr("Textured Mesh Viewer"), this);
        m_actionTexturedMesh->setToolTip(
                tr("View textured meshes with scene debug"));
        m_actionTexturedMesh->setIcon(loadIcon("icon_textured_mesh.svg"));
        connect(m_actionTexturedMesh, &QAction::triggered, this,
                &qSIBR::launchTexturedMeshViewer);
    }
    group.append(m_actionTexturedMesh);

    if (!m_actionPointBased) {
        m_actionPointBased = new QAction(tr("Point-Based Viewer"), this);
        m_actionPointBased->setToolTip(
                tr("View point clouds with point-based rendering"));
        m_actionPointBased->setIcon(loadIcon("icon_point_based.svg"));
        connect(m_actionPointBased, &QAction::triggered, this,
                &qSIBR::launchPointBasedViewer);
    }
    group.append(m_actionPointBased);

#ifdef SIBR_HAS_CUDA
    if (!m_actionGaussian) {
        m_actionGaussian =
                new QAction(tr("3D Gaussian Splatting Viewer"), this);
        m_actionGaussian->setToolTip(
                tr("Real-time 3D Gaussian Splatting viewer (CUDA)"));
        m_actionGaussian->setIcon(loadIcon("icon_gaussian.svg"));
        connect(m_actionGaussian, &QAction::triggered, this,
                &qSIBR::launchGaussianViewer);
    }
    group.append(m_actionGaussian);
#endif

#ifdef SIBR_HAS_REMOTE
    if (!m_actionRemoteGaussian) {
        m_actionRemoteGaussian =
                new QAction(tr("Remote Gaussian Viewer"), this);
        m_actionRemoteGaussian->setToolTip(
                tr("Connect to remote Gaussian Splatting training server"));
        m_actionRemoteGaussian->setIcon(loadIcon("icon_remote.svg"));
        connect(m_actionRemoteGaussian, &QAction::triggered, this,
                &qSIBR::launchRemoteGaussianViewer);
    }
    group.append(m_actionRemoteGaussian);
#endif

    if (!m_actionQuickView) {
        m_actionQuickView = new QAction(tr("Quick View in SIBR"), this);
        m_actionQuickView->setToolTip(
                tr("Auto-detect and open selected entity in the "
                   "best-matching SIBR viewer"));
        m_actionQuickView->setIcon(loadIcon("icon.svg"));
        m_actionQuickView->setEnabled(false);
        connect(m_actionQuickView, &QAction::triggered, this,
                &qSIBR::launchQuickView);
    }
    group.append(m_actionQuickView);

    if (!m_actionDatasetTools) {
        m_actionDatasetTools = new QAction(tr("Dataset Tools"), this);
        m_actionDatasetTools->setToolTip(tr("SIBR dataset preprocessing"));
        m_actionDatasetTools->setIcon(loadIcon("icon_tools.svg"));

        m_datasetToolsMenu = new QMenu();

        auto addTool = [this](const QString& name, const QString& tooltip,
                              void (qSIBR::*slot)()) {
            QAction* action = m_datasetToolsMenu->addAction(name);
            action->setToolTip(tooltip);
            connect(action, &QAction::triggered, this, slot);
        };

        addTool(tr("Prepare COLMAP for SIBR"),
                tr("Prepare COLMAP reconstruction"),
                &qSIBR::launchPrepareColmap4Sibr);
        addTool(tr("Tonemapper"), tr("Tonemap HDR images"),
                &qSIBR::launchTonemapper);
        addTool(tr("Unwrap Mesh"), tr("Unwrap mesh UVs"),
                &qSIBR::launchUnwrapMesh);
        addTool(tr("Texture Mesh"), tr("Apply textures to mesh"),
                &qSIBR::launchTextureMesh);
        addTool(tr("Clipping Planes"), tr("Set clipping planes"),
                &qSIBR::launchClippingPlanes);
        addTool(tr("Crop From Center"), tr("Crop images from center"),
                &qSIBR::launchCropFromCenter);
        addTool(tr("NVM to SIBR"), tr("Convert NVM to SIBR"),
                &qSIBR::launchNvmToSIBR);
        addTool(tr("Distortion Crop"), tr("Crop distorted regions"),
                &qSIBR::launchDistordCrop);
        addTool(tr("Camera Converter"), tr("Convert camera formats"),
                &qSIBR::launchCameraConverter);
        addTool(tr("Align Meshes"), tr("Align meshes"),
                &qSIBR::launchAlignMeshes);

        m_actionDatasetTools->setMenu(m_datasetToolsMenu);
    }
    group.append(m_actionDatasetTools);

    return group;
}

void qSIBR::launchInMemoryGaussianViewer(QByteArray plyBytes,
                                         QByteArray camerasJson,
                                         int shDegree) {
#ifdef SIBR_HAS_CUDA
    if (plyBytes.isEmpty() || camerasJson.isEmpty()) {
        if (m_app) {
            m_app->dispToConsole(
                    tr("[SIBR] In-memory Gaussian payload is empty."),
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        }
        return;
    }

    auto* thread = new SIBRViewerThread(
            SIBRViewerThread::ViewerMode::GaussianSplatting, QString(), 1920,
            1080, this);
    thread->setGaussianLargePointView(true);
    thread->setGaussianAutoTour(true);
    thread->setGaussianShDegree(shDegree > 0 ? shDegree : 3);
    thread->setGaussianWhiteBackground(false);
    thread->setGaussianCamerasJson(camerasJson.toStdString());
    std::vector<uint8_t> plyMemory(
            reinterpret_cast<const uint8_t*>(plyBytes.constData()),
            reinterpret_cast<const uint8_t*>(plyBytes.constData()) +
                    static_cast<size_t>(plyBytes.size()));
    thread->setGaussianPlyMemory(std::move(plyMemory));
    thread->setObjectName("GS_MEM_" +
                          QString::number(QDateTime::currentMSecsSinceEpoch()));
    launchViewer(thread);
#else
    (void)plyBytes;
    (void)camerasJson;
    (void)shDegree;
    if (m_app) {
        m_app->dispToConsole(
                tr("[SIBR] CUDA not available - Gaussian viewer disabled"),
                ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
    }
#endif
}

void qSIBR::launchViewer(SIBRViewerThread* thread) {
    // GLFW and Input::global() are process-wide singletons that are
    // not thread-safe.  Only one SIBR viewer may be active at a time.
    if (SIBRViewerThread::hasActiveViewer()) {
        if (m_app)
            m_app->dispToConsole(tr("[SIBR] Another viewer is still running. "
                                    "Please close it first."),
                                 ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
        delete thread;
        return;
    }

    connect(thread, &SIBRViewerThread::viewerStarted, this,
            &qSIBR::onViewerStarted);
    connect(thread, &SIBRViewerThread::viewerFinished, this,
            &qSIBR::onViewerFinished);
    connect(thread, &SIBRViewerThread::viewerError, this,
            &qSIBR::onViewerError);
    connect(thread, &SIBRViewerThread::viewerLog, this, &qSIBR::onViewerLog);
    connect(thread, &SIBRViewerThread::viewerResultReady, this,
            &qSIBR::onViewerResultReady);

    QString key = thread->objectName();
    if (m_activeViewers.contains(key)) {
        m_activeViewers[key]->requestStop();
        m_activeViewers[key]->wait(2000);
        delete m_activeViewers[key];
    }
    m_activeViewers[key] = thread;
    thread->start();
}

void qSIBR::onViewerStarted(const QString& modeName) {
    if (m_app) {
        m_app->dispToConsole(tr("[SIBR] %1 started").arg(modeName),
                             ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }
}

void qSIBR::onViewerFinished(const QString& modeName, int exitCode) {
    if (m_app) {
        auto level = (exitCode == 0) ? ecvMainAppInterface::STD_CONSOLE_MESSAGE
                                     : ecvMainAppInterface::WRN_CONSOLE_MESSAGE;
        m_app->dispToConsole(
                tr("[SIBR] %1 finished (exit=%2)").arg(modeName).arg(exitCode),
                level);
    }
    for (auto it = m_activeViewers.begin(); it != m_activeViewers.end(); ++it) {
        if (it.value()->isFinished()) {
            it.value()->deleteLater();
            m_activeViewers.erase(it);
            break;
        }
    }
}

void qSIBR::onViewerError(const QString& error) {
    if (m_app) {
        m_app->dispToConsole(error, ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }
}

void qSIBR::onViewerLog(const QString& message) {
    if (m_app) {
        m_app->dispToConsole(message, ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }
}

void qSIBR::onViewerResultReady(const QString& resultPath,
                                const QString& description) {
    if (!m_app) return;

    QFileInfo fi(resultPath);
    if (!fi.exists() || !fi.isFile()) {
        m_app->dispToConsole(
                tr("[SIBR] Result file not found: %1").arg(resultPath),
                ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
        return;
    }

    auto reply = QMessageBox::question(
            m_app->getMainWindow(), tr("Import SIBR Result"),
            tr("Import \"%1\" into ACloudViewer?\n\nFile: %2")
                    .arg(description)
                    .arg(fi.fileName()),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);

    if (reply != QMessageBox::Yes) return;

    ccHObject* loaded = m_app->loadFile(resultPath, true);
    if (loaded) {
        loaded->setFullPath(resultPath);
        m_app->addToDB(loaded,
                       /*updateZoom=*/true,
                       /*autoExpandDBTree=*/true,
                       /*checkDimensions=*/true,
                       /*autoRedraw=*/true);
        m_app->setSelectedInDB(loaded, true);
        m_app->dispToConsole(tr("[SIBR] Imported: %1").arg(fi.fileName()),
                             ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    } else {
        m_app->dispToConsole(
                tr("[SIBR] Failed to import: %1").arg(fi.fileName()),
                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }
}

void qSIBR::launchSimpleViewer(int viewerMode,
                               int dialogType,
                               const QString& prefix) {
    auto mode = static_cast<SIBRViewerThread::ViewerMode>(viewerMode);
    auto dlgType = static_cast<SIBROptionsDialog::ViewerType>(dialogType);
    SIBROptionsDialog dlg(dlgType, m_app ? m_app->getMainWindow() : nullptr);
    if (!m_selectedEntities.empty()) {
        ccHObject* e = m_selectedEntities.front();
        QString src = detectEntitySourcePath(e);
        if (!src.isEmpty())
            dlg.setInitialPaths(QFileInfo(src).absolutePath(), QString());
        dlg.setSelectedEntityInfo(e->getName(),
                                  ccHObjectCaster::ToGenericMesh(e)
                                          ? tr("mesh")
                                          : tr("point cloud"));
    }
    if (dlg.exec() != QDialog::Accepted) return;
    auto opts = dlg.getOptions();
    if (opts.datasetPath.isEmpty()) return;

    auto* thread = new SIBRViewerThread(mode, opts.datasetPath, opts.width,
                                        opts.height, this);
    thread->setObjectName(prefix + "_" +
                          QString::number(QDateTime::currentMSecsSinceEpoch()));
    launchViewer(thread);
}

void qSIBR::launchULRViewer() {
    launchSimpleViewer(static_cast<int>(SIBRViewerThread::ViewerMode::ULR),
                       static_cast<int>(SIBROptionsDialog::ULR), "ULR");
}

void qSIBR::launchULRv2Viewer() {
    launchSimpleViewer(static_cast<int>(SIBRViewerThread::ViewerMode::ULRv2),
                       static_cast<int>(SIBROptionsDialog::ULRv2), "ULRv2");
}

void qSIBR::launchTexturedMeshViewer() {
    launchSimpleViewer(
            static_cast<int>(SIBRViewerThread::ViewerMode::TexturedMesh),
            static_cast<int>(SIBROptionsDialog::TexturedMesh), "TM");
}

void qSIBR::launchPointBasedViewer() {
    launchSimpleViewer(
            static_cast<int>(SIBRViewerThread::ViewerMode::PointBased),
            static_cast<int>(SIBROptionsDialog::PointBased), "PB");
}

void qSIBR::launchGaussianViewer() {
#ifdef SIBR_HAS_CUDA
    SIBROptionsDialog dlg(SIBROptionsDialog::GaussianSplatting,
                          m_app ? m_app->getMainWindow() : nullptr);
    if (!m_selectedEntities.empty()) {
        ccHObject* e = m_selectedEntities.front();
        dlg.setSelectedEntityInfo(e->getName(),
                                  ccHObjectCaster::ToGenericMesh(e)
                                          ? tr("mesh")
                                          : tr("point cloud"));
        QString src = detectEntitySourcePath(e);
        if (!src.isEmpty() && looksLikeGaussianModelDir(src)) {
            QFileInfo fi(src);
            QString modelDir = fi.isDir() ? src : fi.absolutePath();
            for (int i = 0; i < 3; ++i) {
                if (QDir(modelDir).exists("cfg_args") ||
                    QDir(modelDir).exists("point_cloud"))
                    break;
                QDir d(modelDir);
                if (!d.cdUp()) break;
                modelDir = d.absolutePath();
            }
            dlg.setInitialPaths(QString(), modelDir);
        } else if (!src.isEmpty()) {
            dlg.setInitialPaths(QFileInfo(src).absolutePath(), QString());
        }
    }
    if (dlg.exec() != QDialog::Accepted) return;
    auto opts = dlg.getOptions();
    if (opts.modelPath.isEmpty()) {
        if (m_app)
            m_app->dispToConsole(
                    tr("[SIBR] Model path is required for Gaussian viewer"),
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return;
    }

    auto* thread = new SIBRViewerThread(
            SIBRViewerThread::ViewerMode::GaussianSplatting, opts.datasetPath,
            opts.width, opts.height, this);
    thread->setModelPath(opts.modelPath);
    thread->setIteration(opts.iteration);
    thread->setCudaDevice(opts.device);
    thread->setNoInterop(opts.noInterop);
    thread->setObjectName("GS_" +
                          QString::number(QDateTime::currentMSecsSinceEpoch()));
    launchViewer(thread);
#else
    if (m_app) {
        m_app->dispToConsole(
                tr("[SIBR] CUDA not available - Gaussian viewer disabled"),
                ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
    }
#endif
}

void qSIBR::launchRemoteGaussianViewer() {
#ifdef SIBR_HAS_REMOTE
    SIBROptionsDialog dlg(SIBROptionsDialog::RemoteGaussian,
                          m_app ? m_app->getMainWindow() : nullptr);
    if (!m_selectedEntities.empty()) {
        ccHObject* e = m_selectedEntities.front();
        QString src = detectEntitySourcePath(e);
        if (!src.isEmpty())
            dlg.setInitialPaths(QFileInfo(src).absolutePath(), QString());
        dlg.setSelectedEntityInfo(e->getName(),
                                  ccHObjectCaster::ToGenericMesh(e)
                                          ? tr("mesh")
                                          : tr("point cloud"));
    }
    if (dlg.exec() != QDialog::Accepted) return;
    auto opts = dlg.getOptions();

    auto* thread = new SIBRViewerThread(
            SIBRViewerThread::ViewerMode::RemoteGaussian, opts.datasetPath,
            opts.width, opts.height, this);
    thread->setRemoteAddress(opts.remoteIP, opts.remotePort);
    thread->setObjectName("RG_" +
                          QString::number(QDateTime::currentMSecsSinceEpoch()));
    launchViewer(thread);
#else
    if (m_app) {
        m_app->dispToConsole(
                tr("[SIBR] Remote viewer not available in this build"),
                ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
    }
#endif
}

void qSIBR::launchDatasetTool(const QString& toolName) {
    QString datasetPath = QFileDialog::getExistingDirectory(
            m_app ? m_app->getMainWindow() : nullptr,
            tr("Select Dataset Directory"), QString(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (datasetPath.isEmpty()) return;
    launchDatasetToolWithArgs(toolName, {"--path", datasetPath});
}

void qSIBR::launchDatasetToolWithArgs(const QString& toolName,
                                      const QStringList& args) {
    if (m_activeToolThread && m_activeToolThread->isRunning()) {
        if (m_app)
            m_app->dispToConsole(
                    tr("[SIBR] A dataset tool is already running. "
                       "Please wait for it to finish before launching "
                       "another."),
                    ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
        return;
    }

    auto* thread = new SIBRViewerThread(toolName, args, this);
    m_activeToolThread = thread;

    connect(thread, &SIBRViewerThread::viewerLog, [this](const QString& msg) {
        if (m_app)
            m_app->dispToConsole(msg, ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    });
    connect(thread, &SIBRViewerThread::viewerError, [this](const QString& msg) {
        if (m_app)
            m_app->dispToConsole(msg, ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    });
    connect(thread, &SIBRViewerThread::viewerFinished,
            [this](const QString& name, int exitCode) {
                if (m_app) {
                    auto level =
                            (exitCode == 0)
                                    ? ecvMainAppInterface::STD_CONSOLE_MESSAGE
                                    : ecvMainAppInterface::WRN_CONSOLE_MESSAGE;
                    m_app->dispToConsole(tr("[SIBR] %1 finished (exit=%2)")
                                                 .arg(name)
                                                 .arg(exitCode),
                                         level);
                }
            });
    connect(thread, &SIBRViewerThread::finished, this, [this, thread]() {
        if (m_activeToolThread == thread) m_activeToolThread = nullptr;
        thread->deleteLater();
    });

    if (m_app) {
        m_app->dispToConsole(tr("[SIBR] Launching %1 ...").arg(toolName),
                             ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }
    thread->start();
}

// ---------------------------------------------------------------------------
// Context-sensitive quick actions
// ---------------------------------------------------------------------------

void qSIBR::launchQuickView() {
    if (!m_app || m_selectedEntities.empty()) {
        if (m_app)
            m_app->dispToConsole(tr("[SIBR] No entity selected"),
                                 ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
        return;
    }

    ccHObject* entity = m_selectedEntities.front();
    QString sourcePath = detectEntitySourcePath(entity);
    bool isMesh = (ccHObjectCaster::ToGenericMesh(entity) != nullptr);

#ifdef SIBR_HAS_CUDA
    if (!sourcePath.isEmpty() && looksLikeGaussianModelDir(sourcePath)) {
        QFileInfo fi(sourcePath);
        QString modelDir = fi.isDir() ? sourcePath : fi.absolutePath();
        for (int i = 0; i < 3; ++i) {
            if (QDir(modelDir).exists("cfg_args") ||
                QDir(modelDir).exists("point_cloud"))
                break;
            QDir d(modelDir);
            if (!d.cdUp()) break;
            modelDir = d.absolutePath();
        }

        SIBROptionsDialog dlg(SIBROptionsDialog::GaussianSplatting,
                              m_app->getMainWindow());
        dlg.setInitialPaths(QString(), modelDir);
        if (dlg.exec() != QDialog::Accepted) return;
        auto opts = dlg.getOptions();
        if (opts.modelPath.isEmpty()) return;

        auto* thread = new SIBRViewerThread(
                SIBRViewerThread::ViewerMode::GaussianSplatting,
                opts.datasetPath, opts.width, opts.height, this);
        thread->setModelPath(opts.modelPath);
        thread->setIteration(opts.iteration);
        thread->setCudaDevice(opts.device);
        thread->setNoInterop(opts.noInterop);
        thread->setObjectName(
                "GS_" + QString::number(QDateTime::currentMSecsSinceEpoch()));
        launchViewer(thread);
        return;
    }
#endif

    if (isMesh) {
        SIBROptionsDialog dlg(SIBROptionsDialog::TexturedMesh,
                              m_app->getMainWindow());
        if (!sourcePath.isEmpty())
            dlg.setInitialPaths(QFileInfo(sourcePath).absolutePath(),
                                QString());
        if (dlg.exec() != QDialog::Accepted) return;
        auto opts = dlg.getOptions();
        if (opts.datasetPath.isEmpty()) return;

        auto* thread = new SIBRViewerThread(
                SIBRViewerThread::ViewerMode::TexturedMesh, opts.datasetPath,
                opts.width, opts.height, this);
        thread->setObjectName(
                "TM_" + QString::number(QDateTime::currentMSecsSinceEpoch()));
        launchViewer(thread);
    } else {
        SIBROptionsDialog dlg(SIBROptionsDialog::PointBased,
                              m_app->getMainWindow());
        if (!sourcePath.isEmpty())
            dlg.setInitialPaths(QFileInfo(sourcePath).absolutePath(),
                                QString());
        if (dlg.exec() != QDialog::Accepted) return;
        auto opts = dlg.getOptions();
        if (opts.datasetPath.isEmpty()) return;

        auto* thread = new SIBRViewerThread(
                SIBRViewerThread::ViewerMode::PointBased, opts.datasetPath,
                opts.width, opts.height, this);
        thread->setObjectName(
                "PB_" + QString::number(QDateTime::currentMSecsSinceEpoch()));
        launchViewer(thread);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

QString qSIBR::detectEntitySourcePath(ccHObject* entity) const {
    if (!entity) return {};

    // Prefer the full path stored by the I/O system at load time.
    auto checkFullPath = [](ccHObject* obj) -> QString {
        QString fp = obj->getFullPath();
        if (!fp.isEmpty()) {
            QFileInfo fi(fp);
            if (fi.exists()) return fi.absoluteFilePath();
        }
        return {};
    };

    QString result = checkFullPath(entity);
    if (!result.isEmpty()) return result;

    // Walk up the parent hierarchy (container name = file path).
    ccHObject* cur = entity->getParent();
    while (cur) {
        result = checkFullPath(cur);
        if (!result.isEmpty()) return result;

        QFileInfo fi(cur->getName());
        if (fi.exists() && fi.isFile()) return fi.absoluteFilePath();

        if (cur == m_app->dbRootObject()) break;
        cur = cur->getParent();
    }

    // Fallback: entity name may itself be a path.
    QFileInfo fi(entity->getName());
    if (fi.exists() && fi.isFile()) return fi.absoluteFilePath();

    return {};
}

bool qSIBR::looksLikeGaussianModelDir(const QString& path) {
    QFileInfo fi(path);
    QString dir = fi.isDir() ? path : fi.absolutePath();

    for (int i = 0; i < 4; ++i) {
        QDir d(dir);
        if (d.exists("cfg_args")) return true;
        if (d.exists("point_cloud") &&
            QDir(d.filePath("point_cloud"))
                            .entryList({"iteration_*"}, QDir::Dirs)
                            .size() > 0)
            return true;
        if (!d.cdUp()) break;
        dir = d.absolutePath();
    }
    return false;
}

// ---------------------------------------------------------------------------
// Dataset tool launchers
// ---------------------------------------------------------------------------

void qSIBR::launchPrepareColmap4Sibr() {
    launchDatasetTool("prepareColmap4Sibr");
}
void qSIBR::launchTonemapper() { launchDatasetTool("tonemapper"); }
void qSIBR::launchClippingPlanes() { launchDatasetTool("clippingPlanes"); }
void qSIBR::launchNvmToSIBR() { launchDatasetTool("nvmToSIBR"); }
void qSIBR::launchDistordCrop() { launchDatasetTool("distordCrop"); }

void qSIBR::launchTextureMesh() {
    auto* win = m_app ? m_app->getMainWindow() : nullptr;
    QString datasetPath = QFileDialog::getExistingDirectory(
            win, tr("Select Dataset Directory"), QString(),
            QFileDialog::ShowDirsOnly);
    if (datasetPath.isEmpty()) return;

    auto meshReply = QMessageBox::question(
            win, tr("Texture Mesh"),
            tr("Do you want to specify a mesh file?\n\n"
               "Click 'Yes' to select a mesh, or 'No' to let the tool\n"
               "auto-discover one in the dataset directory."),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);

    QStringList args = {"--path", datasetPath};
    if (meshReply == QMessageBox::Yes) {
        QString meshPath = QFileDialog::getOpenFileName(
                win, tr("Select Mesh File"), datasetPath,
                tr("Meshes (*.ply *.obj *.off)"));
        if (meshPath.isEmpty()) return;
        args << "--mesh" << meshPath;
    }

    QString outputPath = QFileDialog::getSaveFileName(
            win, tr("Output Texture Image"), datasetPath + "/texture.png",
            tr("Images (*.png *.jpg)"));
    if (outputPath.isEmpty()) return;
    args << "--output" << outputPath;
    launchDatasetToolWithArgs("textureMesh", args);
}

void qSIBR::launchUnwrapMesh() {
    auto* win = m_app ? m_app->getMainWindow() : nullptr;
    QString meshPath = QFileDialog::getOpenFileName(
            win, tr("Select Mesh to Unwrap"), QString(),
            tr("Meshes (*.ply *.obj *.off)"));
    if (meshPath.isEmpty()) return;
    QString outputPath = QFileDialog::getSaveFileName(
            win, tr("Output Unwrapped Mesh"), meshPath + ".unwrapped.obj",
            tr("OBJ (*.obj)"));
    if (outputPath.isEmpty()) return;
    launchDatasetToolWithArgs("unwrapMesh",
                              {"--path", meshPath, "--output", outputPath});
}

void qSIBR::launchCropFromCenter() {
    auto* win = m_app ? m_app->getMainWindow() : nullptr;
    QString inputFile =
            QFileDialog::getOpenFileName(win, tr("Select Image List File"),
                                         QString(), tr("Text files (*.txt)"));
    if (inputFile.isEmpty()) return;
    QString outputPath = QFileDialog::getExistingDirectory(
            win, tr("Select Output Directory"), QString(),
            QFileDialog::ShowDirsOnly);
    if (outputPath.isEmpty()) return;
    launchDatasetToolWithArgs("cropFromCenter",
                              {"--inputFile", inputFile, "--outputPath",
                               outputPath, "--cropResolution", "0x0"});
}

void qSIBR::launchCameraConverter() {
    auto* win = m_app ? m_app->getMainWindow() : nullptr;
    QString inputPath = QFileDialog::getExistingDirectory(
            win, tr("Select Input Camera Directory"), QString(),
            QFileDialog::ShowDirsOnly);
    if (inputPath.isEmpty()) return;
    QString outputPath = QFileDialog::getSaveFileName(
            win, tr("Output Camera File"), inputPath + "/cameras_out.txt",
            tr("Text files (*.txt);;Bundle files (*.out)"));
    if (outputPath.isEmpty()) return;
    launchDatasetToolWithArgs("cameraConverter",
                              {"--input", inputPath, "--output", outputPath});
}

void qSIBR::launchAlignMeshes() {
    auto* win = m_app ? m_app->getMainWindow() : nullptr;
    QString refPath = QFileDialog::getExistingDirectory(
            win, tr("Select Reference Scene"), QString(),
            QFileDialog::ShowDirsOnly);
    if (refPath.isEmpty()) return;
    QString alignPath = QFileDialog::getExistingDirectory(
            win, tr("Select Scene to Align"), QString(),
            QFileDialog::ShowDirsOnly);
    if (alignPath.isEmpty()) return;
    QString outPath = QFileDialog::getExistingDirectory(
            win, tr("Select Output Directory"), QString(),
            QFileDialog::ShowDirsOnly);
    if (outPath.isEmpty()) return;
    launchDatasetToolWithArgs("alignMeshes",
                              {"--pathRef", refPath, "--path2Align", alignPath,
                               "--out", outPath});
}
