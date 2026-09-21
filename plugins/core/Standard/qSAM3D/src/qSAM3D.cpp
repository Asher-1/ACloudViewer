// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qSAM3D.h"

#include <ecvMainAppInterface.h>
#include <ecvMesh.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>

#include <QFileInfo>
#include <QMainWindow>
#include <QMessageBox>

#include "Sam3dGenerateCommand.h"
#include "ecvPersistentSettings.h"

qSAM3D::qSAM3D(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qSAM3D/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qSAM3D"));
    qRegisterMetaType<Sam3dRunResult>("Sam3dRunResult");
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
    m_dialog->setRunning(true);
    m_dialog->appendLog(tr("[SAM3D] Starting generation from '%1'...")
                                .arg(settings.imagePath));

    m_worker = new Sam3dWorker(this);
    m_worker->configure(settings);
    connect(m_worker, &Sam3dWorker::logMessage, m_dialog,
            &Sam3dDialog::appendLog);
    connect(m_worker, &Sam3dWorker::stageChanged, this, &qSAM3D::onWorkerStage);
    connect(m_worker, &Sam3dWorker::resultReady, this, &qSAM3D::onResultReady);
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

void qSAM3D::onTaskFinished(bool success) {
    if (m_worker) {
        if (!success) {
            const QString error = m_worker->error();
            m_dialog->appendLog(tr("[SAM3D] Task failed: %1").arg(error));
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
    const QString sourceName = QFileInfo(result.sourceImage).completeBaseName();
    const QString entityName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("SAM3D_%1_%2")
                    .arg(sourceName, ecvPluginDbNaming::deviceTagFromName(
                                             result.backend.isEmpty()
                                                     ? QStringLiteral("auto")
                                                     : result.backend)),
            m_app);

    ccMesh* mesh = nullptr;
    if (result.meshTriangleCount > 0) {
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
                meshEntity->showColors(false);
                mesh = meshEntity;
            }
        }
    }

    if (mesh) {
        mesh->setName(entityName);
        mesh->setMetaData(QStringLiteral("Source"), result.sourceImage);
        mesh->setMetaData(QStringLiteral("Backend"), result.backend);
        mesh->setMetaData(QStringLiteral("Quantization"), result.dtype);
        mesh->setMetaData(QStringLiteral("Pipeline (ms)"), result.e2eMs);
        mesh->setMetaData(QStringLiteral("Gaussians"), result.gaussianCount);
        mesh->setDisplay(m_app->getActiveGLDisplay());
        mesh->setVisible(true);
        m_app->addToDB(mesh, false, true, false, true);
        m_dialog->appendLog(
                tr("[SAM3D] Added mesh '%1' (%2 vertices, %3 faces).")
                        .arg(entityName)
                        .arg(result.meshVertexCount)
                        .arg(result.meshTriangleCount));
    } else {
        m_dialog->appendLog(tr(
                "[SAM3D] Mesh unavailable or allocation failed; the Gaussian "
                "PLY is still exported."));
    }

    if (!result.plyPath.isEmpty() && QFileInfo::exists(result.plyPath)) {
        m_dialog->appendLog(tr("[SAM3D] Gaussian PLY written to %1 "
                               "(%2 Gaussians).")
                                    .arg(result.plyPath)
                                    .arg(result.gaussianCount));
    }
    m_dialog->appendLog(tr("[SAM3D] Pipeline wall time: %1 s")
                                .arg(result.e2eMs / 1000.0, 0, 'f', 1));
}
