// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvStdPluginInterface.h>

#include <QAction>

#include "Sam3dDialog.h"
#include "Sam3dWorker.h"

class ccHObject;
class ccMesh;
class ecvMainAppInterface;

class qSAM3D : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qSAM3D" FILE
                          "../info.json")

public:
    explicit qSAM3D(QObject* parent = nullptr);

    QList<QAction*> getActions() override;
    /** Headless CLI: registers -SAM3D_GENERATE (batch Gaussian PLY /
     *  mesh generation over N images, single model load, RESULT_JSON
     *  manifest output). */
    void registerCommands(ccCommandLineInterface* cmd) override;

private slots:
    void showDialog();
    void executeTask(const Sam3dDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const Sam3dRunResult& result);
    void onSceneReady(const Sam3dSceneResult& result);
    void onTaskFinished(bool success);
    void onWorkerStage(int stage, int step, int total);

private:
    void addResultToDb(const Sam3dRunResult& result);
    //! Scene mode: one group per run; every object becomes its own textured
    //! GLB entity (the official per-object mesh deliverable) and the composed
    //! splat cloud lands as one colored point cloud in the same group.
    void addSceneResultToDb(const Sam3dSceneResult& result);
    //! Import a baked GLB through the shared assimp glTF filter so the DB
    //! entity carries the full textured material. Returns null on failure
    //! (caller falls back to the vertex-color mesh). Glb may be empty.
    ccHObject* importGlbEntity(const QByteArray& glb,
                               const Sam3dRunResult& result,
                               const QString& entityName);
    //! File-backed GLB import (scene mode: per-object GLB files).
    ccHObject* importGlbFile(const QString& glbPath,
                             const Sam3dSceneResult& scene,
                             const Sam3dRunResult& object,
                             const QString& entityName);

    QAction* m_action = nullptr;
    Sam3dDialog* m_dialog = nullptr;
    Sam3dWorker* m_worker = nullptr;
    // Settings of the running/most recent run (DB-import / PLY-export flags
    // steer the result handling in onResultReady). One worker runs at a time.
    Sam3dDialog::Settings m_lastSettings;
};
