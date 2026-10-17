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
    //! Incremental scene delivery: one finished object lands in the DB tree
    //! immediately (queued from the worker thread).
    void onObjectReady(const Sam3dRunResult& object);
    void onSceneReady(const Sam3dSceneResult& result);
    //! Scene-mode diagnostic (queued from the worker thread, before the
    //! per-object runs start): the masks-over-image preview as one image
    //! entity in the DB tree.
    void onSegmentationPreview(const QImage& preview, int maskCount);
    void onTaskFinished(bool success);

private:
    void addResultToDb(const Sam3dRunResult& result);
    //! Scene mode: one group per run; every object becomes its own textured
    //! GLB entity plus its own colored splat point cloud, imported the
    //! moment the object finishes (see onObjectReady).
    void addObjectToDb(const Sam3dRunResult& object);
    //! Run-completion summary (counts already collected by addObjectToDb).
    void addSceneResultToDb(const Sam3dSceneResult& result);
    //! Lazily create / fetch the per-run scene group.
    ccHObject* sceneGroupRef(const Sam3dRunResult& object);
    //! Import a baked GLB through the shared assimp glTF filter so the DB
    //! entity carries the full textured material. Returns null on failure
    //! (caller falls back to the vertex-color mesh). Glb may be empty.
    ccHObject* importGlbEntity(const QByteArray& glb,
                               const Sam3dRunResult& result,
                               const QString& entityName);
    //! File-backed GLB import (scene mode: per-object GLB files).
    ccHObject* importGlbFile(const QString& glbPath,
                             const Sam3dRunResult& object,
                             const QString& entityName);

    QAction* m_action = nullptr;
    Sam3dDialog* m_dialog = nullptr;
    Sam3dWorker* m_worker = nullptr;
    // Settings of the running/most recent run (DB-import / PLY-export flags
    // steer the result handling in onResultReady). One worker runs at a time.
    Sam3dDialog::Settings m_lastSettings;
    // Incremental scene delivery state: the per-run group is created on the
    // FIRST objectReady and kept for the whole run (repeated makeUnique
    // calls would otherwise suffix the group name); the counters feed the
    // run-completion summary in onSceneReady. Meshes and point clouds land
    // in two dedicated sub-groups so the two artifact families stay apart.
    ccHObject* m_sceneGroup = nullptr;
    ccHObject* m_meshesGroup = nullptr;
    ccHObject* m_splatsGroup = nullptr;
    QString m_sceneGroupName;
    QString m_sceneSourceImage;
    QString m_sceneBackend;
    QString m_sceneDtype;
    int m_sceneMeshCount = 0;
    int m_sceneCloudCount = 0;
    int m_sceneObjectTotal = 0;
};
