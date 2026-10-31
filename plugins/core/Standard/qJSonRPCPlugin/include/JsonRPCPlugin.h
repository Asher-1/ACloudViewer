// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QElapsedTimer>
#include <functional>

#include "ecvStdPluginInterface.h"
#include "jsonrpcserver.h"

struct RpcMethodEntry {
    QString description;
    std::function<JsonRPCResult(const QMap<QString, QVariant>&)> handler;
};

class JsonRPCPlugin : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.JsonRPC" FILE
                          "../info.json")

public:
    explicit JsonRPCPlugin(QObject* parent = nullptr);
    /** Kills RPC child processes still running (their completions
     *  reference this plugin). */
    ~JsonRPCPlugin() override;

    QList<QAction*> getActions() override;
    /** Auto-starts the WebSocket server once the app interface is bound:
     *  the agent toolchain (cli-anything-acloudviewer, MCP, headless test
     *  drivers) expects the RPC endpoint to be reachable as soon as the
     *  GUI is up. The checkable menu action stays as the manual on/off
     *  switch and is synced to the auto-started state. */
    void setMainAppInterface(ecvMainAppInterface* app) override;

public slots:
    void triggered(bool checked);
    JsonRPCResult execute(QString method, QMap<QString, QVariant> params);

private:
    /** Listens on ACV_RPC_PORT (default 6001); returns false and logs an
     *  error when the port cannot be bound (e.g. another instance). */
    bool startServer();
    void registerMethods();
    void reg(const QString& name,
             const QString& desc,
             std::function<JsonRPCResult(const QMap<QString, QVariant>&)> fn);

    // --- helpers ---
    ccHObject* findEntity(unsigned id);
    ccPointCloud* findCloud(unsigned id, JsonRPCResult& err);
    ccMesh* findMesh(unsigned id, JsonRPCResult& err);
    ecvGenericGLDisplay* resolveView(const QMap<QString, QVariant>& params);
    void redraw();
    void logRequest(const QString& method,
                    const QMap<QString, QVariant>& params);
    void logResponse(const QString& method,
                     const JsonRPCResult& result,
                     qint64 elapsedMs);

    // --- JsonRPCMethodsScene.cpp: files, scene, entities ---
    JsonRPCResult rpcOpen(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcExport(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcFileConvert(const QMap<QString, QVariant>& params);
    // --- Spirula Studio (BUILD_SPIRULA; spirula_base) ---
    JsonRPCResult rpcSpirulaVersion(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSpirulaDevices(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSpirulaLaunch(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSpirulaCli(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSpirulaImport(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSpirulaCancel(const QMap<QString, QVariant>& params);
    // --- Scene tree ---
    JsonRPCResult rpcSceneList(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSceneInfo(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSceneRemove(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSceneSetVisible(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSceneSelect(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcClear(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcTransformApply(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcEntityRename(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcEntitySetColor(const QMap<QString, QVariant>& params);
    // --- JsonRPCMethodsUi.cpp: generic GUI drivers ---
    /** Generic driver: trigger a QAction matched by its text (opens plugin
     *  dialogs etc.). Neutral — knows no concrete plugin type. */
    JsonRPCResult rpcUiTriggerAction(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcUiListActions(const QMap<QString, QVariant>& params);
    /** Generic driver: Qt-reflection invoke — find the unique QObject whose
     *  class name contains \p class_contains and which declares an invokable
     *  \p method taking 0..3 QVariant parameters, and invoke it with \p args.
     *  Neutral — lets agent tests drive plugin dialogs without coupling this
     *  plugin to any of them. */
    JsonRPCResult rpcQtInvokeMethod(const QMap<QString, QVariant>& params);
    // --- JsonRPCMethodsView.cpp: view control, picking ---
    JsonRPCResult rpcViewList(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewSetOrientation(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewZoomFit(const QMap<QString, QVariant>& params);
    /** Test/agent probe: resolve the entity under a viewport pixel through
     *  the REAL entity-pick chain (pickObject -> pickLeafViewId, DPR-aware).
     *  Params: {x, y} in Qt-logical pixels. Returns viewId/entity_id/name. */
    JsonRPCResult rpcViewPickEntity(const QMap<QString, QVariant>& params);
    /** Test/agent probe: drive the CPU point-picking (octree) chain through
     *  the REAL StartPicking entry with a one-shot itemPicked capture.
     *  Params: {x, y} in Qt-logical pixels. Returns hit/entity/point. */
    JsonRPCResult rpcViewPickPoint(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewRefresh(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewSelectRegion(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewSetPerspective(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewSetPointSize(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewScreenshot(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewGetCamera(const QMap<QString, QVariant>& params);
    // --- JsonRPCMethodsCloud.cpp: color, filter, analysis ---
    JsonRPCResult rpcCloudPaintUniform(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudPaintByHeight(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudPaintByScalarField(
            const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSubsample(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudCrop(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRemoveRgb(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudMerge(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudDensity(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudCurvature(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRoughness(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudGeometricFeature(
            const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudApproxDensity(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudColorBanding(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSorFilter(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudExtractConnectedComponents(
            const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudBestFitPlane(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudDelaunay(const QMap<QString, QVariant>& params);
    // --- JsonRPCMethodsCloudScalarFields.cpp: scalar fields ---
    JsonRPCResult rpcCloudGetScalarFields(
            const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSetActiveSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRemoveSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRemoveAllSfs(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRenameSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudFilterSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudCoordToSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSfArithmetic(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSfOperation(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSfGradient(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSfConvertToRGB(const QMap<QString, QVariant>& params);
    // --- JsonRPCMethodsCloudNormals.cpp: normals ---
    JsonRPCResult rpcCloudComputeNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRemoveNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudInvertNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudOctreeNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudOrientNormalsMST(
            const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudClearNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudNormalsToSFs(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudNormalsToDip(const QMap<QString, QVariant>& params);
    // --- JsonRPCMethodsMesh.cpp: mesh processing ---
    JsonRPCResult rpcMeshSimplify(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshSmooth(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshSubdivide(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshSamplePoints(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshExtractVertices(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshFlipTriangles(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshVolume(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshMerge(const QMap<QString, QVariant>& params);
    // --- JsonRPCMethodsProcess.cpp: COLMAP and CLI children ---
    JsonRPCResult rpcColmapReconstruct(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcColmapRun(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcProcessRunCli(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcProcessCsf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcProcessM3c2(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcProcessRansac(const QMap<QString, QVariant>& params);
    // --- JsonRPCMethodsAI.cpp: AICore children ---
    /** Typed bridge to registered AICore plugin batch commands. Spawns an
     *  isolated headless child and returns its RESULT_JSON manifest. */
    JsonRPCResult rpcAICoreBatch(const QMap<QString, QVariant>& params);
    /** Spawns a headless ACloudViewer -SILENT -YOLO_TRACK child process
     *  (same offscreen pattern as rpcProcessRunCli). */
    JsonRPCResult rpcYoloTrack(const QMap<QString, QVariant>& params);
    /** Spawns a headless ACloudViewer -SILENT -SAM3D_GENERATE child
     *  process: batch image-to-3D (Gaussian PLY + optional mesh) with a
     *  single model load, RESULT_JSON manifest (same offscreen pattern). */
    JsonRPCResult rpcSam3dGenerate(const QMap<QString, QVariant>& params);
    /** mapanything.reconstruct: batch multi-view reconstruction (one model
     *  load, merged world-frame PLY + RESULT_JSON). */
    JsonRPCResult rpcMapAnythingReconstruct(
            const QMap<QString, QVariant>& params);

    QAction* m_action{nullptr};
    JsonRPCServer rpc_server;
    QMap<QString, RpcMethodEntry> m_methods;
};
