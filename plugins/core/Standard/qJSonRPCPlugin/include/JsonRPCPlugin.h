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
    ~JsonRPCPlugin() override = default;

    QList<QAction*> getActions() override;

public slots:
    void triggered(bool checked);
    JsonRPCResult execute(QString method, QMap<QString, QVariant> params);

private:
    void registerMethods();
    void reg(const QString& name,
             const QString& desc,
             std::function<JsonRPCResult(const QMap<QString, QVariant>&)> fn);

    // --- helpers ---
    ccHObject* findEntity(unsigned id);
    ccPointCloud* findCloud(unsigned id, JsonRPCResult& err);
    ccMesh* findMesh(unsigned id, JsonRPCResult& err);
    void redraw();
    void logRequest(const QString& method,
                    const QMap<QString, QVariant>& params);
    void logResponse(const QString& method,
                     const JsonRPCResult& result,
                     qint64 elapsedMs);

    // --- File I/O ---
    JsonRPCResult rpcOpen(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcExport(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcFileConvert(const QMap<QString, QVariant>& params);
    // --- Scene tree ---
    JsonRPCResult rpcSceneList(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSceneInfo(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSceneRemove(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSceneSetVisible(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcSceneSelect(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcClear(const QMap<QString, QVariant>& params);
    // --- View control ---
    JsonRPCResult rpcViewSetOrientation(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewZoomFit(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewRefresh(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewSetPerspective(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewSetPointSize(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewScreenshot(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcViewGetCamera(const QMap<QString, QVariant>& params);
    // --- Transform ---
    JsonRPCResult rpcTransformApply(const QMap<QString, QVariant>& params);
    // --- Entity properties ---
    JsonRPCResult rpcEntityRename(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcEntitySetColor(const QMap<QString, QVariant>& params);
    // --- Cloud colorization ---
    JsonRPCResult rpcCloudPaintUniform(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudPaintByHeight(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudPaintByScalarField(
            const QMap<QString, QVariant>& params);
    // --- Cloud processing ---
    JsonRPCResult rpcCloudComputeNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSubsample(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudCrop(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudGetScalarFields(
            const QMap<QString, QVariant>& params);
    // --- Cloud scalar-field management ---
    JsonRPCResult rpcCloudSetActiveSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRemoveSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRemoveAllSfs(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRenameSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudFilterSf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudCoordToSf(const QMap<QString, QVariant>& params);
    // --- Cloud geometry ---
    JsonRPCResult rpcCloudRemoveRgb(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudRemoveNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudInvertNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudMerge(const QMap<QString, QVariant>& params);
    // --- Cloud geometric analysis (NEW) ---
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
    // --- Cloud scalar field operations (NEW) ---
    JsonRPCResult rpcCloudSfArithmetic(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSfOperation(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSfGradient(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudSfConvertToRGB(const QMap<QString, QVariant>& params);
    // --- Cloud normals advanced (NEW) ---
    JsonRPCResult rpcCloudOctreeNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudOrientNormalsMST(
            const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudClearNormals(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudNormalsToSFs(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcCloudNormalsToDip(const QMap<QString, QVariant>& params);
    // --- Mesh processing ---
    JsonRPCResult rpcMeshSimplify(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshSmooth(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshSubdivide(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshSamplePoints(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshExtractVertices(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshFlipTriangles(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshVolume(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcMeshMerge(const QMap<QString, QVariant>& params);
    // --- CLI Processing ---
    JsonRPCResult rpcProcessRunCli(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcProcessCsf(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcProcessM3c2(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcProcessRansac(const QMap<QString, QVariant>& params);
    // --- Reconstruction ---
    JsonRPCResult rpcColmapReconstruct(const QMap<QString, QVariant>& params);
    JsonRPCResult rpcColmapRun(const QMap<QString, QVariant>& params);

    QAction* m_action{nullptr};
    JsonRPCServer rpc_server;
    QMap<QString, RpcMethodEntry> m_methods;
};
