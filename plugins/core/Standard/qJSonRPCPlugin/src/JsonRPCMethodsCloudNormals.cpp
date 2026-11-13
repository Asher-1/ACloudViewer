// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Point-cloud normal methods: cloud.*Normals*, cloud.normalsTo*.

#include <CVConst.h>
#include <ecvNormalVectors.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QJsonDocument>
#include <QJsonObject>

#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::D;

// ═══════════════════════════════════════════════════════════════════════════
// cloud.computeNormals
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudComputeNormals(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    double radius = params.value("radius", 0.0).toDouble();
    bool success = cloud->computeNormalsWithOctree(
            LS, ccNormalVectors::UNDEFINED,
            static_cast<PointCoordinateType>(radius));

    if (!success) {
        return JsonRPCResult::error(
                5, "Normal computation failed",
                D("entity_id", static_cast<qint64>(entityId), "radius",
                  radius));
    }

    cloud->showNormals(true);
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["has_normals"] = cloud->hasNormals();
    result["point_count"] = static_cast<qint64>(cloud->size());
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.removeNormals (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudRemoveNormals(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    bool hadNormals = cloud->hasNormals();
    cloud->unallocateNorms();
    cloud->showNormals(false);
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["had_normals"] = hadNormals;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.invertNormals (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudInvertNormals(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    if (!cloud->hasNormals()) {
        return JsonRPCResult::error(
                9, "Cloud has no normals to invert",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    cloud->invertNormals();
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["point_count"] = static_cast<qint64>(cloud->size());
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.octreeNormals - Compute normals using octree
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudOctreeNormals(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString radiusStr = params.value("radius", "AUTO").toString().toUpper();
    PointCoordinateType radius = 0.05;

    if (radiusStr != "AUTO") {
        radius = radiusStr.toDouble();
        if (radius <= 0) {
            radius = 0.05;
        }
    }

    bool result = cloud->computeNormalsWithOctree(
            CV_LOCAL_MODEL_TYPES::LS, ccNormalVectors::UNDEFINED, radius);

    if (!result) {
        return JsonRPCResult::error(
                5, "Octree normals computation failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    cloud->showNormals(true);
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["has_normals"] = cloud->hasNormals();
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.orientNormalsMST - Orient normals using MST
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudOrientNormalsMST(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    if (!cloud->hasNormals()) {
        return JsonRPCResult::error(
                5, "Cloud has no normals",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    int knn = params.value("knn", 6).toInt();

    bool result = cloud->orientNormalsWithMST(knn, nullptr);

    if (!result) {
        return JsonRPCResult::error(
                5, "MST normal orientation failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    cloud->showNormals(true);
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["has_normals"] = cloud->hasNormals();
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.clearNormals - Remove all normals
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudClearNormals(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    cloud->unallocateNorms();
    cloud->showNormals(false);
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["has_normals"] = cloud->hasNormals();
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.normalsToSFs - Export normals to scalar fields
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudNormalsToSFs(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    if (!cloud->hasNormals()) {
        return JsonRPCResult::error(
                5, "Cloud has no normals",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    bool exportDims[3] = {true, true, true};
    bool result = cloud->exportNormalToSF(exportDims);

    if (!result) {
        return JsonRPCResult::error(
                5, "Normals to SF export failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    int sfCount = static_cast<int>(cloud->getNumberOfScalarFields());
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = sfCount;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.normalsToDip - Convert normals to dip/dip-direction
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudNormalsToDip(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    if (!cloud->hasNormals()) {
        return JsonRPCResult::error(
                5, "Cloud has no normals",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    // Create two new scalar fields for dip and dip direction
    int dipIdx = cloud->addScalarField("Dip");
    int dipDirIdx = cloud->addScalarField("DipDir");

    if (dipIdx < 0 || dipDirIdx < 0) {
        return JsonRPCResult::error(
                5, "Failed to create scalar fields",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    ccScalarField* dipSF =
            static_cast<ccScalarField*>(cloud->getScalarField(dipIdx));
    ccScalarField* dipDirSF =
            static_cast<ccScalarField*>(cloud->getScalarField(dipDirIdx));

    bool result = cloud->convertNormalToDipDirSFs(dipSF, dipDirSF);

    if (!result) {
        return JsonRPCResult::error(
                5, "Normals to dip conversion failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    int sfCount = static_cast<int>(cloud->getNumberOfScalarFields());
    cloud->setCurrentDisplayedScalarField(dipIdx);
    cloud->showSF(true);
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = sfCount;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}
