// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Mesh methods: mesh.*.

#include <ecvMainAppInterface.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <cmath>

#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::D;
using jsonrpc::entityToJson;

// ═══════════════════════════════════════════════════════════════════════════
// mesh.simplify
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcMeshSimplify(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* mesh = findMesh(entityId, err);
    if (!mesh) return err;

    QString method = params.value("method", "quadric").toString();
    int target = params.value("target_triangles", 10000).toInt();
    double voxelSize = params.value("voxel_size", 0.05).toDouble();

    std::shared_ptr<ccMesh> result;
    if (method == "quadric") {
        result = mesh->SimplifyQuadricDecimation(target);
    } else if (method == "vertex_clustering") {
        result = mesh->SimplifyVertexClustering(voxelSize);
    } else {
        return JsonRPCResult::error(
                -32602, "method must be 'quadric' or 'vertex_clustering'",
                D("method", method));
    }

    if (!result) {
        return JsonRPCResult::error(
                5, "Simplification failed",
                D("entity_id", static_cast<qint64>(entityId), "method",
                  method));
    }

    ccMesh* simplified = new ccMesh(*result);
    simplified->setName(mesh->getName() + "_simplified");
    m_app->addToDB(simplified);
    redraw();

    QJsonObject info = entityToJson(simplified);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// mesh.smooth
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcMeshSmooth(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* mesh = findMesh(entityId, err);
    if (!mesh) return err;

    QString method = params.value("method", "laplacian").toString();
    int iterations = params.value("iterations", 5).toInt();
    double lambda = params.value("lambda", 0.5).toDouble();

    std::shared_ptr<ccMesh> result;
    if (method == "laplacian") {
        result = mesh->FilterSmoothLaplacian(iterations, lambda);
    } else if (method == "taubin") {
        double mu = params.value("mu", -0.53).toDouble();
        result = mesh->FilterSmoothTaubin(iterations, lambda, mu);
    } else if (method == "simple") {
        result = mesh->FilterSmoothSimple(iterations);
    } else {
        return JsonRPCResult::error(
                -32602, "method must be 'laplacian', 'taubin', or 'simple'",
                D("method", method));
    }

    if (!result) {
        return JsonRPCResult::error(
                5, "Smoothing failed",
                D("entity_id", static_cast<qint64>(entityId), "method",
                  method));
    }

    ccMesh* smoothed = new ccMesh(*result);
    smoothed->setName(mesh->getName() + "_smoothed");
    m_app->addToDB(smoothed);
    redraw();

    QJsonObject info = entityToJson(smoothed);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// mesh.subdivide
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcMeshSubdivide(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* mesh = findMesh(entityId, err);
    if (!mesh) return err;

    QString method = params.value("method", "midpoint").toString();
    int iterations = params.value("iterations", 1).toInt();

    std::shared_ptr<ccMesh> result;
    if (method == "midpoint") {
        result = mesh->SubdivideMidpoint(iterations);
    } else if (method == "loop") {
        result = mesh->SubdivideLoop(iterations);
    } else {
        return JsonRPCResult::error(-32602,
                                    "method must be 'midpoint' or 'loop'",
                                    D("method", method));
    }

    if (!result) {
        return JsonRPCResult::error(
                5, "Subdivision failed",
                D("entity_id", static_cast<qint64>(entityId), "method",
                  method));
    }

    ccMesh* subdivided = new ccMesh(*result);
    subdivided->setName(mesh->getName() + "_subdivided");
    m_app->addToDB(subdivided);
    redraw();

    QJsonObject info = entityToJson(subdivided);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// mesh.samplePoints
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcMeshSamplePoints(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* mesh = findMesh(entityId, err);
    if (!mesh) return err;

    QString method = params.value("method", "uniform").toString();
    unsigned count = params.value("count", 100000).toUInt();

    std::shared_ptr<ccPointCloud> cloud;
    if (method == "uniform") {
        cloud = mesh->SamplePointsUniformly(count);
    } else if (method == "poisson_disk") {
        cloud = mesh->SamplePointsPoissonDisk(count);
    } else {
        return JsonRPCResult::error(
                -32602, "method must be 'uniform' or 'poisson_disk'",
                D("method", method));
    }

    if (!cloud) {
        return JsonRPCResult::error(
                5, "Sampling failed",
                D("entity_id", static_cast<qint64>(entityId), "method",
                  method));
    }

    ccPointCloud* sampled = new ccPointCloud(*cloud);
    sampled->setName(mesh->getName() + "_sampled");
    m_app->addToDB(sampled);
    redraw();

    QJsonObject info = entityToJson(sampled);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// mesh.extractVertices (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcMeshExtractVertices(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* mesh = findMesh(entityId, err);
    if (!mesh) return err;

    auto* verts = mesh->getAssociatedCloud();
    if (!verts) {
        return JsonRPCResult::error(
                5, "Mesh has no associated vertices",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    auto* pcCloud = ccHObjectCaster::ToPointCloud(verts);
    if (!pcCloud) {
        return JsonRPCResult::error(
                5, "Cannot cast vertices to point cloud",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    ccPointCloud* extracted = new ccPointCloud(*pcCloud);
    extracted->setName(mesh->getName() + "_vertices");
    m_app->addToDB(extracted);
    redraw();

    QJsonObject result = entityToJson(extracted);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// mesh.flipTriangles (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcMeshFlipTriangles(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* mesh = findMesh(entityId, err);
    if (!mesh) return err;

    mesh->flipTriangles();
    mesh->notifyGeometryUpdate();
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["triangle_count"] = static_cast<qint64>(mesh->size());
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// mesh.volume (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcMeshVolume(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* mesh = findMesh(entityId, err);
    if (!mesh) return err;

    double volume = 0.0;
    unsigned triCount = mesh->size();
    auto* verts = mesh->getAssociatedCloud();
    if (!verts) {
        return JsonRPCResult::error(
                5, "Mesh has no vertices",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    for (unsigned i = 0; i < triCount; ++i) {
        cloudViewer::VerticesIndexes* tri = mesh->getTriangleVertIndexes(i);
        if (!tri) continue;
        const CCVector3* v0 = verts->getPoint(tri->i1);
        const CCVector3* v1 = verts->getPoint(tri->i2);
        const CCVector3* v2 = verts->getPoint(tri->i3);
        volume += static_cast<double>(v0->dot(v1->cross(*v2)));
    }
    volume /= 6.0;

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["volume"] = std::abs(volume);
    result["signed_volume"] = volume;
    result["triangle_count"] = static_cast<qint64>(triCount);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// mesh.merge (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcMeshMerge(
        const QMap<QString, QVariant>& params) {
    QList<QVariant> ids = params["entity_ids"].toList();
    if (ids.size() < 2) {
        return JsonRPCResult::error(-32602,
                                    "Need at least 2 entity_ids to merge",
                                    D("count", ids.size()));
    }

    QJsonArray mergedIds;
    std::vector<ccMesh*> meshes;

    for (const auto& v : ids) {
        unsigned id = v.toUInt();
        ccHObject* e = findEntity(id);
        if (!e) continue;
        auto* mesh = ccHObjectCaster::ToMesh(e);
        if (!mesh) continue;
        meshes.push_back(mesh);
        mergedIds.append(static_cast<qint64>(id));
    }

    if (meshes.size() < 2) {
        return JsonRPCResult::error(
                5, "Need at least 2 valid meshes to merge",
                D("found_count", static_cast<int>(meshes.size())));
    }

    ccHObject* mergedGroup = new ccHObject("merged_mesh");
    for (auto* m : meshes) {
        mergedGroup->addChild(new ccMesh(*m));
    }

    m_app->addToDB(mergedGroup);
    redraw();

    QJsonObject result = entityToJson(mergedGroup, true);
    result["merged_from"] = mergedIds;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}
