// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Point-cloud colorization, filtering, sampling and geometric analysis:
// cloud.* except scalar-field and normal methods.

#include <AutoSegmentationTools.h>
#include <CloudSamplingTools.h>
#include <GeometricalAnalysisTools.h>
#include <Neighbourhood.h>
#include <ReferenceCloud.h>
#include <ecvMainAppInterface.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <algorithm>
#include <cmath>

#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::D;
using jsonrpc::entityToJson;

// ═══════════════════════════════════════════════════════════════════════════
// cloud.paintUniform
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudPaintUniform(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int r = params.value("r", 255).toInt();
    int g = params.value("g", 255).toInt();
    int b = params.value("b", 255).toInt();
    ecvColor::Rgb color(static_cast<ColorCompType>(r),
                        static_cast<ColorCompType>(g),
                        static_cast<ColorCompType>(b));

    if (!cloud->resizeTheRGBTable(false))
        return JsonRPCResult::error(
                5, "Failed to allocate color array",
                D("entity_id", static_cast<qint64>(entityId)));

    for (unsigned i = 0; i < cloud->size(); ++i) cloud->setPointColor(i, color);

    cloud->showColors(true);
    redraw();

    QJsonObject info;
    info["entity_id"] = static_cast<qint64>(entityId);
    info["points_colored"] = static_cast<qint64>(cloud->size());
    info["color"] = QJsonArray({r, g, b});
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.paintByHeight
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudPaintByHeight(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;
    if (cloud->size() == 0)
        return JsonRPCResult::error(
                4, "Point cloud is empty",
                D("entity_id", static_cast<qint64>(entityId)));

    QString axis = params.value("axis", "z").toString().toLower();
    int axisIdx = (axis == "x") ? 0 : (axis == "y") ? 1 : 2;

    PointCoordinateType minVal = cloud->getPoint(0)->u[axisIdx];
    PointCoordinateType maxVal = minVal;
    for (unsigned i = 1; i < cloud->size(); ++i) {
        PointCoordinateType v = cloud->getPoint(i)->u[axisIdx];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }

    if (!cloud->resizeTheRGBTable(false))
        return JsonRPCResult::error(
                5, "Failed to allocate color array",
                D("entity_id", static_cast<qint64>(entityId)));

    PointCoordinateType range = maxVal - minVal;
    if (range < 1e-10) range = 1;

    for (unsigned i = 0; i < cloud->size(); ++i) {
        float t = static_cast<float>((cloud->getPoint(i)->u[axisIdx] - minVal) /
                                     range);
        ColorCompType cr =
                static_cast<ColorCompType>(std::min(255.0f, t * 2.0f * 255.0f));
        ColorCompType cg = static_cast<ColorCompType>(
                std::min(255.0f, (1.0f - std::abs(t - 0.5f) * 2.0f) * 255.0f));
        ColorCompType cb = static_cast<ColorCompType>(
                std::min(255.0f, (1.0f - t) * 2.0f * 255.0f));
        cloud->setPointColor(i, ecvColor::Rgb(cr, cg, cb));
    }
    cloud->showColors(true);
    redraw();

    QJsonObject info;
    info["entity_id"] = static_cast<qint64>(entityId);
    info["axis"] = axis;
    info["min"] = static_cast<double>(minVal);
    info["max"] = static_cast<double>(maxVal);
    info["points_colored"] = static_cast<qint64>(cloud->size());
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.paintByScalarField
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudPaintByScalarField(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString sfName = params.value("field_name", "").toString();
    int sfIdx = -1;

    if (!sfName.isEmpty()) {
        sfIdx = cloud->getScalarFieldIndexByName(sfName.toStdString().c_str());
    } else if (params.contains("field_index")) {
        sfIdx = params["field_index"].toInt();
    } else {
        sfIdx = cloud->getCurrentDisplayedScalarFieldIndex();
    }

    if (sfIdx < 0 ||
        sfIdx >= static_cast<int>(cloud->getNumberOfScalarFields())) {
        return JsonRPCResult::error(
                6, "Scalar field not found",
                D("entity_id", static_cast<qint64>(entityId), "requested",
                  sfName.isEmpty() ? QJsonValue(sfIdx) : QJsonValue(sfName),
                  "available_count",
                  static_cast<int>(cloud->getNumberOfScalarFields())));
    }

    cloud->setCurrentDisplayedScalarField(sfIdx);
    cloud->showSF(true);
    redraw();

    cloudViewer::ScalarField* sf = cloud->getScalarField(sfIdx);
    QJsonObject info;
    info["entity_id"] = static_cast<qint64>(entityId);
    info["field_name"] = sf->getName();
    info["field_index"] = sfIdx;
    sf->computeMinAndMax();
    info["min"] = static_cast<double>(sf->getMin());
    info["max"] = static_cast<double>(sf->getMax());
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.subsample
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudSubsample(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString method = params.value("method", "spatial").toString();
    double step = params.value("step", 0.05).toDouble();

    ccPointCloud* subsampled = nullptr;
    if (method == "spatial") {
        cloudViewer::CloudSamplingTools::SFModulationParams modParams(false);
        cloudViewer::ReferenceCloud* refCloud =
                cloudViewer::CloudSamplingTools::resampleCloudSpatially(
                        cloud, static_cast<PointCoordinateType>(step),
                        modParams, nullptr, nullptr);
        if (refCloud) {
            subsampled = cloud->partialClone(refCloud);
            delete refCloud;
        }
    } else if (method == "random") {
        unsigned count = params.value("count", 10000u).toUInt();
        cloudViewer::ReferenceCloud* refCloud =
                cloudViewer::CloudSamplingTools::subsampleCloudRandomly(
                        cloud, count, nullptr);
        if (refCloud) {
            subsampled = cloud->partialClone(refCloud);
            delete refCloud;
        }
    } else {
        return JsonRPCResult::error(-32602,
                                    "method must be 'spatial' or 'random'",
                                    D("method", method));
    }

    if (!subsampled) {
        return JsonRPCResult::error(
                5, "Subsampling failed",
                D("entity_id", static_cast<qint64>(entityId), "method",
                  method));
    }

    subsampled->setName(cloud->getName() + "_subsampled");
    m_app->addToDB(subsampled);
    redraw();

    QJsonObject result = entityToJson(subsampled);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.crop
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudCrop(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    double minX = params["min_x"].toDouble();
    double minY = params["min_y"].toDouble();
    double minZ = params["min_z"].toDouble();
    double maxX = params["max_x"].toDouble();
    double maxY = params["max_y"].toDouble();
    double maxZ = params["max_z"].toDouble();

    ccBBox box(CCVector3(minX, minY, minZ), CCVector3(maxX, maxY, maxZ));
    cloudViewer::ReferenceCloud* refCloud = cloud->crop(box, true);

    if (!refCloud || refCloud->size() == 0) {
        delete refCloud;
        return JsonRPCResult::error(
                5, "Crop produced empty result",
                D("entity_id", static_cast<qint64>(entityId), "min",
                  QJsonValue(QJsonArray({minX, minY, minZ})), "max",
                  QJsonValue(QJsonArray({maxX, maxY, maxZ}))));
    }

    ccPointCloud* cropped = cloud->partialClone(refCloud);
    delete refCloud;
    if (!cropped) {
        return JsonRPCResult::error(
                5, "Crop clone failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    cropped->setName(cloud->getName() + "_cropped");
    m_app->addToDB(cropped);
    redraw();

    QJsonObject result = entityToJson(cropped);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.removeRgb (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudRemoveRgb(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    bool hadColors = cloud->hasColors();
    cloud->unallocateColors();
    cloud->showColors(false);
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["had_colors"] = hadColors;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.merge (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudMerge(
        const QMap<QString, QVariant>& params) {
    QList<QVariant> ids = params["entity_ids"].toList();
    if (ids.size() < 2) {
        return JsonRPCResult::error(-32602,
                                    "Need at least 2 entity_ids to merge",
                                    D("count", ids.size()));
    }

    ccPointCloud* merged = nullptr;
    QJsonArray mergedIds;

    for (const auto& v : ids) {
        unsigned id = v.toUInt();
        ccHObject* e = findEntity(id);
        if (!e) continue;
        auto* cloud = ccHObjectCaster::ToPointCloud(e);
        if (!cloud) continue;

        if (!merged) {
            merged = new ccPointCloud(*cloud);
        } else {
            *merged += cloud;
        }
        mergedIds.append(static_cast<qint64>(id));
    }

    if (!merged || mergedIds.isEmpty()) {
        return JsonRPCResult::error(
                5, "No valid point clouds found for merge",
                D("requested_ids", QJsonArray::fromVariantList(ids)));
    }

    merged->setName("merged_cloud");
    m_app->addToDB(merged);
    redraw();

    QJsonObject result = entityToJson(merged);
    result["merged_from"] = mergedIds;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.density - Compute local density
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudDensity(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    double radius = params.value("radius", 0.1).toDouble();

    int result = cloudViewer::GeometricalAnalysisTools::ComputeCharactersitic(
            cloudViewer::GeometricalAnalysisTools::LocalDensity,
            cloudViewer::GeometricalAnalysisTools::DENSITY_3D, cloud,
            static_cast<PointCoordinateType>(radius), nullptr, nullptr,
            nullptr);

    if (result != cloudViewer::GeometricalAnalysisTools::NoError) {
        return JsonRPCResult::error(
                5, "Density computation failed",
                D("entity_id", static_cast<qint64>(entityId), "error_code",
                  result));
    }

    int sfCount = static_cast<int>(cloud->getNumberOfScalarFields());
    if (sfCount > 0) {
        cloud->setCurrentDisplayedScalarField(sfCount - 1);
        cloud->showSF(true);
    }
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = sfCount;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.curvature - Compute surface curvature
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudCurvature(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString typeStr = params.value("type", "MEAN").toString().toUpper();
    double radius = params.value("radius", 0.05).toDouble();

    int curvatureType = cloudViewer::Neighbourhood::MEAN_CURV;
    if (typeStr == "GAUSS" || typeStr == "GAUSSIAN") {
        curvatureType = cloudViewer::Neighbourhood::GAUSSIAN_CURV;
    }

    int result = cloudViewer::GeometricalAnalysisTools::ComputeCharactersitic(
            cloudViewer::GeometricalAnalysisTools::Curvature, curvatureType,
            cloud, static_cast<PointCoordinateType>(radius), nullptr, nullptr,
            nullptr);

    if (result != cloudViewer::GeometricalAnalysisTools::NoError) {
        return JsonRPCResult::error(
                5, "Curvature computation failed",
                D("entity_id", static_cast<qint64>(entityId), "error_code",
                  result));
    }

    int sfCount = static_cast<int>(cloud->getNumberOfScalarFields());
    if (sfCount > 0) {
        cloud->setCurrentDisplayedScalarField(sfCount - 1);
        cloud->showSF(true);
    }
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = sfCount;
    response["type"] = typeStr;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.roughness - Compute surface roughness
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudRoughness(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    double radius = params.value("radius", 0.1).toDouble();

    int result = cloudViewer::GeometricalAnalysisTools::ComputeCharactersitic(
            cloudViewer::GeometricalAnalysisTools::Roughness, 0, cloud,
            static_cast<PointCoordinateType>(radius), nullptr, nullptr,
            nullptr);

    if (result != cloudViewer::GeometricalAnalysisTools::NoError) {
        return JsonRPCResult::error(
                5, "Roughness computation failed",
                D("entity_id", static_cast<qint64>(entityId), "error_code",
                  result));
    }

    int sfCount = static_cast<int>(cloud->getNumberOfScalarFields());
    if (sfCount > 0) {
        cloud->setCurrentDisplayedScalarField(sfCount - 1);
        cloud->showSF(true);
    }
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = sfCount;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.geometricFeature - Compute geometric features
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudGeometricFeature(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString typeStr =
            params.value("type", "SURFACE_VARIATION").toString().toUpper();
    double kernelSize = params.value("kernel_size", 0.1).toDouble();

    int featureType = cloudViewer::Neighbourhood::SurfaceVariation;
    if (typeStr.contains("EIGEN")) {
        int idx = typeStr.right(1).toInt();
        if (idx >= 1 && idx <= 3) {
            featureType = cloudViewer::Neighbourhood::EigenValue1 + (idx - 1);
        }
    }

    int result = cloudViewer::GeometricalAnalysisTools::ComputeCharactersitic(
            cloudViewer::GeometricalAnalysisTools::Feature, featureType, cloud,
            static_cast<PointCoordinateType>(kernelSize), nullptr, nullptr,
            nullptr);

    if (result != cloudViewer::GeometricalAnalysisTools::NoError) {
        return JsonRPCResult::error(
                5, "Geometric feature computation failed",
                D("entity_id", static_cast<qint64>(entityId), "error_code",
                  result));
    }

    int sfCount = static_cast<int>(cloud->getNumberOfScalarFields());
    if (sfCount > 0) {
        cloud->setCurrentDisplayedScalarField(sfCount - 1);
        cloud->showSF(true);
    }
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = sfCount;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.approxDensity - Compute approximate density
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudApproxDensity(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString densityTypeStr =
            params.value("density_type", "PRECISE").toString().toUpper();

    cloudViewer::GeometricalAnalysisTools::Density densityType =
            cloudViewer::GeometricalAnalysisTools::DENSITY_3D;
    if (densityTypeStr == "2D") {
        densityType = cloudViewer::GeometricalAnalysisTools::DENSITY_2D;
    } else if (densityTypeStr == "KNN") {
        densityType = cloudViewer::GeometricalAnalysisTools::DENSITY_KNN;
    }

    int result =
            cloudViewer::GeometricalAnalysisTools::ComputeLocalDensityApprox(
                    cloud, densityType, nullptr, nullptr);

    if (result != cloudViewer::GeometricalAnalysisTools::NoError) {
        return JsonRPCResult::error(
                5, "Approximate density computation failed",
                D("entity_id", static_cast<qint64>(entityId), "error_code",
                  result));
    }

    int sfCount = static_cast<int>(cloud->getNumberOfScalarFields());
    if (sfCount > 0) {
        cloud->setCurrentDisplayedScalarField(sfCount - 1);
        cloud->showSF(true);
    }
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = sfCount;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.colorBanding - Apply color banding
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudColorBanding(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString axis = params.value("axis", "Z").toString().toUpper();
    double frequency = params.value("frequency", 10.0).toDouble();

    // Use coordinate-to-SF + paintByScalarField instead of direct color
    // manipulation This is more stable and uses existing tested code paths

    // First, convert coordinate to scalar field
    int dimension = 2;  // Z
    if (axis == "X")
        dimension = 0;
    else if (axis == "Y")
        dimension = 1;

    QString sfName = QString("Coord_%1").arg(axis);
    int sfIdx = cloud->getScalarFieldIndexByName(qPrintable(sfName));
    if (sfIdx < 0) {
        sfIdx = cloud->addScalarField(qPrintable(sfName));
        if (sfIdx < 0) {
            return JsonRPCResult::error(
                    5, "Failed to create scalar field",
                    D("entity_id", static_cast<qint64>(entityId)));
        }

        ccScalarField* sf =
                static_cast<ccScalarField*>(cloud->getScalarField(sfIdx));
        if (!sf || !sf->resizeSafe(cloud->size())) {
            return JsonRPCResult::error(
                    5, "Failed to allocate scalar field",
                    D("entity_id", static_cast<qint64>(entityId)));
        }

        for (unsigned i = 0; i < cloud->size(); ++i) {
            const CCVector3* P = cloud->getPoint(i);
            if (!P) continue;

            ScalarType val = 0;
            if (dimension == 0)
                val = P->x;
            else if (dimension == 1)
                val = P->y;
            else
                val = P->z;

            sf->setValue(i, val);
        }
        sf->computeMinAndMax();
    }

    // Now use paintByScalarField
    cloud->setCurrentDisplayedScalarField(sfIdx);

    QMap<QString, QVariant> paintParams;
    paintParams["entity_id"] = entityId;
    paintParams["field_index"] = sfIdx;

    return rpcCloudPaintByScalarField(paintParams);
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.sorFilter - Statistical outlier removal
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudSorFilter(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int knn = params.value("knn", 6).toInt();
    double sigma = params.value("sigma", 1.0).toDouble();

    cloudViewer::ReferenceCloud* selection =
            cloudViewer::CloudSamplingTools::sorFilter(cloud, knn, sigma,
                                                       nullptr, nullptr);

    if (!selection) {
        return JsonRPCResult::error(
                5, "SOR filter failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    ccPointCloud* filtered = cloud->partialClone(selection);
    delete selection;

    if (!filtered) {
        return JsonRPCResult::error(
                5, "Failed to create filtered cloud",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    filtered->setName(cloud->getName() + "_filtered");
    m_app->addToDB(filtered);
    redraw();

    QJsonObject response = entityToJson(filtered);
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.extractConnectedComponents - Extract connected components
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudExtractConnectedComponents(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int minPoints = params.value("min_points", 10).toInt();
    int octreeLevel = params.value("octree_level", 6).toInt();

    ccOctree::Shared octree = cloud->getOctree();
    if (!octree) {
        octree = cloud->computeOctree();
    }

    if (!octree) {
        return JsonRPCResult::error(
                5, "Failed to compute octree",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    cloudViewer::ReferenceCloudContainer components;
    bool result = cloudViewer::AutoSegmentationTools::labelConnectedComponents(
            cloud, static_cast<unsigned char>(octreeLevel), false, nullptr,
            octree.data());

    if (!result) {
        return JsonRPCResult::error(
                5, "Connected components extraction failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    int sfIndex = cloud->getCurrentDisplayedScalarFieldIndex();
    if (sfIndex >= 0) {
        cloud->setCurrentDisplayedScalarField(sfIndex);
        cloud->showSF(true);
    }
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = static_cast<int>(cloud->getNumberOfScalarFields());
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.bestFitPlane - Compute best fit plane
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudBestFitPlane(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    bool makeHoriz = params.value("make_horiz", false).toBool();

    cloudViewer::Neighbourhood neighbourhood(cloud);
    const PointCoordinateType* planeEquation = neighbourhood.getLSPlane();

    if (!planeEquation) {
        return JsonRPCResult::error(
                5, "Best fit plane computation failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    const CCVector3* gravityCenter = neighbourhood.getGravityCenter();
    CCVector3 C = gravityCenter ? *gravityCenter : CCVector3(0, 0, 0);
    CCVector3 N(planeEquation[0], planeEquation[1], planeEquation[2]);

    // Compute RMS (simple approximation)
    double rms = 0;
    unsigned count = cloud->size();
    for (unsigned i = 0; i < count; ++i) {
        const CCVector3* P = cloud->getPoint(i);
        CCVector3 AP = *P - C;
        double dist = std::abs(AP.dot(N));
        rms += dist * dist;
    }
    rms = std::sqrt(rms / count);

    if (makeHoriz && N.z < 0) {
        N = -N;
    }

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["center_x"] = C.x;
    response["center_y"] = C.y;
    response["center_z"] = C.z;
    response["normal_x"] = N.x;
    response["normal_y"] = N.y;
    response["normal_z"] = N.z;
    response["rms"] = rms;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.delaunay - Delaunay 2.5D triangulation
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudDelaunay(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    // Use Neighbourhood to perform Delaunay triangulation
    cloudViewer::Neighbourhood neighbourhood(cloud);
    std::string errorStr;

    cloudViewer::GenericIndexedMesh* genericMesh =
            neighbourhood.triangulateOnPlane(
                    cloudViewer::Neighbourhood::DO_NOT_DUPLICATE_VERTICES,
                    cloudViewer::Neighbourhood::IGNORE_MAX_EDGE_LENGTH,
                    errorStr);

    if (!genericMesh) {
        return JsonRPCResult::error(
                5, "Delaunay triangulation failed",
                D("entity_id", static_cast<qint64>(entityId), "error",
                  QString::fromStdString(errorStr)));
    }

    // Convert to ccMesh
    ccMesh* mesh = new ccMesh(genericMesh, cloud);
    delete genericMesh;

    if (!mesh) {
        return JsonRPCResult::error(
                5, "Failed to create mesh",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    mesh->setName(cloud->getName() + "_mesh");
    mesh->setAssociatedCloud(cloud);
    mesh->addChild(cloud);
    m_app->addToDB(mesh);
    redraw();

    QJsonObject response = entityToJson(mesh);
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}
