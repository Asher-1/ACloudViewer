// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Point-cloud scalar-field methods: cloud.*Sf*, cloud.getScalarFields,
// cloud.coordToSF.

#include <ReferenceCloud.h>
#include <ScalarFieldTools.h>
#include <ecvColorScalesManager.h>
#include <ecvMainAppInterface.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <algorithm>
#include <cmath>

#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::D;
using jsonrpc::entityToJson;

// ═══════════════════════════════════════════════════════════════════════════
// cloud.getScalarFields
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudGetScalarFields(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QJsonArray fields;
    for (unsigned i = 0; i < cloud->getNumberOfScalarFields(); ++i) {
        auto* sf = cloud->getScalarField(i);
        if (!sf) continue;
        QJsonObject f;
        f["index"] = static_cast<int>(i);
        f["name"] = sf->getName();
        f["min"] = sf->getMin();
        f["max"] = sf->getMax();
        ScalarType mean = 0;
        sf->computeMeanAndVariance(mean);
        f["mean"] = mean;
        fields.append(f);
    }
    return JsonRPCResult::success(QJsonDocument(fields).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.setActiveSf (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudSetActiveSf(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int sfIdx = -1;
    if (params.contains("field_name")) {
        QString name = params["field_name"].toString();
        sfIdx = cloud->getScalarFieldIndexByName(name.toStdString().c_str());
        if (sfIdx < 0) {
            return JsonRPCResult::error(
                    6, "Scalar field not found by name",
                    D("entity_id", static_cast<qint64>(entityId), "field_name",
                      name));
        }
    } else {
        sfIdx = params.value("field_index", -1).toInt();
    }

    if (sfIdx < 0 ||
        sfIdx >= static_cast<int>(cloud->getNumberOfScalarFields())) {
        return JsonRPCResult::error(
                6, "Scalar field index out of range",
                D("entity_id", static_cast<qint64>(entityId), "field_index",
                  sfIdx, "count",
                  static_cast<int>(cloud->getNumberOfScalarFields())));
    }

    cloud->setCurrentDisplayedScalarField(sfIdx);
    cloud->showSF(true);
    redraw();

    auto* sf = cloud->getScalarField(sfIdx);
    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["field_index"] = sfIdx;
    result["field_name"] = sf ? sf->getName() : "";
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.removeSf (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudRemoveSf(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int sfIdx = -1;
    QString removedName;
    if (params.contains("field_name")) {
        QString name = params["field_name"].toString();
        sfIdx = cloud->getScalarFieldIndexByName(name.toStdString().c_str());
        if (sfIdx < 0) {
            return JsonRPCResult::error(
                    6, "Scalar field not found by name",
                    D("entity_id", static_cast<qint64>(entityId), "field_name",
                      name));
        }
        removedName = name;
    } else {
        sfIdx = params.value("field_index", -1).toInt();
    }

    if (sfIdx < 0 ||
        sfIdx >= static_cast<int>(cloud->getNumberOfScalarFields())) {
        return JsonRPCResult::error(
                6, "Scalar field index out of range",
                D("entity_id", static_cast<qint64>(entityId), "field_index",
                  sfIdx, "count",
                  static_cast<int>(cloud->getNumberOfScalarFields())));
    }

    if (removedName.isEmpty()) {
        auto* sf = cloud->getScalarField(sfIdx);
        if (sf) removedName = sf->getName();
    }

    cloud->deleteScalarField(sfIdx);
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["removed_field"] = removedName;
    result["remaining_count"] =
            static_cast<int>(cloud->getNumberOfScalarFields());
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.removeAllSfs (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudRemoveAllSfs(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int count = static_cast<int>(cloud->getNumberOfScalarFields());
    cloud->deleteAllScalarFields();
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["removed_count"] = count;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.renameSf (NEW)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudRenameSf(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString newName = params["new_name"].toString();
    if (newName.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'new_name'",
                                    D("param", "new_name"));
    }

    int sfIdx = -1;
    if (params.contains("old_name")) {
        QString oldName = params["old_name"].toString();
        sfIdx = cloud->getScalarFieldIndexByName(oldName.toStdString().c_str());
        if (sfIdx < 0) {
            return JsonRPCResult::error(
                    6, "Scalar field not found by name",
                    D("entity_id", static_cast<qint64>(entityId), "old_name",
                      oldName));
        }
    } else {
        sfIdx = params.value("field_index", -1).toInt();
    }

    if (sfIdx < 0 ||
        sfIdx >= static_cast<int>(cloud->getNumberOfScalarFields())) {
        return JsonRPCResult::error(
                6, "Scalar field index out of range",
                D("entity_id", static_cast<qint64>(entityId), "field_index",
                  sfIdx));
    }

    auto* sf = cloud->getScalarField(sfIdx);
    QString oldName = sf->getName();
    sf->setName(newName.toStdString().c_str());
    m_app->updateUI();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["field_index"] = sfIdx;
    result["old_name"] = oldName;
    result["new_name"] = newName;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.filterSf (NEW) — keep points where SF value is in [min, max]
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudFilterSf(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int sfIdx = -1;
    if (params.contains("field_name")) {
        QString name = params["field_name"].toString();
        sfIdx = cloud->getScalarFieldIndexByName(name.toStdString().c_str());
    } else {
        sfIdx = params.value("field_index",
                             cloud->getCurrentDisplayedScalarFieldIndex())
                        .toInt();
    }

    if (sfIdx < 0 ||
        sfIdx >= static_cast<int>(cloud->getNumberOfScalarFields())) {
        return JsonRPCResult::error(
                6, "Scalar field not found",
                D("entity_id", static_cast<qint64>(entityId), "field_index",
                  sfIdx));
    }

    auto* sf = cloud->getScalarField(sfIdx);
    sf->computeMinAndMax();
    double sfMin =
            params.value("min", static_cast<double>(sf->getMin())).toDouble();
    double sfMax =
            params.value("max", static_cast<double>(sf->getMax())).toDouble();

    cloudViewer::ReferenceCloud refCloud(cloud);
    for (unsigned i = 0; i < cloud->size(); ++i) {
        ScalarType val = sf->getValue(i);
        if (val >= sfMin && val <= sfMax) {
            refCloud.addPointIndex(i);
        }
    }

    if (refCloud.size() == 0) {
        return JsonRPCResult::error(
                5, "Filter produced empty result",
                D("entity_id", static_cast<qint64>(entityId), "min", sfMin,
                  "max", sfMax));
    }

    ccPointCloud* filtered = cloud->partialClone(&refCloud);
    if (!filtered) {
        return JsonRPCResult::error(
                5, "Filter clone failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    filtered->setName(cloud->getName() + "_filtered");
    m_app->addToDB(filtered);
    redraw();

    QJsonObject result = entityToJson(filtered);
    result["original_count"] = static_cast<qint64>(cloud->size());
    result["filter_range"] = QJsonArray({sfMin, sfMax});
    result["field_name"] = sf->getName();
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.coordToSF — create scalar field from X/Y/Z coordinate
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudCoordToSf(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    QString dim = params.value("dimension", "z").toString().toLower();
    int dimIdx = (dim == "x") ? 0 : (dim == "y") ? 1 : 2;
    QString sfName = (dim == "x")   ? "Coord. X"
                     : (dim == "y") ? "Coord. Y"
                                    : "Coord. Z";

    int sfIdx = cloud->getScalarFieldIndexByName(sfName.toStdString().c_str());
    if (sfIdx < 0) {
        sfIdx = cloud->addScalarField(sfName.toStdString().c_str());
        if (sfIdx < 0) {
            return JsonRPCResult::error(
                    5, "Failed to create scalar field",
                    D("entity_id", static_cast<qint64>(entityId), "dimension",
                      dim));
        }
    }

    auto* sf = cloud->getScalarField(sfIdx);
    for (unsigned i = 0; i < cloud->size(); ++i) {
        sf->setValue(i, static_cast<ScalarType>(cloud->getPoint(i)->u[dimIdx]));
    }
    sf->computeMinAndMax();

    cloud->setCurrentDisplayedScalarField(sfIdx);
    cloud->showSF(true);
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["field_name"] = sfName;
    result["field_index"] = sfIdx;
    result["dimension"] = dim;
    result["min"] = static_cast<double>(sf->getMin());
    result["max"] = static_cast<double>(sf->getMax());
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.sfArithmetic - Scalar field arithmetic operations
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudSfArithmetic(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int sfIndex = params.value("sf_index", 0).toInt();
    QString operation = params.value("operation", "SQRT").toString().toUpper();

    ccScalarField* sf =
            static_cast<ccScalarField*>(cloud->getScalarField(sfIndex));
    if (!sf) {
        return JsonRPCResult::error(
                5, "Scalar field not found",
                D("entity_id", static_cast<qint64>(entityId), "sf_index",
                  sfIndex));
    }

    unsigned count = sf->size();
    for (unsigned i = 0; i < count; ++i) {
        ScalarType val = sf->getValue(i);
        ScalarType newVal = val;

        if (operation == "SQRT") {
            newVal = std::sqrt(std::max(0.0, static_cast<double>(val)));
        } else if (operation == "ABS") {
            newVal = std::abs(val);
        } else if (operation == "LOG" || operation == "LOG10") {
            newVal = val > 0 ? std::log10(val) : 0;
        } else if (operation == "LN") {
            newVal = val > 0 ? std::log(val) : 0;
        } else if (operation == "EXP") {
            newVal = std::exp(val);
        } else if (operation == "SQUARE") {
            newVal = val * val;
        }

        sf->setValue(i, newVal);
    }

    sf->computeMinAndMax();
    cloud->setCurrentDisplayedScalarField(sfIndex);
    cloud->showSF(true);
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_index"] = sfIndex;
    response["operation"] = operation;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.sfOperation - Scalar field operation with constant
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudSfOperation(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int sfIndex = params.value("sf_index", 0).toInt();
    QString operation =
            params.value("operation", "MULTIPLY").toString().toUpper();
    double value = params.value("value", 1.0).toDouble();

    ccScalarField* sf =
            static_cast<ccScalarField*>(cloud->getScalarField(sfIndex));
    if (!sf) {
        return JsonRPCResult::error(
                5, "Scalar field not found",
                D("entity_id", static_cast<qint64>(entityId), "sf_index",
                  sfIndex));
    }

    unsigned count = sf->size();
    for (unsigned i = 0; i < count; ++i) {
        ScalarType val = sf->getValue(i);
        ScalarType newVal = val;

        if (operation == "ADD" || operation == "PLUS") {
            newVal = val + value;
        } else if (operation == "SUBTRACT" || operation == "MINUS") {
            newVal = val - value;
        } else if (operation == "MULTIPLY" || operation == "TIMES") {
            newVal = val * value;
        } else if (operation == "DIVIDE") {
            newVal = (value != 0) ? (val / value) : 0;
        }

        sf->setValue(i, newVal);
    }

    sf->computeMinAndMax();
    cloud->setCurrentDisplayedScalarField(sfIndex);
    cloud->showSF(true);
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_index"] = sfIndex;
    response["operation"] = operation;
    response["value"] = value;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.sfGradient - Compute scalar field gradient
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudSfGradient(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int sfIndex = params.value("sf_index", -1).toInt();
    if (sfIndex < 0) {
        sfIndex = cloud->getCurrentDisplayedScalarFieldIndex();
    }

    ccScalarField* sf =
            static_cast<ccScalarField*>(cloud->getScalarField(sfIndex));
    if (!sf) {
        return JsonRPCResult::error(
                5, "Scalar field not found",
                D("entity_id", static_cast<qint64>(entityId), "sf_index",
                  sfIndex));
    }

    cloud->setCurrentDisplayedScalarField(sfIndex);
    cloud->showSF(true);

    PointCoordinateType radius = params.value("radius", 0.0).toDouble();
    int result = cloudViewer::ScalarFieldTools::computeScalarFieldGradient(
            cloud, radius, false, false, nullptr, nullptr);

    if (result < 0) {
        return JsonRPCResult::error(
                5, "Gradient computation failed",
                D("entity_id", static_cast<qint64>(entityId), "error_code",
                  result));
    }

    int newSfCount = static_cast<int>(cloud->getNumberOfScalarFields());
    cloud->setCurrentDisplayedScalarField(newSfCount - 1);
    cloud->showSF(true);
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["sf_count"] = newSfCount;
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// cloud.sfConvertToRGB - Convert scalar field to RGB colors
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcCloudSfConvertToRGB(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    JsonRPCResult err;
    auto* cloud = findCloud(entityId, err);
    if (!cloud) return err;

    int sfIndex = params.value("sf_index", -1).toInt();
    if (sfIndex < 0) {
        sfIndex = cloud->getCurrentDisplayedScalarFieldIndex();
    }

    if (sfIndex < 0 ||
        sfIndex >= static_cast<int>(cloud->getNumberOfScalarFields())) {
        return JsonRPCResult::error(
                5, "Invalid scalar field index",
                D("entity_id", static_cast<qint64>(entityId), "sf_index",
                  sfIndex));
    }

    cloud->setCurrentDisplayedScalarField(sfIndex);

    // Get default color scale
    ccColorScale::Shared colorScale = ccColorScalesManager::GetDefaultScale();
    if (!colorScale) {
        colorScale = ccColorScalesManager::GetUniqueInstance()->getDefaultScale(
                ccColorScalesManager::BGYR);
    }

    bool result = cloud->setRGBColorByHeight(2, colorScale);  // 2 = Z dimension

    if (!result) {
        return JsonRPCResult::error(
                5, "SF to RGB conversion failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    cloud->showColors(true);
    cloud->showSF(false);
    redraw();

    QJsonObject response;
    response["entity_id"] = static_cast<qint64>(entityId);
    response["has_colors"] = cloud->hasColors();
    return JsonRPCResult::success(QJsonDocument(response).toVariant());
}
