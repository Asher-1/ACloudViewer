// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// File I/O, scene tree and entity property methods: open, export,
// file.convert, scene.*, clear, transform.apply, entity.*.

#include <FileIOFilter.h>
#include <ecvGenericPointCloud.h>
#include <ecvMainAppInterface.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::D;
using jsonrpc::entityToJson;

// ═══════════════════════════════════════════════════════════════════════════
// open — load file into DB
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcOpen(const QMap<QString, QVariant>& params) {
    QString filename = params["filename"].toString();
    if (filename.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'filename' parameter",
                                    D("param", "filename"));
    }
    if (!QFile::exists(filename)) {
        return JsonRPCResult::error(-32602, "File not found: " + filename,
                                    D("filename", filename));
    }

    CCVector3d loadCoordinatesShift(0, 0, 0);
    bool loadCoordinatesTransEnabled = false;

    FileIOFilter::LoadParameters parameters;
    parameters.alwaysDisplayLoadDialog = false;
    parameters.shiftHandlingMode = ecvGlobalShiftManager::NO_DIALOG_AUTO_SHIFT;
    parameters.coordinatesShift = &loadCoordinatesShift;
    parameters.coordinatesShiftEnabled = &loadCoordinatesTransEnabled;
    parameters.parentWidget = nullptr;

    CC_FILE_ERROR res = CC_FERR_NO_ERROR;
    ccHObject* newGroup = FileIOFilter::LoadFromFile(
            filename, parameters, res, params["filter"].toString());

    if (!newGroup) {
        return JsonRPCResult::error(
                1, "Failed to load file",
                D("filename", filename, "error_code", static_cast<int>(res)));
    }

    ccHObject::Container clouds;
    newGroup->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD);
    for (ccHObject* cloud : clouds) {
        if (cloud) {
            static_cast<ccGenericPointCloud*>(cloud)->showNormals(false);
        }
    }

    QList<QVariant> transformation = params["transformation"].toList();
    if (transformation.size() == 16) {
        std::vector<double> values(16);
        bool success = true;
        for (unsigned i = 0; i < 16; ++i) {
            values[((i % 4) * 4) + (i / 4)] =
                    transformation[i].toDouble(&success);
            if (!success) break;
        }
        if (success) {
            ccGLMatrix mat(values.data());
            newGroup->setGLTransformation(mat);
            newGroup->applyGLTransformation_recursive();
        }
    }

    m_app->addToDB(newGroup);
    redraw();

    QJsonObject info = entityToJson(newGroup, true);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// export — save entity to file
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcExport(const QMap<QString, QVariant>& params) {
    QString filename = params["filename"].toString();
    if (filename.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'filename' parameter",
                                    D("param", "filename"));
    }

    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntity(entityId);
    if (!entity) {
        return JsonRPCResult::error(
                2, "Entity not found",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;
    saveParams.parentWidget = m_app->getActiveWindow();

    CC_FILE_ERROR err = CC_FERR_UNKNOWN_FILE;
    QString filterName = params["filter"].toString();
    if (!filterName.isEmpty()) {
        err = FileIOFilter::SaveToFile(entity, filename, saveParams,
                                       filterName);
    } else {
        QString ext = QFileInfo(filename).suffix().toLower();
        auto filter = FileIOFilter::FindBestFilterForExtension(ext);
        if (filter) {
            err = FileIOFilter::SaveToFile(entity, filename, saveParams,
                                           filter);
        }
    }

    if (err != CC_FERR_NO_ERROR) {
        return JsonRPCResult::error(3, "Export failed",
                                    D("filename", filename, "entity_id",
                                      static_cast<qint64>(entityId),
                                      "error_code", static_cast<int>(err)));
    }
    QJsonObject result;
    result["filename"] = filename;
    result["entity_id"] = static_cast<qint64>(entityId);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// file.convert — load a file and re-export in a different format
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcFileConvert(
        const QMap<QString, QVariant>& params) {
    QString inputFile = params["input"].toString();
    QString outputFile = params["output"].toString();
    if (inputFile.isEmpty() || outputFile.isEmpty()) {
        return JsonRPCResult::error(
                -32602, "Missing 'input' and/or 'output' parameters",
                D("input", inputFile, "output", outputFile));
    }
    if (!QFile::exists(inputFile)) {
        return JsonRPCResult::error(-32602, "File not found: " + inputFile,
                                    D("filename", inputFile));
    }

    CCVector3d shift(0, 0, 0);
    bool shiftEnabled = false;
    FileIOFilter::LoadParameters loadParams;
    loadParams.alwaysDisplayLoadDialog = false;
    loadParams.shiftHandlingMode = ecvGlobalShiftManager::NO_DIALOG_AUTO_SHIFT;
    loadParams.coordinatesShift = &shift;
    loadParams.coordinatesShiftEnabled = &shiftEnabled;

    CC_FILE_ERROR loadErr = CC_FERR_NO_ERROR;
    ccHObject* loaded = FileIOFilter::LoadFromFile(
            inputFile, loadParams, loadErr, params["input_filter"].toString());
    if (!loaded) {
        return JsonRPCResult::error(1, "Failed to load file",
                                    D("filename", inputFile, "error_code",
                                      static_cast<int>(loadErr)));
    }

    ccHObject* toSave = loaded;
    if (loaded->isA(CV_TYPES::HIERARCHY_OBJECT)) {
        unsigned nChildren = loaded->getChildrenNumber();
        if (nChildren == 1) {
            toSave = loaded->getChild(0);
        } else if (nChildren > 1) {
            for (unsigned i = 0; i < nChildren; ++i) {
                ccHObject* child = loaded->getChild(i);
                if (child->isKindOf(CV_TYPES::POINT_CLOUD) ||
                    child->isKindOf(CV_TYPES::MESH)) {
                    toSave = child;
                    break;
                }
            }
        }
    }

    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;

    CC_FILE_ERROR saveErr = CC_FERR_UNKNOWN_FILE;
    QString filterName = params["output_filter"].toString();
    if (!filterName.isEmpty()) {
        saveErr = FileIOFilter::SaveToFile(toSave, outputFile, saveParams,
                                           filterName);
    } else {
        QString ext = QFileInfo(outputFile).suffix().toLower();
        auto filter = FileIOFilter::FindBestFilterForExtension(ext);
        if (filter) {
            saveErr = FileIOFilter::SaveToFile(toSave, outputFile, saveParams,
                                               filter);
        }
    }

    if (ecvViewManager::instance().getEffectiveView()) {
        loaded->removeFromRenderScreen(true);
    }
    delete loaded;

    if (saveErr != CC_FERR_NO_ERROR) {
        return JsonRPCResult::error(3, "Failed to save file",
                                    D("input", inputFile, "output", outputFile,
                                      "error_code", static_cast<int>(saveErr)));
    }

    QJsonObject result;
    result["input"] = inputFile;
    result["output"] = outputFile;
    result["status"] = "converted";
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// scene.list
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcSceneList(
        const QMap<QString, QVariant>& params) {
    bool recursive = params.value("recursive", true).toBool();
    auto* root = m_app->dbRootObject();
    QJsonArray entities;
    for (unsigned i = 0; i < root->getChildrenNumber(); ++i) {
        entities.append(entityToJson(root->getChild(i), recursive));
    }
    return JsonRPCResult::success(QJsonDocument(entities).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// scene.info
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcSceneInfo(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntity(entityId);
    if (!entity) {
        return JsonRPCResult::error(
                2, "Entity not found",
                D("entity_id", static_cast<qint64>(entityId)));
    }
    QJsonObject info = entityToJson(entity, true);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// scene.remove
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcSceneRemove(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntity(entityId);
    if (!entity) {
        return JsonRPCResult::error(
                2, "Entity not found",
                D("entity_id", static_cast<qint64>(entityId)));
    }
    QString name = entity->getName();
    m_app->removeFromDB(entity, true);
    redraw();
    QJsonObject result;
    result["removed_name"] = name;
    result["entity_id"] = static_cast<qint64>(entityId);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// scene.setVisible
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcSceneSetVisible(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    bool visible = params.value("visible", true).toBool();
    ccHObject* entity = findEntity(entityId);
    if (!entity) {
        return JsonRPCResult::error(
                2, "Entity not found",
                D("entity_id", static_cast<qint64>(entityId)));
    }
    entity->setEnabled(visible);
    const qint64 t0 = QDateTime::currentMSecsSinceEpoch();
    {
        // Visibility is a STATE change: the pushes above already landed and
        // the plain redraw pass re-evaluates draw gates. A FORCED pass here
        // re-converted every entity's geometry (~250ms for 286 cameras) for
        // zero fidelity gain.
        ecvRedrawScope scope(false, false);
    }
    const qint64 t1 = QDateTime::currentMSecsSinceEpoch();
    {
        CVLog::Print(QStringLiteral("[rpcTiming] scene.setVisible: "
                                    "setEnabled=0ms redraw=%1ms")
                             .arg(t1 - t0));
    }
    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["visible"] = visible;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// scene.select
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcSceneSelect(
        const QMap<QString, QVariant>& params) {
    QList<QVariant> ids = params["entity_ids"].toList();
    QJsonArray selected;
    const qint64 t0 = QDateTime::currentMSecsSinceEpoch();
    for (const auto& v : ids) {
        unsigned id = v.toUInt();
        ccHObject* entity = findEntity(id);
        if (entity) {
            m_app->setSelectedInDB(entity, true);
            selected.append(static_cast<qint64>(id));
        }
    }
    const qint64 t1 = QDateTime::currentMSecsSinceEpoch();
    m_app->updateUI();
    const qint64 t2 = QDateTime::currentMSecsSinceEpoch();
    {
        CVLog::Print(QStringLiteral("[rpcTiming] scene.select: dbSelect=%1ms "
                                    "updateUI=%2ms")
                             .arg(t1 - t0)
                             .arg(t2 - t1));
    }
    QJsonObject result;
    result["selected"] = selected;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// clear
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcClear(const QMap<QString, QVariant>& params) {
    auto* root = m_app->dbRootObject();
    int removed = 0;
    ccHObject* child;
    while ((child = root->getChild(0)) != nullptr) {
        m_app->removeFromDB(child, true);
        ++removed;
    }

    CC_DRAW_CONTEXT ctx;
    ctx.removeEntityType = ENTITY_TYPE::ECV_ALL;
    if (auto* eff = resolveView(params)) {
        ctx.display = eff;
        eff->removeEntities(ctx);
    }
    redraw();

    QJsonObject result;
    result["removed_count"] = removed;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// transform.apply
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcTransformApply(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntity(entityId);
    if (!entity) {
        return JsonRPCResult::error(
                2, "Entity not found",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    QList<QVariant> matrix = params["matrix"].toList();
    if (matrix.size() != 16) {
        return JsonRPCResult::error(
                -32602, "Matrix must be 16 elements (4x4 column-major)",
                D("received_size", matrix.size()));
    }

    std::vector<double> values(16);
    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        values[((i % 4) * 4) + (i / 4)] = matrix[i].toDouble(&ok);
        if (!ok) break;
    }
    if (!ok) {
        return JsonRPCResult::error(-32602, "Invalid matrix values");
    }

    ccGLMatrix mat(values.data());
    entity->setGLTransformation(mat);
    entity->applyGLTransformation_recursive();
    entity->notifyGeometryUpdate();
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["name"] = entity->getName();
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// entity.rename
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcEntityRename(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    QString name = params["name"].toString();
    ccHObject* entity = findEntity(entityId);
    if (!entity) {
        return JsonRPCResult::error(
                2, "Entity not found",
                D("entity_id", static_cast<qint64>(entityId)));
    }
    if (name.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'name'",
                                    D("param", "name"));
    }
    QString oldName = entity->getName();
    entity->setName(name);
    m_app->updateUI();
    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["old_name"] = oldName;
    result["new_name"] = name;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// entity.setColor
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcEntitySetColor(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntity(entityId);
    if (!entity) {
        return JsonRPCResult::error(
                2, "Entity not found",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    int r = params.value("r", 255).toInt();
    int g = params.value("g", 255).toInt();
    int b = params.value("b", 255).toInt();
    entity->setTempColor(ecvColor::Rgb(static_cast<ColorCompType>(r),
                                       static_cast<ColorCompType>(g),
                                       static_cast<ColorCompType>(b)));
    entity->enableTempColor(true);
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["color"] = QJsonArray({r, g, b});
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}
