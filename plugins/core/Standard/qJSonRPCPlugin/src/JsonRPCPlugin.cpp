// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "JsonRPCPlugin.h"

#include <CVConst.h>
#include <CloudSamplingTools.h>
#include <FileIOFilter.h>
#include <ReferenceCloud.h>
#include <ecvDisplayTools.h>
#include <ecvGenericPointCloud.h>
#include <ecvMainAppInterface.h>
#include <ecvMesh.h>
#include <ecvNormalVectors.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtGui>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

QJsonObject entityToJson(ccHObject* obj, bool recursive = false) {
    QJsonObject o;
    if (!obj) return o;
    o["id"] = static_cast<qint64>(obj->getUniqueID());
    o["name"] = obj->getName();
    o["type"] = obj->getClassID() == CV_TYPES::POINT_CLOUD ? "POINT_CLOUD"
                : obj->getClassID() == CV_TYPES::MESH      ? "MESH"
                : obj->getClassID() == CV_TYPES::POLY_LINE ? "POLYLINE"
                : obj->getClassID() == CV_TYPES::HIERARCHY_OBJECT ? "GROUP"
                                                                  : "OTHER";
    o["visible"] = obj->isEnabled();
    o["children_count"] = static_cast<int>(obj->getChildrenNumber());

    ccBBox bb = obj->getOwnBB();
    if (bb.isValid()) {
        QJsonObject bbox;
        bbox["min_x"] = bb.minCorner().x;
        bbox["min_y"] = bb.minCorner().y;
        bbox["min_z"] = bb.minCorner().z;
        bbox["max_x"] = bb.maxCorner().x;
        bbox["max_y"] = bb.maxCorner().y;
        bbox["max_z"] = bb.maxCorner().z;
        o["bbox"] = bbox;
    }

    auto* cloud = ccHObjectCaster::ToPointCloud(obj);
    if (cloud) {
        o["point_count"] = static_cast<qint64>(cloud->size());
        o["has_normals"] = cloud->hasNormals();
        o["has_colors"] = cloud->hasColors();
        o["scalar_field_count"] =
                static_cast<int>(cloud->getNumberOfScalarFields());
    }

    auto* mesh = ccHObjectCaster::ToMesh(obj);
    if (mesh) {
        o["triangle_count"] = static_cast<qint64>(mesh->size());
        auto* verts = mesh->getAssociatedCloud();
        if (verts) o["vertex_count"] = static_cast<qint64>(verts->size());
    }

    if (recursive && obj->getChildrenNumber() > 0) {
        QJsonArray children;
        for (unsigned i = 0; i < obj->getChildrenNumber(); ++i) {
            children.append(entityToJson(obj->getChild(i), true));
        }
        o["children"] = children;
    }
    return o;
}

ccHObject* findEntityById(ccHObject* root, unsigned id) {
    if (!root) return nullptr;
    if (root->getUniqueID() == id) return root;
    for (unsigned i = 0; i < root->getChildrenNumber(); ++i) {
        auto* found = findEntityById(root->getChild(i), id);
        if (found) return found;
    }
    return nullptr;
}

CC_VIEW_ORIENTATION viewOrientationFromString(const QString& s) {
    static const QMap<QString, CC_VIEW_ORIENTATION> map = {
            {"top", CC_TOP_VIEW},     {"bottom", CC_BOTTOM_VIEW},
            {"front", CC_FRONT_VIEW}, {"back", CC_BACK_VIEW},
            {"left", CC_LEFT_VIEW},   {"right", CC_RIGHT_VIEW},
            {"iso1", CC_ISO_VIEW_1},  {"iso2", CC_ISO_VIEW_2},
    };
    return map.value(s.toLower(), CC_FRONT_VIEW);
}

}  // namespace

// ---------------------------------------------------------------------------
// Plugin lifecycle
// ---------------------------------------------------------------------------

JsonRPCPlugin::JsonRPCPlugin(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/JsonRPCPlugin/info.json") {
    connect(&rpc_server, &JsonRPCServer::execute, this,
            &JsonRPCPlugin::execute);
}

QList<QAction*> JsonRPCPlugin::getActions() {
    CVLog::PrintDebug("JsonRPCPlugin::getActions");
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());
        m_action->setCheckable(true);
        m_action->setChecked(false);
        m_action->setEnabled(true);
        connect(m_action, &QAction::triggered, this, &JsonRPCPlugin::triggered);
    }
    return {m_action};
}

void JsonRPCPlugin::triggered(bool checked) {
    CVLog::Print(QString("JsonRPCPlugin::triggered: checked(%1)").arg(checked));
    if (checked) {
        rpc_server.listen(6001);
    } else {
        rpc_server.close();
    }
}

// ---------------------------------------------------------------------------
// RPC method dispatch
// ---------------------------------------------------------------------------

JsonRPCResult JsonRPCPlugin::execute(QString method,
                                     QMap<QString, QVariant> params) {
    QStringList paramParts;
    for (auto it = params.constBegin(); it != params.constEnd(); ++it) {
        const QVariant& v = it.value();
        QString typeTag;
        switch (v.type()) {
            case QVariant::Int:
                typeTag = "int";
                break;
            case QVariant::LongLong:
                typeTag = "int64";
                break;
            case QVariant::Double:
                typeTag = "double";
                break;
            case QVariant::Bool:
                typeTag = "bool";
                break;
            case QVariant::String:
                typeTag = "string";
                break;
            case QVariant::List:
                typeTag = "list";
                break;
            case QVariant::Map:
                typeTag = "map";
                break;
            default:
                typeTag = v.typeName();
                break;
        }
        paramParts << QString("  %1 [%2] = %3")
                              .arg(it.key(), typeTag, v.toString());
    }
    QString paramStr = paramParts.isEmpty()
                               ? QStringLiteral("(none)")
                               : QString("{\n%1\n}").arg(paramParts.join("\n"));
    CVLog::Print(QString("[JsonRPC] execute  method: \"%1\"  params: %2")
                         .arg(method, paramStr));
    if (m_app == nullptr) {
        return JsonRPCResult::error(-32603, "Application not ready");
    }

    // --- File I/O ---
    if (method == "open") return rpcOpen(params);
    if (method == "export") return rpcExport(params);
    if (method == "file.convert") return rpcFileConvert(params);

    // --- Scene tree ---
    if (method == "scene.list") return rpcSceneList(params);
    if (method == "scene.info") return rpcSceneInfo(params);
    if (method == "scene.remove") return rpcSceneRemove(params);
    if (method == "scene.setVisible") return rpcSceneSetVisible(params);
    if (method == "scene.select") return rpcSceneSelect(params);
    if (method == "clear") return rpcClear(params);

    // --- View control ---
    if (method == "view.setOrientation") return rpcViewSetOrientation(params);
    if (method == "view.zoomFit") return rpcViewZoomFit(params);
    if (method == "view.refresh") return rpcViewRefresh(params);
    if (method == "view.setPerspective") return rpcViewSetPerspective(params);
    if (method == "view.setPointSize") return rpcViewSetPointSize(params);

    // --- Transform ---
    if (method == "transform.apply") return rpcTransformApply(params);

    // --- Entity properties ---
    if (method == "entity.rename") return rpcEntityRename(params);
    if (method == "entity.setColor") return rpcEntitySetColor(params);

    // --- Cloud colorization ---
    if (method == "cloud.paintUniform") return rpcCloudPaintUniform(params);
    if (method == "cloud.paintByHeight") return rpcCloudPaintByHeight(params);
    if (method == "cloud.paintByScalarField")
        return rpcCloudPaintByScalarField(params);

    // --- Cloud processing (GUI-side) ---
    if (method == "cloud.computeNormals") return rpcCloudComputeNormals(params);
    if (method == "cloud.subsample") return rpcCloudSubsample(params);
    if (method == "cloud.crop") return rpcCloudCrop(params);
    if (method == "cloud.getScalarFields")
        return rpcCloudGetScalarFields(params);

    // --- Mesh processing (GUI-side) ---
    if (method == "mesh.simplify") return rpcMeshSimplify(params);
    if (method == "mesh.smooth") return rpcMeshSmooth(params);
    if (method == "mesh.subdivide") return rpcMeshSubdivide(params);
    if (method == "mesh.samplePoints") return rpcMeshSamplePoints(params);

    // --- View capture ---
    if (method == "view.screenshot") return rpcViewScreenshot(params);
    if (method == "view.getCamera") return rpcViewGetCamera(params);

    // --- Reconstruction (Colmap) ---
    if (method == "colmap.reconstruct") return rpcColmapReconstruct(params);

    // --- Introspection ---
    if (method == "methods.list") return rpcMethodsList(params);
    if (method == "ping") return JsonRPCResult::success("pong");

    return JsonRPCResult::error(-32601, "Method not found: " + method);
}

// ---------------------------------------------------------------------------
// open — load file into DB
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcOpen(const QMap<QString, QVariant>& params) {
    QString filename = params["filename"].toString();
    if (filename.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'filename' parameter");
    }

    CCVector3d loadCoordinatesShift(0, 0, 0);
    bool loadCoordinatesTransEnabled = false;

    FileIOFilter::LoadParameters parameters;
    parameters.alwaysDisplayLoadDialog = !params.contains("silent");
    parameters.shiftHandlingMode = ecvGlobalShiftManager::DIALOG_IF_NECESSARY;
    parameters.coordinatesShift = &loadCoordinatesShift;
    parameters.coordinatesShiftEnabled = &loadCoordinatesTransEnabled;
    parameters.parentWidget = m_app->getActiveWindow();

    CC_FILE_ERROR res = CC_FERR_NO_ERROR;
    ccHObject* newGroup = FileIOFilter::LoadFromFile(
            filename, parameters, res, params["filter"].toString());

    if (!newGroup) {
        return JsonRPCResult::error(1, "Failed to load file: " + filename);
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

// ---------------------------------------------------------------------------
// export — save entity to file
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcExport(const QMap<QString, QVariant>& params) {
    QString filename = params["filename"].toString();
    if (filename.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'filename' parameter");
    }

    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) {
        return JsonRPCResult::error(
                2, "Entity not found: " + QString::number(entityId));
    }

    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;
    saveParams.parentWidget = m_app->getActiveWindow();

    CC_FILE_ERROR err = FileIOFilter::SaveToFile(entity, filename, saveParams,
                                                 params["filter"].toString());

    if (err != CC_FERR_NO_ERROR) {
        return JsonRPCResult::error(
                3, "Export failed with error code " + QString::number(err));
    }
    QJsonObject result;
    result["filename"] = filename;
    result["entity_id"] = static_cast<qint64>(entityId);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ---------------------------------------------------------------------------
// file.convert — load a file and re-export in a different format
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcFileConvert(
        const QMap<QString, QVariant>& params) {
    QString inputFile = params["input"].toString();
    QString outputFile = params["output"].toString();
    if (inputFile.isEmpty() || outputFile.isEmpty()) {
        return JsonRPCResult::error(
                -32602, "Missing 'input' and/or 'output' parameters");
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
        return JsonRPCResult::error(1, "Failed to load: " + inputFile +
                                               " (error " +
                                               QString::number(loadErr) + ")");
    }

    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;

    CC_FILE_ERROR saveErr = CC_FERR_UNKNOWN_FILE;
    QString filterName = params["output_filter"].toString();
    if (!filterName.isEmpty()) {
        saveErr = FileIOFilter::SaveToFile(loaded, outputFile, saveParams,
                                           filterName);
    } else {
        QString ext = QFileInfo(outputFile).suffix().toLower();
        auto filter = FileIOFilter::FindBestFilterForExtension(ext);
        if (filter) {
            saveErr = FileIOFilter::SaveToFile(loaded, outputFile, saveParams,
                                               filter);
        }
    }

    delete loaded;

    if (saveErr != CC_FERR_NO_ERROR) {
        return JsonRPCResult::error(3, "Failed to save: " + outputFile +
                                               " (error " +
                                               QString::number(saveErr) + ")");
    }

    QJsonObject result;
    result["input"] = inputFile;
    result["output"] = outputFile;
    result["status"] = "converted";
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ---------------------------------------------------------------------------
// scene.list — list all entities in the DB tree
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// scene.info — get detailed info for an entity
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcSceneInfo(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) {
        return JsonRPCResult::error(2, "Entity not found");
    }
    QJsonObject info = entityToJson(entity, true);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ---------------------------------------------------------------------------
// scene.remove — remove entity from DB
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcSceneRemove(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) {
        return JsonRPCResult::error(2, "Entity not found");
    }
    m_app->removeFromDB(entity, true);
    redraw();
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// scene.setVisible — toggle entity visibility
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcSceneSetVisible(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    bool visible = params.value("visible", true).toBool();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) {
        return JsonRPCResult::error(2, "Entity not found");
    }
    entity->setEnabled(visible);
    redraw();
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// scene.select — select entities
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcSceneSelect(
        const QMap<QString, QVariant>& params) {
    QList<QVariant> ids = params["entity_ids"].toList();
    for (const auto& v : ids) {
        ccHObject* entity = findEntityById(m_app->dbRootObject(), v.toUInt());
        if (entity) {
            m_app->setSelectedInDB(entity, true);
        }
    }
    m_app->updateUI();
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// clear — remove all entities
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcClear(const QMap<QString, QVariant>&) {
    auto* root = m_app->dbRootObject();
    ccHObject* child;
    while ((child = root->getChild(0)) != nullptr) {
        m_app->removeFromDB(child, true);
    }
    redraw();
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// view.setOrientation — set camera view
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcViewSetOrientation(
        const QMap<QString, QVariant>& params) {
    QString orientation = params.value("orientation", "front").toString();
    m_app->setView(viewOrientationFromString(orientation));
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// view.zoomFit — zoom to fit all or selected
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcViewZoomFit(
        const QMap<QString, QVariant>& params) {
    if (params.contains("entity_id")) {
        unsigned id = params["entity_id"].toUInt();
        ccHObject* entity = findEntityById(m_app->dbRootObject(), id);
        if (entity) {
            m_app->zoomOnEntities(entity);
        }
    } else {
        m_app->setGlobalZoom();
    }
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// view.refresh — force redraw
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcViewRefresh(const QMap<QString, QVariant>&) {
    redraw();
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// view.setPerspective — toggle perspective mode
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcViewSetPerspective(
        const QMap<QString, QVariant>& params) {
    QString mode = params.value("mode", "object").toString();
    if (mode == "viewer") {
        m_app->toggleActiveWindowViewerBasedPerspective();
    } else {
        m_app->toggleActiveWindowCenteredPerspective();
    }
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// view.setPointSize — adjust point display size
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcViewSetPointSize(
        const QMap<QString, QVariant>& params) {
    QString action = params.value("action", "increase").toString();
    if (action == "increase") {
        m_app->increasePointSize();
    } else {
        m_app->decreasePointSize();
    }
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// transform.apply — apply 4x4 transformation matrix
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcTransformApply(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) {
        return JsonRPCResult::error(2, "Entity not found");
    }

    QList<QVariant> matrix = params["matrix"].toList();
    if (matrix.size() != 16) {
        return JsonRPCResult::error(
                -32602, "Matrix must be 16 elements (4x4 column-major)");
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
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// mesh.simplify — decimate a mesh
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcMeshSimplify(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* mesh = ccHObjectCaster::ToMesh(entity);
    if (!mesh) return JsonRPCResult::error(8, "Entity is not a mesh");

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
                -32602, "method must be 'quadric' or 'vertex_clustering'");
    }

    if (!result) return JsonRPCResult::error(5, "Simplification failed");

    ccMesh* simplified = new ccMesh(*result);
    simplified->setName(mesh->getName() + "_simplified");
    m_app->addToDB(simplified);
    redraw();

    QJsonObject info = entityToJson(simplified);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ---------------------------------------------------------------------------
// mesh.smooth — smooth a mesh
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcMeshSmooth(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* mesh = ccHObjectCaster::ToMesh(entity);
    if (!mesh) return JsonRPCResult::error(8, "Entity is not a mesh");

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
                -32602, "method must be 'laplacian', 'taubin', or 'simple'");
    }

    if (!result) return JsonRPCResult::error(5, "Smoothing failed");

    ccMesh* smoothed = new ccMesh(*result);
    smoothed->setName(mesh->getName() + "_smoothed");
    m_app->addToDB(smoothed);
    redraw();

    QJsonObject info = entityToJson(smoothed);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ---------------------------------------------------------------------------
// mesh.subdivide — subdivide a mesh
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcMeshSubdivide(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* mesh = ccHObjectCaster::ToMesh(entity);
    if (!mesh) return JsonRPCResult::error(8, "Entity is not a mesh");

    QString method = params.value("method", "midpoint").toString();
    int iterations = params.value("iterations", 1).toInt();

    std::shared_ptr<ccMesh> result;
    if (method == "midpoint") {
        result = mesh->SubdivideMidpoint(iterations);
    } else if (method == "loop") {
        result = mesh->SubdivideLoop(iterations);
    } else {
        return JsonRPCResult::error(-32602,
                                    "method must be 'midpoint' or 'loop'");
    }

    if (!result) return JsonRPCResult::error(5, "Subdivision failed");

    ccMesh* subdivided = new ccMesh(*result);
    subdivided->setName(mesh->getName() + "_subdivided");
    m_app->addToDB(subdivided);
    redraw();

    QJsonObject info = entityToJson(subdivided);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ---------------------------------------------------------------------------
// mesh.samplePoints — sample points from a mesh surface
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcMeshSamplePoints(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* mesh = ccHObjectCaster::ToMesh(entity);
    if (!mesh) return JsonRPCResult::error(8, "Entity is not a mesh");

    QString method = params.value("method", "uniform").toString();
    unsigned count = params.value("count", 100000).toUInt();

    std::shared_ptr<ccPointCloud> cloud;
    if (method == "uniform") {
        cloud = mesh->SamplePointsUniformly(count);
    } else if (method == "poisson_disk") {
        cloud = mesh->SamplePointsPoissonDisk(count);
    } else {
        return JsonRPCResult::error(
                -32602, "method must be 'uniform' or 'poisson_disk'");
    }

    if (!cloud) return JsonRPCResult::error(5, "Sampling failed");

    ccPointCloud* sampled = new ccPointCloud(*cloud);
    sampled->setName(mesh->getName() + "_sampled");
    m_app->addToDB(sampled);
    redraw();

    QJsonObject info = entityToJson(sampled);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ---------------------------------------------------------------------------
// entity.rename — rename an entity
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcEntityRename(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    QString name = params["name"].toString();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");
    if (name.isEmpty()) return JsonRPCResult::error(-32602, "Missing 'name'");
    entity->setName(name);
    m_app->updateUI();
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// entity.setColor — set entity display color
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcEntitySetColor(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    int r = params.value("r", 255).toInt();
    int g = params.value("g", 255).toInt();
    int b = params.value("b", 255).toInt();
    entity->setTempColor(ecvColor::Rgb(static_cast<ColorCompType>(r),
                                       static_cast<ColorCompType>(g),
                                       static_cast<ColorCompType>(b)));
    entity->enableTempColor(true);
    redraw();
    return JsonRPCResult::success(0);
}

// ---------------------------------------------------------------------------
// cloud.computeNormals — estimate normals for a point cloud
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcCloudComputeNormals(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* cloud = ccHObjectCaster::ToPointCloud(entity);
    if (!cloud) return JsonRPCResult::error(4, "Entity is not a point cloud");

    double radius = params.value("radius", 0.0).toDouble();
    bool success = cloud->computeNormalsWithOctree(
            LS, ccNormalVectors::UNDEFINED,
            static_cast<PointCoordinateType>(radius));

    if (!success) return JsonRPCResult::error(5, "Normal computation failed");

    cloud->showNormals(true);
    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["has_normals"] = cloud->hasNormals();
    result["point_count"] = static_cast<qint64>(cloud->size());
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ---------------------------------------------------------------------------
// cloud.subsample — subsample a point cloud
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcCloudSubsample(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* cloud = ccHObjectCaster::ToPointCloud(entity);
    if (!cloud) return JsonRPCResult::error(4, "Entity is not a point cloud");

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
    }

    if (!subsampled) return JsonRPCResult::error(5, "Subsampling failed");

    subsampled->setName(cloud->getName() + "_subsampled");
    m_app->addToDB(subsampled);
    redraw();

    QJsonObject result = entityToJson(subsampled);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ---------------------------------------------------------------------------
// cloud.crop — crop a point cloud by bounding box
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcCloudCrop(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* cloud = ccHObjectCaster::ToPointCloud(entity);
    if (!cloud) return JsonRPCResult::error(4, "Entity is not a point cloud");

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
        return JsonRPCResult::error(5, "Crop produced empty result");
    }

    ccPointCloud* cropped = cloud->partialClone(refCloud);
    delete refCloud;
    if (!cropped) return JsonRPCResult::error(5, "Crop clone failed");

    cropped->setName(cloud->getName() + "_cropped");
    m_app->addToDB(cropped);
    redraw();

    QJsonObject result = entityToJson(cropped);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ---------------------------------------------------------------------------
// cloud.getScalarFields — list scalar fields on a point cloud
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcCloudGetScalarFields(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* cloud = ccHObjectCaster::ToPointCloud(entity);
    if (!cloud) return JsonRPCResult::error(4, "Entity is not a point cloud");

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

// ---------------------------------------------------------------------------
// view.screenshot — capture the active viewport to an image file
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcViewScreenshot(
        const QMap<QString, QVariant>& params) {
    QString filename = params["filename"].toString();
    if (filename.isEmpty())
        return JsonRPCResult::error(-32602, "Missing 'filename'");

    QWidget* win = m_app->getActiveWindow();
    if (!win) return JsonRPCResult::error(6, "No active window");

    QPixmap pixmap = win->grab();
    bool saved = pixmap.save(filename);
    if (!saved)
        return JsonRPCResult::error(7,
                                    "Failed to save screenshot: " + filename);

    QJsonObject result;
    result["filename"] = filename;
    result["width"] = pixmap.width();
    result["height"] = pixmap.height();
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ---------------------------------------------------------------------------
// view.getCamera — get current camera parameters
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcViewGetCamera(const QMap<QString, QVariant>&) {
    ccGLMatrixd viewMat = ecvDisplayTools::GetViewportParameters().viewMat;
    QJsonArray mat;
    const double* data = viewMat.data();
    for (int i = 0; i < 16; ++i) mat.append(data[i]);

    auto vp = ecvDisplayTools::GetViewportParameters();
    QJsonObject result;
    result["view_matrix"] = mat;
    result["fov_deg"] = vp.fov_deg;
    result["perspective"] = vp.perspectiveView;
    result["object_centered"] = vp.objectCenteredView;
    result["near_clipping"] = vp.zNear;
    result["far_clipping"] = vp.zFar;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ---------------------------------------------------------------------------
// cloud.paintUniform — set all points to a single RGB color
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcCloudPaintUniform(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* cloud = ccHObjectCaster::ToPointCloud(entity);
    if (!cloud) return JsonRPCResult::error(4, "Entity is not a point cloud");

    int r = params.value("r", 255).toInt();
    int g = params.value("g", 255).toInt();
    int b = params.value("b", 255).toInt();
    ecvColor::Rgb color(static_cast<ColorCompType>(r),
                        static_cast<ColorCompType>(g),
                        static_cast<ColorCompType>(b));

    if (!cloud->resizeTheRGBTable(false))
        return JsonRPCResult::error(5, "Failed to allocate color array");

    for (unsigned i = 0; i < cloud->size(); ++i) {
        cloud->setPointColor(i, color);
    }
    cloud->showColors(true);
    redraw();

    QJsonObject info;
    info["entity_id"] = static_cast<qint64>(entityId);
    info["points_colored"] = static_cast<qint64>(cloud->size());
    info["color"] = QJsonArray{r, g, b};
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ---------------------------------------------------------------------------
// cloud.paintByHeight — color points by coordinate gradient
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcCloudPaintByHeight(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* cloud = ccHObjectCaster::ToPointCloud(entity);
    if (!cloud) return JsonRPCResult::error(4, "Entity is not a point cloud");
    if (cloud->size() == 0)
        return JsonRPCResult::error(4, "Point cloud is empty");

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
        return JsonRPCResult::error(5, "Failed to allocate color array");

    PointCoordinateType range = maxVal - minVal;
    if (range < 1e-10) range = 1;

    for (unsigned i = 0; i < cloud->size(); ++i) {
        float t = static_cast<float>((cloud->getPoint(i)->u[axisIdx] - minVal) /
                                     range);
        ColorCompType r =
                static_cast<ColorCompType>(std::min(255.0f, t * 2.0f * 255.0f));
        ColorCompType g = static_cast<ColorCompType>(
                std::min(255.0f, (1.0f - std::abs(t - 0.5f) * 2.0f) * 255.0f));
        ColorCompType b = static_cast<ColorCompType>(
                std::min(255.0f, (1.0f - t) * 2.0f * 255.0f));
        cloud->setPointColor(i, ecvColor::Rgb(r, g, b));
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

// ---------------------------------------------------------------------------
// cloud.paintByScalarField — color by an existing scalar field with ramp
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcCloudPaintByScalarField(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    ccHObject* entity = findEntityById(m_app->dbRootObject(), entityId);
    if (!entity) return JsonRPCResult::error(2, "Entity not found");

    auto* cloud = ccHObjectCaster::ToPointCloud(entity);
    if (!cloud) return JsonRPCResult::error(4, "Entity is not a point cloud");

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
        sfIdx >= static_cast<int>(cloud->getNumberOfScalarFields()))
        return JsonRPCResult::error(6, "Scalar field not found");

    cloud->setCurrentDisplayedScalarField(sfIdx);
    cloud->showSF(true);
    redraw();

    cloudViewer::ScalarField* sf = cloud->getScalarField(sfIdx);
    QJsonObject info;
    info["entity_id"] = static_cast<qint64>(entityId);
    info["field_name"] = sf->getName();
    info["field_index"] = sfIdx;
    ScalarType minVal, maxVal;
    sf->computeMinAndMax();
    minVal = sf->getMin();
    maxVal = sf->getMax();
    info["min"] = static_cast<double>(minVal);
    info["max"] = static_cast<double>(maxVal);
    return JsonRPCResult::success(QJsonDocument(info).toVariant());
}

// ---------------------------------------------------------------------------
// colmap.reconstruct — launch Colmap automatic_reconstructor as subprocess
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcColmapReconstruct(
        const QMap<QString, QVariant>& params) {
    QString imagePath = params.value("image_path").toString();
    QString workspace = params.value("workspace_path").toString();
    if (imagePath.isEmpty() || workspace.isEmpty()) {
        return JsonRPCResult::error(
                -32602,
                "Missing 'image_path' and/or 'workspace_path' parameters");
    }

    QString colmapBin = params.value("colmap_binary", "colmap").toString();
    QString quality = params.value("quality", "HIGH").toString();
    QString dataType = params.value("data_type", "INDIVIDUAL").toString();
    QString mesher = params.value("mesher", "POISSON").toString();
    bool useGpu = params.value("use_gpu", true).toBool();

    QDir().mkpath(workspace);

    QStringList args;
    args << "automatic_reconstructor"
         << "--workspace_path" << workspace << "--image_path" << imagePath
         << "--quality" << quality << "--data_type" << dataType << "--mesher"
         << mesher;
    if (!useGpu) {
        args << "--use_gpu" << "0";
    }

    QProcess process;
    process.setProgram(colmapBin);
    process.setArguments(args);
    process.start();

    int timeoutMs = params.value("timeout_ms", 7200000).toInt();
    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        return JsonRPCResult::error(
                3,
                "Colmap timed out after " + QString::number(timeoutMs) + "ms");
    }

    if (process.exitCode() != 0) {
        QString err = process.readAllStandardError().trimmed();
        if (err.isEmpty()) err = process.readAllStandardOutput().trimmed();
        return JsonRPCResult::error(
                3, "Colmap failed (exit " +
                           QString::number(process.exitCode()) +
                           "): " + err.left(500));
    }

    QJsonObject result;
    result["workspace"] = workspace;
    result["image_path"] = imagePath;
    result["quality"] = quality;
    result["status"] = "completed";

    QString fusedPly = workspace + "/dense/0/fused.ply";
    if (QFile::exists(fusedPly)) {
        result["fused_ply"] = fusedPly;
    }

    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ---------------------------------------------------------------------------
// methods.list — introspection: return all supported RPC methods
// ---------------------------------------------------------------------------
JsonRPCResult JsonRPCPlugin::rpcMethodsList(const QMap<QString, QVariant>&) {
    QJsonArray methods;
    auto add = [&](const QString& name, const QString& desc) {
        QJsonObject m;
        m["method"] = name;
        m["description"] = desc;
        methods.append(m);
    };
    add("ping", "Health check, returns 'pong'");
    add("open", "Load file: {filename, ?silent, ?filter, ?transformation}");
    add("export", "Export entity: {entity_id, filename, ?filter}");
    add("file.convert",
        "Convert format: {input, output, ?input_filter, ?output_filter}");
    add("clear", "Remove all entities from scene");
    add("scene.list", "List entities: {?recursive}");
    add("scene.info", "Entity details: {entity_id}");
    add("scene.remove", "Remove entity: {entity_id}");
    add("scene.setVisible", "Toggle visibility: {entity_id, visible}");
    add("scene.select", "Select entities: {entity_ids}");
    add("entity.rename", "Rename entity: {entity_id, name}");
    add("entity.setColor", "Set display color: {entity_id, r, g, b}");
    add("cloud.paintUniform", "Paint all points: {entity_id, r, g, b}");
    add("cloud.paintByHeight", "Color by height gradient: {entity_id, ?axis}");
    add("cloud.paintByScalarField",
        "Color by scalar field: {entity_id, ?field_name, ?field_index}");
    add("cloud.computeNormals", "Estimate normals: {entity_id, ?radius}");
    add("cloud.subsample",
        "Subsample cloud: {entity_id, method, ?step, ?count}");
    add("cloud.crop", "Crop by bbox: {entity_id, min_x..max_z}");
    add("cloud.getScalarFields", "List scalar fields: {entity_id}");
    add("view.setOrientation",
        "Set view: {orientation: top|bottom|front|back|left|right|iso1|iso2}");
    add("view.zoomFit", "Zoom to fit: {?entity_id}");
    add("view.refresh", "Force redraw");
    add("view.setPerspective", "Perspective mode: {mode: object|viewer}");
    add("view.setPointSize", "Point size: {action: increase|decrease}");
    add("mesh.simplify",
        "Simplify mesh: {entity_id, method, ?target_triangles, ?voxel_size}");
    add("mesh.smooth",
        "Smooth mesh: {entity_id, method, ?iterations, ?lambda, ?mu}");
    add("mesh.subdivide", "Subdivide mesh: {entity_id, method, ?iterations}");
    add("mesh.samplePoints",
        "Sample points from mesh: {entity_id, method, ?count}");
    add("view.screenshot", "Capture viewport: {filename}");
    add("view.getCamera", "Get camera parameters");
    add("transform.apply", "Apply 4x4 matrix: {entity_id, matrix[16]}");
    add("colmap.reconstruct",
        "Run Colmap automatic_reconstructor: {image_path, workspace_path, "
        "?quality, ?data_type, ?mesher, ?use_gpu, ?colmap_binary, "
        "?timeout_ms}");
    add("methods.list", "List available RPC methods");
    return JsonRPCResult::success(QJsonDocument(methods).toVariant());
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------
void JsonRPCPlugin::redraw() {
    QWidget* win = m_app->getActiveWindow();
    if (win) {
        ecvDisplayTools::RedrawDisplay();
    }
}
