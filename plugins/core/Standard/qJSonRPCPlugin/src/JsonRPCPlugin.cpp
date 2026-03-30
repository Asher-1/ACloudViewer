// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "JsonRPCPlugin.h"

#include <AutoSegmentationTools.h>
#include <CVConst.h>
#include <CloudSamplingTools.h>
#include <Delaunay2dMesh.h>
#include <DgmOctree.h>
#include <DistanceComputationTools.h>
#include <FileIOFilter.h>
#include <GeometricalAnalysisTools.h>
#include <Neighbourhood.h>
#include <ReferenceCloud.h>
#include <ScalarFieldTools.h>
#include <ecvColorScalesManager.h>
#include <ecvDisplayTools.h>
#include <ecvGenericPointCloud.h>
#include <ecvMainAppInterface.h>
#include <ecvMesh.h>
#include <ecvNormalVectors.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#ifdef HAS_PCV_PLUGIN
#include <PCV.h>
#include <ecvHObjectCaster.h>
#endif

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QtGui>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

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

QJsonObject D() { return QJsonObject(); }
QJsonObject D(const QString& k, const QJsonValue& v) {
    QJsonObject o;
    o[k] = v;
    return o;
}
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2) {
    QJsonObject o;
    o[k1] = v1;
    o[k2] = v2;
    return o;
}
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3) {
    QJsonObject o;
    o[k1] = v1;
    o[k2] = v2;
    o[k3] = v3;
    return o;
}
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3,
              const QString& k4,
              const QJsonValue& v4) {
    QJsonObject o;
    o[k1] = v1;
    o[k2] = v2;
    o[k3] = v3;
    o[k4] = v4;
    return o;
}
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3,
              const QString& k4,
              const QJsonValue& v4,
              const QString& k5,
              const QJsonValue& v5) {
    QJsonObject o;
    o[k1] = v1;
    o[k2] = v2;
    o[k3] = v3;
    o[k4] = v4;
    o[k5] = v5;
    return o;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Logging helpers
// ═══════════════════════════════════════════════════════════════════════════

static QString variantToLogString(const QVariant& v, int depth = 0) {
    if (v.isNull() || !v.isValid()) return QStringLiteral("null");

    switch (static_cast<int>(v.type())) {
        case QVariant::Bool:
            return v.toBool() ? QStringLiteral("true")
                              : QStringLiteral("false");
        case QVariant::Int:
            return QString::number(v.toInt());
        case QVariant::UInt:
            return QString::number(v.toUInt());
        case QVariant::LongLong:
            return QString::number(v.toLongLong());
        case QVariant::ULongLong:
            return QString::number(v.toULongLong());
        case QVariant::Double: {
            double d = v.toDouble();
            if (d == static_cast<long long>(d) && std::abs(d) < 1e15)
                return QString::number(static_cast<long long>(d));
            return QString::number(d, 'g', 10);
        }
        case QVariant::String:
            return QStringLiteral("\"%1\"").arg(v.toString());
        case QVariant::List: {
            if (depth > 5)
                return QStringLiteral("[...]");  // Increased from 3 to 5
            const QVariantList list = v.toList();
            if (list.isEmpty()) return QStringLiteral("[]");
            if (list.size() > 100)  // Increased from 20 to 100
                return QStringLiteral("[%1 items]").arg(list.size());
            QStringList items;
            items.reserve(list.size());
            for (const QVariant& item : list)
                items << variantToLogString(item, depth + 1);
            return QStringLiteral("[%1]").arg(items.join(", "));
        }
        case QVariant::Map: {
            if (depth > 5)
                return QStringLiteral("{...}");  // Increased from 3 to 5
            const QVariantMap map = v.toMap();
            if (map.isEmpty()) return QStringLiteral("{}");
            QStringList items;
            for (auto it = map.constBegin(); it != map.constEnd(); ++it)
                items << QStringLiteral("%1: %2").arg(
                        it.key(), variantToLogString(it.value(), depth + 1));
            return QStringLiteral("{%1}").arg(items.join(", "));
        }
        case QVariant::StringList: {
            const QStringList sl = v.toStringList();
            if (sl.isEmpty()) return QStringLiteral("[]");
            QStringList quoted;
            quoted.reserve(sl.size());
            for (const QString& s : sl)
                quoted << QStringLiteral("\"%1\"").arg(s);
            return QStringLiteral("[%1]").arg(quoted.join(", "));
        }
        case QVariant::ByteArray:
            return QStringLiteral("<bytes:%1>").arg(v.toByteArray().size());
        case QVariant::Char:
            return QStringLiteral("'%1'").arg(v.toChar());
        default: {
            QString s = v.toString();
            if (!s.isEmpty()) return QStringLiteral("\"%1\"").arg(s);
            return QStringLiteral("<%1>").arg(v.typeName());
        }
    }
}

static QString variantTypeTag(const QVariant& v) {
    if (v.isNull() || !v.isValid()) return QStringLiteral("null");
    switch (static_cast<int>(v.type())) {
        case QVariant::Bool:
            return QStringLiteral("bool");
        case QVariant::Int:
            return QStringLiteral("int");
        case QVariant::UInt:
            return QStringLiteral("uint");
        case QVariant::LongLong:
            return QStringLiteral("int64");
        case QVariant::ULongLong:
            return QStringLiteral("uint64");
        case QVariant::Double: {
            double d = v.toDouble();
            if (d == static_cast<long long>(d) && std::abs(d) < 1e15)
                return QStringLiteral("int");
            return QStringLiteral("double");
        }
        case QVariant::String:
            return QStringLiteral("string");
        case QVariant::List:
            return QStringLiteral("list");
        case QVariant::Map:
            return QStringLiteral("map");
        case QVariant::StringList:
            return QStringLiteral("stringlist");
        case QVariant::ByteArray:
            return QStringLiteral("bytes");
        case QVariant::Char:
            return QStringLiteral("char");
        default:
            return QString::fromLatin1(v.typeName());
    }
}

static constexpr const char* kLogPrefix = "      | ";
static constexpr const char* kLogIndent = "      |   ";

static int measureMaxCol(const QVariant& v,
                         int prefixLen,
                         int depth,
                         int maxDepth) {
    if (v.isNull() || !v.isValid() || depth >= maxDepth) return 0;
    const int vtype = static_cast<int>(v.type());
    int best = 0;

    if (vtype == QVariant::Map) {
        const QVariantMap map = v.toMap();
        for (auto it = map.constBegin(); it != map.constEnd(); ++it) {
            int col = prefixLen + it.key().length() + 1;
            best = qMax(best, col);
            const int ct = static_cast<int>(it.value().type());
            if (ct == QVariant::Map || ct == QVariant::List) {
                best = qMax(best, measureMaxCol(it.value(), prefixLen + 2,
                                                depth + 1, maxDepth));
            }
        }
    } else if (vtype == QVariant::List || vtype == QVariant::StringList) {
        const QVariantList list = v.toList();
        const int show = qMin(list.size(), 50);
        for (int i = 0; i < show; ++i) {
            const int ct = static_cast<int>(list[i].type());
            if (ct == QVariant::Map || ct == QVariant::List) {
                best = qMax(best, measureMaxCol(list[i], prefixLen + 2,
                                                depth + 1, maxDepth));
            }
        }
    }
    return best;
}

static void prettyAppend(QStringList& out,
                         const QVariant& v,
                         const QString& prefix,
                         int depth,
                         int maxDepth,
                         int globalValCol) {
    if (v.isNull() || !v.isValid()) return;
    if (depth >= maxDepth) {
        out << prefix + variantToLogString(v, depth);
        return;
    }

    const int vtype = static_cast<int>(v.type());

    if (vtype == QVariant::Map) {
        const QVariantMap map = v.toMap();
        if (map.isEmpty()) return;

        const int localPadTo = qMax(0, globalValCol - prefix.length());
        const QString child = prefix + QStringLiteral("  ");
        for (auto it = map.constBegin(); it != map.constEnd(); ++it) {
            const int ct = static_cast<int>(it.value().type());
            const bool isMapOrList =
                    (ct == QVariant::Map || ct == QVariant::List);

            if (isMapOrList && depth + 1 < maxDepth) {
                out << prefix + it.key() + QStringLiteral(":");
                prettyAppend(out, it.value(), child, depth + 1, maxDepth,
                             globalValCol);
            } else {
                QString keyColon =
                        QString(it.key() + QStringLiteral(":"))
                                .leftJustified(localPadTo);
                out << prefix + keyColon + QStringLiteral(" ") +
                                variantToLogString(it.value(), depth + 1);
            }
        }
    } else if (vtype == QVariant::List || vtype == QVariant::StringList) {
        const QVariantList list = v.toList();
        if (list.isEmpty()) return;

        bool allSimple = true;
        for (const QVariant& item : list) {
            const int t = static_cast<int>(item.type());
            if (t == QVariant::Map || t == QVariant::List) {
                allSimple = false;
                break;
            }
        }
        if (allSimple) {
            QString compact = variantToLogString(v, depth);
            if (compact.length() <= 200) {
                out << prefix + compact;
                return;
            }
        }

        const int show = qMin(list.size(), 50);
        if (list.size() > 50)
            out << prefix + QStringLiteral("(%1 items, first 50)")
                                    .arg(list.size());

        const QString child = prefix + QStringLiteral("  ");
        for (int i = 0; i < show; ++i) {
            const int ct = static_cast<int>(list[i].type());
            const bool nested = (ct == QVariant::Map || ct == QVariant::List);
            if (nested && depth + 1 < maxDepth) {
                out << prefix + QStringLiteral("[%1]:").arg(i);
                prettyAppend(out, list[i], child, depth + 1, maxDepth,
                             globalValCol);
            } else {
                out << prefix + QStringLiteral("[%1]: ").arg(i) +
                                variantToLogString(list[i], depth + 1);
            }
        }
        if (list.size() > 50)
            out << prefix + QStringLiteral("...");
    } else {
        out << prefix + variantToLogString(v, depth);
    }
}

static void prettyFormatLines(QStringList& out, const QVariant& v) {
    const int vtype = static_cast<int>(v.type());
    if (vtype != QVariant::Map && vtype != QVariant::List &&
        vtype != QVariant::StringList) {
        return;
    }
    const int baseLen = static_cast<int>(qstrlen(kLogIndent));
    int valCol = measureMaxCol(v, baseLen, 0, 6) + 1;
    prettyAppend(out, v, QLatin1String(kLogIndent), 0, 6, valCol);
}

// ═══════════════════════════════════════════════════════════════════════════
// Plugin lifecycle
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCPlugin::JsonRPCPlugin(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/JsonRPCPlugin/info.json") {
    connect(&rpc_server, &JsonRPCServer::execute, this,
            &JsonRPCPlugin::execute);
    registerMethods();
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
        CVLog::Print(QString("[JsonRPC] Server listening on port 6001 "
                             "(%1 methods registered)")
                             .arg(m_methods.size()));
    } else {
        rpc_server.close();
        CVLog::Print("[JsonRPC] Server stopped");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Method registry
// ═══════════════════════════════════════════════════════════════════════════

void JsonRPCPlugin::reg(
        const QString& name,
        const QString& desc,
        std::function<JsonRPCResult(const QMap<QString, QVariant>&)> fn) {
    m_methods.insert(name, {desc, std::move(fn)});
}

void JsonRPCPlugin::registerMethods() {
    // clang-format off
    // --- Connectivity ---
    reg("ping",              "Health check, returns 'pong'",
        [](const QMap<QString,QVariant>&){ return JsonRPCResult::success("pong"); });

    // --- File I/O ---
    reg("open",              "Load file into scene: {filename, ?filter, ?transformation}",
        [this](auto& p){ return rpcOpen(p); });
    reg("export",            "Export entity to file: {entity_id, filename, ?filter}",
        [this](auto& p){ return rpcExport(p); });
    reg("file.convert",      "Convert file format: {input, output, ?input_filter, ?output_filter}",
        [this](auto& p){ return rpcFileConvert(p); });

    // --- Scene tree ---
    reg("scene.list",        "List all entities: {?recursive}",
        [this](auto& p){ return rpcSceneList(p); });
    reg("scene.info",        "Get entity details: {entity_id}",
        [this](auto& p){ return rpcSceneInfo(p); });
    reg("scene.remove",      "Remove entity: {entity_id}",
        [this](auto& p){ return rpcSceneRemove(p); });
    reg("scene.setVisible",  "Toggle visibility: {entity_id, visible}",
        [this](auto& p){ return rpcSceneSetVisible(p); });
    reg("scene.select",      "Select entities: {entity_ids}",
        [this](auto& p){ return rpcSceneSelect(p); });
    reg("clear",             "Remove all entities from scene",
        [this](auto& p){ return rpcClear(p); });

    // --- View control ---
    reg("view.setOrientation", "Set camera view: {orientation: top|bottom|front|back|left|right|iso1|iso2}",
        [this](auto& p){ return rpcViewSetOrientation(p); });
    reg("view.zoomFit",      "Zoom to fit: {?entity_id}",
        [this](auto& p){ return rpcViewZoomFit(p); });
    reg("view.refresh",      "Force redraw",
        [this](auto& p){ return rpcViewRefresh(p); });
    reg("view.setPerspective","Perspective mode: {mode: object|viewer}",
        [this](auto& p){ return rpcViewSetPerspective(p); });
    reg("view.setPointSize", "Adjust point size: {action: increase|decrease}",
        [this](auto& p){ return rpcViewSetPointSize(p); });
    reg("view.screenshot",   "Capture viewport: {filename}",
        [this](auto& p){ return rpcViewScreenshot(p); });
    reg("view.getCamera",    "Get camera parameters",
        [this](auto& p){ return rpcViewGetCamera(p); });

    // --- Transform ---
    reg("transform.apply",   "Apply 4x4 matrix: {entity_id, matrix[16]}",
        [this](auto& p){ return rpcTransformApply(p); });

    // --- Entity properties ---
    reg("entity.rename",     "Rename entity: {entity_id, name}",
        [this](auto& p){ return rpcEntityRename(p); });
    reg("entity.setColor",   "Set display color: {entity_id, r, g, b}",
        [this](auto& p){ return rpcEntitySetColor(p); });

    // --- Cloud colorization ---
    reg("cloud.paintUniform","Paint all points: {entity_id, r, g, b}",
        [this](auto& p){ return rpcCloudPaintUniform(p); });
    reg("cloud.paintByHeight","Color by height gradient: {entity_id, ?axis}",
        [this](auto& p){ return rpcCloudPaintByHeight(p); });
    reg("cloud.paintByScalarField","Color by scalar field: {entity_id, ?field_name, ?field_index}",
        [this](auto& p){ return rpcCloudPaintByScalarField(p); });

    // --- Cloud processing ---
    reg("cloud.computeNormals","Estimate normals: {entity_id, ?radius}",
        [this](auto& p){ return rpcCloudComputeNormals(p); });
    reg("cloud.subsample",   "Subsample: {entity_id, method, ?step, ?count}",
        [this](auto& p){ return rpcCloudSubsample(p); });
    reg("cloud.crop",        "Crop by bbox: {entity_id, min_x..max_z}",
        [this](auto& p){ return rpcCloudCrop(p); });
    reg("cloud.getScalarFields","List scalar fields: {entity_id}",
        [this](auto& p){ return rpcCloudGetScalarFields(p); });

    // --- Cloud scalar-field management (NEW) ---
    reg("cloud.setActiveSf", "Set active scalar field: {entity_id, field_index | field_name}",
        [this](auto& p){ return rpcCloudSetActiveSf(p); });
    reg("cloud.removeSf",    "Remove one scalar field: {entity_id, field_index | field_name}",
        [this](auto& p){ return rpcCloudRemoveSf(p); });
    reg("cloud.removeAllSfs","Remove all scalar fields: {entity_id}",
        [this](auto& p){ return rpcCloudRemoveAllSfs(p); });
    reg("cloud.renameSf",    "Rename a scalar field: {entity_id, field_index | old_name, new_name}",
        [this](auto& p){ return rpcCloudRenameSf(p); });
    reg("cloud.filterSf",    "Filter by scalar field range: {entity_id, field_index | field_name, min, max}",
        [this](auto& p){ return rpcCloudFilterSf(p); });
    reg("cloud.coordToSF",   "Coordinate to scalar field: {entity_id, dimension: x|y|z}",
        [this](auto& p){ return rpcCloudCoordToSf(p); });

    // --- Cloud geometry (NEW) ---
    reg("cloud.removeRgb",   "Remove color data: {entity_id}",
        [this](auto& p){ return rpcCloudRemoveRgb(p); });
    reg("cloud.removeNormals","Remove normals: {entity_id}",
        [this](auto& p){ return rpcCloudRemoveNormals(p); });
    reg("cloud.invertNormals","Invert normal directions: {entity_id}",
        [this](auto& p){ return rpcCloudInvertNormals(p); });
    reg("cloud.merge",       "Merge multiple clouds: {entity_ids}",
        [this](auto& p){ return rpcCloudMerge(p); });

    // --- Cloud geometric analysis (NEW) ---
    reg("cloud.density",     "Compute local density: {entity_id, radius}",
        [this](auto& p){ return rpcCloudDensity(p); });
    reg("cloud.curvature",   "Compute surface curvature: {entity_id, type, radius}",
        [this](auto& p){ return rpcCloudCurvature(p); });
    reg("cloud.roughness",   "Compute surface roughness: {entity_id, radius}",
        [this](auto& p){ return rpcCloudRoughness(p); });
    reg("cloud.geometricFeature", "Compute geometric features: {entity_id, type, kernel_size}",
        [this](auto& p){ return rpcCloudGeometricFeature(p); });
    reg("cloud.approxDensity","Compute approximate density: {entity_id, density_type}",
        [this](auto& p){ return rpcCloudApproxDensity(p); });
    reg("cloud.colorBanding","Apply color banding: {entity_id, axis, frequency}",
        [this](auto& p){ return rpcCloudColorBanding(p); });
    reg("cloud.sorFilter",   "Statistical outlier removal: {entity_id, knn, sigma}",
        [this](auto& p){ return rpcCloudSorFilter(p); });
    reg("cloud.extractConnectedComponents", "Extract connected components: {entity_id, min_points, octree_level}",
        [this](auto& p){ return rpcCloudExtractConnectedComponents(p); });
    reg("cloud.bestFitPlane","Compute best fit plane: {entity_id, make_horiz}",
        [this](auto& p){ return rpcCloudBestFitPlane(p); });
    reg("cloud.delaunay",    "Delaunay 2.5D triangulation: {entity_id}",
        [this](auto& p){ return rpcCloudDelaunay(p); });

    // --- Cloud scalar field operations (NEW) ---
    reg("cloud.sfArithmetic","Scalar field arithmetic: {entity_id, sf_index, operation}",
        [this](auto& p){ return rpcCloudSfArithmetic(p); });
    reg("cloud.sfOperation", "Scalar field with constant: {entity_id, sf_index, operation, value}",
        [this](auto& p){ return rpcCloudSfOperation(p); });
    reg("cloud.sfGradient",  "Scalar field gradient: {entity_id, sf_index?, radius?}",
        [this](auto& p){ return rpcCloudSfGradient(p); });
    reg("cloud.sfConvertToRGB","Convert SF to RGB: {entity_id, sf_index}",
        [this](auto& p){ return rpcCloudSfConvertToRGB(p); });

    // --- Cloud normals advanced (NEW) ---
    reg("cloud.octreeNormals","Compute normals with octree: {entity_id, radius}",
        [this](auto& p){ return rpcCloudOctreeNormals(p); });
    reg("cloud.orientNormalsMST","Orient normals with MST: {entity_id, knn}",
        [this](auto& p){ return rpcCloudOrientNormalsMST(p); });
    reg("cloud.clearNormals","Clear all normals: {entity_id}",
        [this](auto& p){ return rpcCloudClearNormals(p); });
    reg("cloud.normalsToSFs","Export normals to scalar fields: {entity_id}",
        [this](auto& p){ return rpcCloudNormalsToSFs(p); });
    reg("cloud.normalsToDip","Convert normals to dip/dip-direction: {entity_id}",
        [this](auto& p){ return rpcCloudNormalsToDip(p); });

    // --- Mesh processing ---
    reg("mesh.simplify",     "Simplify mesh: {entity_id, method, ?target_triangles, ?voxel_size}",
        [this](auto& p){ return rpcMeshSimplify(p); });
    reg("mesh.smooth",       "Smooth mesh: {entity_id, method, ?iterations, ?lambda, ?mu}",
        [this](auto& p){ return rpcMeshSmooth(p); });
    reg("mesh.subdivide",    "Subdivide mesh: {entity_id, method, ?iterations}",
        [this](auto& p){ return rpcMeshSubdivide(p); });
    reg("mesh.samplePoints", "Sample points: {entity_id, method, ?count}",
        [this](auto& p){ return rpcMeshSamplePoints(p); });
    reg("mesh.extractVertices","Extract vertices as cloud: {entity_id}",
        [this](auto& p){ return rpcMeshExtractVertices(p); });
    reg("mesh.flipTriangles","Flip triangle winding: {entity_id}",
        [this](auto& p){ return rpcMeshFlipTriangles(p); });
    reg("mesh.volume",       "Compute mesh volume: {entity_id}",
        [this](auto& p){ return rpcMeshVolume(p); });
    reg("mesh.merge",        "Merge multiple meshes: {entity_ids}",
        [this](auto& p){ return rpcMeshMerge(p); });

#ifdef HAS_PCV_PLUGIN
    // --- Ambient Occlusion (PCV/ShadeVis) ---
    reg("process.pcv",       "Compute ambient occlusion (PCV): {entity_id, ?ray_count, ?resolution, ?mode_360, ?mesh_closed}",
        [this](auto& p){ return rpcProcessPcv(p); });
#endif

    // --- Reconstruction ---
    reg("colmap.reconstruct","Run automatic_reconstructor: {image_path, workspace_path, ...}",
        [this](auto& p){ return rpcColmapReconstruct(p); });
    reg("colmap.run",        "Run any COLMAP subcommand: {command, args, ?colmap_binary, ?timeout_ms}",
        [this](auto& p){ return rpcColmapRun(p); });

    // --- Introspection (auto-generated) ---
    reg("methods.list",      "List all available RPC methods with descriptions",
        [this](const QMap<QString,QVariant>&) {
            QJsonArray arr;
            for (auto it = m_methods.constBegin(); it != m_methods.constEnd(); ++it) {
                QJsonObject m;
                m["method"] = it.key();
                m["description"] = it.value().description;
                arr.append(m);
            }
            return JsonRPCResult::success(QJsonDocument(arr).toVariant());
        });
    // clang-format on
}

// ═══════════════════════════════════════════════════════════════════════════
// Entity lookup helpers (with structured error data)
// ═══════════════════════════════════════════════════════════════════════════

ccHObject* JsonRPCPlugin::findEntity(unsigned id) {
    return findEntityById(m_app->dbRootObject(), id);
}

ccPointCloud* JsonRPCPlugin::findCloud(unsigned id, JsonRPCResult& err) {
    ccHObject* e = findEntity(id);
    if (!e) {
        QJsonObject d;
        d["entity_id"] = static_cast<qint64>(id);
        err = JsonRPCResult::error(2, "Entity not found", d);
        return nullptr;
    }
    auto* cloud = ccHObjectCaster::ToPointCloud(e);
    if (!cloud) {
        QJsonObject d;
        d["entity_id"] = static_cast<qint64>(id);
        d["actual_type"] = entityToJson(e)["type"].toString();
        err = JsonRPCResult::error(4, "Entity is not a point cloud", d);
        return nullptr;
    }
    return cloud;
}

ccMesh* JsonRPCPlugin::findMesh(unsigned id, JsonRPCResult& err) {
    ccHObject* e = findEntity(id);
    if (!e) {
        QJsonObject d;
        d["entity_id"] = static_cast<qint64>(id);
        err = JsonRPCResult::error(2, "Entity not found", d);
        return nullptr;
    }
    auto* mesh = ccHObjectCaster::ToMesh(e);
    if (!mesh) {
        QJsonObject d;
        d["entity_id"] = static_cast<qint64>(id);
        d["actual_type"] = entityToJson(e)["type"].toString();
        err = JsonRPCResult::error(8, "Entity is not a mesh", d);
        return nullptr;
    }
    return mesh;
}

// ═══════════════════════════════════════════════════════════════════════════
// Beautiful logging: input & output to both log and GUI console
// ═══════════════════════════════════════════════════════════════════════════

void JsonRPCPlugin::logRequest(const QString& method,
                               const QMap<QString, QVariant>& params) {
    if (method == "ping") {
        CVLog::PrintDebug("[JsonRPC] >> ping");
        return;
    }

    if (params.isEmpty()) {
        CVLog::Print(QString("[JsonRPC] >> %1()").arg(method));
        return;
    }

    int maxKeyLen = 0;
    for (auto it = params.constBegin(); it != params.constEnd(); ++it)
        maxKeyLen = qMax(maxKeyLen, it.key().length());

    const int colonCol = maxKeyLen + 1;
    QStringList lines;
    lines << QStringLiteral("[JsonRPC] >> %1").arg(method);
    for (auto it = params.constBegin(); it != params.constEnd(); ++it) {
        QString valStr = variantToLogString(it.value());
        QString tag = variantTypeTag(it.value());
        QString keyColon =
                QString(it.key() + QStringLiteral(":"))
                        .leftJustified(colonCol);

        if (valStr.length() <= 150) {
            lines << QStringLiteral("%1%2 %3  (%4)")
                             .arg(QLatin1String(kLogPrefix), keyColon, valStr,
                                  tag);
        } else {
            lines << QStringLiteral("%1%2  (%3)")
                             .arg(QLatin1String(kLogPrefix), keyColon, tag);
            const int baseLen = static_cast<int>(qstrlen(kLogIndent));
            int vc = measureMaxCol(it.value(), baseLen, 0, 6) + 1;
            prettyAppend(lines, it.value(), QLatin1String(kLogIndent), 0, 6,
                         vc);
        }
    }
    CVLog::Print(lines.join("\n"));
}

void JsonRPCPlugin::logResponse(const QString& method,
                                const JsonRPCResult& result,
                                qint64 elapsedMs) {
    if (method == "ping") {
        CVLog::PrintDebug("[JsonRPC] << ping OK");
        return;
    }

    auto appendResultBlock = [](QStringList& lines, const QString& label,
                                const QVariant& v) {
        const int vt = static_cast<int>(v.type());
        const bool isComplex =
                (vt == QVariant::Map || vt == QVariant::List ||
                 vt == QVariant::StringList);

        if (isComplex) {
            lines << QStringLiteral("%1%2:")
                             .arg(QLatin1String(kLogPrefix), label);
            prettyFormatLines(lines, v);
        } else {
            lines << QStringLiteral("%1%2: %3")
                             .arg(QLatin1String(kLogPrefix), label,
                                  variantToLogString(v, 0));
        }
    };

    if (result.isError) {
        QStringList lines;
        lines << QStringLiteral("[JsonRPC] << %1 FAILED [%2] (%3ms): %4")
                         .arg(method)
                         .arg(result.error_code)
                         .arg(elapsedMs)
                         .arg(result.error_message);
        if (!result.error_data.isNull() && result.error_data.isValid()) {
            appendResultBlock(lines, QStringLiteral("data"),
                              result.error_data);
        }
        CVLog::Warning(lines.join(QChar('\n')));
    } else {
        QStringList lines;
        lines << QStringLiteral("[JsonRPC] << %1 OK (%2ms)")
                         .arg(method)
                         .arg(elapsedMs);
        appendResultBlock(lines, QStringLiteral("result"), result.result);
        CVLog::Print(lines.join(QChar('\n')));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RPC dispatch (registry-based)
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::execute(QString method,
                                     QMap<QString, QVariant> params) {
    logRequest(method, params);
    QElapsedTimer timer;
    timer.start();

    auto it = m_methods.constFind(method);
    if (it == m_methods.constEnd()) {
        auto result =
                JsonRPCResult::error(-32601, "Method not found: " + method,
                                     D("available_count", m_methods.size()));
        logResponse(method, result, timer.elapsed());
        return result;
    }

    if (method != "ping" && method != "methods.list" && m_app == nullptr) {
        auto result = JsonRPCResult::error(-32603, "Application not ready");
        logResponse(method, result, timer.elapsed());
        return result;
    }

    auto result = it.value().handler(params);
    logResponse(method, result, timer.elapsed());
    return result;
}

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

    if (ecvDisplayTools::HasInstance()) {
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
    redraw();
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
    for (const auto& v : ids) {
        unsigned id = v.toUInt();
        ccHObject* entity = findEntity(id);
        if (entity) {
            m_app->setSelectedInDB(entity, true);
            selected.append(static_cast<qint64>(id));
        }
    }
    m_app->updateUI();
    QJsonObject result;
    result["selected"] = selected;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// clear
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcClear(const QMap<QString, QVariant>&) {
    auto* root = m_app->dbRootObject();
    int removed = 0;
    ccHObject* child;
    while ((child = root->getChild(0)) != nullptr) {
        m_app->removeFromDB(child, true);
        ++removed;
    }

    CC_DRAW_CONTEXT ctx;
    ctx.removeEntityType = ENTITY_TYPE::ECV_ALL;
    ecvDisplayTools::RemoveEntities(ctx);
    redraw();

    QJsonObject result;
    result["removed_count"] = removed;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// view.setOrientation
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcViewSetOrientation(
        const QMap<QString, QVariant>& params) {
    QString orientation = params.value("orientation", "front").toString();
    m_app->setView(viewOrientationFromString(orientation));
    QJsonObject result;
    result["orientation"] = orientation;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// view.zoomFit
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcViewZoomFit(
        const QMap<QString, QVariant>& params) {
    if (params.contains("entity_id")) {
        unsigned id = params["entity_id"].toUInt();
        ccHObject* entity = findEntity(id);
        if (entity) {
            m_app->zoomOnEntities(entity);
        }
    } else {
        m_app->setGlobalZoom();
    }
    return JsonRPCResult::success(0);
}

// ═══════════════════════════════════════════════════════════════════════════
// view.refresh
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcViewRefresh(const QMap<QString, QVariant>&) {
    redraw();
    return JsonRPCResult::success(0);
}

// ═══════════════════════════════════════════════════════════════════════════
// view.setPerspective
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcViewSetPerspective(
        const QMap<QString, QVariant>& params) {
    QString mode = params.value("mode", "object").toString();
    if (mode == "viewer") {
        m_app->toggleActiveWindowViewerBasedPerspective();
    } else {
        m_app->toggleActiveWindowCenteredPerspective();
    }
    QJsonObject result;
    result["mode"] = mode;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// view.setPointSize
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcViewSetPointSize(
        const QMap<QString, QVariant>& params) {
    QString action = params.value("action", "increase").toString();
    if (action == "increase") {
        m_app->increasePointSize();
    } else {
        m_app->decreasePointSize();
    }
    QJsonObject result;
    result["action"] = action;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// view.screenshot
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcViewScreenshot(
        const QMap<QString, QVariant>& params) {
    QString filename = params["filename"].toString();
    if (filename.isEmpty())
        return JsonRPCResult::error(-32602, "Missing 'filename'",
                                    D("param", "filename"));

    QWidget* win = m_app->getActiveWindow();
    if (!win) return JsonRPCResult::error(6, "No active window");

    QPixmap pixmap = win->grab();
    bool saved = pixmap.save(filename);
    if (!saved)
        return JsonRPCResult::error(7, "Failed to save screenshot",
                                    D("filename", filename));

    QJsonObject result;
    result["filename"] = filename;
    result["width"] = pixmap.width();
    result["height"] = pixmap.height();
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// view.getCamera
// ═══════════════════════════════════════════════════════════════════════════

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
// cloud.coordToSf (NEW) — create scalar field from X/Y/Z coordinate
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

// ═══════════════════════════════════════════════════════════════════════════
// colmap.reconstruct
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcColmapReconstruct(
        const QMap<QString, QVariant>& params) {
    QString imagePath = params.value("image_path").toString();
    QString workspace = params.value("workspace_path").toString();
    if (imagePath.isEmpty() || workspace.isEmpty()) {
        return JsonRPCResult::error(
                -32602,
                "Missing 'image_path' and/or 'workspace_path' parameters",
                D("image_path", imagePath, "workspace_path", workspace));
    }

    QString colmapBin = params.value("colmap_binary", "colmap").toString();
    QString quality = params.value("quality", "HIGH").toString();
    QString dataType = params.value("data_type", "INDIVIDUAL").toString();
    QString mesher = params.value("mesher", "POISSON").toString();
    QString cameraModel = params.value("camera_model", "").toString();
    bool singleCamera = params.value("single_camera", false).toBool();
    bool useGpu = params.value("use_gpu", true).toBool();

    QDir().mkpath(workspace);

    QStringList args;
    args << "automatic_reconstructor"
         << "--workspace_path" << workspace << "--image_path" << imagePath
         << "--quality" << quality << "--data_type" << dataType << "--mesher"
         << mesher;
    if (!cameraModel.isEmpty()) args << "--camera_model" << cameraModel;
    if (singleCamera) args << "--single_camera" << "1";
    if (!useGpu) args << "--use_gpu" << "0";

    QProcess process;
    process.setProgram(colmapBin);
    process.setArguments(args);
    process.start();

    int timeoutMs = params.value("timeout_ms", 7200000).toInt();
    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        return JsonRPCResult::error(
                3, "Colmap timed out",
                D("timeout_ms", timeoutMs, "command",
                  QString(colmapBin + " " + args.join(" "))));
    }

    if (process.exitCode() != 0) {
        QString errOut = process.readAllStandardError().trimmed();
        if (errOut.isEmpty())
            errOut = process.readAllStandardOutput().trimmed();
        return JsonRPCResult::error(
                3, "Colmap failed",
                D("exit_code", process.exitCode(), "stderr", errOut.left(500),
                  "command", QString(colmapBin + " " + args.join(" "))));
    }

    QJsonObject result;
    result["workspace"] = workspace;
    result["image_path"] = imagePath;
    result["quality"] = quality;
    result["status"] = "completed";

    QJsonObject outputs;
    for (int idx = 0; idx < 10; ++idx) {
        QString denseDir = workspace + "/dense/" + QString::number(idx);
        if (!QDir(denseDir).exists()) break;

        QString fusedPly = denseDir + "/fused.ply";
        if (QFile::exists(fusedPly)) {
            QJsonArray arr = outputs.value("fused_ply").toArray();
            arr.append(fusedPly);
            outputs["fused_ply"] = arr;
        }
        for (const QString& meshName :
             {"meshed-poisson.ply", "meshed-delaunay.ply"}) {
            QString meshPath = denseDir + "/" + meshName;
            if (QFile::exists(meshPath)) {
                QJsonArray arr = outputs.value("mesh").toArray();
                arr.append(meshPath);
                outputs["mesh"] = arr;
            }
        }
        QString texturedObj = denseDir + "/textured.obj";
        if (QFile::exists(texturedObj)) {
            QJsonArray arr = outputs.value("textured_mesh").toArray();
            arr.append(texturedObj);
            outputs["textured_mesh"] = arr;
        }
    }
    result["outputs"] = outputs;

    bool importResults = params.value("import_results", true).toBool();
    if (importResults && m_app) {
        QJsonArray imported;
        auto tryImport = [&](const QString& key) {
            QJsonArray paths = outputs.value(key).toArray();
            for (const QJsonValue& v : paths) {
                QString path = v.toString();
                if (!QFile::exists(path)) continue;
                QMap<QString, QVariant> openParams;
                openParams["filename"] = path;
                openParams["silent"] = true;
                auto openResult = rpcOpen(openParams);
                if (!openResult.isError) imported.append(path);
            }
        };
        tryImport("textured_mesh");
        tryImport("mesh");
        tryImport("fused_ply");
        result["imported"] = imported;
    }

    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

// ═══════════════════════════════════════════════════════════════════════════
// colmap.run (NEW) — generic COLMAP subcommand executor
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcColmapRun(
        const QMap<QString, QVariant>& params) {
    QString command = params["command"].toString();
    if (command.isEmpty()) {
        return JsonRPCResult::error(-32602, "Missing 'command' parameter",
                                    D("param", "command"));
    }

    QString colmapBin = params.value("colmap_binary", "colmap").toString();
    int timeoutMs = params.value("timeout_ms", 3600000).toInt();

    QStringList argList;
    argList << command;
    QVariantList userArgs = params.value("args", QVariant()).toList();
    for (const auto& a : userArgs) {
        argList << a.toString();
    }

    QVariantMap kvArgs = params.value("kwargs", QVariant()).toMap();
    for (auto it = kvArgs.constBegin(); it != kvArgs.constEnd(); ++it) {
        argList << ("--" + it.key()) << it.value().toString();
    }

    QProcess process;
    process.setProgram(colmapBin);
    process.setArguments(argList);
    process.start();

    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        return JsonRPCResult::error(
                3, "Colmap command timed out",
                D("command", command, "timeout_ms", timeoutMs));
    }

    QString stdOut = process.readAllStandardOutput().trimmed();
    QString stdErr = process.readAllStandardError().trimmed();

    if (process.exitCode() != 0) {
        return JsonRPCResult::error(
                3, "Colmap command failed",
                D("command", command, "exit_code", process.exitCode(), "stdout",
                  stdOut.left(2000), "stderr", stdErr.left(2000),
                  "full_command",
                  QString(colmapBin + " " + argList.join(" "))));
    }

    QJsonObject result;
    result["command"] = command;
    result["exit_code"] = 0;
    result["stdout"] = stdOut.left(5000);
    result["stderr"] = stdErr.left(2000);
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

#ifdef HAS_PCV_PLUGIN
// ═══════════════════════════════════════════════════════════════════════════
// Ambient Occlusion (PCV / ShadeVis)
// ═══════════════════════════════════════════════════════════════════════════

static constexpr char PCV_SF_NAME[] = "Illuminance (PCV)";

JsonRPCResult JsonRPCPlugin::rpcProcessPcv(
        const QMap<QString, QVariant>& params) {
    unsigned entityId = params["entity_id"].toUInt();
    if (entityId == 0) {
        return JsonRPCResult::error(
                -32602, "Missing required parameter: entity_id");
    }

    unsigned rayCount = params.value("ray_count", 256).toUInt();
    unsigned resolution = params.value("resolution", 1024).toUInt();
    bool mode360 = params.value("mode_360", true).toBool();
    bool meshClosed = params.value("mesh_closed", false).toBool();

    ccHObject* entity = findEntity(entityId);
    if (!entity) {
        return JsonRPCResult::error(
                -32602,
                "Entity not found: " + QString::number(entityId));
    }

    ccPointCloud* cloud = nullptr;
    ccGenericMesh* mesh = nullptr;

    if (entity->isA(CV_TYPES::POINT_CLOUD)) {
        cloud = ccHObjectCaster::ToPointCloud(entity);
    } else if (entity->isKindOf(CV_TYPES::MESH)) {
        mesh = ccHObjectCaster::ToGenericMesh(entity);
        if (mesh) {
            cloud = ccHObjectCaster::ToPointCloud(
                    mesh->getAssociatedCloud());
        }
    }

    if (!cloud) {
        return JsonRPCResult::error(
                -32602,
                "Entity is not a point cloud or mesh: " +
                        QString::number(entityId));
    }

    std::vector<CCVector3d> rays;
    if (!PCV::GenerateRays(rayCount, rays, mode360)) {
        return JsonRPCResult::error(5, "Failed to generate ray set");
    }

    int sfIdx = cloud->getScalarFieldIndexByName(PCV_SF_NAME);
    if (sfIdx < 0) {
        sfIdx = cloud->addScalarField(PCV_SF_NAME);
    }
    if (sfIdx < 0) {
        return JsonRPCResult::error(
                5, "Failed to allocate PCV scalar field");
    }
    cloud->setCurrentScalarField(sfIdx);

    bool success = PCV::Launch(rays, cloud, mesh, meshClosed, resolution,
                               resolution, nullptr, cloud->getName());

    if (!success) {
        cloud->deleteScalarField(sfIdx);
        return JsonRPCResult::error(
                5, "PCV computation failed",
                D("entity_id", static_cast<qint64>(entityId)));
    }

    ccScalarField* sf =
            static_cast<ccScalarField*>(cloud->getScalarField(sfIdx));
    if (sf) {
        sf->computeMinAndMax();
        cloud->setCurrentDisplayedScalarField(sfIdx);
        sf->setColorScale(ccColorScalesManager::GetDefaultScale(
                ccColorScalesManager::GREY));
        entity->showNormals(false);
        entity->showSF(true);
        if (entity != cloud) {
            cloud->showSF(true);
        }
    }

    redraw();

    QJsonObject result;
    result["entity_id"] = static_cast<qint64>(entityId);
    result["ray_count"] = static_cast<qint64>(rays.size());
    result["resolution"] = static_cast<qint64>(resolution);
    result["mode_360"] = mode360;
    result["point_count"] = static_cast<qint64>(cloud->size());
    if (sf) {
        result["sf_min"] = sf->getMin();
        result["sf_max"] = sf->getMax();
    }
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// Internal helper
// ═══════════════════════════════════════════════════════════════════════════

void JsonRPCPlugin::redraw() {
    QWidget* win = m_app->getActiveWindow();
    if (win) {
        ecvDisplayTools::RedrawDisplay();
    }
}
