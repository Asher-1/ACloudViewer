// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Plugin lifecycle, method registry, dispatch and request logging. The
// handlers live in JsonRPCMethods*.cpp, one file per method family.

#include "JsonRPCPlugin.h"

#include <cvViewSelectionManager.h>
#include <ecvGenericGLDisplay.h>
#include <ecvMainAppInterface.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

#include <QAction>
#include <QDateTime>
#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <cmath>
#include <memory>

#include "JsonRPCChildProcess.h"
#include "JsonRPCCommon.h"

using jsonrpc::D;
using jsonrpc::entityToJson;

namespace {

ccHObject* findEntityById(ccHObject* root, unsigned id) {
    if (!root) return nullptr;
    if (root->getUniqueID() == id) return root;
    for (unsigned i = 0; i < root->getChildrenNumber(); ++i) {
        auto* found = findEntityById(root->getChild(i), id);
        if (found) return found;
    }
    return nullptr;
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
                QString keyColon = QString(it.key() + QStringLiteral(":"))
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
        if (list.size() > 50) out << prefix + QStringLiteral("...");
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

JsonRPCPlugin::~JsonRPCPlugin() { jsonrpc::abortChildProcesses(this); }

QList<QAction*> JsonRPCPlugin::getActions() {
    CVLog::PrintDebug("JsonRPCPlugin::getActions");
    if (!m_action) {
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());
        m_action->setCheckable(true);
        // setMainAppInterface() may already have started the server.
        m_action->setChecked(rpc_server.isListening());
        m_action->setEnabled(true);
        connect(m_action, &QAction::triggered, this, &JsonRPCPlugin::triggered);
    }
    return {m_action};
}

void JsonRPCPlugin::setMainAppInterface(ecvMainAppInterface* app) {
    ccStdPluginInterface::setMainAppInterface(app);
    if (!app) return;
    const bool listening = startServer();
    if (m_action) {
        m_action->setChecked(listening);
    }
}

bool JsonRPCPlugin::startServer() {
    // Clients select the port through ACV_RPC_URL; ACV_RPC_PORT lets a second
    // instance (CI, tests next to an interactive session) take another one.
    bool ok = false;
    const int requested = qEnvironmentVariableIntValue("ACV_RPC_PORT", &ok);
    const quint16 port = ok && requested > 0 && requested <= 65535
                                 ? static_cast<quint16>(requested)
                                 : static_cast<quint16>(6001);
    if (!rpc_server.listen(port)) {
        // Warning, not Error: errors open a modal box, which would block an
        // unattended (offscreen) start that merely lost the port race.
        CVLog::Warning(QStringLiteral("[JsonRPC] Cannot listen on port %1: %2")
                               .arg(port)
                               .arg(rpc_server.errorString()));
        return false;
    }
    CVLog::Print(QStringLiteral("[JsonRPC] Server listening on port %1 "
                                "(%2 methods registered)")
                         .arg(port)
                         .arg(m_methods.size()));
    return true;
}

void JsonRPCPlugin::triggered(bool checked) {
    CVLog::Print(QString("JsonRPCPlugin::triggered: checked(%1)").arg(checked));
    if (checked) {
        if (!startServer() && m_action) m_action->setChecked(false);
    } else {
        rpc_server.close();
        // Nobody is left to receive their answers.
        jsonrpc::abortChildProcesses(this);
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

    // --- Generic UI drivers (agent-test integration; plugin-agnostic) ---
    reg("ui.triggerAction",  "Trigger a QAction by its text: {text}",
        [this](auto& p){ return rpcUiTriggerAction(p); });
    reg("ui.listActions",    "List every triggerable QAction with text/objectName/enabled/checkable/shortcut",
        [this](auto& p){ return rpcUiListActions(p); });
    reg("qt.invokeMethod",   "Qt-reflection invoke: {class_contains, method, ?args[0..3], ?object_contains} — the target method must be invokable (Q_INVOKABLE/slot) with QVariant parameters",
        [this](auto& p){ return rpcQtInvokeMethod(p); });

    // --- View control ---
    reg("view.list",          "List all views: returns [{view_id, active}]",
        [this](auto& p){ return rpcViewList(p); });
    reg("view.setOrientation", "Set camera view: {orientation, ?view_id}",
        [this](auto& p){ return rpcViewSetOrientation(p); });
    reg("view.zoomFit",      "Zoom to fit: {?entity_id}",
        [this](auto& p){ return rpcViewZoomFit(p); });
    // --- Pick-layer probes (agent-test integration): each call drives ONE
    // layer of the real pick chain so a failing click can be attributed to
    // event dispatch vs entity resolution vs octree point picking. ---
    reg("view.pickEntity",  "Entity pick probe: {x, y} logical px -> pickObject/pickLeafViewId chain",
        [this](auto& p){ return rpcViewPickEntity(p); });
    reg("view.pickPoint",   "Point-pick probe: {x, y} logical px -> StartPicking(POINT_PICKING) octree chain",
        [this](auto& p){ return rpcViewPickPoint(p); });
    reg("view.selectRegion", "Selection-tools probe: {x1,y1,x2,y2} logical px (or single x,y point), field=points|cells -> cvViewSelectionManager hardware selection chain",
        [this](auto& p){ return rpcViewSelectRegion(p); });
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

    // --- CLI Processing ---
    reg("aicore.batch",     "Run a registered AICore plugin batch command in an isolated headless child: {command, options{}, ?result_json, ?timeout_ms}; list values repeat an option, HELP:1 returns its schema",
        [this](auto& p){ return rpcAICoreBatch(p); });
    reg("process.run_cli",   "Run any CLI processing command: {input_path, output_path, args[], ?timeout_ms}",
        [this](auto& p){ return rpcProcessRunCli(p); });
    reg("yolo.track",        "Headless multi-object tracking (six official tracker modes): {model, ?video, ?frames_dir, ?tracker: bytetrack|botsort|ocsort|deepocsort|fasttrack|tracktrack, ?gmc, ?conf, ?iou, ?max_det, ?reid: 0|1, ?device, ?threads, tracks_json, ?timeout_ms}",
        [this](auto& p){ return rpcYoloTrack(p); });
    reg("sam3d.generate",    "Batch SAM 3D Objects image-to-3D (Gaussian PLY + optional FlexiCubes mesh, single model load): {images[], ?masks[], output_dir, ?device: auto|cpu|cuda|vulkan, ?dtype: f16|q8_0|q4_k, ?steps, ?seed, ?mesh: 0|1, ?rmbg: 0|1, ?moge_cache: 0|1, ?threads, ?models_dir, ?result_json, ?timeout_ms}",
        [this](auto& p){ return rpcSam3dGenerate(p); });
    reg("mapanything.reconstruct", "Multi-view metric 3D reconstruction (MapAnything family, single model load): {images[>=2], output_dir, ?model: path|catalog filename (default vggt-omega-1b-512-q8_0), ?device, ?image_size, ?resize_mode: balanced|max_size|square (default: catalog entry), ?min_conf (default: catalog entry), ?max_frames_per_pass (default 64; larger sets stream overlap-2 Sim3-aligned chunks, halved automatically when the device cannot hold a pass; same driver as the GUI), ?threads, ?cloud_ply, ?per_view, ?raw_output_dir, ?stage_dump_dir, ?gt_meta, ?ref_images_path: COLMAP model_aligner text '<image_name> X Y Z' per line, ?ref_is_gps: 0|1 (required with ref_images_path), ?result_json, ?timeout_ms}",
        [this](auto& p){ return rpcMapAnythingReconstruct(p); });
    reg("process.csf",       "Cloth Simulation Filter: {input_path, output_path, ?scene, ?cloth_resolution, ?max_iterations, ?class_threshold, ?export_ground, ?export_offground}",
        [this](auto& p){ return rpcProcessCsf(p); });
    reg("process.m3c2",      "M3C2 distance computation: {cloud1_path, cloud2_path, params_file, output_path}",
        [this](auto& p){ return rpcProcessM3c2(p); });
    reg("process.ransac",    "RANSAC shape detection: {input_path, output_path, ?epsilon, ?bitmap_epsilon, ?support_points, ?max_normal_dev, ?probability, ?primitives[]}",
        [this](auto& p){ return rpcProcessRansac(p); });

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
// View resolution helper
// ═══════════════════════════════════════════════════════════════════════════

ecvGenericGLDisplay* JsonRPCPlugin::resolveView(
        const QMap<QString, QVariant>& params) {
    if (params.contains("view_id")) {
        int viewId = params["view_id"].toInt();
        auto* view = ecvViewManager::instance().findView(viewId);
        return view;
    }
    return ecvViewManager::instance().getEffectiveView();
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
                QString(it.key() + QStringLiteral(":")).leftJustified(colonCol);

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
        const bool isComplex = (vt == QVariant::Map || vt == QVariant::List ||
                                vt == QVariant::StringList);

        if (isComplex) {
            lines << QStringLiteral("%1%2:").arg(QLatin1String(kLogPrefix),
                                                 label);
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
            appendResultBlock(lines, QStringLiteral("data"), result.error_data);
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
    // Dispatch-entry stamp: pairs with the handler-internal [rpcTiming]
    // prints to split queueing from handler work (interaction-latency
    // forensics).
    const qint64 dispatchMs = QDateTime::currentMSecsSinceEpoch();
    if (method == QStringLiteral("scene.select") ||
        method == QStringLiteral("scene.setVisible")) {
        CVLog::Print(QStringLiteral("[rpcDispatch] %1 @ %2")
                             .arg(method)
                             .arg(dispatchMs));
    }

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
    if (result.pending) {
        // Logged when the child ends, with the whole duration.
        auto logged = std::make_shared<JsonRPCDeferred>();
        result.pending->onComplete(
                [this, method, timer, logged](const JsonRPCResult& done) {
                    logResponse(method, done, timer.elapsed());
                    logged->complete(done);
                });
        return JsonRPCResult::deferred(logged);
    }
    logResponse(method, result, timer.elapsed());
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helper
// ═══════════════════════════════════════════════════════════════════════════

void JsonRPCPlugin::redraw() {
    if (m_app->getActiveWindow()) {
        {
            ecvRedrawScope scope;
        }
    }
}
