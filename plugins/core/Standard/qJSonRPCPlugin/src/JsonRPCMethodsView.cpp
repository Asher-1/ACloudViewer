// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// View control and picking probes: view.*.

#include <VtkVis.h>
#include <cvHardwareSelector.h>
#include <cvSelectionPipeline.h>
#include <cvViewSelectionManager.h>
#include <ecvDisplayCoordinates.h>
#include <ecvDisplayTools.h>
#include <ecvGenericGLDisplay.h>
#include <ecvMainAppInterface.h>
#include <ecvViewManager.h>
#include <vtkActor.h>
#include <vtkDataObject.h>
#include <vtkGLView.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkProp.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPixmap>
#include <QWidget>
#include <algorithm>

#include "JsonRPCCommon.h"
#include "JsonRPCPlugin.h"

using jsonrpc::D;

namespace {

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

// ═══════════════════════════════════════════════════════════════════════════
// view.list
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcViewList(const QMap<QString, QVariant>&) {
    auto* activeView = ecvViewManager::instance().getEffectiveView();
    const auto& allViews = ecvViewManager::instance().getAllViews();
    QJsonArray arr;
    for (auto* view : allViews) {
        QJsonObject v;
        v["view_id"] = view->getUniqueID();
        v["title"] = view->getTitle();
        v["active"] = (view == activeView);
        v["width"] = view->glWidth();
        v["height"] = view->glHeight();
        arr.append(v);
    }
    return JsonRPCResult::success(QJsonDocument(arr).toVariant());
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
// view.pickEntity / view.pickPoint / view.refresh
// ═══════════════════════════════════════════════════════════════════════════

JsonRPCResult JsonRPCPlugin::rpcViewPickEntity(
        const QMap<QString, QVariant>& params) {
    const double x = params.value("x", -1.0).toDouble();
    const double y = params.value("y", -1.0).toDouble();
    ecvGenericGLDisplay* view = m_app ? m_app->getActiveGLDisplay() : nullptr;
    if (!view) return JsonRPCResult::error(1, "no active view");
    const QString viewID = view->pickObject(x, y);
    QJsonObject result;
    result["viewId"] = viewID;
    bool ok = false;
    const unsigned uid = viewID.toUInt(&ok);
    if (ok && m_app && m_app->dbRootObject()) {
        if (ccHObject* ent = m_app->dbRootObject()->find(uid)) {
            result["entity_id"] = static_cast<qint64>(uid);
            result["name"] = ent->getName();
        }
    }
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

JsonRPCResult JsonRPCPlugin::rpcViewPickPoint(
        const QMap<QString, QVariant>& params) {
    const int x = params.value("x", -1).toInt();
    const int y = params.value("y", -1).toInt();
    ecvDisplayTools* dt = ecvViewManager::instance().displayTools();
    if (!dt) return JsonRPCResult::error(1, "no display tools");

    ccHObject* picked = nullptr;
    unsigned sub = 0;
    CCVector3 P(0, 0, 0);
    bool got = false;
    QMetaObject::Connection conn = connect(
            dt, &ecvDisplayTools::itemPicked, this,
            [&](ccHObject* e, unsigned s, int, int, const CCVector3& p) {
                got = true;
                picked = e;
                sub = s;
                P = p;
            },
            Qt::DirectConnection);

    ecvDisplayTools::PickingParameters pp(ecvGenericGLDisplay::POINT_PICKING, x,
                                          y, 5, 5);
    ecvDisplayTools::StartPicking(pp);
    disconnect(conn);

    QJsonObject result;
    result["hit"] = got;
    if (got && picked) {
        result["entity_id"] = static_cast<qint64>(picked->getUniqueID());
        result["name"] = picked->getName();
        result["point_index"] = static_cast<qint64>(sub);
        QJsonArray pt;
        pt.append(static_cast<double>(P.x));
        pt.append(static_cast<double>(P.y));
        pt.append(static_cast<double>(P.z));
        result["point"] = pt;
    }
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}

JsonRPCResult JsonRPCPlugin::rpcViewRefresh(const QMap<QString, QVariant>&) {
    redraw();
    return JsonRPCResult::success(0);
}

// ═══════════════════════════════════════════════════════════════════════════
// view.selectRegion — Selection-tools hardware-selection probe (agent-test
// integration). Drives the REAL cvViewSelectionManager chain the ParaView-
// style selection tools use, so a failing tool click can be attributed to
// hardware selection vs composite-block conversion vs highlight overlay.
// Params: {x1,y1,x2,y2} Qt-logical px (top-left origin) or a single {x,y}
// point pick; field: "points" (default) | "cells".
// ═══════════════════════════════════════════════════════════════════════════
JsonRPCResult JsonRPCPlugin::rpcViewSelectRegion(
        const QMap<QString, QVariant>& params) {
    const int x1 = params.value("x1", params.value("x", -1)).toInt();
    const int y1 = params.value("y1", params.value("y", -1)).toInt();
    const int x2 = params.value("x2", x1).toInt();
    const int y2 = params.value("y2", y1).toInt();
    const bool wantCells =
            params.value("field", "points").toString() == "cells";

    ecvGenericGLDisplay* view = m_app ? m_app->getActiveGLDisplay() : nullptr;
    if (!view) return JsonRPCResult::error(1, "no active view");
    QWidget* widget = view->asWidget();
    if (!widget) return JsonRPCResult::error(1, "no view widget");

    // Logical top-left Qt px -> VTK bottom-left PHYSICAL px. The hardware
    // selector samples the GL framebuffer, whose size is the viewport widget
    // size x DPR (this 4K env measures DPR ~1.43, offscreen render
    // 3282x1752): feeding logical values made every probe coordinate land
    // at ~0.7x of the intended position (fitted via selection_region_fit.py).
    const qreal dpr = ecvDisplayCoordinates::dprOf(widget);
    const int Hp = static_cast<int>(widget->height() * dpr);
    int region[4] = {
            static_cast<int>(x1 * dpr), Hp - 1 - static_cast<int>(y2 * dpr),
            static_cast<int>(x2 * dpr), Hp - 1 - static_cast<int>(y1 * dpr)};

    cvViewSelectionManager* mgr = cvViewSelectionManager::instance();
    if (!mgr) return JsonRPCResult::error(1, "no selection manager");
    // Bind to the ACTIVE VIEW's VtkVis (the renderer that actually paints).
    // ecvDisplayTools::getVisualizer3D() returns the display-tools singleton's
    // own VtkVis, whose renderer is NOT the viewport's — binding that made
    // every hardware selection silently miss, even on dedicated actors.
    // MainWindow's rebindToolsToActiveView() does the same for the GUI path.
    if (auto* glView = dynamic_cast<vtkGLView*>(view)) {
        if (auto* vis = glView->getVisualizer3D()) {
            mgr->setVisualizer(vis);
        }
    }

    const cvSelectionPipeline::SelectionType type =
            wantCells ? cvSelectionPipeline::SelectionType::SURFACE_CELLS
                      : cvSelectionPipeline::SelectionType::SURFACE_POINTS;
    // NOTE: cvGenericSelectionTool::getSelectionPipeline() is unusable here —
    // it forwards to m_manager which is never set on the manager itself
    // (the manager IS the pipeline owner). Use its own accessor.
    cvSelectionPipeline* pipeline = mgr->getPipeline();
    if (!pipeline) return JsonRPCResult::error(1, "no selection pipeline");
    cvSelectionData sel;
    QJsonArray pipelineNodeDetails;
    if (vtkSmartPointer<vtkSelection> vtkSel =
                pipeline->executeRectangleSelection(region, type)) {
        sel = cvSelectionPipeline::convertToCvSelectionData(
                vtkSel,
                wantCells ? cvSelectionPipeline::FIELD_ASSOCIATION_CELLS
                          : cvSelectionPipeline::FIELD_ASSOCIATION_POINTS);
        // Dump the PIPELINE's own selection nodes for comparison with the
        // raw pass above — any field that differs is what the converter
        // filters on.
        QJsonArray pnodes;
        const unsigned pn = std::min<unsigned>(vtkSel->GetNumberOfNodes(), 6);
        for (unsigned ni = 0; ni < pn; ++ni) {
            vtkSelectionNode* node = vtkSel->GetNode(ni);
            if (!node) continue;
            QJsonObject o;
            vtkInformation* props = node->GetProperties();
            o["fieldType"] = node->GetFieldType();
            o["hasProp"] = props && props->Has(vtkSelectionNode::PROP());
            if (props && props->Has(vtkSelectionNode::PROP())) {
                vtkActor* a = vtkActor::SafeDownCast(vtkProp::SafeDownCast(
                        props->Get(vtkSelectionNode::PROP())));
                o["propIsActor"] = (a != nullptr);
            }
            o["hasCompositeIndex"] =
                    props && props->Has(vtkSelectionNode::COMPOSITE_INDEX());
            if (o["hasCompositeIndex"].toBool()) {
                o["compositeIndex"] = static_cast<qint64>(
                        props->Get(vtkSelectionNode::COMPOSITE_INDEX()));
            }
            vtkIdTypeArray* lst =
                    vtkIdTypeArray::SafeDownCast(node->GetSelectionList());
            o["listSize"] = lst ? static_cast<qint64>(lst->GetNumberOfTuples())
                                : static_cast<qint64>(-1);
            pnodes.append(o);
        }
        pipelineNodeDetails = pnodes;
    }
    QJsonObject result;
    result["pipelineNodes"] = pipelineNodeDetails;
    // Layer-1 probe: raw hardware selection straight against the active
    // view's renderer, bypassing the pipeline. rawNodes>0 with count==0
    // means the break is in the conversion; rawNodes==0 means the GL
    // selection pass itself sees nothing.
    if (auto* glView = dynamic_cast<vtkGLView*>(view)) {
        if (auto* vis = glView->getVisualizer3D()) {
            if (vtkRenderer* ren = vis->getCurrentRenderer()) {
                cvHardwareSelector* raw = cvHardwareSelector::New();
                raw->SetRenderer(ren);
                raw->SetFieldAssociation(
                        wantCells ? vtkDataObject::FIELD_ASSOCIATION_CELLS
                                  : vtkDataObject::FIELD_ASSOCIATION_POINTS);
                int rawRegion[4] = {region[0], region[1], region[2], region[3]};
                vtkSmartPointer<vtkSelection> rawSel;
                rawSel.TakeReference(raw->Select(rawRegion));
                result["rawNodes"] =
                        rawSel ? static_cast<qint64>(rawSel->GetNumberOfNodes())
                               : static_cast<qint64>(-1);
                // Dump per-node properties: PROP type, field type, selection
                // list size and COMPOSITE_INDEX — the exact fields
                // convertToCvSelectionData filters on.
                QJsonArray nodes;
                if (rawSel) {
                    const unsigned n =
                            std::min<unsigned>(rawSel->GetNumberOfNodes(), 6);
                    for (unsigned ni = 0; ni < n; ++ni) {
                        vtkSelectionNode* node = rawSel->GetNode(ni);
                        if (!node) continue;
                        QJsonObject o;
                        vtkInformation* props = node->GetProperties();
                        o["fieldType"] = node->GetFieldType();
                        o["hasProp"] =
                                props && props->Has(vtkSelectionNode::PROP());
                        if (props && props->Has(vtkSelectionNode::PROP())) {
                            vtkProp* p = vtkProp::SafeDownCast(
                                    props->Get(vtkSelectionNode::PROP()));
                            vtkActor* a = vtkActor::SafeDownCast(p);
                            vtkActor* mapperHost = a;
                            o["propIsActor"] = (a != nullptr);
                            o["propIsCompositeMapper"] =
                                    (mapperHost && mapperHost->GetMapper() &&
                                     mapperHost->GetMapper()->IsA(
                                             "vtkCompositePolyDataMapper2"));
                        }
                        o["hasCompositeIndex"] =
                                props &&
                                props->Has(vtkSelectionNode::COMPOSITE_INDEX());
                        if (o["hasCompositeIndex"].toBool()) {
                            o["compositeIndex"] = static_cast<
                                    qint64>(props->Get(
                                    vtkSelectionNode::COMPOSITE_INDEX()));
                        }
                        vtkIdTypeArray* lst = vtkIdTypeArray::SafeDownCast(
                                node->GetSelectionList());
                        o["listSize"] = lst ? static_cast<qint64>(
                                                      lst->GetNumberOfTuples())
                                            : static_cast<qint64>(-1);
                        nodes.append(o);
                    }
                }
                result["rawNodeDetails"] = nodes;
                raw->Delete();
            }
        }
    }
    result["count"] = static_cast<qint64>(sel.count());
    result["actorCount"] = static_cast<qint64>(sel.actorCount());
    result["field"] = wantCells ? "cells" : "points";
    QJsonArray infos;
    for (const auto& info : sel.actorInfos()) {
        QJsonObject o;
        o["blockIndex"] = static_cast<qint64>(info.blockIndex);
        o["ids"] = static_cast<qint64>(info.selectedIds.size());
        o["hasPolyData"] = (info.polyData != nullptr);
        infos.append(o);
    }
    result["actors"] = infos;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
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

JsonRPCResult JsonRPCPlugin::rpcViewGetCamera(
        const QMap<QString, QVariant>& params) {
    ccGLMatrixd viewMat;
    ecvViewportParameters vp;
    auto* view = resolveView(params);
    if (view) {
        vp = view->getViewportParameters();
        viewMat = vp.viewMat;
    }
    QJsonArray mat;
    const double* data = viewMat.data();
    for (int i = 0; i < 16; ++i) mat.append(data[i]);
    QJsonObject result;
    result["view_matrix"] = mat;
    result["fov_deg"] = vp.fov_deg;
    result["perspective"] = vp.perspectiveView;
    result["object_centered"] = vp.objectCenteredView;
    result["near_clipping"] = vp.zNear;
    result["far_clipping"] = vp.zFar;
    return JsonRPCResult::success(QJsonDocument(result).toVariant());
}
