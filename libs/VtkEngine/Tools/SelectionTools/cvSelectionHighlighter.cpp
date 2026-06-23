// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionHighlighter.h"

#include "cvSelectionPipeline.h"
#include "cvViewSelectionManager.h"

// LOCAL
#include <QVector>
#include <algorithm>
#include <vector>

#include "Visualization/VtkVis.h"

// CV_DB_LIB
#include <ecvGenericVisualizer3D.h>
#include <ecvViewManager.h>

// LOCAL
#include "Visualization/vtkGLView.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <VTKExtensions/Views/vtkPVLODActor.h>
#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>
#include <VtkRendering/Core/VtkLODHelper.h>
#include <VtkRendering/Core/VtkRenderingUtils.h>

#include <QCoreApplication>
#include <QTimer>
#include <QWidget>

// VTK
#include <vtkAbstractArray.h>
#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkAppendFilter.h>
#include <vtkCellArray.h>
#include <vtkCellCenters.h>
#include <vtkCellData.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDataSetMapper.h>
#include <vtkExtractSelection.h>
#include <vtkGeometryFilter.h>
#include <vtkIdTypeArray.h>
#include <vtkLabeledDataMapper.h>
#include <vtkLinearTransform.h>
#include <vtkMapper.h>
#include <vtkMaskPoints.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkTextProperty.h>
#include <vtkTransform.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>

static vtkSmartPointer<vtkActor> cloneHighlightActor(vtkActor* source) {
    if (!source) return {};
    vtkSmartPointer<vtkActor> clone;
    if (auto* srcLod = vtkPVLODActor::SafeDownCast(source)) {
        auto lodClone = vtkSmartPointer<vtkPVLODActor>::New();
        lodClone->ShallowCopy(srcLod);
        if (auto* srcMapper =
                    vtkDataSetMapper::SafeDownCast(srcLod->GetMapper())) {
            auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
            mapper->ShallowCopy(srcMapper);
            lodClone->SetMapper(mapper);
        }
        clone = lodClone.GetPointer();
    } else {
        clone = vtkSmartPointer<vtkActor>::New();
        clone->ShallowCopy(source);
        if (auto* srcMapper =
                    vtkDataSetMapper::SafeDownCast(source->GetMapper())) {
            auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
            mapper->ShallowCopy(srcMapper);
            clone->SetMapper(mapper);
        }
    }
    clone->SetPickable(0);
    return clone;
}

void cvSelectionHighlighter::prepareForShutdown() {
    if (m_shuttingDown) return;
    m_shuttingDown = true;
    m_highlightInstances.clear();
    m_hoverActor = nullptr;
    m_preselectedActor = nullptr;
    m_selectedActor = nullptr;
    m_boundaryActor = nullptr;
    setVisualizer(nullptr);
}

void cvSelectionHighlighter::scheduleAllViewsUpdate() const {
    if (m_shuttingDown) return;
    if (!QCoreApplication::instance()) return;

    Visualization::VtkVis* vis = getVtkVis();
    const auto& views = ecvViewManager::instance().getAllViews();
    for (auto* view : views) {
        auto* glView = dynamic_cast<vtkGLView*>(view);
        if (!glView) continue;
        if (vis && glView->getVisualizer3D() != vis) continue;
        glView->toBeRefreshed();
        glView->redraw(false, true);
        return;
    }

    if (vis) vis->UpdateScreen();
}

// Template highlight actors must never stay in any renderer (breaks HW
// selection when the same vtkActor was attached to multiple GL contexts /
// renderers).
static void stripActorFromRenderWindow(vtkActor* actor, vtkRenderWindow* rw) {
    if (!actor || !rw) return;
    vtkRendererCollection* rens = rw->GetRenderers();
    if (!rens) return;
    rens->InitTraversal();
    while (vtkRenderer* ren = rens->GetNextItem()) {
        ren->RemoveViewProp(actor);
    }
}

static void stripHighlightTemplatesFromVisualizer(
        Visualization::VtkVis* vis,
        const vtkSmartPointer<vtkActor>& hover,
        const vtkSmartPointer<vtkActor>& preselected,
        const vtkSmartPointer<vtkActor>& selected,
        const vtkSmartPointer<vtkActor>& boundary) {
    if (!vis) return;
    vtkRenderWindow* rw = vis->getRenderWindow();
    if (!rw) return;
    stripActorFromRenderWindow(hover, rw);
    stripActorFromRenderWindow(preselected, rw);
    stripActorFromRenderWindow(selected, rw);
    stripActorFromRenderWindow(boundary, rw);
}

// Remove layer-1 overlay renderers created by earlier selection experiments.
// Sharing the scene camera on a full-window layer-1 renderer breaks
// vtkOrientationMarkerWidget (axes scale with wheel zoom).
static const char* kSelectionOverlayId = "__highlight_selected_overlay__";
static const char* kHoverOverlayId = "__highlight_hover_overlay__";
static const char* kPreselectOverlayId = "__highlight_preselect_overlay__";

static const char* overlayIdForMode(
        cvSelectionHighlighter::HighlightMode mode) {
    switch (mode) {
        case cvSelectionHighlighter::HOVER:
            return kHoverOverlayId;
        case cvSelectionHighlighter::PRESELECTED:
            return kPreselectOverlayId;
        case cvSelectionHighlighter::SELECTED:
        default:
            return kSelectionOverlayId;
    }
}

static vtkSmartPointer<vtkPolyData> extractSelectionPolyData(
        vtkPolyData* polyData,
        vtkIdTypeArray* selection,
        int fieldAssociation) {
    if (!polyData || !selection || selection->GetNumberOfTuples() == 0) {
        return {};
    }

    const bool isPointSelection = (fieldAssociation != 0);
    const vtkIdType maxValidId = isPointSelection
                                         ? polyData->GetNumberOfPoints()
                                         : polyData->GetNumberOfCells();

    vtkSmartPointer<vtkIdTypeArray> validSelection =
            vtkSmartPointer<vtkIdTypeArray>::New();
    for (vtkIdType i = 0; i < selection->GetNumberOfTuples(); ++i) {
        const vtkIdType id = selection->GetValue(i);
        if (id >= 0 && id < maxValidId) {
            validSelection->InsertNextValue(id);
        }
    }
    if (validSelection->GetNumberOfTuples() == 0) {
        return {};
    }

    vtkSmartPointer<vtkSelectionNode> selectionNode =
            vtkSmartPointer<vtkSelectionNode>::New();
    selectionNode->SetContentType(vtkSelectionNode::INDICES);
    selectionNode->SetFieldType(isPointSelection ? vtkSelectionNode::POINT
                                                 : vtkSelectionNode::CELL);
    selectionNode->SetSelectionList(validSelection);

    vtkSmartPointer<vtkSelection> vtkSel = vtkSmartPointer<vtkSelection>::New();
    vtkSel->AddNode(selectionNode);

    vtkSmartPointer<vtkExtractSelection> extractor =
            vtkSmartPointer<vtkExtractSelection>::New();
    extractor->SetInputData(0, polyData);
    extractor->SetInputData(1, vtkSel);
    extractor->Update();

    vtkDataSet* extracted =
            vtkDataSet::SafeDownCast(extractor->GetOutputDataObject(0));
    if (!extracted) return {};

    auto result = vtkSmartPointer<vtkPolyData>::New();
    if (auto* poly = vtkPolyData::SafeDownCast(extracted)) {
        result->ShallowCopy(poly);
    } else {
        vtkNew<vtkGeometryFilter> geom;
        geom->SetInputData(extracted);
        geom->Update();
        result->ShallowCopy(geom->GetOutput());
    }
    if (result->GetNumberOfCells() == 0 && result->GetNumberOfPoints() == 0) {
        return {};
    }
    return result;
}

static vtkSmartPointer<vtkPolyData> buildSelectionMarkerPolyData(
        vtkPolyData* polyData,
        vtkIdTypeArray* selection,
        int fieldAssociation) {
    if (!polyData || !selection || selection->GetNumberOfTuples() == 0) {
        return {};
    }

    const bool isPointSelection = (fieldAssociation != 0);
    vtkSmartPointer<vtkPolyData> marker = vtkSmartPointer<vtkPolyData>::New();
    vtkNew<vtkPoints> pts;
    vtkNew<vtkCellArray> verts;

    if (isPointSelection) {
        for (vtkIdType i = 0; i < selection->GetNumberOfTuples(); ++i) {
            const vtkIdType pid = selection->GetValue(i);
            if (pid < 0 || pid >= polyData->GetNumberOfPoints()) continue;
            double p[3];
            polyData->GetPoint(pid, p);
            const vtkIdType newId = pts->InsertNextPoint(p);
            verts->InsertNextCell(1);
            verts->InsertCellPoint(newId);
        }
    } else {
        vtkNew<vtkCellCenters> centers;
        centers->SetInputData(polyData);
        centers->VertexCellsOn();
        centers->Update();
        auto* centersPoly = centers->GetOutput();
        if (!centersPoly) return {};
        for (vtkIdType i = 0; i < selection->GetNumberOfTuples(); ++i) {
            const vtkIdType cid = selection->GetValue(i);
            if (cid < 0 || cid >= centersPoly->GetNumberOfPoints()) continue;
            double p[3];
            centersPoly->GetPoint(cid, p);
            const vtkIdType newId = pts->InsertNextPoint(p);
            verts->InsertNextCell(1);
            verts->InsertCellPoint(newId);
        }
    }

    if (pts->GetNumberOfPoints() == 0) return {};
    marker->SetPoints(pts);
    marker->SetVerts(verts);
    return marker;
}

void cvSelectionHighlighter::clearSelectionOverlay(Visualization::VtkVis* vis,
                                                   const char* overlayId) {
    if (!vis) return;
    const char* id = overlayId ? overlayId : kSelectionOverlayId;
    if (vis->contains(id)) {
        vis->removePointCloud(id);
    }
}

void cvSelectionHighlighter::clearAllSelectionOverlays(
        Visualization::VtkVis* vis) {
    if (!vis) return;
    clearSelectionOverlay(vis, kSelectionOverlayId);
    clearSelectionOverlay(vis, kHoverOverlayId);
    clearSelectionOverlay(vis, kPreselectOverlayId);
}

void cvSelectionHighlighter::styleSelectionOverlayProperty(
        vtkProperty* prop, SelectionOverlayKind kind) {
    if (!prop) return;
    prop->SetColor(1.0, 0.0, 1.0);
    prop->SetOpacity(1.0);
    prop->SetAmbient(1.0);
    prop->SetDiffuse(0.0);
    prop->SetSpecular(0.0);
    prop->LightingOff();
    prop->SetRenderLinesAsTubes(false);
    prop->SetRenderPointsAsSpheres(false);
    if (kind == SelectionOverlayPoints) {
        prop->SetRepresentationToPoints();
        prop->SetPointSize(5.0f);
        prop->EdgeVisibilityOff();
    } else if (kind == SelectionOverlaySurface) {
        prop->SetRepresentationToWireframe();
        prop->EdgeVisibilityOff();
        prop->SetLineWidth(2.0f);
    }
}

vtkSmartPointer<vtkActor> cvSelectionHighlighter::createSelectionOverlayActor(
        vtkPolyData* poly, SelectionOverlayKind kind) {
    if (!poly) return {};

    if (kind == SelectionOverlayPoints) {
        if (poly->GetNumberOfPoints() == 0) return {};
        auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
        mapper->SetInputData(poly);
        mapper->ScalarVisibilityOff();
        auto actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        styleSelectionOverlayProperty(actor->GetProperty(), kind);
        actor->SetPickable(0);
        return actor;
    }

    if (kind == SelectionOverlaySurface) {
        if (poly->GetNumberOfCells() == 0) return {};
        vtkSmartPointer<vtkPVLODActor> actor;
        VtkRendering::CreateActorFromVTKDataSet(poly, actor);
        if (!actor) return {};
        if (auto* mapper = vtkDataSetMapper::SafeDownCast(actor->GetMapper())) {
            mapper->ScalarVisibilityOff();
            mapper->SetResolveCoincidentTopologyToPolygonOffset();
            mapper->SetRelativeCoincidentTopologyPolygonOffsetParameters(-1,
                                                                         -1);
        }
        styleSelectionOverlayProperty(actor->GetProperty(), kind);
        actor->SetPickable(0);
        return actor.GetPointer();
    }

    return {};
}

void cvSelectionHighlighter::applySelectionOverlay(Visualization::VtkVis* vis,
                                                   vtkPolyData* poly,
                                                   SelectionOverlayKind kind,
                                                   const char* overlayId,
                                                   bool interactiveColor) {
    if (!vis || !poly) return;
    const char* id = overlayId ? overlayId : kSelectionOverlayId;
    clearSelectionOverlay(vis, id);

    const double cr = interactiveColor ? 0.5 : 1.0;
    const double cg = 0.0;
    const double cb = 1.0;

    if (kind == SelectionOverlayPoints) {
        if (!vis->addPointCloud(poly, id)) {
            CVLog::Warning(
                    "[cvSelectionHighlighter] Failed to add point selection "
                    "overlay");
            return;
        }
        vis->setPointCloudUniqueColor(cr, cg, cb, id);
        vis->setPointCloudRenderingProperties(
                Visualization::VtkVis::PCL_VISUALIZER_POINT_SIZE, 5.0, id);
        vis->setPointCloudRenderingProperties(
                Visualization::VtkVis::PCL_VISUALIZER_REPRESENTATION,
                Visualization::VtkVis::PCL_VISUALIZER_REPRESENTATION_POINTS,
                id);
        if (vtkActor* a = vis->getActorById(id)) {
            styleSelectionOverlayProperty(a->GetProperty(),
                                          SelectionOverlayPoints);
            a->GetProperty()->SetColor(cr, cg, cb);
            a->SetPickable(0);
        }
        CVLog::PrintVerbose(
                QString("[cvSelectionHighlighter] ParaView point overlay: "
                        "%1 pts")
                        .arg(poly->GetNumberOfPoints()));
        return;
    }

    if (kind == SelectionOverlaySurface) {
        if (!vis->addSelectionHighlightSurface(poly, id)) {
            CVLog::Warning(
                    "[cvSelectionHighlighter] Failed to add cell selection "
                    "surface overlay");
            return;
        }
        CVLog::Print(
                QString("[cvSelectionHighlighter] ParaView cell overlay: %1 "
                        "cells %2 pts")
                        .arg(poly->GetNumberOfCells())
                        .arg(poly->GetNumberOfPoints()));
    }
}

static cvSelectionHighlighter::SelectionOverlayKind
upsertParaViewSelectionOverlay(Visualization::VtkVis* vis,
                               vtkPolyData* polyData,
                               vtkIdTypeArray* selection,
                               int fieldAssociation,
                               vtkSmartPointer<vtkPolyData>& outPoly,
                               const char* overlayId = kSelectionOverlayId,
                               bool interactiveColor = false) {
    outPoly = nullptr;
    if (!vis) return cvSelectionHighlighter::SelectionOverlayNone;

    const bool isPointSelection = (fieldAssociation != 0);
    if (isPointSelection) {
        outPoly = buildSelectionMarkerPolyData(polyData, selection,
                                               fieldAssociation);
        if (!outPoly) return cvSelectionHighlighter::SelectionOverlayNone;
        cvSelectionHighlighter::applySelectionOverlay(
                vis, outPoly, cvSelectionHighlighter::SelectionOverlayPoints,
                overlayId, interactiveColor);
        return cvSelectionHighlighter::SelectionOverlayPoints;
    }

    outPoly = extractSelectionPolyData(polyData, selection, fieldAssociation);
    if (!outPoly) return cvSelectionHighlighter::SelectionOverlayNone;
    cvSelectionHighlighter::applySelectionOverlay(
            vis, outPoly, cvSelectionHighlighter::SelectionOverlaySurface,
            overlayId, interactiveColor);
    return cvSelectionHighlighter::SelectionOverlaySurface;
}

static void pruneSelectionOverlayRenderers(vtkRenderer* sceneRenderer) {
    if (!sceneRenderer) return;
    vtkRenderWindow* rw = sceneRenderer->GetRenderWindow();
    if (!rw) return;

    std::vector<vtkRenderer*> toRemove;
    if (vtkRendererCollection* rens = rw->GetRenderers()) {
        rens->InitTraversal();
        while (vtkRenderer* ren = rens->GetNextItem()) {
            if (ren != sceneRenderer && ren->GetLayer() == 1 &&
                !ren->GetInteractive() &&
                ren->GetNumberOfPropsRendered() == 0 &&
                ren->GetActors()->GetNumberOfItems() == 0) {
                toRemove.push_back(ren);
            }
        }
    }
    for (vtkRenderer* ren : toRemove) {
        rw->RemoveRenderer(ren);
    }
}

vtkActor* cvSelectionHighlighter::resolveRegisteredMeshActor(
        Visualization::VtkVis* vis, vtkActor* picked) {
    if (!vis || !picked) return picked;
    // Do not call getIdByActor(picked): HW-picked props may belong to another
    // VtkVis/render view and can crash or return stale ids.

    vtkPolyData* pickedPoly = nullptr;
    if (picked->GetMapper()) {
        pickedPoly = vtkPolyData::SafeDownCast(picked->GetMapper()->GetInput());
    }
    if (!pickedPoly) return picked;

    for (const auto& kv : *vis->getCloudActorMap()) {
        vtkActor* reg = vtkActor::SafeDownCast(kv.second.actor);
        if (!reg || !reg->GetMapper()) continue;
        if (vtkPolyData::SafeDownCast(reg->GetMapper()->GetInput()) ==
            pickedPoly) {
            return reg;
        }
    }
    for (const auto& kv : *vis->getShapeActorMap()) {
        vtkActor* reg = vtkActor::SafeDownCast(kv.second);
        if (!reg || !reg->GetMapper()) continue;
        if (vtkPolyData::SafeDownCast(reg->GetMapper()->GetInput()) ==
            pickedPoly) {
            return reg;
        }
    }
    return picked;
}

//-----------------------------------------------------------------------------
cvSelectionHighlighter::cvSelectionHighlighter()
    : QObject(),
      cvGenericSelectionTool(),
      m_hoverOpacity(1.0),        // ParaView default: opaque
      m_preselectedOpacity(1.0),  // ParaView default: opaque
      m_selectedOpacity(1.0),     // ParaView default: opaque
      m_boundaryOpacity(1.0),     // ParaView default: opaque
      m_hoverPointSize(5),        // ParaView default: 5 for hover
      m_preselectedPointSize(5),  // ParaView default: 5 for preselection
      m_selectedPointSize(5),     // ParaView default: 5 for final selection
      m_boundaryPointSize(5),     // ParaView default: 5 for boundary
      m_hoverLineWidth(2),
      m_preselectedLineWidth(2),
      m_selectedLineWidth(8),
      m_boundaryLineWidth(2),
      m_enabled(true),
      m_pointLabelVisible(false),
      m_cellLabelVisible(false) {
    // ParaView default colors from utilities_remotingviews.xml ColorPalette:
    // - SelectionColor: 1.0 0.0 1.0 (Magenta)
    // - InteractiveSelectionColor: 0.5 0.0 1.0 (Purple/Violet)

    // Hover: Purple/Violet (InteractiveSelectionColor) - ParaView default
    m_hoverColor[0] = 0.5;  // Red = 127
    m_hoverColor[1] = 0.0;  // Green = 0
    m_hoverColor[2] = 1.0;  // Blue = 255 (Purple/Violet)

    // Pre-selected: Purple/Violet (InteractiveSelectionColor) - ParaView
    // default
    m_preselectedColor[0] = 0.5;
    m_preselectedColor[1] = 0.0;
    m_preselectedColor[2] = 1.0;

    // Selected: Magenta (SelectionColor) - ParaView default
    // Magenta is highly visible against any point cloud color
    m_selectedColor[0] = 1.0;  // Red = 255
    m_selectedColor[1] = 0.0;  // Green = 0
    m_selectedColor[2] = 1.0;  // Blue = 255 (Magenta)

    // Boundary: Magenta (same as SelectionColor) - ParaView default
    m_boundaryColor[0] = 1.0;
    m_boundaryColor[1] = 0.0;
    m_boundaryColor[2] = 1.0;

    m_hoverActorId = "__highlight_hover__";
    m_preselectedActorId = "__highlight_preselected__";
    m_selectedActorId = "__highlight_selected__";
    m_boundaryActorId = "__highlight_boundary__";

    CVLog::PrintVerbose(
            "[cvSelectionHighlighter] Initialized with ParaView default "
            "colors: Hover=Purple(0.5,0,1), Selected=Magenta(1,0,1)");
}

//-----------------------------------------------------------------------------
cvSelectionHighlighter::~cvSelectionHighlighter() {
    m_shuttingDown = true;
    clearHighlights();
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setVisualizer(ecvGenericVisualizer3D* viewer) {
    cvGenericSelectionTool::setVisualizer(viewer);
    pruneSelectionOverlayRenderers(
            getVtkVis() ? getVtkVis()->getCurrentRenderer() : nullptr);
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setHighlightColor(double r,
                                               double g,
                                               double b,
                                               HighlightMode mode) {
    double* color = nullptr;
    vtkSmartPointer<vtkActor>* actor = nullptr;

    switch (mode) {
        case HOVER:
            color = m_hoverColor;
            actor = &m_hoverActor;
            break;
        case PRESELECTED:
            color = m_preselectedColor;
            actor = &m_preselectedActor;
            break;
        case SELECTED:
            color = m_selectedColor;
            actor = &m_selectedActor;
            break;
        case BOUNDARY:
            color = m_boundaryColor;
            actor = &m_boundaryActor;
            break;
    }

    if (color) {
        // Check if color actually changed
        bool changed = (color[0] != r || color[1] != g || color[2] != b);

        color[0] = r;
        color[1] = g;
        color[2] = b;

        // Update existing actor's color immediately for real-time preview
        if (actor && *actor) {
            (*actor)->GetProperty()->SetColor(r, g, b);
            (*actor)->Modified();
            scheduleAllViewsUpdate();
        }

        // Emit signals for property change notification
        if (changed) {
            emit colorChanged(static_cast<int>(mode));
            emit propertiesChanged();
        }

        CVLog::PrintVerbose(
                QString("[cvSelectionHighlighter] Color set for mode %1: "
                        "RGB(%2, %3, %4)")
                        .arg(mode)
                        .arg(r)
                        .arg(g)
                        .arg(b));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setHighlightOpacity(double opacity,
                                                 HighlightMode mode) {
    double oldOpacity = 0.0;
    vtkSmartPointer<vtkActor>* actor = nullptr;

    switch (mode) {
        case HOVER:
            oldOpacity = m_hoverOpacity;
            m_hoverOpacity = opacity;
            actor = &m_hoverActor;
            break;
        case PRESELECTED:
            oldOpacity = m_preselectedOpacity;
            m_preselectedOpacity = opacity;
            actor = &m_preselectedActor;
            break;
        case SELECTED:
            oldOpacity = m_selectedOpacity;
            m_selectedOpacity = opacity;
            actor = &m_selectedActor;
            break;
        case BOUNDARY:
            oldOpacity = m_boundaryOpacity;
            m_boundaryOpacity = opacity;
            actor = &m_boundaryActor;
            break;
    }

    // Update existing actor's opacity immediately for real-time preview
    if (actor && *actor) {
        (*actor)->GetProperty()->SetOpacity(opacity);
        (*actor)->Modified();  // Mark actor as modified for VTK pipeline

        scheduleAllViewsUpdate();
    }

    // Emit signals for property change notification
    if (oldOpacity != opacity) {
        emit opacityChanged(static_cast<int>(mode));
        emit propertiesChanged();
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Opacity set for mode %1: %2")
                    .arg(mode)
                    .arg(opacity));
}

//-----------------------------------------------------------------------------
const double* cvSelectionHighlighter::getHighlightColor(
        HighlightMode mode) const {
    switch (mode) {
        case HOVER:
            return m_hoverColor;
        case PRESELECTED:
            return m_preselectedColor;
        case SELECTED:
            return m_selectedColor;
        case BOUNDARY:
            return m_boundaryColor;
        default:
            return nullptr;
    }
}

//-----------------------------------------------------------------------------
double cvSelectionHighlighter::getHighlightOpacity(HighlightMode mode) const {
    switch (mode) {
        case HOVER:
            return m_hoverOpacity;
        case PRESELECTED:
            return m_preselectedOpacity;
        case SELECTED:
            return m_selectedOpacity;
        case BOUNDARY:
            return m_boundaryOpacity;
        default:
            return 1.0;  // Default to opaque
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setEnabled(bool enabled) {
    m_enabled = enabled;
    if (!m_enabled) {
        clearHighlights();
    }
}

//-----------------------------------------------------------------------------
bool cvSelectionHighlighter::highlightSelection(
        const vtkSmartPointer<vtkIdTypeArray>& selection,
        int fieldAssociation,
        HighlightMode mode) {
    if (!m_enabled) {
        CVLog::Warning("[cvSelectionHighlighter] Highlighter is disabled");
        return false;
    }

    if (!m_viewer) {
        CVLog::Warning("[cvSelectionHighlighter] No viewer available");
        return false;
    }

    if (!selection) {
        CVLog::Warning("[cvSelectionHighlighter] Selection array is null");
        return false;
    }

    vtkPolyData* polyData = nullptr;
    cvViewSelectionManager* mgr = cvViewSelectionManager::instance();
    if (mgr) {
        const cvSelectionData current = mgr->currentSelection();
        if (!current.isEmpty()) {
            polyData = getPolyDataForSelection(&current);
        }
    }
    if (!polyData) {
        polyData = getPolyDataForSelection();
    }
    if (!polyData) {
        CVLog::Error(
                "[cvSelectionHighlighter::highlightSelection] No polyData "
                "available for highlighting - this is a critical error!");
        return false;
    }

    CVLog::PrintVerbose(QString("[cvSelectionHighlighter] Got polyData with %1 "
                                "cells, %2 points")
                                .arg(polyData->GetNumberOfCells())
                                .arg(polyData->GetNumberOfPoints()));

    // Delegate to the explicit polyData overload
    return highlightSelection(polyData, selection, fieldAssociation, mode);
}

//-----------------------------------------------------------------------------
bool cvSelectionHighlighter::highlightSelection(
        vtkPolyData* polyData,
        const vtkSmartPointer<vtkIdTypeArray>& selection,
        int fieldAssociation,
        HighlightMode mode) {
    if (!m_enabled || !m_viewer || !polyData || !selection) {
        return false;
    }

    if (selection->GetNumberOfTuples() == 0) {
        return false;
    }

    // Create highlight actor
    // IMPORTANT: Store as vtkSmartPointer to prevent premature deletion!
    vtkSmartPointer<vtkActor> actor =
            createHighlightActor(polyData, selection, fieldAssociation, mode);
    // If actor is nullptr, it's a normal case (e.g., invalid selection IDs),
    // not an error - just return false silently
    if (!actor) {
        return false;
    }

    // Remove old actor and add new one (ParaView-style multi-level)
    QString actorId;

    switch (mode) {
        case HOVER:
            removeActorFromVisualizer(m_hoverActorId);
            m_hoverActor = actor;
            actorId = m_hoverActorId;
            break;

        case PRESELECTED:
            removeActorFromVisualizer(m_preselectedActorId);
            m_preselectedActor = actor;
            actorId = m_preselectedActorId;
            break;

        case SELECTED:
            removeActorFromVisualizer(m_selectedActorId);
            m_selectedActor = actor;
            actorId = m_selectedActorId;
            break;

        case BOUNDARY:
            removeActorFromVisualizer(m_boundaryActorId);
            m_boundaryActor = actor;
            actorId = m_boundaryActorId;
            break;
    }

    if (mode == SELECTED) {
        // ParaView: visible highlight is the VtkVis overlay (wireframe/points).
        vtkSmartPointer<vtkPolyData> overlayPoly;
        int overlayKind = cvSelectionHighlighter::SelectionOverlayNone;
        Visualization::VtkVis* vis = getVtkVis();
        if (vis) {
            overlayKind = static_cast<int>(upsertParaViewSelectionOverlay(
                    vis, polyData, selection, fieldAssociation, overlayPoly,
                    kSelectionOverlayId, false));
        }
        if (!m_shuttingDown) {
            emit selectionOverlayUpdated(overlayPoly, overlayKind);
            scheduleAllViewsUpdate();
        }
    } else if (mode == HOVER || mode == PRESELECTED) {
        // Hover/preselect: per-view overlay only (never addActorToVisualizer).
        removeActorFromVisualizer(actorId);
        Visualization::VtkVis* vis = getVtkVis();
        if (vis) {
            vtkSmartPointer<vtkPolyData> overlayPoly;
            const bool interactive = (mode == HOVER || mode == PRESELECTED);
            upsertParaViewSelectionOverlay(vis, polyData, selection,
                                           fieldAssociation, overlayPoly,
                                           overlayIdForMode(mode), interactive);
            vis->UpdateScreen();
            scheduleAllViewsUpdate();
        }
    } else {
        addActorToVisualizer(actor, actorId);
    }

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionHighlighter::highlightSelection(
        const cvSelectionData& selectionData, HighlightMode mode) {
    // High-level interface that accepts cvSelectionData directly
    // This keeps UI code (like MainWindow) free from VTK types

    if (selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionHighlighter] Selection data is empty");
        return false;
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Highlighting %1 %2 in mode %3")
                    .arg(selectionData.count())
                    .arg(selectionData.fieldTypeString())
                    .arg(mode));

    vtkPolyData* polyData = getPolyDataForSelection(&selectionData);
    if (!polyData && selectionData.primaryActor()) {
        if (auto* mapper = selectionData.primaryActor()->GetMapper()) {
            polyData = vtkPolyData::SafeDownCast(mapper->GetInput());
        }
    }
    if (!polyData) {
        CVLog::Print(
                "[cvSelectionHighlighter] No polyData for selection highlight "
                "(hasActorInfo=" +
                QString(selectionData.hasActorInfo() ? "yes" : "no") + ")");
        return false;
    }

    Visualization::VtkVis* vis = getVtkVis();
    m_highlightSourceActor =
            resolveRegisteredMeshActor(vis, selectionData.primaryActor());

    const bool ok = highlightSelection(
            polyData, selectionData.vtkArray(),
            static_cast<int>(selectionData.fieldAssociation()), mode);
    m_highlightSourceActor = nullptr;
    CVLog::Print(
            QString("[cvSelectionHighlighter] highlightSelection(%1 ids) -> %2 "
                    "polyCells=%3")
                    .arg(selectionData.count())
                    .arg(ok ? "OK" : "FAIL")
                    .arg(polyData->GetNumberOfCells()));
    return ok;
}

//-----------------------------------------------------------------------------
bool cvSelectionHighlighter::highlightMultiColorSelections(
        vtkPolyData* polyData,
        const QVector<QPair<vtkSmartPointer<vtkIdTypeArray>, QColor>>&
                selectionsWithColors,
        int fieldAssociation,
        HighlightMode mode) {
    if (!m_enabled || !m_viewer || !polyData ||
        selectionsWithColors.isEmpty()) {
        return false;
    }

    bool isPointSelection = (fieldAssociation != 0);
    vtkIdType maxValidId = isPointSelection ? polyData->GetNumberOfPoints()
                                            : polyData->GetNumberOfCells();

    vtkSmartPointer<vtkAppendFilter> appendFilter =
            vtkSmartPointer<vtkAppendFilter>::New();
    appendFilter->MergePointsOff();

    bool hasValidExtraction = false;

    for (const auto& pair : selectionsWithColors) {
        vtkIdTypeArray* selection = pair.first;
        const QColor& color = pair.second;

        if (!selection || selection->GetNumberOfTuples() == 0) {
            continue;
        }

        vtkSmartPointer<vtkIdTypeArray> validSelection =
                vtkSmartPointer<vtkIdTypeArray>::New();
        for (vtkIdType i = 0; i < selection->GetNumberOfTuples(); ++i) {
            vtkIdType id = selection->GetValue(i);
            if (id >= 0 && id < maxValidId) {
                validSelection->InsertNextValue(id);
            }
        }
        if (validSelection->GetNumberOfTuples() == 0) {
            continue;
        }

        vtkSmartPointer<vtkSelectionNode> selectionNode =
                createSelectionNode(validSelection, fieldAssociation);
        if (!selectionNode) {
            continue;
        }

        vtkSmartPointer<vtkSelection> vtkSel =
                vtkSmartPointer<vtkSelection>::New();
        vtkSel->AddNode(selectionNode);

        vtkSmartPointer<vtkExtractSelection> extractor =
                vtkSmartPointer<vtkExtractSelection>::New();
        extractor->SetInputData(0, polyData);
        extractor->SetInputData(1, vtkSel);
        extractor->Update();

        vtkUnstructuredGrid* extracted =
                vtkUnstructuredGrid::SafeDownCast(extractor->GetOutput());
        if (!extracted || (extracted->GetNumberOfCells() == 0 &&
                           extracted->GetNumberOfPoints() == 0)) {
            continue;
        }

        vtkSmartPointer<vtkUnsignedCharArray> colorArray =
                vtkSmartPointer<vtkUnsignedCharArray>::New();
        colorArray->SetName("vtkSelectionColor");
        colorArray->SetNumberOfComponents(3);

        unsigned char r = static_cast<unsigned char>(color.red());
        unsigned char g = static_cast<unsigned char>(color.green());
        unsigned char b = static_cast<unsigned char>(color.blue());

        if (isPointSelection) {
            vtkIdType numPoints = extracted->GetNumberOfPoints();
            colorArray->SetNumberOfTuples(numPoints);
            for (vtkIdType i = 0; i < numPoints; ++i) {
                colorArray->SetTuple3(i, r, g, b);
            }
            extracted->GetPointData()->SetScalars(colorArray);
        } else {
            vtkIdType numCells = extracted->GetNumberOfCells();
            colorArray->SetNumberOfTuples(numCells);
            for (vtkIdType i = 0; i < numCells; ++i) {
                colorArray->SetTuple3(i, r, g, b);
            }
            extracted->GetCellData()->SetScalars(colorArray);
        }

        appendFilter->AddInputData(extracted);
        hasValidExtraction = true;
    }

    if (!hasValidExtraction) {
        return false;
    }

    appendFilter->Update();
    vtkDataSet* merged = appendFilter->GetOutput();
    if (!merged ||
        (merged->GetNumberOfCells() == 0 && merged->GetNumberOfPoints() == 0)) {
        return false;
    }

    vtkSmartPointer<vtkDataSetMapper> mapper =
            vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(merged);
    mapper->ScalarVisibilityOn();
    mapper->SetColorModeToDirectScalars();
    if (isPointSelection) {
        mapper->SetScalarModeToUsePointData();
    } else {
        mapper->SetScalarModeToUseCellData();
    }

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkProperty* prop = actor->GetProperty();
    double opacity = 1.0;

    switch (mode) {
        case HOVER:
            opacity = m_hoverOpacity;
            prop->SetLineWidth(static_cast<float>(m_hoverLineWidth));
            prop->SetPointSize(static_cast<float>(m_hoverPointSize));
            break;
        case PRESELECTED:
            opacity = m_preselectedOpacity;
            prop->SetLineWidth(static_cast<float>(m_preselectedLineWidth));
            prop->SetPointSize(static_cast<float>(m_preselectedPointSize));
            break;
        case SELECTED:
            opacity = m_selectedOpacity;
            prop->SetLineWidth(static_cast<float>(m_selectedLineWidth));
            prop->SetPointSize(static_cast<float>(m_selectedPointSize));
            break;
        case BOUNDARY:
            opacity = m_boundaryOpacity;
            prop->SetLineWidth(static_cast<float>(m_boundaryLineWidth));
            prop->SetPointSize(static_cast<float>(m_boundaryPointSize));
            break;
    }

    prop->SetOpacity(opacity);
    prop->SetRenderLinesAsTubes(true);
    prop->SetAmbient(0.6);
    prop->SetDiffuse(0.8);
    prop->SetSpecular(0.5);
    prop->SetSpecularPower(30.0);
    prop->SetRenderLinesAsTubes(true);
    prop->SetRenderPointsAsSpheres(true);
    prop->EdgeVisibilityOn();
    prop->SetEdgeColor(1.0, 1.0, 1.0);

    if (fieldAssociation == 0) {
        prop->SetRepresentationToWireframe();
    } else {
        prop->SetRepresentationToPoints();
    }

    actor->SetPickable(false);

    QString actorId;
    switch (mode) {
        case HOVER:
            removeActorFromVisualizer(m_hoverActorId);
            m_hoverActor = actor;
            actorId = m_hoverActorId;
            break;
        case PRESELECTED:
            removeActorFromVisualizer(m_preselectedActorId);
            m_preselectedActor = actor;
            actorId = m_preselectedActorId;
            break;
        case SELECTED:
            removeActorFromVisualizer(m_selectedActorId);
            m_selectedActor = actor;
            actorId = m_selectedActorId;
            break;
        case BOUNDARY:
            removeActorFromVisualizer(m_boundaryActorId);
            m_boundaryActor = actor;
            actorId = m_boundaryActorId;
            break;
    }

    addActorToVisualizer(actor, actorId);

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Multi-color highlight: %1 "
                    "sub-selections in mode %2")
                    .arg(selectionsWithColors.size())
                    .arg(mode));

    return true;
}

//-----------------------------------------------------------------------------
bool cvSelectionHighlighter::highlightElement(vtkPolyData* polyData,
                                              vtkIdType elementId,
                                              int fieldAssociation) {
    if (!m_enabled) {
        CVLog::Warning(
                "[cvSelectionHighlighter::highlightElement] Highlighter is "
                "disabled");
        return false;
    }
    if (!m_viewer) {
        CVLog::Warning(
                "[cvSelectionHighlighter::highlightElement] No viewer set - "
                "call setVisualizer() first!");
        return false;
    }
    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionHighlighter::highlightElement] No polyData "
                "provided");
        return false;
    }

    // Create single-element selection array
    vtkSmartPointer<vtkIdTypeArray> selection =
            vtkSmartPointer<vtkIdTypeArray>::New();
    selection->InsertNextValue(elementId);

    // Use HOVER mode for single element highlighting
    bool result =
            highlightSelection(polyData, selection, fieldAssociation, HOVER);

    // Smart pointer handles cleanup automatically
    return result;
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::clearHighlights() {
    if (m_shuttingDown) {
        m_hoverActor = nullptr;
        m_preselectedActor = nullptr;
        m_selectedActor = nullptr;
        m_boundaryActor = nullptr;
        m_highlightInstances.clear();
        return;
    }

    pruneSelectionOverlayRenderers(
            getVtkVis() ? getVtkVis()->getCurrentRenderer() : nullptr);

    if (Visualization::VtkVis* vis = getVtkVis()) {
        clearAllSelectionOverlays(vis);
    }

    removeActorFromVisualizer(m_hoverActorId);
    removeActorFromVisualizer(m_preselectedActorId);
    removeActorFromVisualizer(m_selectedActorId);
    removeActorFromVisualizer(m_boundaryActorId);

    stripHighlightTemplatesFromVisualizer(getVtkVis(), m_hoverActor,
                                          m_preselectedActor, m_selectedActor,
                                          m_boundaryActor);

    m_hoverActor = nullptr;
    m_preselectedActor = nullptr;
    m_selectedActor = nullptr;
    m_boundaryActor = nullptr;

    if (!m_shuttingDown) {
        emit selectionOverlayUpdated(nullptr, SelectionOverlayNone);
        emit highlightsCleared();
        scheduleAllViewsUpdate();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::clearHighlight(HighlightMode mode) {
    switch (mode) {
        case HOVER:
            removeActorFromVisualizer(m_hoverActorId);
            m_hoverActor = nullptr;
            break;
        case PRESELECTED:
            removeActorFromVisualizer(m_preselectedActorId);
            m_preselectedActor = nullptr;
            break;
        case SELECTED:
            removeActorFromVisualizer(m_selectedActorId);
            m_selectedActor = nullptr;
            break;
        case BOUNDARY:
            removeActorFromVisualizer(m_boundaryActorId);
            m_boundaryActor = nullptr;
            break;
    }

    scheduleAllViewsUpdate();
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::clearHoverHighlight() {
    removeActorFromVisualizer(m_hoverActorId);
    m_hoverActor = nullptr;
    if (Visualization::VtkVis* vis = getVtkVis()) {
        clearSelectionOverlay(vis, kHoverOverlayId);
        vis->UpdateScreen();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setHighlightsVisible(bool visible) {
    auto setVis = [visible](const QString& id,
                            QHash<QString, QVector<HighlightActorInstance>>&
                                    instances) {
        auto it = instances.find(id);
        if (it == instances.end()) return;
        for (const auto& inst : it.value()) {
            if (inst.actor) inst.actor->SetVisibility(visible ? 1 : 0);
        }
    };
    setVis(m_hoverActorId, m_highlightInstances);
    setVis(m_preselectedActorId, m_highlightInstances);
    setVis(m_selectedActorId, m_highlightInstances);
    setVis(m_boundaryActorId, m_highlightInstances);

    if (!visible) {
        stripHighlightTemplatesFromVisualizer(getVtkVis(), m_hoverActor,
                                              m_preselectedActor,
                                              m_selectedActor, m_boundaryActor);
        if (cvViewSelectionManager* mgr = cvViewSelectionManager::instance()) {
            if (cvSelectionPipeline* pipeline = mgr->getPipeline()) {
                pipeline->invalidateCachedSelection();
            }
        }
    }

    scheduleAllViewsUpdate();
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::applyParaViewHighlightStyle(
        vtkProperty* prop,
        int fieldAssociation,
        HighlightMode mode,
        bool applySolidColor) const {
    if (!prop) return;

    const double* color = m_selectedColor;
    float lineWidth = static_cast<float>(m_selectedLineWidth);
    float pointSize = static_cast<float>(m_selectedPointSize);
    double opacity = m_selectedOpacity;

    switch (mode) {
        case HOVER:
            color = m_hoverColor;
            lineWidth = static_cast<float>(m_hoverLineWidth);
            pointSize = static_cast<float>(m_hoverPointSize);
            opacity = m_hoverOpacity;
            break;
        case PRESELECTED:
            color = m_preselectedColor;
            lineWidth = static_cast<float>(m_preselectedLineWidth);
            pointSize = static_cast<float>(m_preselectedPointSize);
            opacity = m_preselectedOpacity;
            break;
        case SELECTED:
            color = m_selectedColor;
            lineWidth = static_cast<float>(m_selectedLineWidth);
            pointSize = static_cast<float>(m_selectedPointSize);
            opacity = m_selectedOpacity;
            break;
        case BOUNDARY:
            color = m_boundaryColor;
            lineWidth = static_cast<float>(m_boundaryLineWidth);
            pointSize = static_cast<float>(m_boundaryPointSize);
            opacity = m_boundaryOpacity;
            break;
    }

    if (applySolidColor) {
        prop->SetColor(color[0], color[1], color[2]);
    }
    prop->SetOpacity(opacity);
    prop->SetLineWidth(lineWidth);
    prop->SetPointSize(pointSize);
    prop->SetAmbient(1.0);
    prop->SetDiffuse(0.0);
    prop->SetSpecular(0.0);
    prop->SetSpecularPower(1.0);
    prop->SetRenderLinesAsTubes(false);
    prop->SetRenderPointsAsSpheres(false);

    prop->LightingOff();
    prop->BackfaceCullingOff();

    if (fieldAssociation == 0) {
        // ParaView Render View: cell selections are a colored wireframe.
        prop->SetRepresentationToWireframe();
        prop->EdgeVisibilityOff();
    } else {
        prop->EdgeVisibilityOff();
        prop->SetRepresentationToPoints();
    }
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkActor> cvSelectionHighlighter::createHighlightActor(
        vtkPolyData* polyData,
        vtkIdTypeArray* selection,
        int fieldAssociation,
        HighlightMode mode) {
    if (!polyData || !selection) {
        return {};
    }

    // Validate selection IDs against polyData bounds and filter out invalid IDs
    // This prevents extraction failures and error logs for invalid selections
    // (e.g., when selection IDs are out of range or data has changed)
    bool isPointSelection =
            (fieldAssociation != 0);  // 0 = CELL, non-zero = POINT
    vtkIdType maxValidId = isPointSelection ? polyData->GetNumberOfPoints()
                                            : polyData->GetNumberOfCells();

    // Filter out invalid IDs to prevent extraction failures
    vtkSmartPointer<vtkIdTypeArray> validSelection =
            vtkSmartPointer<vtkIdTypeArray>::New();
    for (vtkIdType i = 0; i < selection->GetNumberOfTuples(); ++i) {
        vtkIdType id = selection->GetValue(i);
        if (id >= 0 && id < maxValidId) {
            validSelection->InsertNextValue(id);
        }
    }

    // If no valid IDs remain, this is a normal case (invalid selection),
    // not an error - return nullptr silently
    if (validSelection->GetNumberOfTuples() == 0) {
        CVLog::PrintVerbose(
                QString("[cvSelectionHighlighter::createHighlightActor] No "
                        "valid "
                        "selection IDs (all %1 IDs are out of range [0, %2))")
                        .arg(selection->GetNumberOfTuples())
                        .arg(maxValidId));
        return {};
    }

    // Create selection node using validated IDs
    vtkSmartPointer<vtkSelectionNode> selectionNode =
            createSelectionNode(validSelection, fieldAssociation);
    if (!selectionNode) {
        CVLog::Error(
                "[cvSelectionHighlighter::createHighlightActor] Failed to "
                "create selection node");
        return {};
    }

    // Create selection
    vtkSmartPointer<vtkSelection> vtkSel = vtkSmartPointer<vtkSelection>::New();
    vtkSel->AddNode(selectionNode);

    // Extract selected elements
    vtkSmartPointer<vtkExtractSelection> extractor =
            vtkSmartPointer<vtkExtractSelection>::New();
    extractor->SetInputData(0, polyData);
    extractor->SetInputData(1, vtkSel);

    extractor->Update();

    vtkDataSet* extracted =
            vtkDataSet::SafeDownCast(extractor->GetOutputDataObject(0));
    if (!extracted) {
        CVLog::Error(
                "[cvSelectionHighlighter::createHighlightActor] Extraction "
                "failed: extracted is nullptr");
        return {};
    }

    vtkSmartPointer<vtkPolyData> highlightPoly =
            vtkSmartPointer<vtkPolyData>::New();
    if (auto* poly = vtkPolyData::SafeDownCast(extracted)) {
        highlightPoly->ShallowCopy(poly);
    } else {
        auto geom = vtkSmartPointer<vtkGeometryFilter>::New();
        geom->SetInputData(extracted);
        geom->Update();
        highlightPoly->ShallowCopy(geom->GetOutput());
    }

    if (highlightPoly->GetNumberOfCells() == 0 &&
        highlightPoly->GetNumberOfPoints() == 0) {
        CVLog::Print(
                "[cvSelectionHighlighter::createHighlightActor] extract empty");
        return {};
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] extract geometry: cells=%1 "
                    "points=%2")
                    .arg(highlightPoly->GetNumberOfCells())
                    .arg(highlightPoly->GetNumberOfPoints()));

    vtkSmartPointer<vtkDataSetMapper> mapper =
            vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(highlightPoly);
    mapper->ScalarVisibilityOff();

    vtkSmartPointer<vtkPVLODActor> actor =
            vtkSmartPointer<vtkPVLODActor>::New();
    actor->SetMapper(mapper);
    VtkRendering::BuildAndAttachLODMapper(actor, highlightPoly);
    applyParaViewHighlightStyle(actor->GetProperty(), fieldAssociation, mode);
    if (fieldAssociation == 0 && mode == SELECTED) {
        mapper->SetResolveCoincidentTopologyToPolygonOffset();
        mapper->SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -2);
    }

    // CRITICAL: Set highlight actor as NOT pickable!
    // This prevents hardware selection from picking highlight actors.
    // When users try to add/subtract selection in highlighted areas,
    // we want them to select from the original data, not the highlight actor.
    // Without this, subtract selection would select IDs from the highlight
    // actor (0 to N-1) instead of the original point cloud IDs.
    actor->SetPickable(false);

    return actor;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelectionNode> cvSelectionHighlighter::createSelectionNode(
        vtkIdTypeArray* selection, int fieldAssociation) {
    if (!selection) {
        return {};
    }

    vtkSmartPointer<vtkSelectionNode> node =
            vtkSmartPointer<vtkSelectionNode>::New();
    node->SetContentType(vtkSelectionNode::INDICES);

    if (fieldAssociation == 0) {
        node->SetFieldType(vtkSelectionNode::CELL);
    } else {
        node->SetFieldType(vtkSelectionNode::POINT);
    }

    node->SetSelectionList(selection);

    return node;
}

// Match primary mesh actor transform so highlight geometry aligns in world
// space.
static void syncHighlightActorTransform(vtkActor* highlight, vtkActor* source) {
    if (!highlight || !source) return;
    highlight->SetPosition(source->GetPosition());
    highlight->SetOrientation(source->GetOrientation());
    highlight->SetScale(source->GetScale());
    highlight->SetOrigin(source->GetOrigin());
    if (vtkLinearTransform* linear = source->GetUserTransform()) {
        auto copy = vtkSmartPointer<vtkTransform>::New();
        copy->SetMatrix(linear->GetMatrix());
        highlight->SetUserTransform(copy);
    } else {
        highlight->SetUserTransform(nullptr);
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::addActorToVisualizer(vtkActor* actor,
                                                  const QString& id) {
    if (!m_viewer) {
        CVLog::Error(
                "[cvSelectionHighlighter::addActorToVisualizer] No viewer!");
        return;
    }

    if (!actor) {
        CVLog::Error(
                "[cvSelectionHighlighter::addActorToVisualizer] No actor!");
        return;
    }

    Visualization::VtkVis* pclVis = getVtkVis();
    if (!pclVis) {
        CVLog::Error(
                "[cvSelectionHighlighter::addActorToVisualizer] Visualizer is "
                "not VtkVis");
        return;
    }

    vtkActor* sourceActor = m_highlightSourceActor;
    // Never resolve mgr->currentSelection().primaryActor() here — it may
    // belong to a different render view and caused SIGSEGV in getIdByActor.

    QVector<HighlightActorInstance> instances;

    vtkRenderer* sceneRenderer = pclVis->getCurrentRenderer();
    pruneSelectionOverlayRenderers(sceneRenderer);

    if (sceneRenderer) {
        HighlightActorInstance inst;
        inst.renderer = sceneRenderer;
        inst.actor = cloneHighlightActor(actor);
        if (inst.actor) {
            syncHighlightActorTransform(inst.actor, sourceActor);
            inst.actor->ForceOpaqueOn();
            inst.actor->ForceTranslucentOff();
            sceneRenderer->AddActor(inst.actor);
            inst.actor->SetVisibility(1);
            instances.append(inst);
            double b[6] = {0, 0, 0, 0, 0, 0};
            inst.actor->GetBounds(b);
            CVLog::Print(
                    QString("[cvSelectionHighlighter] Added highlight to scene "
                            "(layer=0) id=%1 bounds=[%2,%3,%4,%5,%6,%7] "
                            "rendererActors=%8")
                            .arg(id)
                            .arg(b[0])
                            .arg(b[1])
                            .arg(b[2])
                            .arg(b[3])
                            .arg(b[4])
                            .arg(b[5])
                            .arg(sceneRenderer->GetActors()
                                         ? sceneRenderer->GetActors()
                                                   ->GetNumberOfItems()
                                         : 0));
        }
    }

    stripHighlightTemplatesFromVisualizer(pclVis, actor, nullptr, nullptr,
                                          nullptr);

    m_highlightInstances[id] = instances;
    if (instances.isEmpty()) {
        CVLog::Warning(QString("[cvSelectionHighlighter] No renderer accepted "
                               "highlight clone for id=%1")
                               .arg(id));
    }

    if (!m_shuttingDown) {
        emit highlightActorAdded(actor);
        scheduleAllViewsUpdate();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::removeActorFromVisualizer(const QString& id) {
    if (m_shuttingDown) {
        m_highlightInstances.remove(id);
        if (id == m_hoverActorId)
            m_hoverActor = nullptr;
        else if (id == m_preselectedActorId)
            m_preselectedActor = nullptr;
        else if (id == m_selectedActorId)
            m_selectedActor = nullptr;
        else if (id == m_boundaryActorId)
            m_boundaryActor = nullptr;
        return;
    }

    if (!m_viewer) {
        return;
    }

    vtkActor* actor = nullptr;
    if (id == m_hoverActorId && m_hoverActor) {
        actor = m_hoverActor;
    } else if (id == m_preselectedActorId && m_preselectedActor) {
        actor = m_preselectedActor;
    } else if (id == m_selectedActorId && m_selectedActor) {
        actor = m_selectedActor;
    } else if (id == m_boundaryActorId && m_boundaryActor) {
        actor = m_boundaryActor;
    }

    auto it = m_highlightInstances.find(id);
    if (it != m_highlightInstances.end()) {
        for (const auto& inst : it.value()) {
            if (inst.renderer && inst.actor) {
                inst.renderer->RemoveActor(inst.actor);
            }
        }
        m_highlightInstances.remove(id);
    }

    if (!actor) return;

    if (id == m_selectedActorId) {
        clearSelectionOverlay(getVtkVis());
        if (!m_shuttingDown) {
            emit selectionOverlayUpdated(nullptr, SelectionOverlayNone);
        }
    }

    if (!m_shuttingDown) {
        emit highlightActorRemoved(actor);
        scheduleAllViewsUpdate();
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Removed highlight actor: %1")
                    .arg(id));
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setPointSize(int size, HighlightMode mode) {
    int oldSize = 0;
    vtkSmartPointer<vtkActor>* actor = nullptr;

    switch (mode) {
        case HOVER:
            oldSize = m_hoverPointSize;
            m_hoverPointSize = size;
            actor = &m_hoverActor;
            break;
        case PRESELECTED:
            oldSize = m_preselectedPointSize;
            m_preselectedPointSize = size;
            actor = &m_preselectedActor;
            break;
        case SELECTED:
            oldSize = m_selectedPointSize;
            m_selectedPointSize = size;
            actor = &m_selectedActor;
            break;
        case BOUNDARY:
            oldSize = m_boundaryPointSize;
            m_boundaryPointSize = size;
            actor = &m_boundaryActor;
            break;
    }

    // Update existing actor's point size immediately for real-time preview
    if (actor && *actor) {
        (*actor)->GetProperty()->SetPointSize(static_cast<float>(size));
    }

    // Emit signals for property change notification
    if (oldSize != size) {
        emit pointSizeChanged(static_cast<int>(mode));
        emit propertiesChanged();
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Point size set for mode %1: %2")
                    .arg(mode)
                    .arg(size));
}

//-----------------------------------------------------------------------------
int cvSelectionHighlighter::getPointSize(HighlightMode mode) const {
    switch (mode) {
        case HOVER:
            return m_hoverPointSize;
        case PRESELECTED:
            return m_preselectedPointSize;
        case SELECTED:
            return m_selectedPointSize;
        case BOUNDARY:
            return m_boundaryPointSize;
        default:
            return 5;  // ParaView default: SelectionPointSize = 5
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setLineWidth(int width, HighlightMode mode) {
    int oldWidth = 0;
    vtkSmartPointer<vtkActor>* actor = nullptr;

    switch (mode) {
        case HOVER:
            oldWidth = m_hoverLineWidth;
            m_hoverLineWidth = width;
            actor = &m_hoverActor;
            break;
        case PRESELECTED:
            oldWidth = m_preselectedLineWidth;
            m_preselectedLineWidth = width;
            actor = &m_preselectedActor;
            break;
        case SELECTED:
            oldWidth = m_selectedLineWidth;
            m_selectedLineWidth = width;
            actor = &m_selectedActor;
            break;
        case BOUNDARY:
            oldWidth = m_boundaryLineWidth;
            m_boundaryLineWidth = width;
            actor = &m_boundaryActor;
            break;
    }

    // Update existing actor's line width immediately for real-time preview
    if (actor && *actor) {
        (*actor)->GetProperty()->SetLineWidth(static_cast<float>(width));
    }

    // Emit signals for property change notification
    if (oldWidth != width) {
        emit lineWidthChanged(static_cast<int>(mode));
        emit propertiesChanged();
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Line width set for mode %1: %2")
                    .arg(mode)
                    .arg(width));
}

//-----------------------------------------------------------------------------
int cvSelectionHighlighter::getLineWidth(HighlightMode mode) const {
    switch (mode) {
        case HOVER:
            return m_hoverLineWidth;
        case PRESELECTED:
            return m_preselectedLineWidth;
        case SELECTED:
            return m_selectedLineWidth;
        case BOUNDARY:
            return m_boundaryLineWidth;
        default:
            return 2;
    }
}

//-----------------------------------------------------------------------------
const SelectionLabelProperties& cvSelectionHighlighter::getLabelProperties(
        bool interactive) const {
    return interactive ? m_interactiveLabelProperties : m_labelProperties;
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setLabelProperties(
        const SelectionLabelProperties& props, bool interactive) {
    SelectionLabelProperties& target =
            interactive ? m_interactiveLabelProperties : m_labelProperties;

    // Check if properties changed (including label font properties)
    bool changed = (target.opacity != props.opacity ||
                    target.pointSize != props.pointSize ||
                    target.lineWidth != props.lineWidth ||
                    target.pointLabelFontSize != props.pointLabelFontSize ||
                    target.pointLabelColor != props.pointLabelColor ||
                    target.pointLabelBold != props.pointLabelBold ||
                    target.pointLabelItalic != props.pointLabelItalic ||
                    target.pointLabelShadow != props.pointLabelShadow ||
                    target.cellLabelFontSize != props.cellLabelFontSize ||
                    target.cellLabelColor != props.cellLabelColor ||
                    target.cellLabelBold != props.cellLabelBold ||
                    target.cellLabelItalic != props.cellLabelItalic ||
                    target.cellLabelShadow != props.cellLabelShadow);

    target = props;

    // Apply relevant properties to highlight modes
    HighlightMode mode = interactive ? HOVER : SELECTED;
    setHighlightOpacity(props.opacity, mode);
    setPointSize(props.pointSize, mode);
    setLineWidth(props.lineWidth, mode);

    // Also apply to related mode
    if (interactive) {
        setHighlightOpacity(props.opacity, PRESELECTED);
        setPointSize(props.pointSize, PRESELECTED);
        setLineWidth(props.lineWidth, PRESELECTED);
    } else {
        setHighlightOpacity(props.opacity, BOUNDARY);
        setPointSize(props.pointSize, BOUNDARY);
        setLineWidth(props.lineWidth, BOUNDARY);
    }

    if (changed) {
        // Update existing label actors with new properties
        if (m_pointLabelVisible) {
            updateLabelActor(true);  // true = point labels
        }
        if (m_cellLabelVisible) {
            updateLabelActor(false);  // false = cell labels
        }

        emit labelPropertiesChanged(interactive);
        emit propertiesChanged();
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionPropertiesWidget] Label properties applied: "
                    "opacity=%1, pointSize=%2, lineWidth=%3")
                    .arg(props.opacity)
                    .arg(props.pointSize)
                    .arg(props.lineWidth));
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setPointLabelArray(const QString& arrayName,
                                                bool visible) {
    m_pointLabelArrayName = arrayName;
    m_pointLabelVisible = visible && !arrayName.isEmpty();

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Point label array set: "
                    "'%1', visible=%2")
                    .arg(arrayName)
                    .arg(m_pointLabelVisible));

    // Update label rendering
    updateLabelActor(true);  // true = point labels

    // Trigger re-render
    emit labelPropertiesChanged(false);
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setCellLabelArray(const QString& arrayName,
                                               bool visible) {
    m_cellLabelArrayName = arrayName;
    m_cellLabelVisible = visible && !arrayName.isEmpty();

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Cell label array set: "
                    "'%1', visible=%2")
                    .arg(arrayName)
                    .arg(m_cellLabelVisible));

    // Update label rendering
    updateLabelActor(false);  // false = cell labels

    // Trigger re-render
    emit labelPropertiesChanged(false);
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::updateLabelActor(bool isPointLabels) {
    Visualization::VtkVis* pclVis = getVtkVis();
    if (!pclVis) {
        CVLog::Warning("[cvSelectionHighlighter] No visualizer for labels");
        return;
    }

    vtkRenderer* renderer = pclVis->getRendererCollection()->GetFirstRenderer();
    if (!renderer) {
        CVLog::Warning("[cvSelectionHighlighter] No renderer for labels");
        return;
    }

    // Get current label configuration
    QString arrayName =
            isPointLabels ? m_pointLabelArrayName : m_cellLabelArrayName;
    bool visible = isPointLabels ? m_pointLabelVisible : m_cellLabelVisible;
    vtkSmartPointer<vtkActor2D>& labelActor =
            isPointLabels ? m_pointLabelActor : m_cellLabelActor;

    // Remove existing label actor
    if (labelActor) {
        renderer->RemoveActor2D(labelActor);
        labelActor = nullptr;
    }

    if (!visible || arrayName.isEmpty()) {
        // Labels disabled - ensure proper cleanup and refresh
        CVLog::PrintVerbose(
                QString("[cvSelectionHighlighter] Clearing %1 labels")
                        .arg(isPointLabels ? "point" : "cell"));
        // Force render to update the view
        scheduleAllViewsUpdate();
        return;
    }

    // Get the current selection highlight actor's data
    vtkSmartPointer<vtkActor>& highlightActor = m_selectedActor;
    if (!highlightActor) {
        CVLog::Warning(
                "[cvSelectionHighlighter] No highlight actor for labels");
        return;
    }

    vtkMapper* mapper = highlightActor->GetMapper();
    if (!mapper) {
        return;
    }

    vtkDataSet* data = mapper->GetInput();
    if (!data || data->GetNumberOfPoints() == 0) {
        return;
    }

    // ParaView-style: Use vtkMaskPoints to limit number of labels for
    // performance This prevents lag when there are many selected points/cells
    // Reference: ParaView/Remoting/Views/vtkDataLabelRepresentation.cxx
    const int maxLabels =
            m_labelProperties.maxTooltipAttributes > 0
                    ? qMin(100, m_labelProperties.maxTooltipAttributes * 10)
                    : 100;  // ParaView default: MaximumNumberOfLabels = 100

    vtkSmartPointer<vtkMaskPoints> maskFilter =
            vtkSmartPointer<vtkMaskPoints>::New();

    // IMPORTANT: For cell labels, we need to convert cells to their center
    // points first ParaView does this with vtkCellCenters (line 43-58 in
    // vtkDataLabelRepresentation.cxx)
    if (!isPointLabels) {
        // Cell labels: convert cells to center points
        vtkSmartPointer<vtkCellCenters> cellCenters =
                vtkSmartPointer<vtkCellCenters>::New();
        cellCenters->SetInputData(data);
        cellCenters->Update();

        CVLog::PrintVerbose(
                QString("[cvSelectionHighlighter] Cell centers: %1 cells -> %2 "
                        "points")
                        .arg(data->GetNumberOfCells())
                        .arg(cellCenters->GetOutput()->GetNumberOfPoints()));

        // Feed cell centers to mask filter
        maskFilter->SetInputConnection(cellCenters->GetOutputPort());
    } else {
        // Point labels: use data directly
        maskFilter->SetInputData(data);
    }

    maskFilter->SetMaximumNumberOfPoints(maxLabels);
    maskFilter->SetOnRatio(1);
    maskFilter->RandomModeOn();  // ParaView uses random sampling
    maskFilter->Update();

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Label mask: %1 points "
                    "-> %2 labels (max %3)")
                    .arg(data->GetNumberOfPoints())
                    .arg(maskFilter->GetOutput()->GetNumberOfPoints())
                    .arg(maxLabels));

    // Create label mapper with masked input for performance
    vtkSmartPointer<vtkLabeledDataMapper> labelMapper =
            vtkSmartPointer<vtkLabeledDataMapper>::New();
    labelMapper->SetInputConnection(maskFilter->GetOutputPort());

    // Configure label mode based on array name
    // ParaView uses vtkOriginalPointIds/vtkOriginalCellIds for ID labels
    // Reference: ParaView/Qt/Components/pqFindDataSelectionDisplayFrame.cxx
    // line 155-157
    if (arrayName == "ID" || arrayName == "PointID" || arrayName == "CellID") {
        // Use vtkOriginalPointIds/vtkOriginalCellIds added by
        // vtkExtractSelection This shows the actual Point ID from the
        // spreadsheet, not 0-based index
        labelMapper->SetLabelModeToLabelFieldData();
        const char* idArrayName =
                isPointLabels ? "vtkOriginalPointIds" : "vtkOriginalCellIds";
        labelMapper->SetFieldDataName(idArrayName);
    } else {
        // Verify that the field exists in the data before setting it
        // This prevents VTK warnings "Failed to find match for field 'xxx'"
        vtkDataSetAttributes* attrData = nullptr;
        if (isPointLabels) {
            attrData = maskFilter->GetOutput()->GetPointData();
        } else {
            attrData = maskFilter->GetOutput()
                               ->GetPointData();  // Cell centers are converted
                                                  // to points
        }

        const char* arrayNameCStr = arrayName.toUtf8().constData();
        vtkAbstractArray* array =
                attrData ? attrData->GetArray(arrayNameCStr) : nullptr;

        if (array) {
            // Field exists - set it for labels
            labelMapper->SetLabelModeToLabelFieldData();
            labelMapper->SetFieldDataName(arrayNameCStr);
        } else {
            // Field not found - log warning and disable labels to avoid VTK
            // warnings
            CVLog::Warning(QString("[cvSelectionHighlighter] Label field '%1' "
                                   "not found in %2 data, labels disabled")
                                   .arg(arrayName)
                                   .arg(isPointLabels ? "point" : "cell"));
            // Don't add the label actor if field doesn't exist
            return;
        }
    }

    // Configure text properties (ParaView-style)
    vtkTextProperty* textProp = labelMapper->GetLabelTextProperty();
    if (textProp) {
        const SelectionLabelProperties& props = m_labelProperties;
        if (isPointLabels) {
            textProp->SetFontSize(props.pointLabelFontSize);
            textProp->SetColor(props.pointLabelColor.redF(),
                               props.pointLabelColor.greenF(),
                               props.pointLabelColor.blueF());
            textProp->SetOpacity(props.pointLabelOpacity);
            textProp->SetBold(props.pointLabelBold);
            textProp->SetItalic(props.pointLabelItalic);
            textProp->SetShadow(props.pointLabelShadow);
        } else {
            textProp->SetFontSize(props.cellLabelFontSize);
            textProp->SetColor(props.cellLabelColor.redF(),
                               props.cellLabelColor.greenF(),
                               props.cellLabelColor.blueF());
            textProp->SetOpacity(props.cellLabelOpacity);
            textProp->SetBold(props.cellLabelBold);
            textProp->SetItalic(props.cellLabelItalic);
            textProp->SetShadow(props.cellLabelShadow);
        }
    }

    // Create label actor
    labelActor = vtkSmartPointer<vtkActor2D>::New();
    labelActor->SetMapper(labelMapper);

    // Add to renderer
    renderer->AddActor2D(labelActor);

    CVLog::PrintVerbose(
            QString("[cvSelectionHighlighter] Added %1 labels with array '%2'")
                    .arg(isPointLabels ? "point" : "cell")
                    .arg(arrayName));

    scheduleAllViewsUpdate();
}

//-----------------------------------------------------------------------------
QColor cvSelectionHighlighter::getHighlightQColor(HighlightMode mode) const {
    const double* color = getHighlightColor(mode);
    if (color) {
        return QColor::fromRgbF(color[0], color[1], color[2]);
    }
    return QColor(255, 0, 255);  // Default magenta
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setHighlightQColor(const QColor& color,
                                                HighlightMode mode) {
    setHighlightColor(color.redF(), color.greenF(), color.blueF(), mode);
}

//=============================================================================
// cvTooltipFormatter Implementation (merged from cvTooltipFormatter.cpp)
//=============================================================================

#include <QtCompat.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkStringArray.h>

#include <QStringList>

//-----------------------------------------------------------------------------
cvTooltipFormatter::cvTooltipFormatter() : m_maxAttributes(15) {}

//-----------------------------------------------------------------------------
cvTooltipFormatter::~cvTooltipFormatter() {}

//-----------------------------------------------------------------------------
void cvTooltipFormatter::setMaxAttributes(int maxAttribs) {
    m_maxAttributes = maxAttribs;
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::getTooltipInfo(vtkPolyData* polyData,
                                           vtkIdType elementId,
                                           AssociationType association,
                                           const QString& datasetName) {
    if (!polyData) {
        CVLog::Error("[cvTooltipFormatter] Invalid polyData");
        return QString();
    }

    if (association == POINTS) {
        return formatPointTooltip(polyData, elementId, datasetName);
    } else {
        return formatCellTooltip(polyData, elementId, datasetName);
    }
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::getPlainTooltipInfo(vtkPolyData* polyData,
                                                vtkIdType elementId,
                                                AssociationType association,
                                                const QString& datasetName) {
    // Get HTML tooltip and strip HTML tags
    QString htmlTooltip =
            getTooltipInfo(polyData, elementId, association, datasetName);

    // Simple HTML tag removal
    QString plainText = htmlTooltip;
    plainText.remove(QtCompatRegExp("<[^>]*>"));
    plainText.replace("&nbsp;", " ");

    return plainText;
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::formatPointTooltip(vtkPolyData* polyData,
                                               vtkIdType pointId,
                                               const QString& datasetName) {
    if (pointId < 0 || pointId >= polyData->GetNumberOfPoints()) {
        CVLog::Error("[cvTooltipFormatter] Invalid point ID: %lld", pointId);
        return QString();
    }

    QString tooltip;

    // ParaView format: Dataset name as first line (no "Block:" prefix)
    if (!datasetName.isEmpty()) {
        tooltip += QString("<b>%1</b>").arg(datasetName);
    }

    // ParaView format: Point ID (with 2-space indent like cell tooltip)
    tooltip += QString("\n  Id: %1").arg(pointId);

    // ParaView format: Coordinates with consistent formatting
    double point[3];
    polyData->GetPoint(pointId, point);
    tooltip += QString("\n  Coords: (%1, %2, %3)")
                       .arg(point[0], 0, 'g', 6)
                       .arg(point[1], 0, 'g', 6)
                       .arg(point[2], 0, 'g', 6);

    // ParaView format: Point data arrays
    vtkPointData* pointData = polyData->GetPointData();
    if (pointData) {
        int numArrays = pointData->GetNumberOfArrays();
        int displayedArrays = 0;  // Counter for limiting displayed attributes

        // Show RGB colors if available ("SourceRGB" holds actual per-point
        // RGB even when SF rendering is active; "RGB" may contain SF-mapped
        // colors)
        vtkUnsignedCharArray* colorArray = nullptr;
        const char* colorNames[] = {"SourceRGB", "RGB", "Colors", "rgba",
                                    "rgb"};
        for (const char* name : colorNames) {
            vtkDataArray* arr = pointData->GetArray(name);
            if (arr && (arr->GetNumberOfComponents() == 3 ||
                        arr->GetNumberOfComponents() == 4)) {
                colorArray = vtkUnsignedCharArray::SafeDownCast(arr);
                if (colorArray) break;
            }
        }
        if (colorArray) {
            unsigned char color[4] = {0, 0, 0, 255};
            colorArray->GetTypedTuple(pointId, color);
            tooltip += QString("\n  RGB: (%1, %2, %3)")
                               .arg(color[0])
                               .arg(color[1])
                               .arg(color[2]);
            displayedArrays++;
        }

        // Show normals if available (ParaView style)
        // First try active normals (set via SetNormals), then named "Normals"
        // array
        vtkDataArray* normalsArray = pointData->GetNormals();
        if (!normalsArray) {
            // Fallback: look for array named "Normals" with 3 components
            normalsArray = pointData->GetArray("Normals");
            // Only use if it has 3 components (actual normals, not curvature)
            if (normalsArray && normalsArray->GetNumberOfComponents() != 3) {
                normalsArray = nullptr;
            }
        }
        if (normalsArray && normalsArray->GetNumberOfComponents() == 3) {
            double* normal = normalsArray->GetTuple3(pointId);
            // ParaView format: "Normals: (x, y, z)" with consistent formatting
            tooltip += QString("\n  Normals: (%1, %2, %3)")
                               .arg(normal[0], 0, 'g', 6)
                               .arg(normal[1], 0, 'g', 6)
                               .arg(normal[2], 0, 'g', 6);
            displayedArrays++;
        }

        // Show texture coordinates (ParaView style: "TCoords: (x, y)")
        // Look for texture coordinate arrays - prioritize "TCoords" or arrays
        // starting with "TCoords"
        vtkDataArray* tcoordsArray = nullptr;
        QString tcoordsArrayName;

        for (int i = 0; i < numArrays; ++i) {
            vtkDataArray* array = pointData->GetArray(i);
            if (!array) continue;

            QString arrayName = QString::fromUtf8(array->GetName());
            int numComp = array->GetNumberOfComponents();

            // Check if this is texture coordinates (2 or 3 components)
            if ((numComp == 2 || numComp == 3) &&
                (arrayName.compare("TCoords", Qt::CaseInsensitive) == 0 ||
                 arrayName.startsWith("TCoords", Qt::CaseInsensitive) ||
                 arrayName.contains("texture", Qt::CaseInsensitive) ||
                 arrayName.contains("tcoords", Qt::CaseInsensitive) ||
                 arrayName.contains("uv", Qt::CaseInsensitive))) {
                // Prefer exact "TCoords" match, otherwise use first match
                if (arrayName.compare("TCoords", Qt::CaseInsensitive) == 0 ||
                    !tcoordsArray) {
                    tcoordsArray = array;
                    tcoordsArrayName = arrayName;
                    // If we found exact "TCoords", use it
                    if (arrayName.compare("TCoords", Qt::CaseInsensitive) ==
                        0) {
                        break;
                    }
                }
            }
        }

        if (tcoordsArray && displayedArrays < m_maxAttributes) {
            int numComp = tcoordsArray->GetNumberOfComponents();
            if (numComp == 2) {
                double* tc = tcoordsArray->GetTuple2(pointId);
                // ParaView format: "TCoords: (x, y)" - always use "TCoords" as
                // label
                tooltip += QString("\n  TCoords: (%1, %2)")
                                   .arg(tc[0], 0, 'g', 6)
                                   .arg(tc[1], 0, 'g', 6);
            } else if (numComp == 3) {
                double* tc = tcoordsArray->GetTuple3(pointId);
                tooltip += QString("\n  TCoords: (%1, %2, %3)")
                                   .arg(tc[0], 0, 'g', 6)
                                   .arg(tc[1], 0, 'g', 6)
                                   .arg(tc[2], 0, 'g', 6);
            }
            displayedArrays++;
        }

        // Show scalars if available (after normals and texture coords)
        if (pointData->GetScalars() && displayedArrays < m_maxAttributes) {
            QString scalarName =
                    QString::fromUtf8(pointData->GetScalars()->GetName());
            if (!scalarName.contains("texture", Qt::CaseInsensitive) &&
                !scalarName.contains("tcoord", Qt::CaseInsensitive)) {
                double scalar = pointData->GetScalars()->GetTuple1(pointId);
                tooltip += QString("\n  %1: %2")
                                   .arg(scalarName.isEmpty() ? "Scalars"
                                                             : scalarName)
                                   .arg(formatNumber(scalar));
                displayedArrays++;
            }
        }

        // Add additional point data arrays (custom attributes)
        for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
            if (displayedArrays >= m_maxAttributes) {
                tooltip += QString("\n  <i>... (%1 more attributes "
                                   "hidden)</i>")
                                   .arg(pointData->GetNumberOfArrays() - i);
                break;
            }

            vtkDataArray* array = pointData->GetArray(i);
            if (!array) continue;

            QString arrayName = QString::fromUtf8(array->GetName());

            // Skip already displayed arrays (normals, scalars, textures,
            // colors) Skip "Normals" by name regardless of components (shown
            // specially above or not real normals) Skip "RGB"/"Colors" to avoid
            // showing them in generic loop
            bool isNormalsArray =
                    (array == pointData->GetNormals()) ||
                    (arrayName.compare("Normals", Qt::CaseInsensitive) == 0);
            bool isColorArray =
                    (arrayName.compare("RGB", Qt::CaseInsensitive) == 0) ||
                    (arrayName.compare("SourceRGB", Qt::CaseInsensitive) ==
                     0) ||
                    (arrayName.compare("Colors", Qt::CaseInsensitive) == 0) ||
                    (arrayName.compare("rgba", Qt::CaseInsensitive) == 0);
            bool isTcoordsArray =
                    (arrayName.compare("TCoords", Qt::CaseInsensitive) == 0) ||
                    arrayName.startsWith("TCoords", Qt::CaseInsensitive) ||
                    arrayName.contains("texture", Qt::CaseInsensitive) ||
                    arrayName.contains("tcoord", Qt::CaseInsensitive) ||
                    arrayName.contains("uv", Qt::CaseInsensitive);
            if (arrayName.isEmpty() || isNormalsArray || isColorArray ||
                array == pointData->GetScalars() || isTcoordsArray) {
                continue;
            }

            QString valueStr = formatArrayValue(array, pointId);
            if (!valueStr.isEmpty()) {
                tooltip += QString("\n  %1: %2").arg(arrayName).arg(valueStr);
                displayedArrays++;
            }
        }
    }

    // ParaView format: Field data arrays (with horizontal line separator)
    vtkFieldData* fieldDataPoint = polyData->GetFieldData();
    if (fieldDataPoint && fieldDataPoint->GetNumberOfArrays() > 0) {
        bool hasFieldData = false;
        bool isFirstField = true;
        for (int i = 0; i < fieldDataPoint->GetNumberOfArrays(); ++i) {
            vtkAbstractArray* abstractArray =
                    fieldDataPoint->GetAbstractArray(i);
            if (!abstractArray) continue;

            QString arrayName = QString::fromUtf8(abstractArray->GetName());

            // Skip internal/metadata arrays
            if (arrayName.startsWith("vtk", Qt::CaseInsensitive) ||
                arrayName.startsWith(
                        "Has",
                        Qt::CaseInsensitive) ||  // Skip HasSourceRGB etc.
                arrayName == "DatasetName" ||
                arrayName == "MaterialNames" ||
                arrayName.compare("RGB", Qt::CaseInsensitive) == 0 ||
                arrayName.compare("Colors", Qt::CaseInsensitive) == 0 ||
                arrayName.compare("Normals", Qt::CaseInsensitive) == 0) {
                continue;
            }

            if (!hasFieldData) {
                tooltip += "\n  <hr>";
                hasFieldData = true;
            }

            QString linePrefix = isFirstField ? "  " : "\n  ";

            vtkStringArray* stringArray =
                    vtkStringArray::SafeDownCast(abstractArray);
            if (stringArray && stringArray->GetNumberOfTuples() > 0) {
                if (stringArray->GetNumberOfTuples() == 1) {
                    QString value =
                            QString::fromStdString(stringArray->GetValue(0));
                    tooltip += QString("%1%2: %3")
                                       .arg(linePrefix)
                                       .arg(arrayName)
                                       .arg(value);
                } else {
                    QStringList values;
                    for (vtkIdType j = 0; j < stringArray->GetNumberOfTuples();
                         ++j) {
                        values << QString::fromStdString(
                                stringArray->GetValue(j));
                    }
                    tooltip += QString("%1%2: %3")
                                       .arg(linePrefix)
                                       .arg(arrayName)
                                       .arg(values.join(", "));
                }
                isFirstField = false;
            } else {
                vtkDataArray* array = vtkDataArray::SafeDownCast(abstractArray);
                if (array && array->GetNumberOfTuples() > 0) {
                    if (array->GetNumberOfTuples() == 1) {
                        QString valueStr = formatArrayValue(array, 0);
                        tooltip += QString("%1%2: %3")
                                           .arg(linePrefix)
                                           .arg(arrayName)
                                           .arg(valueStr);
                    } else {
                        QString valueStr = formatArrayValue(array, 0);
                        tooltip += QString("%1%2: %3 (array of %4)")
                                           .arg(linePrefix)
                                           .arg(arrayName)
                                           .arg(valueStr)
                                           .arg(array->GetNumberOfTuples());
                    }
                    isFirstField = false;
                }
            }
        }
    }

    // ParaView style: wrap in <p style='white-space:pre'> to preserve
    // whitespace and newlines
    return QString("<p style='white-space:pre'>%1</p>").arg(tooltip);
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::formatCellTooltip(vtkPolyData* polyData,
                                              vtkIdType cellId,
                                              const QString& datasetName) {
    if (cellId < 0 || cellId >= polyData->GetNumberOfCells()) {
        CVLog::Error("[cvTooltipFormatter] Invalid cell ID: %lld", cellId);
        return QString();
    }

    QString tooltip;

    // ParaView format: Dataset name as first line
    if (!datasetName.isEmpty()) {
        tooltip += QString("<b>%1</b>").arg(datasetName);
    }

    // ParaView format: Cell ID
    tooltip += QString("\n  Id: %1").arg(cellId);

    // ParaView format: Cell type
    vtkCell* cell = polyData->GetCell(cellId);
    if (cell) {
        QString cellType;
        switch (cell->GetCellType()) {
            case VTK_EMPTY_CELL:
                cellType = "Empty Cell";
                break;
            case VTK_VERTEX:
                cellType = "Vertex";
                break;
            case VTK_POLY_VERTEX:
                cellType = "Poly Vertex";
                break;
            case VTK_LINE:
                cellType = "Line";
                break;
            case VTK_POLY_LINE:
                cellType = "Poly Line";
                break;
            case VTK_TRIANGLE:
                cellType = "Triangle";
                break;
            case VTK_TRIANGLE_STRIP:
                cellType = "Triangle Strip";
                break;
            case VTK_POLYGON:
                cellType = "Polygon";
                break;
            case VTK_PIXEL:
                cellType = "Pixel";
                break;
            case VTK_QUAD:
                cellType = "Quad";
                break;
            case VTK_TETRA:
                cellType = "Tetra";
                break;
            case VTK_VOXEL:
                cellType = "Voxel";
                break;
            case VTK_HEXAHEDRON:
                cellType = "Hexahedron";
                break;
            case VTK_WEDGE:
                cellType = "Wedge";
                break;
            case VTK_PYRAMID:
                cellType = "Pyramid";
                break;
            case VTK_PENTAGONAL_PRISM:
                cellType = "Pentagonal Prism";
                break;
            case VTK_HEXAGONAL_PRISM:
                cellType = "Hexagonal Prism";
                break;
            default:
                cellType = QString("Unknown (%1)").arg(cell->GetCellType());
        }
        tooltip += QString("\n  Type: %1").arg(cellType);

        vtkIdType npts = cell->GetNumberOfPoints();
        tooltip += QString("\n  Number of Points: %1").arg(npts);

        if (npts > 0 && npts <= 10) {
            QString pointIds;
            for (vtkIdType i = 0; i < npts; ++i) {
                if (i > 0) pointIds += ", ";
                pointIds += QString::number(cell->GetPointId(i));
            }
            tooltip += QString("\n  Point IDs: [%1]").arg(pointIds);
        }

        // Show cell center/centroid
        double center[3] = {0, 0, 0};
        for (vtkIdType i = 0; i < npts; ++i) {
            double pt[3];
            polyData->GetPoint(cell->GetPointId(i), pt);
            center[0] += pt[0];
            center[1] += pt[1];
            center[2] += pt[2];
        }
        if (npts > 0) {
            center[0] /= npts;
            center[1] /= npts;
            center[2] /= npts;
            tooltip += QString("\n  Center: (%1, %2, %3)")
                               .arg(center[0], 0, 'g', 6)
                               .arg(center[1], 0, 'g', 6)
                               .arg(center[2], 0, 'g', 6);
        }
    }

    // ParaView format: Cell data arrays
    vtkCellData* cellData = polyData->GetCellData();
    if (cellData) {
        int displayedArrays = 0;

        // Show normals - check both active and named "Normals" array with 3
        // components
        vtkDataArray* cellNormalsArray = cellData->GetNormals();
        if (!cellNormalsArray) {
            cellNormalsArray = cellData->GetArray("Normals");
            if (cellNormalsArray &&
                cellNormalsArray->GetNumberOfComponents() != 3) {
                cellNormalsArray = nullptr;
            }
        }
        if (cellNormalsArray &&
            cellNormalsArray->GetNumberOfComponents() == 3) {
            double* normal = cellNormalsArray->GetTuple3(cellId);
            tooltip += QString("\n  Normals: (%1, %2, %3)")
                               .arg(normal[0], 0, 'f', 4)
                               .arg(normal[1], 0, 'f', 4)
                               .arg(normal[2], 0, 'f', 4);
            displayedArrays++;
        }

        if (cellData->GetScalars() && displayedArrays < m_maxAttributes) {
            QString scalarName =
                    QString::fromUtf8(cellData->GetScalars()->GetName());
            double scalar = cellData->GetScalars()->GetTuple1(cellId);
            tooltip +=
                    QString("\n  %1: %2")
                            .arg(scalarName.isEmpty() ? "Scalars" : scalarName)
                            .arg(formatNumber(scalar));
            displayedArrays++;
        }

        for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
            if (displayedArrays >= m_maxAttributes) {
                tooltip += QString("\n<i>... (%1 more attributes "
                                   "hidden)</i>")
                                   .arg(cellData->GetNumberOfArrays() - i);
                break;
            }

            vtkDataArray* array = cellData->GetArray(i);
            if (!array) continue;

            QString arrayName = QString::fromUtf8(array->GetName());

            // Skip already displayed arrays (normals, scalars, colors)
            // Skip "Normals" by name regardless of components
            // Skip "RGB"/"Colors" to avoid showing them in generic loop
            bool isCellNormals =
                    (array == cellData->GetNormals()) ||
                    (arrayName.compare("Normals", Qt::CaseInsensitive) == 0);
            bool isColorArray =
                    (arrayName.compare("RGB", Qt::CaseInsensitive) == 0) ||
                    (arrayName.compare("SourceRGB", Qt::CaseInsensitive) ==
                     0) ||
                    (arrayName.compare("Colors", Qt::CaseInsensitive) == 0) ||
                    (arrayName.compare("rgba", Qt::CaseInsensitive) == 0);
            if (arrayName.isEmpty() || isCellNormals || isColorArray ||
                array == cellData->GetScalars()) {
                continue;
            }

            QString valueStr = formatArrayValue(array, cellId);
            if (!valueStr.isEmpty()) {
                tooltip += QString("\n%1: %2").arg(arrayName).arg(valueStr);
                displayedArrays++;
            }
        }
    }

    // ParaView format: Field data arrays
    vtkFieldData* fieldDataCell = polyData->GetFieldData();
    if (fieldDataCell && fieldDataCell->GetNumberOfArrays() > 0) {
        bool hasFieldData = false;
        bool isFirstField = true;
        for (int i = 0; i < fieldDataCell->GetNumberOfArrays(); ++i) {
            vtkAbstractArray* abstractArray =
                    fieldDataCell->GetAbstractArray(i);
            if (!abstractArray) continue;

            QString arrayName = QString::fromUtf8(abstractArray->GetName());

            // Skip internal/metadata arrays
            if (arrayName.startsWith("vtk", Qt::CaseInsensitive) ||
                arrayName.startsWith(
                        "Has",
                        Qt::CaseInsensitive) ||  // Skip HasSourceRGB etc.
                arrayName == "DatasetName" ||
                arrayName == "MaterialNames" ||
                arrayName.compare("RGB", Qt::CaseInsensitive) == 0 ||
                arrayName.compare("Colors", Qt::CaseInsensitive) == 0 ||
                arrayName.compare("Normals", Qt::CaseInsensitive) == 0) {
                continue;
            }

            if (!hasFieldData) {
                tooltip += "\n<hr>";
                hasFieldData = true;
            }

            QString linePrefix = isFirstField ? "" : "\n";

            vtkStringArray* stringArray =
                    vtkStringArray::SafeDownCast(abstractArray);
            if (stringArray && stringArray->GetNumberOfTuples() > 0) {
                if (stringArray->GetNumberOfTuples() == 1) {
                    QString value =
                            QString::fromStdString(stringArray->GetValue(0));
                    tooltip += QString("%1%2: %3")
                                       .arg(linePrefix)
                                       .arg(arrayName)
                                       .arg(value);
                } else {
                    QStringList values;
                    for (vtkIdType j = 0; j < stringArray->GetNumberOfTuples();
                         ++j) {
                        values << QString::fromStdString(
                                stringArray->GetValue(j));
                    }
                    tooltip += QString("%1%2: %3")
                                       .arg(linePrefix)
                                       .arg(arrayName)
                                       .arg(values.join(", "));
                }
                isFirstField = false;
            } else {
                vtkDataArray* array = vtkDataArray::SafeDownCast(abstractArray);
                if (array && array->GetNumberOfTuples() > 0) {
                    if (array->GetNumberOfTuples() == 1) {
                        QString valueStr = formatArrayValue(array, 0);
                        tooltip += QString("%1%2: %3")
                                           .arg(linePrefix)
                                           .arg(arrayName)
                                           .arg(valueStr);
                    } else {
                        QString valueStr = formatArrayValue(array, 0);
                        tooltip += QString("%1%2: %3 (array of %4)")
                                           .arg(linePrefix)
                                           .arg(arrayName)
                                           .arg(valueStr)
                                           .arg(array->GetNumberOfTuples());
                    }
                    isFirstField = false;
                }
            }
        }
    }

    // ParaView style: wrap in <p style='white-space:pre'> to preserve
    // whitespace and newlines
    return QString("<p style='white-space:pre'>%1</p>").arg(tooltip);
}

//-----------------------------------------------------------------------------
void cvTooltipFormatter::addArrayValues(QString& tooltip,
                                        vtkFieldData* fieldData,
                                        vtkIdType tupleIndex) {
    if (!fieldData) {
        return;
    }

    int numArrays = fieldData->GetNumberOfArrays();

    for (int i = 0; i < numArrays; ++i) {
        vtkDataArray* array = fieldData->GetArray(i);
        if (!array) {
            continue;
        }

        QString arrayName = QString::fromUtf8(array->GetName());
        if (arrayName.isEmpty() ||
            arrayName.startsWith("vtk", Qt::CaseInsensitive) ||
            arrayName == "vtkOriginalPointIds" ||
            arrayName == "vtkOriginalCellIds" ||
            arrayName == "vtkCompositeIndexArray" ||
            arrayName == "vtkGhostType" || arrayName == "vtkValidPointMask") {
            continue;
        }

        QString valueStr = formatArrayValue(array, tupleIndex);
        if (!valueStr.isEmpty()) {
            tooltip += QString("\n%1: %2").arg(arrayName).arg(valueStr);
        }
    }
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::formatArrayValue(vtkDataArray* array,
                                             vtkIdType tupleIndex) {
    if (!array || tupleIndex < 0 || tupleIndex >= array->GetNumberOfTuples()) {
        return QString();
    }

    int numComponents = array->GetNumberOfComponents();
    const int maxDisplayedComp = 9;

    if (numComponents == 1) {
        double value = array->GetTuple1(tupleIndex);
        return formatNumber(value);
    } else {
        QString result;
        if (numComponents > 1) {
            result = "(";
        }

        for (int i = 0; i < std::min(numComponents, maxDisplayedComp); ++i) {
            double value = array->GetComponent(tupleIndex, i);
            result += formatNumber(value);
            if (i + 1 < numComponents && i < maxDisplayedComp) {
                result += ", ";
            }
        }

        if (numComponents > maxDisplayedComp) {
            result += ", ...";
        }

        if (numComponents > 1) {
            result += ")";
        }
        return result;
    }
}

//-----------------------------------------------------------------------------
QString cvTooltipFormatter::formatNumber(double value) {
    double absValue = qAbs(value);

    if (absValue > 0 && (absValue < 1e-4 || absValue >= 1e6)) {
        return QString::number(value, 'e', 4);
    } else {
        return QString::number(value, 'g', 6);
    }
}
