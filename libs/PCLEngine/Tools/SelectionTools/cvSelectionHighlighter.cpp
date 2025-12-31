// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionHighlighter.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// ECV_DB_LIB
#include <ecvGenericVisualizer3D.h>

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkActor.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkExtractSelection.h>
#include <vtkIdTypeArray.h>
#include <vtkMapper.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkUnstructuredGrid.h>

//-----------------------------------------------------------------------------
cvSelectionHighlighter::cvSelectionHighlighter()
    : cvGenericSelectionTool(),
      m_hoverOpacity(0.9),  // High opacity for better visibility (enhanced)
      m_preselectedOpacity(0.8),  // High opacity preselect feedback (enhanced)
      m_selectedOpacity(1.0),     // Opaque final selection
      m_boundaryOpacity(0.85),    // High opacity boundary (enhanced)
      m_hoverPointSize(5),
      m_preselectedPointSize(5),
      m_selectedPointSize(5),
      m_boundaryPointSize(5),
      m_hoverLineWidth(2),
      m_preselectedLineWidth(2),
      m_selectedLineWidth(2),
      m_boundaryLineWidth(2),
      m_enabled(true) {
    // Enhanced colors for better visibility and contrast
    // Using bright, saturated colors that stand out against most backgrounds

    // Hover: Bright Cyan (0, 255, 255) - RGB normalized - highly visible
    m_hoverColor[0] = 0.0;
    m_hoverColor[1] = 1.0;
    m_hoverColor[2] = 1.0;

    // Pre-selected: Bright Yellow (255, 255, 0) - RGB normalized - highly
    // visible
    m_preselectedColor[0] = 1.0;
    m_preselectedColor[1] = 1.0;
    m_preselectedColor[2] = 0.0;

    // Selected: Bright Magenta (255, 0, 255) - RGB normalized - maximum
    // visibility Magenta is highly visible against any point cloud color
    // (nature/buildings)
    m_selectedColor[0] = 1.0;  // Red
    m_selectedColor[1] = 0.0;  // Green
    m_selectedColor[2] = 1.0;  // Blue = Magenta

    // Boundary: Bright Orange (255, 165, 0) - RGB normalized - highly visible
    m_boundaryColor[0] = 1.0;
    m_boundaryColor[1] = 0.65;
    m_boundaryColor[2] = 0.0;

    m_hoverActorId = "__highlight_hover__";
    m_preselectedActorId = "__highlight_preselected__";
    m_selectedActorId = "__highlight_selected__";
    m_boundaryActorId = "__highlight_boundary__";

    CVLog::PrintDebug(
            "[cvSelectionHighlighter] Initialized with enhanced visibility "
            "colors: Hover=Cyan(0,1,1), Selected=Green(0,1,0)");
}

//-----------------------------------------------------------------------------
cvSelectionHighlighter::~cvSelectionHighlighter() { clearHighlights(); }

// setVisualizer is inherited from cvGenericSelectionTool

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
        color[0] = r;
        color[1] = g;
        color[2] = b;

        // Update existing actor's color immediately for real-time preview
        if (actor && *actor) {
            (*actor)->GetProperty()->SetColor(r, g, b);
        }

        CVLog::Print(QString("[cvSelectionHighlighter] Color set for mode %1: "
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
    double* opacityPtr = nullptr;
    vtkSmartPointer<vtkActor>* actor = nullptr;

    switch (mode) {
        case HOVER:
            m_hoverOpacity = opacity;
            opacityPtr = &m_hoverOpacity;
            actor = &m_hoverActor;
            break;
        case PRESELECTED:
            m_preselectedOpacity = opacity;
            opacityPtr = &m_preselectedOpacity;
            actor = &m_preselectedActor;
            break;
        case SELECTED:
            m_selectedOpacity = opacity;
            opacityPtr = &m_selectedOpacity;
            actor = &m_selectedActor;
            break;
        case BOUNDARY:
            m_boundaryOpacity = opacity;
            opacityPtr = &m_boundaryOpacity;
            actor = &m_boundaryActor;
            break;
    }

    // Update existing actor's opacity immediately for real-time preview
    if (actor && *actor) {
        (*actor)->GetProperty()->SetOpacity(opacity);
    }

    CVLog::Print(QString("[cvSelectionHighlighter] Opacity set for mode %1: %2")
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

    // Get polyData using centralized ParaView-style method
    // Note: In ParaView, the highlighter would get data from the
    // representation, but here we use the getPolyDataForSelection() which
    // handles the priority logic.
    vtkPolyData* polyData = getPolyDataForSelection();
    if (!polyData) {
        CVLog::Error(
                "[cvSelectionHighlighter::highlightSelection] No polyData "
                "available for highlighting - this is a critical error!");
        return false;
    }

    CVLog::PrintDebug(QString("[cvSelectionHighlighter] Got polyData with %1 "
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
    if (!actor) {
        CVLog::Error(
                "[cvSelectionHighlighter] Failed to create highlight actor");
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

    addActorToVisualizer(actor, actorId);
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

    CVLog::Print(
            QString("[cvSelectionHighlighter] Highlighting %1 %2 in mode %3")
                    .arg(selectionData.count())
                    .arg(selectionData.fieldTypeString())
                    .arg(mode));

    // Delegate to the VTK-level implementation
    bool success = highlightSelection(selectionData.vtkArray(),
                                      selectionData.fieldAssociation(), mode);

    if (!success) {
        CVLog::Error("[cvSelectionHighlighter] Failed to highlight selection");
    }

    return success;
}

//-----------------------------------------------------------------------------
bool cvSelectionHighlighter::highlightElement(vtkPolyData* polyData,
                                              vtkIdType elementId,
                                              int fieldAssociation) {
    if (!m_enabled || !m_viewer || !polyData) {
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
    // Remove all highlight actors (ParaView-style multi-level)
    removeActorFromVisualizer(m_hoverActorId);
    removeActorFromVisualizer(m_preselectedActorId);
    removeActorFromVisualizer(m_selectedActorId);
    removeActorFromVisualizer(m_boundaryActorId);

    m_hoverActor = nullptr;
    m_preselectedActor = nullptr;
    m_selectedActor = nullptr;
    m_boundaryActor = nullptr;
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::clearHoverHighlight() {
    // Remove ONLY hover highlight, keep selected/preselected/boundary
    // This is used during hover updates to avoid clearing persistent selections
    removeActorFromVisualizer(m_hoverActorId);
    m_hoverActor = nullptr;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkActor> cvSelectionHighlighter::createHighlightActor(
        vtkPolyData* polyData,
        vtkIdTypeArray* selection,
        int fieldAssociation,
        HighlightMode mode) {
    CVLog::Print(
            QString("[cvSelectionHighlighter::createHighlightActor] START: "
                    "fieldAssociation=%1 (%2), mode=%3, selectionCount=%4, "
                    "polyDataCells=%5, polyDataPoints=%6")
                    .arg(fieldAssociation)
                    .arg(fieldAssociation == 0 ? "CELLS" : "POINTS")
                    .arg(mode)
                    .arg(selection ? selection->GetNumberOfTuples() : 0)
                    .arg(polyData ? polyData->GetNumberOfCells() : 0)
                    .arg(polyData ? polyData->GetNumberOfPoints() : 0));

    // Log selection IDs for debugging
    if (selection && selection->GetNumberOfTuples() > 0) {
        QString idsStr;
        vtkIdType numToShow =
                qMin(selection->GetNumberOfTuples(), (vtkIdType)10);
        for (vtkIdType i = 0; i < numToShow; ++i) {
            if (i > 0) idsStr += ", ";
            idsStr += QString::number(selection->GetValue(i));
        }
        if (selection->GetNumberOfTuples() > 10) {
            idsStr += ", ...";
        }
        CVLog::Print(QString("[cvSelectionHighlighter] Selection IDs: [%1]")
                             .arg(idsStr));
    }

    // Create selection node
    vtkSmartPointer<vtkSelectionNode> selectionNode =
            createSelectionNode(selection, fieldAssociation);
    if (!selectionNode) {
        CVLog::Error(
                "[cvSelectionHighlighter::createHighlightActor] Failed to "
                "create selection node");
        return nullptr;
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

    vtkUnstructuredGrid* extracted =
            vtkUnstructuredGrid::SafeDownCast(extractor->GetOutput());

    if (!extracted) {
        CVLog::Error(
                "[cvSelectionHighlighter::createHighlightActor] Extraction "
                "failed: extracted is nullptr");
        return nullptr;
    }

    vtkIdType numCells = extracted->GetNumberOfCells();
    vtkIdType numPoints = extracted->GetNumberOfPoints();

    CVLog::Print(QString("[cvSelectionHighlighter::createHighlightActor] "
                         "Extracted %1 cells, %2 points")
                         .arg(numCells)
                         .arg(numPoints));

    if (numCells == 0 && numPoints == 0) {
        CVLog::Error(
                "[cvSelectionHighlighter::createHighlightActor] Extraction "
                "failed: 0 cells and 0 points extracted - check if selection "
                "IDs are valid");
        return nullptr;
    }

    // Create mapper
    vtkSmartPointer<vtkDataSetMapper> mapper =
            vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(extracted);
    mapper->ScalarVisibilityOff();  // Don't use scalar colors

    // Create actor (use smart pointer to prevent leaks)
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Set properties based on mode (ParaView-style)
    vtkProperty* prop = actor->GetProperty();
    double* color = nullptr;
    double opacity = 1.0;

    switch (mode) {
        case HOVER:
            // Enhanced hover feedback (bright cyan, high opacity)
            color = m_hoverColor;
            opacity = m_hoverOpacity;
            prop->SetLineWidth(3.0);  // Moderate line width
            prop->SetPointSize(8.0);  // Reasonable hover point size
            prop->SetRenderLinesAsTubes(true);
            break;

        case PRESELECTED:
            // Enhanced preselect feedback (bright yellow, high opacity)
            color = m_preselectedColor;
            opacity = m_preselectedOpacity;
            prop->SetLineWidth(4.5);   // Thicker lines
            prop->SetPointSize(12.0);  // Larger points (enhanced)
            prop->SetRenderLinesAsTubes(true);
            break;

        case SELECTED:
            // Final selection (magenta color for visibility)
            color = m_selectedColor;
            opacity = m_selectedOpacity;
            prop->SetLineWidth(5.0);   // Thick lines for final selection
            prop->SetPointSize(12.0);  // Large enough to see, not overwhelming
            prop->SetRenderLinesAsTubes(true);
            break;

        case BOUNDARY:
            // Enhanced boundary highlight (bright orange, high opacity)
            color = m_boundaryColor;
            opacity = m_boundaryOpacity;
            prop->SetLineWidth(4.0);   // Thicker lines
            prop->SetPointSize(10.0);  // Larger points (enhanced)
            prop->SetRenderLinesAsTubes(true);
            break;
    }

    if (color) {
        prop->SetColor(color[0], color[1], color[2]);
    }
    prop->SetOpacity(opacity);

    // Enhanced rendering properties for maximum visibility
    prop->SetAmbient(0.6);   // Increased ambient light for brighter appearance
    prop->SetDiffuse(0.8);   // Increased diffuse for better color saturation
    prop->SetSpecular(0.5);  // Increased specular for more shine/visibility
    prop->SetSpecularPower(
            30.0);  // Higher specular power for sharper highlights
    prop->SetRenderLinesAsTubes(true);     // Render lines as 3D tubes
    prop->SetRenderPointsAsSpheres(true);  // Render points as 3D spheres

    // Additional enhancement: edge visibility
    prop->EdgeVisibilityOn();           // Show edges for better definition
    prop->SetEdgeColor(1.0, 1.0, 1.0);  // White edges for contrast

    // Set representation mode based on field association (enhanced for
    // visibility) 0 = CELLS: show as wireframe (edges only) 1 = POINTS: show as
    // points
    if (fieldAssociation == 0) {
        // CELLS: Render as wireframe (edges only)
        prop->SetRepresentationToWireframe();
        prop->SetLineWidth(
                5.0);  // Much thicker edges for better visibility (enhanced)
    } else {
        // POINTS: Render as points with enhanced visibility
        prop->SetRepresentationToPoints();
        // Point size is already set above based on mode, ensure it's visible
    }

    return actor;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelectionNode> cvSelectionHighlighter::createSelectionNode(
        vtkIdTypeArray* selection, int fieldAssociation) {
    if (!selection) {
        return nullptr;
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

    // Get PCLVis for VTK operations
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (!pclVis) {
        CVLog::Error(
                "[cvSelectionHighlighter::addActorToVisualizer] Visualizer is "
                "not PCLVis");
        return;
    }

    // Get current renderer
    vtkRenderer* renderer = pclVis->getCurrentRenderer();
    if (!renderer) {
        CVLog::Error(
                "[cvSelectionHighlighter::addActorToVisualizer] No renderer!");
        return;
    }

    // CRITICAL: Configure mapper to render highlights ON TOP of main cloud
    // This prevents Z-fighting and ensures highlights are always visible
    vtkMapper* mapper = actor->GetMapper();
    if (mapper) {
        // Use polygon offset to push highlights slightly forward
        mapper->SetResolveCoincidentTopologyToPolygonOffset();
        mapper->SetRelativeCoincidentTopologyPolygonOffsetParameters(-1.0,
                                                                     -1.0);
        mapper->SetResolveCoincidentTopologyPolygonOffsetFaces(1);
    }

    // Add actor to renderer
    renderer->AddActor(actor);
    actor->SetVisibility(1);  // Ensure visibility is on

    CVLog::Print(QString("[cvSelectionHighlighter] ✓ Added highlight actor "
                         "'%1' to renderer")
                         .arg(id));

    // Log actor properties for debugging
    vtkProperty* prop = actor->GetProperty();
    double* color = prop->GetColor();
    CVLog::Print(QString("[cvSelectionHighlighter]   Color: (%1, %2, %3), "
                         "Opacity: %4, PointSize: %5")
                         .arg(color[0], 0, 'f', 2)
                         .arg(color[1], 0, 'f', 2)
                         .arg(color[2], 0, 'f', 2)
                         .arg(prop->GetOpacity(), 0, 'f', 2)
                         .arg(prop->GetPointSize(), 0, 'f', 1));

    // Trigger immediate render update (ParaView-style)
    vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
    if (renderWindow) {
        renderWindow->Render();
        CVLog::PrintDebug("[cvSelectionHighlighter] Triggered render update");
    } else {
        CVLog::Warning("[cvSelectionHighlighter] No render window to update!");
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::removeActorFromVisualizer(const QString& id) {
    if (!m_viewer) {
        return;
    }

    // Get PCLVis for VTK operations
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (!pclVis) {
        return;
    }

    vtkRenderer* renderer = pclVis->getCurrentRenderer();
    if (!renderer) {
        return;
    }

    // Remove corresponding actor (ParaView-style multi-level)
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

    if (actor) {
        renderer->RemoveActor(actor);

        // Trigger immediate render update (ParaView-style)
        vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
        if (renderWindow) {
            renderWindow->Render();
        }

        CVLog::PrintDebug(
                QString("[cvSelectionHighlighter] Removed highlight actor: %1")
                        .arg(id));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setPointSize(int size, HighlightMode mode) {
    int* sizePtr = nullptr;
    vtkSmartPointer<vtkActor>* actor = nullptr;

    switch (mode) {
        case HOVER:
            m_hoverPointSize = size;
            sizePtr = &m_hoverPointSize;
            actor = &m_hoverActor;
            break;
        case PRESELECTED:
            m_preselectedPointSize = size;
            sizePtr = &m_preselectedPointSize;
            actor = &m_preselectedActor;
            break;
        case SELECTED:
            m_selectedPointSize = size;
            sizePtr = &m_selectedPointSize;
            actor = &m_selectedActor;
            break;
        case BOUNDARY:
            m_boundaryPointSize = size;
            sizePtr = &m_boundaryPointSize;
            actor = &m_boundaryActor;
            break;
    }

    // Update existing actor's point size immediately for real-time preview
    if (actor && *actor) {
        (*actor)->GetProperty()->SetPointSize(static_cast<float>(size));
    }

    CVLog::PrintDebug(
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
            return 5;
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setLineWidth(int width, HighlightMode mode) {
    int* widthPtr = nullptr;
    vtkSmartPointer<vtkActor>* actor = nullptr;

    switch (mode) {
        case HOVER:
            m_hoverLineWidth = width;
            widthPtr = &m_hoverLineWidth;
            actor = &m_hoverActor;
            break;
        case PRESELECTED:
            m_preselectedLineWidth = width;
            widthPtr = &m_preselectedLineWidth;
            actor = &m_preselectedActor;
            break;
        case SELECTED:
            m_selectedLineWidth = width;
            widthPtr = &m_selectedLineWidth;
            actor = &m_selectedActor;
            break;
        case BOUNDARY:
            m_boundaryLineWidth = width;
            widthPtr = &m_boundaryLineWidth;
            actor = &m_boundaryActor;
            break;
    }

    // Update existing actor's line width immediately for real-time preview
    if (actor && *actor) {
        (*actor)->GetProperty()->SetLineWidth(static_cast<float>(width));
    }

    CVLog::PrintDebug(
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
