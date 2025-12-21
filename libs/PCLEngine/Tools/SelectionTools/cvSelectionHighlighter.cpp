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

    // Selected: Bright Lime Green (0, 255, 0) - RGB normalized - highly visible
    // Changed from magenta to bright green for better visibility
    m_selectedColor[0] = 0.0;
    m_selectedColor[1] = 1.0;
    m_selectedColor[2] = 0.0;

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
    switch (mode) {
        case HOVER:
            color = m_hoverColor;
            break;
        case PRESELECTED:
            color = m_preselectedColor;
            break;
        case SELECTED:
            color = m_selectedColor;
            break;
        case BOUNDARY:
            color = m_boundaryColor;
            break;
    }
    if (color) {
        color[0] = r;
        color[1] = g;
        color[2] = b;
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
    switch (mode) {
        case HOVER:
            m_hoverOpacity = opacity;
            break;
        case PRESELECTED:
            m_preselectedOpacity = opacity;
            break;
        case SELECTED:
            m_selectedOpacity = opacity;
            break;
        case BOUNDARY:
            m_boundaryOpacity = opacity;
            break;
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
    if (!m_enabled || !m_viewer || !selection) {
        return false;
    }

    // Get polyData using centralized ParaView-style method
    // Note: In ParaView, the highlighter would get data from the
    // representation, but here we use the getPolyDataForSelection() which
    // handles the priority logic.
    vtkPolyData* polyData = getPolyDataForSelection();
    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionHighlighter::highlightSelection] No polyData "
                "available for highlighting");
        return false;
    }

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
    vtkActor* actor =
            createHighlightActor(polyData, selection, fieldAssociation, mode);
    if (!actor) {
        CVLog::Error(
                "[cvSelectionHighlighter] Failed to create highlight actor");
        return false;
    }

    // Remove old actor and add new one (ParaView-style multi-level)
    QString actorId;
    vtkSmartPointer<vtkActor>* targetActor = nullptr;

    switch (mode) {
        case HOVER:
            removeActorFromVisualizer(m_hoverActorId);
            m_hoverActor = actor;
            actorId = m_hoverActorId;
            targetActor = &m_hoverActor;
            break;

        case PRESELECTED:
            removeActorFromVisualizer(m_preselectedActorId);
            m_preselectedActor = actor;
            actorId = m_preselectedActorId;
            targetActor = &m_preselectedActor;
            break;

        case SELECTED:
            removeActorFromVisualizer(m_selectedActorId);
            m_selectedActor = actor;
            actorId = m_selectedActorId;
            targetActor = &m_selectedActor;
            break;

        case BOUNDARY:
            removeActorFromVisualizer(m_boundaryActorId);
            m_boundaryActor = actor;
            actorId = m_boundaryActorId;
            targetActor = &m_boundaryActor;
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

    // Delegate to the VTK-level implementation
    return highlightSelection(selectionData.vtkArray(),
                              selectionData.fieldAssociation(), mode);
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
vtkSmartPointer<vtkActor> cvSelectionHighlighter::createHighlightActor(
        vtkPolyData* polyData,
        vtkIdTypeArray* selection,
        int fieldAssociation,
        HighlightMode mode) {
    CVLog::PrintDebug(
            QString("[cvSelectionHighlighter::createHighlightActor] "
                    "fieldAssociation=%1 (%2), mode=%3, selectionCount=%4")
                    .arg(fieldAssociation)
                    .arg(fieldAssociation == 0 ? "CELLS" : "POINTS")
                    .arg(mode)
                    .arg(selection ? selection->GetNumberOfTuples() : 0));

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

    if (extracted->GetNumberOfCells() == 0) {
        CVLog::Error(
                "[cvSelectionHighlighter::createHighlightActor] Extraction "
                "failed: 0 cells extracted");
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
            // Enhanced hover feedback (bright cyan, high opacity) - highly
            // visible
            color = m_hoverColor;
            opacity = m_hoverOpacity;
            prop->SetLineWidth(
                    5.0);  // Much thicker lines for better visibility
            prop->SetPointSize(10.0);  // Much larger points for better
                                       // visibility (enhanced)
            prop->SetRenderLinesAsTubes(
                    true);  // Enable tube rendering for better visibility
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
            // Enhanced final selection (bright green, opaque)
            color = m_selectedColor;
            opacity = m_selectedOpacity;
            prop->SetLineWidth(6.0);  // Thickest lines for final selection
            prop->SetPointSize(
                    15.0);  // Largest points for final selection (enhanced)
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
    if (!m_viewer || !actor) {
        return;
    }

    // Get PCLVis for VTK operations
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (!pclVis) {
        CVLog::Warning("[cvSelectionHighlighter] Visualizer is not PCLVis");
        return;
    }

    // Get current renderer
    vtkRenderer* renderer = pclVis->getCurrentRenderer();
    if (renderer) {
        renderer->AddActor(actor);

        // Trigger immediate render update (ParaView-style)
        vtkRenderWindow* renderWindow = renderer->GetRenderWindow();
        if (renderWindow) {
            renderWindow->Render();
        }

        CVLog::PrintDebug(
                QString("[cvSelectionHighlighter] Added highlight actor: %1")
                        .arg(id));
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
