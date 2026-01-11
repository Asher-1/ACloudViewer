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
#include <vtkAbstractArray.h>
#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkCellCenters.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDataSetMapper.h>
#include <vtkExtractSelection.h>
#include <vtkIdTypeArray.h>
#include <vtkLabeledDataMapper.h>
#include <vtkMapper.h>
#include <vtkMaskPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkTextProperty.h>
#include <vtkUnstructuredGrid.h>

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
      m_selectedLineWidth(2),
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

    CVLog::PrintDebug(
            "[cvSelectionHighlighter] Initialized with ParaView default "
            "colors: Hover=Purple(0.5,0,1), Selected=Magenta(1,0,1)");
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
        // Check if color actually changed
        bool changed = (color[0] != r || color[1] != g || color[2] != b);

        color[0] = r;
        color[1] = g;
        color[2] = b;

        // Update existing actor's color immediately for real-time preview
        if (actor && *actor) {
            (*actor)->GetProperty()->SetColor(r, g, b);
            (*actor)->Modified();  // Mark actor as modified for VTK pipeline

            // Trigger immediate render to show color change
            PclUtils::PCLVis* pclVis = getPCLVis();
            if (pclVis) {
                vtkRenderWindow* renWin = pclVis->getRenderWindow();
                if (renWin) {
                    renWin->Render();
                }
            }
        }

        // Emit signals for property change notification
        if (changed) {
            emit colorChanged(static_cast<int>(mode));
            emit propertiesChanged();
        }

        CVLog::PrintDebug(
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

        // Trigger immediate render to show opacity change
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis) {
            vtkRenderWindow* renWin = pclVis->getRenderWindow();
            if (renWin) {
                renWin->Render();
            }
        }
    }

    // Emit signals for property change notification
    if (oldOpacity != opacity) {
        emit opacityChanged(static_cast<int>(mode));
        emit propertiesChanged();
    }

    CVLog::PrintDebug(
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

    CVLog::PrintDebug(
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
    removeActorFromVisualizer(m_hoverActorId);
    removeActorFromVisualizer(m_preselectedActorId);
    removeActorFromVisualizer(m_selectedActorId);
    removeActorFromVisualizer(m_boundaryActorId);

    m_hoverActor = nullptr;
    m_preselectedActor = nullptr;
    m_selectedActor = nullptr;
    m_boundaryActor = nullptr;

    // Force render to show changes immediately
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis) {
        vtkRenderWindow* renWin = pclVis->getRenderWindow();
        if (renWin) {
            renWin->Render();
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::clearHoverHighlight() {
    // Remove ONLY hover highlight, keep selected/preselected/boundary
    // This is used during hover updates to avoid clearing persistent selections
    removeActorFromVisualizer(m_hoverActorId);
    m_hoverActor = nullptr;
}

//-----------------------------------------------------------------------------
void cvSelectionHighlighter::setHighlightsVisible(bool visible) {
    // Set visibility of all highlight actors
    // Used to temporarily hide highlights during hardware selection
    // to prevent depth buffer occlusion issues with subtract selection

    if (m_hoverActor) {
        m_hoverActor->SetVisibility(visible ? 1 : 0);
    }
    if (m_preselectedActor) {
        m_preselectedActor->SetVisibility(visible ? 1 : 0);
    }
    if (m_selectedActor) {
        m_selectedActor->SetVisibility(visible ? 1 : 0);
    }
    if (m_boundaryActor) {
        m_boundaryActor->SetVisibility(visible ? 1 : 0);
    }

    // Force render update
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis) {
        vtkRenderWindow* renWin = pclVis->getRenderWindow();
        if (renWin) {
            renWin->Render();
        }
    }
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkActor> cvSelectionHighlighter::createHighlightActor(
        vtkPolyData* polyData,
        vtkIdTypeArray* selection,
        int fieldAssociation,
        HighlightMode mode) {
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
            // Hover feedback (ParaView-style)
            color = m_hoverColor;
            opacity = m_hoverOpacity;
            prop->SetLineWidth(static_cast<float>(m_hoverLineWidth));
            prop->SetPointSize(static_cast<float>(m_hoverPointSize));
            prop->SetRenderLinesAsTubes(true);
            break;

        case PRESELECTED:
            // Preselect feedback (ParaView-style)
            color = m_preselectedColor;
            opacity = m_preselectedOpacity;
            prop->SetLineWidth(static_cast<float>(m_preselectedLineWidth));
            prop->SetPointSize(static_cast<float>(m_preselectedPointSize));
            prop->SetRenderLinesAsTubes(true);
            break;

        case SELECTED:
            // Final selection (ParaView-style)
            color = m_selectedColor;
            opacity = m_selectedOpacity;
            prop->SetLineWidth(static_cast<float>(m_selectedLineWidth));
            prop->SetPointSize(static_cast<float>(m_selectedPointSize));
            prop->SetRenderLinesAsTubes(true);
            break;

        case BOUNDARY:
            // Boundary highlight (ParaView-style)
            color = m_boundaryColor;
            opacity = m_boundaryOpacity;
            prop->SetLineWidth(static_cast<float>(m_boundaryLineWidth));
            prop->SetPointSize(static_cast<float>(m_boundaryPointSize));
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
        // Use the configured line width for the current mode (consistent with
        // hover) Line width is already set above based on mode
        // (hover/preselected/selected/boundary) No need to override it here -
        // keep it consistent
    } else {
        // POINTS: Render as points with enhanced visibility
        prop->SetRepresentationToPoints();
        // Point size is already set above based on mode, ensure it's visible
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

    CVLog::PrintDebug(
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

    CVLog::PrintDebug(QString("[cvSelectionHighlighter] Point label array set: "
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

    CVLog::PrintDebug(QString("[cvSelectionHighlighter] Cell label array set: "
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
    PclUtils::PCLVis* pclVis = getPCLVis();
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
        CVLog::PrintDebug(QString("[cvSelectionHighlighter] Clearing %1 labels")
                                  .arg(isPointLabels ? "point" : "cell"));
        // Force render to update the view
        if (pclVis->getRenderWindow()) {
            pclVis->getRenderWindow()->Modified();
            pclVis->getRenderWindow()->Render();
        }
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

        CVLog::PrintDebug(
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

    CVLog::PrintDebug(QString("[cvSelectionHighlighter] Label mask: %1 points "
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

    CVLog::PrintDebug(
            QString("[cvSelectionHighlighter] Added %1 labels with array '%2'")
                    .arg(isPointLabels ? "point" : "cell")
                    .arg(arrayName));

    // Refresh view
    if (pclVis->getRenderWindow()) {
        pclVis->getRenderWindow()->Render();
    }
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

        // Show RGB colors if available
        vtkUnsignedCharArray* colorArray = nullptr;
        const char* colorNames[] = {"RGB", "Colors", "rgba", "rgb"};
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
                    (arrayName.compare("Colors", Qt::CaseInsensitive) == 0) ||
                    (arrayName.compare("rgba", Qt::CaseInsensitive) == 0);
            // Skip texture coordinate arrays (already displayed above)
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
