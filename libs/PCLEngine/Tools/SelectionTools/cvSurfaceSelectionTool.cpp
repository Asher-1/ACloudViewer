// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSurfaceSelectionTool.h"

#include "cvSelectionData.h"
#include "cvSelectionPipeline.h"
#include "cvSelectionTypes.h"  // For SelectionMode and SelectionModifier enums

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkCellData.h>
#include <vtkDataSet.h>
#include <vtkExtractSelection.h>
#include <vtkHardwareSelector.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkPointData.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>

// QT
#include <QSet>

//-----------------------------------------------------------------------------
cvSurfaceSelectionTool::cvSurfaceSelectionTool(SelectionMode mode,
                                               QObject* parent)
    : cvRenderViewSelectionTool(mode, parent), m_fieldAssociation(0) {
    // Determine field association based on mode
    m_fieldAssociation = isSelectingCells() ? 0 : 1;  // 0=cells, 1=points

    CVLog::Print(
            QString("[cvSurfaceSelectionTool] Created with mode: %1, field: %2")
                    .arg(static_cast<int>(mode))
                    .arg(m_fieldAssociation == 0 ? "CELLS" : "POINTS"));
}

//-----------------------------------------------------------------------------
cvSurfaceSelectionTool::~cvSurfaceSelectionTool() {
    CVLog::Print("[cvSurfaceSelectionTool] Destroyed");
}

// Note: setupInteractorStyle(), setupObservers(), and onSelectionChanged()
// are now fully handled by the base class cvRenderViewSelectionTool.
// No need to override them in this subclass.

//-----------------------------------------------------------------------------
bool cvSurfaceSelectionTool::performSelection(int region[4]) {
    if (!m_viewer || !region) {
        CVLog::Warning("[cvSurfaceSelectionTool] Invalid viewer or region");
        return false;
    }

    CVLog::Print(QString("[cvSurfaceSelectionTool] Perform selection: [%1, %2, "
                         "%3, %4]")
                         .arg(region[0])
                         .arg(region[1])
                         .arg(region[2])
                         .arg(region[3]));

    // Use base class unified hardware selection (ParaView-aligned)
    // Reference: pqRenderView.cxx, selectOnSurface()
    SelectionMode mode =
            isSelectingCells() ? SelectionMode::SELECT_SURFACE_CELLS
                               : SelectionMode::SELECT_SURFACE_POINTS;

    // Get selection modifier
    SelectionModifier modifier = getSelectionModifierFromKeyboard();
    if (modifier == SelectionModifier::SELECTION_DEFAULT &&
        m_modifier != SelectionModifier::SELECTION_DEFAULT) {
        modifier = m_modifier;
    }

    // Perform hardware selection using base class method
    cvSelectionData newSelection = hardwareSelectInRegion(
            region, mode, SelectionModifier::SELECTION_DEFAULT);

    if (newSelection.isEmpty()) {
        CVLog::Print("[cvSurfaceSelectionTool] No items selected");
        // Empty selection is still valid - clear selection
        cvSelectionData emptySelection;
        emit selectionFinished(emptySelection);
        return true;
    }

    CVLog::Print(QString("[cvSurfaceSelectionTool] Selected %1 items")
                         .arg(newSelection.count()));

    // Apply selection modifier if needed
    cvSelectionData finalSelection =
            applySelectionModifier(newSelection, modifier);

    // Emit signal with final selection
    emit selectionFinished(finalSelection);

    // Store current selection for modifier operations
    // Smart pointer handles cleanup automatically
    if (!finalSelection.isEmpty()) {
        m_currentSelection = finalSelection.vtkArray();
    } else {
        m_currentSelection = nullptr;
    }

    return true;
}

//-----------------------------------------------------------------------------
// Note: pickRegion() and extractSelectionIds() methods removed - now using
// base class unified hardware selection methods: hardwareSelectInRegion()

//-----------------------------------------------------------------------------
cvSelectionData cvSurfaceSelectionTool::applySelectionModifier(
        const cvSelectionData& newSelection, SelectionModifier modifier) {
    // Use base class unified method (eliminates code duplication)
    cvSelectionData::FieldAssociation assoc = (m_fieldAssociation == 0)
                                                      ? cvSelectionData::CELLS
                                                      : cvSelectionData::POINTS;
    cvSelectionData currentSel;
    if (m_currentSelection && m_currentSelection->GetNumberOfTuples() > 0) {
        currentSel = cvSelectionData(m_currentSelection, assoc);
    }

    return applySelectionModifierUnified(newSelection, currentSel,
                                         static_cast<int>(modifier),
                                         m_fieldAssociation);
}

//-----------------------------------------------------------------------------
bool cvSurfaceSelectionTool::isSelectingCells() const {
    return (m_mode == SelectionMode::SELECT_SURFACE_CELLS ||
            m_mode == SelectionMode::SELECT_SURFACE_CELLS_POLYGON);
}
