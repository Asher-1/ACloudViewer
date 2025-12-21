// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvPolygonSelectionTool.h"

#include "cvSelectionData.h"
#include "cvSelectionPipeline.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkHardwareSelector.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkIntArray.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>

// QT
#include <QSet>

//-----------------------------------------------------------------------------
cvPolygonSelectionTool::cvPolygonSelectionTool(SelectionMode mode,
                                               QObject* parent)
    : cvRenderViewSelectionTool(mode, parent), m_fieldAssociation(0) {
    // Determine field association based on mode
    m_fieldAssociation = isSelectingCells() ? 0 : 1;

    CVLog::Print(
            QString("[cvPolygonSelectionTool] Created with mode: %1, field: %2")
                    .arg(static_cast<int>(mode))
                    .arg(m_fieldAssociation == 0 ? "CELLS" : "POINTS"));
}

//-----------------------------------------------------------------------------
cvPolygonSelectionTool::~cvPolygonSelectionTool() {
    CVLog::Print("[cvPolygonSelectionTool] Destroyed");
}

// Note: setupInteractorStyle(), setupObservers(), and onSelectionChanged()
// are now fully handled by the base class cvRenderViewSelectionTool.
// No need to override them in this subclass.

//-----------------------------------------------------------------------------
bool cvPolygonSelectionTool::performPolygonSelection(vtkIntArray* polygon) {
    if (!m_viewer || !polygon) {
        CVLog::Warning("[cvPolygonSelectionTool] Invalid viewer or polygon");
        return false;
    }

    CVLog::Print(QString("[cvPolygonSelectionTool] Perform polygon selection: "
                         "%1 vertices")
                         .arg(polygon->GetNumberOfTuples()));

    // Convert polygon to bounding region for hardware selection
    // (full polygon filtering will be done post-selection)
    vtkIdType numPoints = polygon->GetNumberOfTuples() / 2;
    if (numPoints < 3) {
        CVLog::Warning(
                "[cvPolygonSelectionTool] Polygon needs at least 3 points");
        return false;
    }

    // Find bounding box of polygon
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = INT_MIN, maxY = INT_MIN;

    for (vtkIdType i = 0; i < numPoints; ++i) {
        int x = polygon->GetValue(i * 2);
        int y = polygon->GetValue(i * 2 + 1);

        minX = std::min(minX, x);
        minY = std::min(minY, y);
        maxX = std::max(maxX, x);
        maxY = std::max(maxY, y);
    }

    CVLog::Print(
            QString("[cvPolygonSelectionTool] Polygon bounds: [%1, %2, %3, %4]")
                    .arg(minX)
                    .arg(minY)
                    .arg(maxX)
                    .arg(maxY));

    // Use base class unified hardware selection (ParaView-aligned)
    cvGenericSelectionTool::SelectionMode mode =
            isSelectingCells() ? cvGenericSelectionTool::SELECT_SURFACE_CELLS
                               : cvGenericSelectionTool::SELECT_SURFACE_POINTS;

    // Get selection modifier
    SelectionModifier modifier = getSelectionModifierFromKeyboard();
    if (modifier == cvViewSelectionManager::SELECTION_DEFAULT &&
        m_modifier != cvViewSelectionManager::SELECTION_DEFAULT) {
        modifier = m_modifier;
    }

    // Perform hardware selection using base class method
    int region[4] = {minX, minY, maxX, maxY};
    cvSelectionData newSelection = hardwareSelectInRegion(
            region, mode, cvGenericSelectionTool::REPLACE);

    if (newSelection.isEmpty()) {
        CVLog::Print("[cvPolygonSelectionTool] No items selected");
        cvSelectionData emptySelection;
        emit selectionFinished(emptySelection);
        return true;
    }

    // Note: Pixel-precise polygon filtering is now handled by
    // cvSelectionPipeline::executePolygonSelection() using
    // vtkHardwareSelector::GeneratePolygonSelection() (ParaView-aligned)

    CVLog::Print(QString("[cvPolygonSelectionTool] Selected %1 items")
                         .arg(newSelection.count()));

    // Apply selection modifier
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
// Note: pickPolygonRegion() and extractSelectionIds() methods removed - now
// using base class unified hardware selection methods: hardwareSelectInRegion()

//-----------------------------------------------------------------------------
cvSelectionData cvPolygonSelectionTool::applySelectionModifier(
        const cvSelectionData& newSelection, SelectionModifier modifier) {
    // Phase 3: Use Pipeline's unified selection combination logic
    // This eliminates code duplication and ensures consistent behavior

    CVLog::PrintDebug(QString("[cvPolygonSelectionTool] "
                              "applySelectionModifier: modifier=%1")
                              .arg(modifier));

    // Get current selection
    cvSelectionData::FieldAssociation assoc = (m_fieldAssociation == 0)
                                                      ? cvSelectionData::CELLS
                                                      : cvSelectionData::POINTS;
    cvSelectionData currentSel;
    if (m_currentSelection && m_currentSelection->GetNumberOfTuples() > 0) {
        currentSel = cvSelectionData(m_currentSelection, assoc);
    }

    // Map to Pipeline operation
    cvSelectionPipeline::CombineOperation operation;
    switch (modifier) {
        case cvViewSelectionManager::SELECTION_DEFAULT:
            operation = cvSelectionPipeline::OPERATION_DEFAULT;
            break;
        case cvViewSelectionManager::SELECTION_ADDITION:
            operation = cvSelectionPipeline::OPERATION_ADDITION;
            break;
        case cvViewSelectionManager::SELECTION_SUBTRACTION:
            operation = cvSelectionPipeline::OPERATION_SUBTRACTION;
            break;
        case cvViewSelectionManager::SELECTION_TOGGLE:
            operation = cvSelectionPipeline::OPERATION_TOGGLE;
            break;
        default:
            CVLog::Warning(
                    QString("[cvPolygonSelectionTool] Unknown modifier: %1")
                            .arg(modifier));
            return newSelection;
    }

    // Use Pipeline's unified combination logic
    return cvSelectionPipeline::combineSelections(currentSel, newSelection,
                                                  operation);
}

//-----------------------------------------------------------------------------
bool cvPolygonSelectionTool::isSelectingCells() const {
    return (m_mode == cvViewSelectionManager::SELECT_SURFACE_CELLS_POLYGON);
}
