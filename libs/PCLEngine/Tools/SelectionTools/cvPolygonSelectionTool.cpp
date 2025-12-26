// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvPolygonSelectionTool.h"

#include "cvSelectionData.h"
#include "cvSelectionPipeline.h"
#include "cvSelectionTypes.h"  // For SelectionMode and SelectionModifier enums

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

    vtkIdType numPoints = polygon->GetNumberOfTuples() / 2;
    CVLog::Print(QString("[cvPolygonSelectionTool] Perform polygon selection: "
                         "%1 vertices")
                         .arg(numPoints));

    if (numPoints < 3) {
        CVLog::Warning(
                "[cvPolygonSelectionTool] Polygon needs at least 3 points");
        return false;
    }

    // ParaView-style: For custom polygon mode, just emit the polygon
    // and don't perform selection (let the caller handle it)
    // Reference: pqRenderViewSelectionReaction::selectionChanged()
    // SELECT_CUSTOM_POLYGON case
    if (m_mode == SelectionMode::SELECT_CUSTOM_POLYGON) {
        CVLog::Print(
                "[cvPolygonSelectionTool] Custom polygon mode - emitting "
                "polygonCompleted");
        emit polygonCompleted(polygon);
        return true;
    }

    // Get selection modifier
    SelectionModifier modifier = getSelectionModifierFromKeyboard();
    if (modifier == SelectionModifier::SELECTION_DEFAULT &&
        m_modifier != SelectionModifier::SELECTION_DEFAULT) {
        modifier = m_modifier;
    }

    cvSelectionData newSelection;

    // ParaView-aligned: Use pipeline for pixel-precise polygon selection
    // Reference: pqRenderView::selectPolygonPoints/selectPolygonCells
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (pipeline) {
        // Use pixel-precise polygon selection via
        // vtkHardwareSelector::GeneratePolygonSelection
        cvSelectionPipeline::SelectionType pipelineType =
                isSelectingCells() ? cvSelectionPipeline::POLYGON_CELLS
                                   : cvSelectionPipeline::POLYGON_POINTS;

        vtkSmartPointer<vtkSelection> vtkSel =
                pipeline->executePolygonSelection(polygon, pipelineType);

        if (vtkSel) {
            cvSelectionPipeline::FieldAssociation fieldAssoc =
                    isSelectingCells()
                            ? cvSelectionPipeline::FIELD_ASSOCIATION_CELLS
                            : cvSelectionPipeline::FIELD_ASSOCIATION_POINTS;

            newSelection = cvSelectionPipeline::convertToCvSelectionData(
                    vtkSel, fieldAssoc);

            CVLog::Print(QString("[cvPolygonSelectionTool] Pixel-precise "
                                 "polygon selection: %1 items")
                                 .arg(newSelection.count()));
        }
    } else {
        // Fallback: Use bounding box selection (less accurate)
        CVLog::Warning(
                "[cvPolygonSelectionTool] Pipeline not available, "
                "using bounding box fallback");

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

        CVLog::Print(QString("[cvPolygonSelectionTool] Polygon bounds: [%1, "
                             "%2, %3, %4]")
                             .arg(minX)
                             .arg(minY)
                             .arg(maxX)
                             .arg(maxY));

        SelectionMode mode = isSelectingCells()
                                     ? SelectionMode::SELECT_SURFACE_CELLS
                                     : SelectionMode::SELECT_SURFACE_POINTS;

        int region[4] = {minX, minY, maxX, maxY};
        newSelection = hardwareSelectInRegion(
                region, mode, SelectionModifier::SELECTION_DEFAULT);
    }

    if (newSelection.isEmpty()) {
        CVLog::Print("[cvPolygonSelectionTool] No items selected");
        cvSelectionData emptySelection;
        emit selectionFinished(emptySelection);
        return true;
    }

    CVLog::Print(QString("[cvPolygonSelectionTool] Selected %1 items")
                         .arg(newSelection.count()));

    // Apply selection modifier
    cvSelectionData finalSelection =
            applySelectionModifier(newSelection, modifier);

    // Emit signal with final selection
    emit selectionFinished(finalSelection);

    // Store current selection for modifier operations
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
bool cvPolygonSelectionTool::isSelectingCells() const {
    return (m_mode == SelectionMode::SELECT_SURFACE_CELLS_POLYGON);
}
