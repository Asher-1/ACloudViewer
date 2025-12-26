// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvBlockSelectionTool.h"

#include "cvSelectionTypes.h"  // For SelectionMode enum

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkAreaPicker.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkHardwareSelector.h>  // Full definition needed for copy assignment operator
#include <vtkInteractorStyleRubberBandPick.h>
#include <vtkProp.h>
#include <vtkProp3DCollection.h>
#include <vtkPropCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

//-----------------------------------------------------------------------------
cvBlockSelectionTool::cvBlockSelectionTool(SelectionMode mode, QObject* parent)
    : cvRenderViewSelectionTool(mode, parent), m_interactorStyle(nullptr) {
    QString modeStr = (mode == SelectionMode::SELECT_BLOCKS)
                              ? "SELECT_BLOCKS"
                              : "SELECT_FRUSTUM_BLOCKS";
    CVLog::Print(QString("[cvBlockSelectionTool] Created with mode: %1")
                         .arg(modeStr));
}

//-----------------------------------------------------------------------------
cvBlockSelectionTool::~cvBlockSelectionTool() {
    CVLog::Print("[cvBlockSelectionTool] Destroyed");
}

//-----------------------------------------------------------------------------
void cvBlockSelectionTool::setupInteractorStyle() {
    PclUtils::PCLVis* pclVis = getPCLVis();
    vtkRenderWindowInteractor* interactor =
            pclVis ? pclVis->getRenderWindowInteractor() : nullptr;
    if (!interactor) {
        CVLog::Error("[cvBlockSelectionTool] No interactor available!");
        return;
    }

    // Create rubber band pick style for both surface and frustum block
    // selection
    m_interactorStyle =
            vtkSmartPointer<vtkInteractorStyleRubberBandPick>::New();

    // Set the interactor style
    interactor->SetInteractorStyle(m_interactorStyle);

    QString modeStr = (m_mode == SelectionMode::SELECT_BLOCKS)
                              ? "surface"
                              : "frustum";
    CVLog::Print(QString("[cvBlockSelectionTool] Interactor style set for %1 "
                         "block selection")
                         .arg(modeStr));
}

//-----------------------------------------------------------------------------
void cvBlockSelectionTool::setupObservers() {
    PclUtils::PCLVis* pclVis = getPCLVis();
    vtkRenderWindowInteractor* interactor =
            pclVis ? pclVis->getRenderWindowInteractor() : nullptr;
    if (!interactor || !m_interactorStyle) {
        CVLog::Error(
                "[cvBlockSelectionTool] Cannot setup observers - missing "
                "components");
        return;
    }

    // Create callback for selection events
    vtkSmartPointer<vtkCallbackCommand> selectionCallback =
            vtkSmartPointer<vtkCallbackCommand>::New();
    selectionCallback->SetCallback([](vtkObject* caller, unsigned long eventId,
                                      void* clientData, void* callData) {
        auto* tool = static_cast<cvBlockSelectionTool*>(clientData);
        if (tool) {
            tool->onSelectionChanged(caller, eventId, callData);
        }
    });
    selectionCallback->SetClientData(this);

    // Listen for selection end event
    m_interactorStyle->AddObserver(vtkCommand::EndPickEvent, selectionCallback);

    CVLog::Print("[cvBlockSelectionTool] Observers configured");
}

//-----------------------------------------------------------------------------
void cvBlockSelectionTool::onSelectionChanged(vtkObject* caller,
                                              unsigned long eventId,
                                              void* callData) {
    if (eventId != vtkCommand::EndPickEvent) {
        return;
    }

    if (!m_interactorStyle || !getVisualizer()) {
        CVLog::Warning(
                "[cvBlockSelectionTool] No interactor style or viewer for "
                "selection");
        return;
    }

    // Get the selection rectangle from the interactor
    PclUtils::PCLVis* pclVis = getPCLVis();
    vtkRenderWindowInteractor* interactor =
            pclVis ? pclVis->getRenderWindowInteractor() : nullptr;
    if (!interactor) {
        CVLog::Warning("[cvBlockSelectionTool] No interactor available");
        return;
    }

    // Get the current event position (end position)
    int* endPos = interactor->GetEventPosition();
    int* lastPos = interactor->GetLastEventPosition();

    int x1 = lastPos[0];
    int y1 = lastPos[1];
    int x2 = endPos[0];
    int y2 = endPos[1];

    // Ensure x1 < x2 and y1 < y2
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);

    CVLog::Print(QString("[cvBlockSelectionTool] Selection region: (%1,%2) to "
                         "(%3,%4)")
                         .arg(x1)
                         .arg(y1)
                         .arg(x2)
                         .arg(y2));

    // Extract block IDs from the selection
    QVector<int> blockIds = extractBlockIds(x1, y1, x2, y2);

    if (!blockIds.isEmpty()) {
        QString modeStr = (m_mode == SelectionMode::SELECT_BLOCKS)
                                  ? "surface"
                                  : "frustum";
        CVLog::Print(
                QString("[cvBlockSelectionTool] %1 blocks selected (%2 mode)")
                        .arg(blockIds.size())
                        .arg(modeStr));

        emit blockSelectionFinished(blockIds);
    } else {
        CVLog::Print("[cvBlockSelectionTool] No blocks selected");
    }

    emit selectionCompleted();
}

//-----------------------------------------------------------------------------
QVector<int> cvBlockSelectionTool::extractBlockIds(int x1,
                                                   int y1,
                                                   int x2,
                                                   int y2) {
    QVector<int> blockIds;

    PclUtils::PCLVis* pclVis = getPCLVis();
    if (!pclVis) {
        CVLog::Error("[cvBlockSelectionTool] No viewer available");
        return blockIds;
    }

    // NOTE: This is a simplified implementation
    // In a full implementation, we would:
    // 1. Get the renderer and area picker
    // 2. Use vtkAreaPicker to pick within the rectangle
    // 3. For each picked prop, determine its block ID
    // 4. For frustum selection, also check depth/frustum containment

    vtkRenderer* renderer = pclVis->getCurrentRenderer();
    if (!renderer) {
        CVLog::Error("[cvBlockSelectionTool] No renderer available");
        return blockIds;
    }

    // Use area picker to select props within the rectangle
    vtkNew<vtkAreaPicker> picker;
    picker->AreaPick(x1, y1, x2, y2, renderer);

    vtkProp3DCollection* pickedProps = picker->GetProp3Ds();
    if (pickedProps) {
        pickedProps->InitTraversal();
        int blockId = 0;
        while (vtkProp* prop = pickedProps->GetNextProp()) {
            // In a real implementation, we would query the prop's block
            // metadata For now, assign sequential block IDs
            if (isBlockInSelection(blockId, x1, y1, x2, y2)) {
                blockIds.append(blockId);
                CVLog::Print(QString("[cvBlockSelectionTool] Block %1 selected")
                                     .arg(blockId));
            }
            blockId++;
        }
    }

    // If no props picked but selection made, treat as single block selection
    if (blockIds.isEmpty() && (x2 - x1) > 5 && (y2 - y1) > 5) {
        // Simulate block 0 as the main data block
        blockIds.append(0);
        CVLog::Print("[cvBlockSelectionTool] Default block (0) selected");
    }

    return blockIds;
}

//-----------------------------------------------------------------------------
bool cvBlockSelectionTool::isBlockInSelection(
        int blockId, int x1, int y1, int x2, int y2) {
    // For frustum selection, we would check if the block's bounds intersect
    // with the frustum For surface selection, we would check if the block is
    // visible on the surface

    if (m_mode == SelectionMode::SELECT_FRUSTUM_BLOCKS) {
        // Frustum selection: include all blocks that intersect with the frustum
        // (simplified: always include for demonstration)
        return true;
    } else {
        // Surface selection: only include visible blocks
        // (simplified: always include for demonstration)
        return true;
    }
}
