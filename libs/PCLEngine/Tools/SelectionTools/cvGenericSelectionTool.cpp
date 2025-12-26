// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvGenericSelectionTool.h"

#include "PclUtils/PCLVis.h"
#include "cvSelectionBase.h"
#include "cvSelectionPipeline.h"
#include "cvSelectionTypes.h"  // For SelectionMode and SelectionModifier enums
#include "cvViewSelectionManager.h"

// VTK
#include <vtkActor.h>
#include <vtkCellPicker.h>
#include <vtkDataObject.h>
#include <vtkDataSetMapper.h>
#include <vtkHardwareSelector.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>  // For vtkInformation
#include <vtkMapper.h>
#include <vtkPointPicker.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkSmartPointer.h>

// Qt
#include <QSet>  // For QSet

// ECV
#include <CVLog.h>

// STD
#include <algorithm>
#include <map>
#include <set>

//=============================================================================
// Manager and Pipeline Access
//=============================================================================

cvSelectionPipeline* cvGenericSelectionTool::getSelectionPipeline() const {
    if (!m_manager) {
        return nullptr;
    }
    return m_manager->getPipeline();
}

//-----------------------------------------------------------------------------
vtkPolyData* cvGenericSelectionTool::getPolyDataForSelection(
        const cvSelectionData* selectionData) {
    // Priority 1: From provided selection data's actor info (ParaView way!)
    if (selectionData && selectionData->hasActorInfo()) {
        vtkPolyData* polyData = selectionData->primaryPolyData();
        if (polyData) {
            CVLog::PrintDebug(
                    "[cvGenericSelectionTool] Using polyData from provided "
                    "selection actor info");
            return polyData;
        }
    }

    // Priority 2: From current selection's actor info (tool-specific)
    if (m_currentSelection.hasActorInfo()) {
        vtkPolyData* polyData = m_currentSelection.primaryPolyData();
        if (polyData) {
            CVLog::PrintDebug(
                    "[cvGenericSelectionTool] Using polyData from current "
                    "selection actor info");
            return polyData;
        }
    }

    // Priority 3-4: Delegate to base class (manager singleton, then fallback)
    return cvSelectionBase::getPolyDataForSelection(selectionData);
}

//=============================================================================
// Hardware Selection Implementation (ParaView-Aligned)
//=============================================================================

//-----------------------------------------------------------------------------
cvSelectionData cvGenericSelectionTool::hardwareSelectAtPoint(
        int x, int y, SelectionMode mode, SelectionModifier modifier) {
    int region[4] = {x, y, x, y};
    return hardwareSelectInRegion(region, mode, modifier);
}

//-----------------------------------------------------------------------------
cvSelectionData cvGenericSelectionTool::hardwareSelectInRegion(
        const int region[4], SelectionMode mode, SelectionModifier modifier) {
    if (!hasValidPCLVis()) {
        CVLog::Warning("[hardwareSelectInRegion] No valid visualizer");
        return cvSelectionData();
    }

    // Try to use pipeline if available (Phase 1: Pipeline Integration)
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (pipeline) {
        CVLog::PrintDebug("[hardwareSelectInRegion] Using cvSelectionPipeline");

        // Map SelectionMode to Pipeline SelectionType
        cvSelectionPipeline::SelectionType pipelineType;
        switch (mode) {
            case SelectionMode::SELECT_SURFACE_CELLS:
                pipelineType = cvSelectionPipeline::SURFACE_CELLS;
                break;
            case SelectionMode::SELECT_SURFACE_POINTS:
                pipelineType = cvSelectionPipeline::SURFACE_POINTS;
                break;
            case SelectionMode::SELECT_FRUSTUM_CELLS:
                pipelineType = cvSelectionPipeline::FRUSTUM_CELLS;
                break;
            case SelectionMode::SELECT_FRUSTUM_POINTS:
                pipelineType = cvSelectionPipeline::FRUSTUM_POINTS;
                break;
            default:
                pipelineType = cvSelectionPipeline::SURFACE_CELLS;
                break;
        }

        // Execute selection through pipeline (with caching!)
        // ParaView-style: Get vtkSelection first, then convert to
        // cvSelectionData WITH actor info
        vtkSmartPointer<vtkSelection> vtkSel =
                pipeline->executeRectangleSelection(const_cast<int*>(region),
                                                    pipelineType);

        cvSelectionData newSelection;
        if (vtkSel) {
            int fieldAssoc =
                    (mode == SelectionMode::SELECT_SURFACE_POINTS ||
                     mode == SelectionMode::SELECT_FRUSTUM_POINTS)
                            ? cvSelectionPipeline::FIELD_ASSOCIATION_POINTS
                            : cvSelectionPipeline::FIELD_ASSOCIATION_CELLS;

            // Convert to cvSelectionData WITH actor information (ParaView way!)
            newSelection = cvSelectionPipeline::convertToCvSelectionData(
                    vtkSel, static_cast<cvSelectionPipeline::FieldAssociation>(
                                    fieldAssoc));
        }

        // Apply modifier using pipeline's combineSelections
        if (modifier != SelectionModifier::SELECTION_DEFAULT &&
            !m_currentSelection.isEmpty()) {
            cvSelectionPipeline::CombineOperation operation;
            switch (modifier) {
                case SelectionModifier::SELECTION_ADDITION:
                    operation = cvSelectionPipeline::OPERATION_ADDITION;
                    break;
                case SelectionModifier::SELECTION_SUBTRACTION:
                    operation = cvSelectionPipeline::OPERATION_SUBTRACTION;
                    break;
                case SelectionModifier::SELECTION_TOGGLE:
                    operation = cvSelectionPipeline::OPERATION_TOGGLE;
                    break;
                default:
                    operation = cvSelectionPipeline::OPERATION_DEFAULT;
                    break;
            }
            newSelection = cvSelectionPipeline::combineSelections(
                    m_currentSelection, newSelection, operation);
        }

        // Store as current selection
        if (m_multipleSelectionMode) {
            m_currentSelection = newSelection;
        }

        return newSelection;
    }

    // Fallback: Direct VTK implementation (original code)
    CVLog::Warning(
            "[hardwareSelectInRegion] Pipeline not available, using direct "
            "VTK");

    // Determine field association
    int fieldAssociation = vtkDataObject::FIELD_ASSOCIATION_CELLS;
    switch (mode) {
        case SelectionMode::SELECT_SURFACE_POINTS:
        case SelectionMode::SELECT_FRUSTUM_POINTS:
            fieldAssociation = vtkDataObject::FIELD_ASSOCIATION_POINTS;
            break;
        default:
            break;
    }

    // Configure hardware selector (reused for performance)
    vtkHardwareSelector* selector =
            configureHardwareSelector(region, fieldAssociation);
    if (!selector) {
        CVLog::Warning(
                "[hardwareSelectInRegion] Failed to configure hardware "
                "selector");
        return cvSelectionData();
    }

    // Perform selection
    vtkSmartPointer<vtkSelection> vtkSel = selector->Select();
    if (!vtkSel) {
        CVLog::Warning("[hardwareSelectInRegion] Selection failed");
        return cvSelectionData();
    }

    // Extract actor information
    QVector<cvActorSelectionInfo> actorInfos =
            extractActorInfo(selector, vtkSel);

    // Convert to our selection format
    cvSelectionData newSelection = convertSelection(vtkSel, actorInfos);

    // Smart pointer handles cleanup automatically

    // Apply modifier if needed
    if (modifier != SelectionModifier::SELECTION_DEFAULT &&
        !m_currentSelection.isEmpty()) {
        return applyModifier(newSelection, m_currentSelection, modifier);
    }

    // Store as current selection
    if (m_multipleSelectionMode) {
        m_currentSelection = newSelection;
    }

    return newSelection;
}

//-----------------------------------------------------------------------------
QVector<cvActorSelectionInfo> cvGenericSelectionTool::getActorsAtPoint(int x,
                                                                       int y) {
    if (!hasValidPCLVis()) {
        return QVector<cvActorSelectionInfo>();
    }

    int region[4] = {x, y, x, y};
    vtkHardwareSelector* selector = configureHardwareSelector(
            region, vtkDataObject::FIELD_ASSOCIATION_CELLS);

    if (!selector) {
        return QVector<cvActorSelectionInfo>();
    }

    vtkSmartPointer<vtkSelection> vtkSel = selector->Select();
    QVector<cvActorSelectionInfo> actorInfos;

    if (vtkSel) {
        actorInfos = extractActorInfo(selector, vtkSel);
        // Smart pointer handles cleanup automatically
    }

    // Selector is reused, no need to delete
    return actorInfos;
}

//-----------------------------------------------------------------------------
vtkHardwareSelector* cvGenericSelectionTool::getHardwareSelector() {
    if (!m_hardwareSelector) {
        m_hardwareSelector = vtkSmartPointer<vtkHardwareSelector>::New();
        CVLog::PrintDebug(
                "[getHardwareSelector] Created new hardware selector instance");
    }
    return m_hardwareSelector;
}

//-----------------------------------------------------------------------------
vtkHardwareSelector* cvGenericSelectionTool::configureHardwareSelector(
        const int region[4], int fieldAssociation) {
    vtkRenderer* renderer = getRenderer();
    if (!renderer) {
        CVLog::Warning("[configureHardwareSelector] No renderer available");
        return nullptr;
    }

    // Get or create hardware selector (reused for performance)
    vtkHardwareSelector* selector = getHardwareSelector();
    if (!selector) {
        CVLog::Warning(
                "[configureHardwareSelector] Failed to get hardware selector");
        return nullptr;
    }

    // Configure for this selection
    selector->SetRenderer(renderer);
    selector->SetArea(region[0], region[1], region[2], region[3]);
    selector->SetFieldAssociation(fieldAssociation);

    // Capture Z-buffer values for depth sorting
    selector->SetCaptureZValues(m_captureZValues);

    CVLog::PrintDebug(QString("[configureHardwareSelector] Configured: "
                              "region=[%1,%2,%3,%4], "
                              "fieldAssoc=%5, captureZ=%6")
                              .arg(region[0])
                              .arg(region[1])
                              .arg(region[2])
                              .arg(region[3])
                              .arg(fieldAssociation)
                              .arg(m_captureZValues));

    return selector;
}

//-----------------------------------------------------------------------------
QVector<cvActorSelectionInfo> cvGenericSelectionTool::extractActorInfo(
        vtkHardwareSelector* selector, vtkSelection* vtkSel) {
    QVector<cvActorSelectionInfo> actorInfos;

    if (!selector || !vtkSel) {
        return actorInfos;
    }

    // Get renderer for actor lookup
    vtkRenderer* renderer = getRenderer();
    if (!renderer) {
        return actorInfos;
    }

    // Map to store prop information
    std::map<int, cvActorSelectionInfo> propInfoMap;

    // Iterate through selection nodes
    for (unsigned int i = 0; i < vtkSel->GetNumberOfNodes(); ++i) {
        vtkSelectionNode* node = vtkSel->GetNode(i);
        if (!node) continue;

        // Get prop ID
        int propId = -1;
        if (node->GetProperties()->Has(vtkSelectionNode::PROP_ID())) {
            propId = node->GetProperties()->Get(vtkSelectionNode::PROP_ID());
        }

        if (propId < 0) continue;

        // Get the prop/actor
        vtkProp* prop = selector->GetPropFromID(propId);
        vtkActor* actor = vtkActor::SafeDownCast(prop);

        if (!actor) continue;

        // Create actor info
        cvActorSelectionInfo info;
        info.actor = actor;
        info.propId = propId;

        // Get Z-value if available (for depth sorting)
        if (m_captureZValues) {
            unsigned int centerPos[2] = {
                    (selector->GetArea()[0] + selector->GetArea()[2]) / 2,
                    (selector->GetArea()[1] + selector->GetArea()[3]) / 2};
            vtkHardwareSelector::PixelInformation pixelInfo =
                    selector->GetPixelInformation(centerPos, 0);

            if (pixelInfo.Valid && pixelInfo.PropID == propId) {
                // Normalize Z value to [0, 1], smaller = closer to camera
                info.zValue =
                        pixelInfo.AttributeID >= 0
                                ? static_cast<double>(pixelInfo.AttributeID) /
                                          1000000.0
                                : 1.0;
            }
        }

        // Get polyData from actor
        vtkMapper* mapper = actor->GetMapper();
        if (mapper) {
            // Try vtkPolyDataMapper first
            if (vtkPolyDataMapper* pdMapper =
                        vtkPolyDataMapper::SafeDownCast(mapper)) {
                info.polyData = vtkPolyData::SafeDownCast(pdMapper->GetInput());
            }
            // Try vtkDataSetMapper
            else if (vtkDataSetMapper* dsMapper =
                             vtkDataSetMapper::SafeDownCast(mapper)) {
                vtkDataSet* dataset = dsMapper->GetInput();
                if (dataset) {
                    info.polyData = vtkPolyData::SafeDownCast(dataset);
                }
            }
        }

        // Get composite index if available
        if (node->GetProperties()->Has(vtkSelectionNode::COMPOSITE_INDEX())) {
            info.blockIndex = node->GetProperties()->Get(
                    vtkSelectionNode::COMPOSITE_INDEX());
        }

        // Store in map
        propInfoMap[propId] = info;
    }

    // Convert map to vector and sort by Z-value
    actorInfos.reserve(propInfoMap.size());
    for (const auto& pair : propInfoMap) {
        actorInfos.append(pair.second);
    }

    // Sort by Z-value (front to back)
    std::sort(actorInfos.begin(), actorInfos.end(),
              [](const cvActorSelectionInfo& a, const cvActorSelectionInfo& b) {
                  return a.zValue < b.zValue;
              });

    CVLog::PrintDebug(QString("[extractActorInfo] Found %1 actor(s)")
                              .arg(actorInfos.size()));

    return actorInfos;
}

//-----------------------------------------------------------------------------
cvSelectionData cvGenericSelectionTool::convertSelection(
        vtkSelection* vtkSel, const QVector<cvActorSelectionInfo>& actorInfos) {
    cvSelectionData selection;

    if (!vtkSel || vtkSel->GetNumberOfNodes() == 0) {
        return selection;
    }

    // Get the first selection node
    vtkSelectionNode* node = vtkSel->GetNode(0);
    if (!node) {
        return selection;
    }

    // Get selection IDs
    vtkIdTypeArray* selectionIds =
            vtkIdTypeArray::SafeDownCast(node->GetSelectionList());

    if (!selectionIds) {
        return selection;
    }

    // Determine field association
    int fieldAssociation = node->GetFieldType();

    // Create selection data
    selection = cvSelectionData(selectionIds, fieldAssociation);

    // Attach actor information
    for (const cvActorSelectionInfo& info : actorInfos) {
        selection.addActorInfo(info);
    }

    CVLog::PrintDebug(
            QString("[convertSelection] Selection: %1 items, %2 actors")
                    .arg(selection.count())
                    .arg(selection.actorCount()));

    return selection;
}

//-----------------------------------------------------------------------------
cvSelectionData cvGenericSelectionTool::applyModifier(
        const cvSelectionData& newSelection,
        const cvSelectionData& currentSelection,
        SelectionModifier modifier) {
    if (newSelection.isEmpty()) {
        return currentSelection;
    }

    if (currentSelection.isEmpty() ||
        modifier == SelectionModifier::SELECTION_DEFAULT) {
        return newSelection;
    }

    // Get ID sets
    QVector<qint64> newIds = newSelection.ids();
    QVector<qint64> currentIds = currentSelection.ids();
    QSet<qint64> resultSet(currentIds.begin(), currentIds.end());

    switch (modifier) {
        case SelectionModifier::SELECTION_ADDITION: {
            // Add new IDs to current selection
            for (qint64 id : newIds) {
                resultSet.insert(id);
            }
            break;
        }

        case SelectionModifier::SELECTION_SUBTRACTION: {
            // Remove new IDs from current selection
            for (qint64 id : newIds) {
                resultSet.remove(id);
            }
            break;
        }

        case SelectionModifier::SELECTION_TOGGLE: {
            // Toggle: add if not present, remove if present
            for (qint64 id : newIds) {
                if (resultSet.contains(id)) {
                    resultSet.remove(id);
                } else {
                    resultSet.insert(id);
                }
            }
            break;
        }

        default:
            break;
    }

    // Convert back to vector
    QVector<qint64> resultIds(resultSet.begin(), resultSet.end());

    // Create result selection
    cvSelectionData result(resultIds, newSelection.fieldAssociation());

    // Merge actor information
    if (modifier == SelectionModifier::SELECTION_ADDITION ||
        modifier == SelectionModifier::SELECTION_TOGGLE) {
        for (int i = 0; i < currentSelection.actorCount(); ++i) {
            result.addActorInfo(currentSelection.actorInfo(i));
        }
    }
    for (int i = 0; i < newSelection.actorCount(); ++i) {
        result.addActorInfo(newSelection.actorInfo(i));
    }

    return result;
}

//-----------------------------------------------------------------------------
vtkRenderer* cvGenericSelectionTool::getRenderer() {
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (!pclVis) {
        return nullptr;
    }
    return pclVis->getCurrentRenderer();
}

//=============================================================================
// Software Picking Implementation (Unified from Subclasses)
//=============================================================================

//-----------------------------------------------------------------------------
void cvGenericSelectionTool::initializePickers() {
    if (!m_cellPicker) {
        m_cellPicker = vtkSmartPointer<vtkCellPicker>::New();
        m_cellPicker->SetTolerance(0.005);  // Default tolerance for cells
    }

    if (!m_pointPicker) {
        m_pointPicker = vtkSmartPointer<vtkPointPicker>::New();
        m_pointPicker->SetTolerance(0.01);  // Default tolerance for points
    }

    CVLog::PrintDebug("[initializePickers] Pickers initialized");
}

//-----------------------------------------------------------------------------
vtkIdType cvGenericSelectionTool::pickAtPosition(int x,
                                                 int y,
                                                 bool selectCells) {
    // Ensure pickers are initialized
    initializePickers();

    // Get renderer (try from member variable first, then from visualizer)
    vtkRenderer* renderer = m_renderer;
    if (!renderer) {
        renderer = getRenderer();
    }

    if (!renderer) {
        CVLog::Warning("[pickAtPosition] No renderer available");
        return -1;
    }

    vtkIdType pickedId = -1;

    if (selectCells) {
        // Pick cells
        if (m_cellPicker->Pick(x, y, 0, renderer)) {
            pickedId = m_cellPicker->GetCellId();

            // Get the picked actor for multi-actor support
            vtkActor* pickedActor = m_cellPicker->GetActor();

            CVLog::PrintDebug(
                    QString("[pickAtPosition] Picked cell ID: %1 from actor %2")
                            .arg(pickedId)
                            .arg((quintptr)pickedActor, 0, 16));
        }
    } else {
        // Pick points
        if (m_pointPicker->Pick(x, y, 0, renderer)) {
            pickedId = m_pointPicker->GetPointId();

            // Get the picked actor for multi-actor support
            vtkActor* pickedActor = m_pointPicker->GetActor();

            CVLog::PrintDebug(QString("[pickAtPosition] Picked point ID: %1 "
                                      "from actor %2")
                                      .arg(pickedId)
                                      .arg((quintptr)pickedActor, 0, 16));
        }
    }

    return pickedId;
}

//-----------------------------------------------------------------------------
vtkIdType cvGenericSelectionTool::pickAtCursor(bool selectCells) {
    if (!m_interactor) {
        CVLog::Warning("[pickAtCursor] No interactor set");
        return -1;
    }

    // Get cursor position from interactor
    int* cursorPos = m_interactor->GetEventPosition();
    return pickAtPosition(cursorPos[0], cursorPos[1], selectCells);
}

//-----------------------------------------------------------------------------
void cvGenericSelectionTool::setPickerTolerance(double cellTolerance,
                                                double pointTolerance) {
    initializePickers();

    if (m_cellPicker) {
        m_cellPicker->SetTolerance(cellTolerance);
    }

    if (m_pointPicker) {
        m_pointPicker->SetTolerance(pointTolerance);
    }

    CVLog::PrintDebug(QString("[setPickerTolerance] Cell: %1, Point: %2")
                              .arg(cellTolerance)
                              .arg(pointTolerance));
}

//-----------------------------------------------------------------------------
vtkActor* cvGenericSelectionTool::getPickedActor(bool selectCells) {
    if (selectCells && m_cellPicker) {
        return m_cellPicker->GetActor();
    } else if (!selectCells && m_pointPicker) {
        return m_pointPicker->GetActor();
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
vtkPolyData* cvGenericSelectionTool::getPickedPolyData(bool selectCells) {
    vtkActor* actor = getPickedActor(selectCells);
    if (!actor) {
        return nullptr;
    }

    vtkMapper* mapper = actor->GetMapper();
    if (!mapper) {
        return nullptr;
    }

    // Try vtkPolyDataMapper
    vtkPolyDataMapper* pdMapper = vtkPolyDataMapper::SafeDownCast(mapper);
    if (pdMapper) {
        return vtkPolyData::SafeDownCast(pdMapper->GetInput());
    }

    // Try vtkDataSetMapper
    vtkDataSetMapper* dsMapper = vtkDataSetMapper::SafeDownCast(mapper);
    if (dsMapper) {
        vtkDataSet* dataset = dsMapper->GetInput();
        return vtkPolyData::SafeDownCast(dataset);
    }

    return nullptr;
}

//-----------------------------------------------------------------------------
bool cvGenericSelectionTool::getPickedPosition(bool selectCells,
                                               double position[3]) {
    if (selectCells && m_cellPicker) {
        m_cellPicker->GetPickPosition(position);
        return true;
    } else if (!selectCells && m_pointPicker) {
        m_pointPicker->GetPickPosition(position);
        return true;
    }
    return false;
}

//-----------------------------------------------------------------------------
cvSelectionData cvGenericSelectionTool::createSelectionFromPick(
        vtkIdType pickedId, bool selectCells) {
    cvSelectionData selection;

    if (pickedId < 0) {
        return selection;  // Empty selection
    }

    // Create selection with single ID
    QVector<qint64> ids;
    ids.append(pickedId);

    cvSelectionData::FieldAssociation association =
            selectCells ? cvSelectionData::CELLS : cvSelectionData::POINTS;

    selection = cvSelectionData(ids, association);

    // Attach actor information
    vtkActor* actor = getPickedActor(selectCells);
    vtkPolyData* polyData = getPickedPolyData(selectCells);

    if (actor && polyData) {
        double worldPos[3];
        getPickedPosition(selectCells, worldPos);

        // Create actor info
        cvActorSelectionInfo info;
        info.actor = actor;
        info.polyData = polyData;

        // Estimate Z value from world position (not as accurate as hardware
        // selector) This is a simplified approximation
        info.zValue = worldPos[2];  // Use world Z as approximation

        selection.addActorInfo(info);

        CVLog::PrintDebug(QString("[createSelectionFromPick] Created "
                                  "selection: ID=%1, actor=%2")
                                  .arg(pickedId)
                                  .arg((quintptr)actor, 0, 16));
    }

    return selection;
}

//-----------------------------------------------------------------------------
cvSelectionData cvGenericSelectionTool::applySelectionModifierUnified(
        const cvSelectionData& newSelection,
        const cvSelectionData& currentSelection,
        int modifier,
        int fieldAssociation) {
    // ParaView-aligned: Use cvSelectionPipeline::combineSelections()
    // This eliminates code duplication between tools

    CVLog::PrintDebug(QString("[cvGenericSelectionTool] "
                              "applySelectionModifierUnified: modifier=%1")
                              .arg(modifier));

    // Map view manager modifier to pipeline operation
    cvSelectionPipeline::CombineOperation operation;
    switch (modifier) {
        case 0:  // SELECTION_DEFAULT
            operation = cvSelectionPipeline::OPERATION_DEFAULT;
            break;
        case 1:  // SELECTION_ADDITION
            operation = cvSelectionPipeline::OPERATION_ADDITION;
            break;
        case 2:  // SELECTION_SUBTRACTION
            operation = cvSelectionPipeline::OPERATION_SUBTRACTION;
            break;
        case 3:  // SELECTION_TOGGLE
            operation = cvSelectionPipeline::OPERATION_TOGGLE;
            break;
        default:
            CVLog::Warning(
                    QString("[cvGenericSelectionTool] Unknown modifier: %1")
                            .arg(modifier));
            return newSelection;
    }

    // Use Pipeline's unified combination logic
    return cvSelectionPipeline::combineSelections(currentSelection,
                                                  newSelection, operation);
}
