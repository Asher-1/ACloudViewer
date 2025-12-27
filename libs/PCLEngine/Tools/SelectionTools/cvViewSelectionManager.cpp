// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvViewSelectionManager.h"

#include "cvBlockSelectionTool.h"
#include "cvFrustumSelectionTool.h"
#include "cvPolygonSelectionTool.h"
#include "cvRenderViewSelectionTool.h"
#include "cvSelectionData.h"
#include "cvSelectionTypes.h"  // For SelectionMode and SelectionModifier enums
#include "cvSurfaceSelectionTool.h"
#include "cvTooltipSelectionTool.h"
#include "cvZoomBoxSelectionTool.h"

// Selection utility modules
#include "cvSelectionAlgebra.h"
#include "cvSelectionAnnotation.h"
#include "cvSelectionBookmarks.h"
#include "cvSelectionFilter.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionHistory.h"
#include "cvSelectionPipeline.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkHardwareSelector.h>  // Full definition needed for copy assignment operator
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkPolyData.h>

// QT
#include <QSet>

//-----------------------------------------------------------------------------
cvViewSelectionManager* cvViewSelectionManager::instance() {
    static cvViewSelectionManager _instance;
    return &_instance;
}

//-----------------------------------------------------------------------------
cvViewSelectionManager::cvViewSelectionManager(QObject* parent)
    : QObject(parent),
      cvGenericSelectionTool(),
      m_currentMode(static_cast<SelectionMode>(-1)),
      m_currentModifier(SelectionModifier::SELECTION_DEFAULT),
      m_isActive(false),
      m_currentTool(nullptr),
      m_currentSelection(nullptr),
      m_currentSelectionFieldAssociation(0),
      m_history(nullptr),
      m_algebra(nullptr),
      m_pipeline(nullptr),
      m_filter(nullptr),
      m_bookmarks(nullptr),
      m_annotations(nullptr),
      m_highlighter(nullptr) {
    // Initialize utility modules (ParaView-style architecture)
    m_history = new cvSelectionHistory(this);
    m_algebra = new cvSelectionAlgebra(this);
    m_pipeline = new cvSelectionPipeline(this);
    m_filter = new cvSelectionFilter(this);
    m_bookmarks = new cvSelectionBookmarks(this);
    m_annotations = new cvSelectionAnnotationManager(this);
    m_highlighter =
            new cvSelectionHighlighter();  // Shared highlighter for all tools

    // Connect history signals
    connect(m_history, &cvSelectionHistory::selectionRestored, this,
            [this](const cvSelectionData& data) { setCurrentSelection(data); });
    connect(m_history, &cvSelectionHistory::historyChanged, this,
            [this]() { emit selectionChanged(); });

    CVLog::PrintDebug(
            "[cvViewSelectionManager] Initialized with utility modules");
}

//-----------------------------------------------------------------------------
cvViewSelectionManager::~cvViewSelectionManager() {
    // Disable current selection if active
    if (m_isActive) {
        disableSelection();
    }

    // Clean up selection data
    // Smart pointer handles cleanup automatically
    m_currentSelection = nullptr;

    // Clean up all cached tools
    for (auto tool : m_toolCache) {
        if (tool) {
            delete tool;
        }
    }
    m_toolCache.clear();

    // Clean up highlighter (not a QObject, so manual cleanup)
    delete m_highlighter;
    m_highlighter = nullptr;

    // Utility modules will be automatically deleted by Qt parent-child
    // relationship
    CVLog::PrintDebug("[cvViewSelectionManager] Destroyed");
}

//-----------------------------------------------------------------------------
// Override setVisualizer to handle tool cache updates
void cvViewSelectionManager::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (getVisualizer() == viewer) {
        return;
    }

    // Disable current selection if active
    if (m_isActive) {
        disableSelection();
    }

    cvGenericSelectionTool::setVisualizer(viewer);

    // Update visualizer for all cached tools
    for (auto tool : m_toolCache) {
        if (tool) {
            tool->setVisualizer(viewer);
        }
    }

    // Update visualizer for utility modules
    if (m_annotations && getPCLVis()) {
        m_annotations->setVisualizer(getPCLVis());
    }
    if (m_pipeline && getPCLVis()) {
        m_pipeline->setVisualizer(getPCLVis());

        // ParaView-style: Invalidate cached selection when visualizer changes
        // Reference: pqRenderViewSelectionReaction connects to dataUpdated
        // signal
        m_pipeline->invalidateCachedSelection();
    }

    // Update visualizer for shared highlighter
    if (m_highlighter && getPCLVis()) {
        m_highlighter->setVisualizer(getPCLVis());
    }
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::enableSelection(SelectionMode mode) {
    // Special handling for non-interactive modes
    if (mode == SelectionMode::CLEAR_SELECTION) {
        clearSelection();
        return;
    }
    if (mode == SelectionMode::GROW_SELECTION) {
        growSelection();
        return;
    }
    if (mode == SelectionMode::SHRINK_SELECTION) {
        shrinkSelection();
        return;
    }

    // Check if this is compatible with current mode
    if (m_isActive && m_currentTool) {
        if (!isCompatible(m_currentMode, mode)) {
            // Not compatible, disable current first
            disableSelection();
        }
    }

    // Get or create the tool
    cvRenderViewSelectionTool* tool = getOrCreateTool(mode);
    if (!tool) {
        CVLog::Warning(QString("[cvViewSelectionManager] Failed to create tool "
                               "for mode %1")
                               .arg(static_cast<int>(mode)));
        return;
    }

    // Set the selection modifier
    tool->setSelectionModifier(m_currentModifier);

    // Disable previous tool if it's different
    if (m_currentTool && m_currentTool != tool) {
        m_currentTool->disable();
    }

    // Enable the new tool
    m_currentTool = tool;
    m_currentMode = mode;
    m_isActive = true;

    tool->enable();

    emit modeChanged(mode);
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::disableSelection() {
    if (!m_isActive || !m_currentTool) {
        return;
    }

    m_currentTool->disable();
    m_isActive = false;
    m_currentMode = static_cast<SelectionMode>(-1);

    emit modeChanged(m_currentMode);
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::isSelectionActive() const {
    return m_isActive && m_currentTool && m_currentTool->isEnabled();
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setSelectionModifier(SelectionModifier modifier) {
    if (m_currentModifier == modifier) {
        return;
    }

    m_currentModifier = modifier;

    // Update current tool if active
    if (m_currentTool) {
        m_currentTool->setSelectionModifier(modifier);
    }

    emit modifierChanged(modifier);
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::clearSelection() {
    // Clear current selection
    setCurrentSelection(nullptr, 0);

    // Push to history
    if (m_history) {
        m_history->pushSelection(cvSelectionData(), "Clear Selection");
    }

    CVLog::PrintDebug("[cvViewSelectionManager] Clear selection");
    emit selectionChanged();
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::growSelection() {
    // ParaView-style: Use expand with +1 layer
    expandSelection(1, m_growRemoveSeed, m_growRemoveIntermediateLayers);
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::shrinkSelection() {
    // ParaView-style: Use expand with -1 layer
    expandSelection(-1, false, false);
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::expandSelection(int layers,
                                             bool removeSeed,
                                             bool removeIntermediateLayers) {
    // ParaView-compatible expand selection
    // Reference: pqRenderViewSelectionReaction::actionTriggered()
    // GROW_SELECTION case

    if (!m_algebra || !hasSelection()) {
        CVLog::Warning(
                "[cvViewSelectionManager] No selection or algebra module to "
                "expand");
        return;
    }

    // Get polyData
    vtkPolyData* polyData = getPolyData();
    if (!polyData) {
        CVLog::Warning(
                "[cvViewSelectionManager] No polyData available for expand "
                "operation");
        return;
    }

    // Get current selection as cvSelectionData
    cvSelectionData current = currentSelection();

    // Use algebra module to expand selection
    cvSelectionData result = m_algebra->expandSelection(
            polyData, current, layers, removeSeed, removeIntermediateLayers);

    if (!result.isEmpty() || layers < 0) {
        // For shrink (layers < 0), empty result is valid (shrunk to nothing)
        setCurrentSelection(result);

        // Push to history
        QString operationName =
                layers > 0 ? "Grow Selection" : "Shrink Selection";
        if (m_history) {
            m_history->pushSelection(result, operationName);
        }

        CVLog::PrintDebug(QString("[cvViewSelectionManager] %1: %2 -> %3")
                                  .arg(operationName)
                                  .arg(current.count())
                                  .arg(result.count()));
        emit selectionChanged();
    } else if (layers > 0) {
        CVLog::Warning(
                "[cvViewSelectionManager] Grow resulted in empty selection");
    }
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setGrowSelectionRemoveSeed(bool remove) {
    m_growRemoveSeed = remove;
    CVLog::Print(
            QString("[cvViewSelectionManager] GrowSelectionRemoveSeed = %1")
                    .arg(remove));
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setGrowSelectionRemoveIntermediateLayers(
        bool remove) {
    m_growRemoveIntermediateLayers = remove;
    CVLog::PrintDebug(QString("[cvViewSelectionManager] "
                              "GrowSelectionRemoveIntermediateLayers = %1")
                              .arg(remove));
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::isCompatible(SelectionMode mode1,
                                          SelectionMode mode2) const {
    // Same mode is always compatible
    if (mode1 == mode2) {
        return true;
    }

    // Cell selection modes are compatible with each other
    // Reference: pqRenderViewSelectionReaction.cxx, line 1037-1060
    QSet<SelectionMode> cellModes = {
            SelectionMode::SELECT_SURFACE_CELLS,
            SelectionMode::SELECT_SURFACE_CELLS_POLYGON,
            SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY,
            SelectionMode::SELECT_FRUSTUM_CELLS};

    if (cellModes.contains(mode1) && cellModes.contains(mode2)) {
        return true;
    }

    // Point selection modes are compatible with each other
    QSet<SelectionMode> pointModes = {
            SelectionMode::SELECT_SURFACE_POINTS,
            SelectionMode::SELECT_SURFACE_POINTS_POLYGON,
            SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY,
            SelectionMode::SELECT_FRUSTUM_POINTS};

    if (pointModes.contains(mode1) && pointModes.contains(mode2)) {
        return true;
    }

    // Block selection modes are compatible
    QSet<SelectionMode> blockModes = {SelectionMode::SELECT_BLOCKS,
                                      SelectionMode::SELECT_FRUSTUM_BLOCKS};

    if (blockModes.contains(mode1) && blockModes.contains(mode2)) {
        return true;
    }

    return false;
}

//-----------------------------------------------------------------------------
cvRenderViewSelectionTool* cvViewSelectionManager::getOrCreateTool(
        SelectionMode mode) {
    // Check cache first
    if (m_toolCache.contains(mode) && m_toolCache[mode]) {
        return m_toolCache[mode];
    }

    // Create new tool based on mode
    // ParaView-style: use specific tool classes for different modes
    cvRenderViewSelectionTool* tool = nullptr;

    switch (mode) {
        // Tooltip and Interactive modes: use cvTooltipSelectionTool
        // (unified class supporting both modes)
        // Reference: pqRenderViewSelectionReaction lines 92-109
        case SelectionMode::HOVER_CELLS_TOOLTIP:
        case SelectionMode::HOVER_POINTS_TOOLTIP:
        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
            tool = new cvTooltipSelectionTool(mode, this);
            // NOTE: ESC key handling is centralized in
            // MainWindow::handleEscapeKey()
            CVLog::PrintDebug(QString("[cvViewSelectionManager] Created "
                                      "cvTooltipSelectionTool for mode %1")
                                      .arg(static_cast<int>(mode)));
            break;

        // Surface selection modes: use cvSurfaceSelectionTool
        case SelectionMode::SELECT_SURFACE_CELLS:
        case SelectionMode::SELECT_SURFACE_POINTS:
            tool = new cvSurfaceSelectionTool(mode, this);
            CVLog::PrintDebug(QString("[cvViewSelectionManager] Created "
                                      "cvSurfaceSelectionTool for mode %1")
                                      .arg(static_cast<int>(mode)));
            break;

        // Polygon selection modes: use cvPolygonSelectionTool
        case SelectionMode::SELECT_SURFACE_CELLS_POLYGON:
        case SelectionMode::SELECT_SURFACE_POINTS_POLYGON:
        case SelectionMode::SELECT_CUSTOM_POLYGON:  // Custom polygon also uses
                                                    // polygon tool
            tool = new cvPolygonSelectionTool(mode, this);
            // For custom polygon, connect the custom signal
            if (mode == SelectionMode::SELECT_CUSTOM_POLYGON) {
                connect(qobject_cast<cvPolygonSelectionTool*>(tool),
                        &cvPolygonSelectionTool::polygonCompleted, this,
                        &cvViewSelectionManager::customPolygonSelected);
            }
            CVLog::PrintDebug(QString("[cvViewSelectionManager] Created "
                                      "cvPolygonSelectionTool for mode %1")
                                      .arg(static_cast<int>(mode)));
            break;

        // Frustum selection modes: use cvFrustumSelectionTool
        case SelectionMode::SELECT_FRUSTUM_CELLS:
        case SelectionMode::SELECT_FRUSTUM_POINTS:
        case SelectionMode::SELECT_FRUSTUM_BLOCKS:
            tool = new cvFrustumSelectionTool(mode, this);
            CVLog::PrintDebug(QString("[cvViewSelectionManager] Created "
                                      "cvFrustumSelectionTool for mode %1")
                                      .arg(static_cast<int>(mode)));
            break;

        // Block selection mode: use cvBlockSelectionTool
        case SelectionMode::SELECT_BLOCKS:
            tool = new cvBlockSelectionTool(mode, this);
            CVLog::PrintDebug(QString("[cvViewSelectionManager] Created "
                                      "cvBlockSelectionTool for mode %1")
                                      .arg(static_cast<int>(mode)));
            break;

        // Zoom to box mode: use cvZoomBoxSelectionTool
        case SelectionMode::ZOOM_TO_BOX: {
            cvZoomBoxSelectionTool* zoomTool = new cvZoomBoxSelectionTool(this);
            // Connect zoom completed signal to manager's signal
            connect(zoomTool, &cvZoomBoxSelectionTool::zoomToBoxCompleted, this,
                    [this](int xmin, int ymin, int xmax, int ymax) {
                        int region[4] = {xmin, ymin, xmax, ymax};
                        emit zoomToBoxRequested(region);
                        emit zoomToBoxRequested(xmin, ymin, xmax, ymax);
                    });
            tool = zoomTool;
            CVLog::Print(
                    "[cvViewSelectionManager] Created cvZoomBoxSelectionTool");
            break;
        }

        // Fallback for unknown modes: use base class
        default:
            tool = new cvRenderViewSelectionTool(mode, this);
            CVLog::Warning(QString("[cvViewSelectionManager] Using base class "
                                   "for unknown mode %1")
                                   .arg(static_cast<int>(mode)));
            break;
    }

    if (!tool) {
        CVLog::Error(QString("[cvViewSelectionManager] Failed to create tool "
                             "for mode %1")
                             .arg(static_cast<int>(mode)));
        return nullptr;
    }

    if (getVisualizer()) {
        tool->setVisualizer(getVisualizer());
    }

    // Set manager reference (for pipeline access)
    tool->setSelectionManager(this);

    // Connect common signals
    connect(tool, &cvRenderViewSelectionTool::selectionCompleted, this,
            &cvViewSelectionManager::onToolSelectionCompleted);
    connect(tool, &cvRenderViewSelectionTool::selectionChanged, this,
            &cvViewSelectionManager::onToolSelectionChanged);

    // Cache the tool
    m_toolCache[mode] = tool;

    // Clean up if too many tools cached
    if (m_toolCache.size() > MAX_CACHED_TOOLS) {
        cleanupInactiveTools();
    }

    return tool;
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::cleanupInactiveTools() {
    // Remove inactive tools that are not recently used
    QList<SelectionMode> toRemove;

    for (auto it = m_toolCache.begin(); it != m_toolCache.end(); ++it) {
        if (!it.value()) {
            toRemove.append(it.key());
        } else if (!it.value()->isEnabled() && it.value() != m_currentTool) {
            // Delete the tool
            delete it.value();
            toRemove.append(it.key());
        }
    }

    for (SelectionMode mode : toRemove) {
        m_toolCache.remove(mode);
    }
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::onToolSelectionCompleted() {
    emit selectionCompleted();
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::onToolSelectionChanged() {
    emit selectionChanged();
}

//-----------------------------------------------------------------------------
const cvSelectionData& cvViewSelectionManager::currentSelection() const {
    static cvSelectionData emptySelection;
    if (!m_currentSelection) {
        return emptySelection;
    }

    // Create a temporary cvSelectionData from current VTK array
    static thread_local cvSelectionData cachedSelection;
    cachedSelection = cvSelectionData(m_currentSelection,
                                      m_currentSelectionFieldAssociation);
    return cachedSelection;
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setCurrentSelection(
        const cvSelectionData& selectionData) {
    // Convert cvSelectionData to VTK array for internal storage
    setCurrentSelection(selectionData.vtkArray(),
                        static_cast<int>(selectionData.fieldAssociation()));
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setCurrentSelection(
        const vtkSmartPointer<vtkIdTypeArray>& selection,
        int fieldAssociation) {
    // CRITICAL FIX: Validate selection pointer before use
    if (selection && selection.GetPointer() == nullptr) {
        CVLog::Error(
                "[cvViewSelectionManager] Selection SmartPointer contains null "
                "pointer!");
        return;
    }

    // Additional validation: check if the object is valid
    vtkIdType newCount = 0;
    if (selection) {
        try {
            // Test if we can safely access the array
            newCount = selection->GetNumberOfTuples();
            if (newCount < 0) {
                CVLog::Error(
                        "[cvViewSelectionManager] Selection array has invalid "
                        "tuple count!");
                return;
            }
        } catch (...) {
            CVLog::Error(
                    "[cvViewSelectionManager] Selection array access failed - "
                    "invalid pointer!");
            return;
        }
    }

    // CRITICAL FIX: Check if selection actually changed to prevent infinite
    // recursion Compare with current selection
    bool hasChanged = false;
    if (!m_currentSelection && newCount == 0) {
        // Both empty - no change
        CVLog::PrintDebug(
                "[cvViewSelectionManager] Selection unchanged (both empty)");
        return;
    } else if (!m_currentSelection && newCount > 0) {
        // Was empty, now has content
        hasChanged = true;
    } else if (m_currentSelection && newCount == 0) {
        // Had content, now empty
        hasChanged = true;
    } else if (m_currentSelection) {
        // Both have content - check if they're identical
        vtkIdType oldCount = m_currentSelection->GetNumberOfTuples();
        if (oldCount != newCount ||
            m_currentSelectionFieldAssociation != fieldAssociation) {
            hasChanged = true;
        } else {
            // Same count and field association - check if IDs are identical
            bool idsMatch = true;
            for (vtkIdType i = 0; i < newCount; ++i) {
                if (m_currentSelection->GetValue(i) != selection->GetValue(i)) {
                    idsMatch = false;
                    break;
                }
            }
            hasChanged = !idsMatch;
        }
    }

    if (!hasChanged) {
        CVLog::PrintDebug(
                "[cvViewSelectionManager] Selection unchanged, skipping "
                "update");
        return;
    }

    // Clean up old selection
    // Smart pointer handles cleanup automatically
    m_currentSelection = nullptr;

    // Store new selection
    if (selection && newCount > 0) {
        m_currentSelection = vtkSmartPointer<vtkIdTypeArray>::New();

        try {
            m_currentSelection->DeepCopy(selection);
            m_currentSelectionFieldAssociation = fieldAssociation;

            CVLog::PrintDebug(
                    QString("[cvViewSelectionManager] Selection updated: %1 "
                            "%2 selected")
                            .arg(newCount)
                            .arg(fieldAssociation == 0 ? "cells" : "points"));
        } catch (const std::exception& e) {
            CVLog::Error(QString("[cvViewSelectionManager] DeepCopy failed: %1")
                                 .arg(e.what()));
            m_currentSelection = nullptr;
            m_currentSelectionFieldAssociation = 0;
            return;
        } catch (...) {
            CVLog::Error(
                    "[cvViewSelectionManager] DeepCopy failed with unknown "
                    "exception!");
            m_currentSelection = nullptr;
            m_currentSelectionFieldAssociation = 0;
            return;
        }
    } else {
        m_currentSelectionFieldAssociation = 0;
        CVLog::PrintDebug("[cvViewSelectionManager] Selection cleared");
    }

    // Create selection data object
    cvSelectionData selectionData(m_currentSelection,
                                  m_currentSelectionFieldAssociation);

    // Push to history (only if not empty and history exists)
    if (m_history && !selectionData.isEmpty()) {
        m_history->pushSelection(selectionData,
                                 QString("%1 %2 selected")
                                         .arg(selectionData.count())
                                         .arg(selectionData.fieldTypeString()));
    }

    // Emit both new and legacy signals
    emit selectionChanged(selectionData);
    emit selectionChanged();  // Legacy signal
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::hasSelection() const {
    return m_currentSelection && m_currentSelection->GetNumberOfTuples() > 0;
}

//-----------------------------------------------------------------------------
// Undo/redo operations (using cvSelectionHistory)
//-----------------------------------------------------------------------------

bool cvViewSelectionManager::undo() {
    if (!m_history || !m_history->canUndo()) {
        return false;
    }

    cvSelectionData restored = m_history->undo();
    // Note: setCurrentSelection is already connected to history signals
    // So it will be called automatically
    return true;
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::redo() {
    if (!m_history || !m_history->canRedo()) {
        return false;
    }

    cvSelectionData restored = m_history->redo();
    return true;
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::canUndo() const {
    return m_history && m_history->canUndo();
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::canRedo() const {
    return m_history && m_history->canRedo();
}

//-----------------------------------------------------------------------------
// Algebra operations (using cvSelectionAlgebra)
//-----------------------------------------------------------------------------

cvSelectionData cvViewSelectionManager::performAlgebraOperation(
        int op,
        const cvSelectionData& selectionA,
        const cvSelectionData& selectionB) {
    if (!m_algebra) {
        CVLog::Warning("[cvViewSelectionManager] Algebra module not available");
        return cvSelectionData();
    }

    vtkPolyData* polyData = getPolyData();

    cvSelectionData result = m_algebra->performOperation(
            static_cast<cvSelectionAlgebra::Operation>(op), selectionA,
            selectionB, polyData);

    if (!result.isEmpty()) {
        // Push to history
        if (m_history) {
            QString desc = QString("Algebra Operation %1").arg(op);
            m_history->pushSelection(result, desc);
        }
    }

    return result;
}

//-----------------------------------------------------------------------------
vtkPolyData* cvViewSelectionManager::getPolyData() const {
    // ParaView-style: Try to get from last selection result first (from
    // pipeline)
    if (m_pipeline && m_pipeline->getLastSelection()) {
        vtkDataSet* data = cvSelectionPipeline::getPrimaryDataFromSelection(
                m_pipeline->getLastSelection());
        if (data) {
            vtkPolyData* polyData = vtkPolyData::SafeDownCast(data);
            if (polyData) {
                CVLog::PrintDebug(
                        "[cvViewSelectionManager::getPolyData] Got polyData "
                        "from last selection");
                return polyData;
            }
        }
    }

    // Fallback: Get from visualizer's data actors
    // Note: Cast away const to call non-const methods
    cvViewSelectionManager* mutableThis =
            const_cast<cvViewSelectionManager*>(this);
    QList<vtkActor*> actors = mutableThis->getDataActors();
    if (!actors.isEmpty()) {
        vtkDataSet* data = mutableThis->getDataFromActor(actors.first());
        if (data) {
            vtkPolyData* polyData = vtkPolyData::SafeDownCast(data);
            if (polyData) {
                CVLog::PrintDebug(
                        "[cvViewSelectionManager::getPolyData] Got polyData "
                        "from first data actor");
                return polyData;
            }
        }
    }

    CVLog::Warning(
            "[cvViewSelectionManager::getPolyData] No polyData available");
    return nullptr;
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::notifyDataUpdated() {
    // ParaView-style: Invalidate cached selection when data changes
    // Reference: pqRenderViewSelectionReaction::clearSelectionCache()
    if (m_pipeline) {
        m_pipeline->invalidateCachedSelection();
        CVLog::Print(
                "[cvViewSelectionManager] Data updated - selection cache "
                "invalidated");
    }
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setPointPickingRadius(unsigned int radius) {
    if (m_pipeline) {
        m_pipeline->setPointPickingRadius(radius);
    }
}

//-----------------------------------------------------------------------------
unsigned int cvViewSelectionManager::getPointPickingRadius() const {
    if (m_pipeline) {
        return m_pipeline->getPointPickingRadius();
    }
    return 5;  // Default value
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::clearCurrentSelection() {
    // CRITICAL FIX: Clear current selection to prevent crashes from dangling
    // pointers This is called when objects might have been deleted

    CVLog::Print(
            "[cvViewSelectionManager] Clearing current selection (preventing "
            "stale references)");

    // Clear the stored selection data
    if (m_currentSelection) {
        m_currentSelection = nullptr;
    }

    m_currentSelectionFieldAssociation = -1;

    // Clear pipeline cache
    if (m_pipeline) {
        m_pipeline->invalidateCachedSelection();
    }

    // Emit signal to notify listeners
    cvSelectionData emptySelection;
    emit selectionChanged(emptySelection);
    emit selectionChanged();  // Legacy signal
}
