// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvViewSelectionManager.h"

#include "cvSelectionData.h"
// cvSelectionTypes.h merged into cvSelectionData.h  // For SelectionMode and
// SelectionModifier enums

// Selection utility modules
#include "cvSelectionAlgebra.h"  // Contains cvSelectionFilter
#include "cvSelectionAnnotation.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionPipeline.h"
#include "cvSelectionStorage.h"  // Contains cvSelectionHistory and cvSelectionBookmarks

// LOCAL
#include "PclUtils/PCLVis.h"

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>

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
      m_currentSelection(nullptr),
      m_currentSelectionFieldAssociation(0),
      m_history(nullptr),
      m_algebra(nullptr),
      m_pipeline(nullptr),
      m_filter(nullptr),
      m_bookmarks(nullptr),
      m_annotations(nullptr),
      m_highlighter(nullptr),
      m_sourceObject(nullptr) {
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

    // Clean up highlighter (not a QObject, so manual cleanup)
    delete m_highlighter;
    m_highlighter = nullptr;

    // Utility modules will be automatically deleted by Qt parent-child
    // relationship
    CVLog::PrintDebug("[cvViewSelectionManager] Destroyed");
}

//-----------------------------------------------------------------------------
// Override setVisualizer to handle utility module updates
void cvViewSelectionManager::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (getVisualizer() == viewer) {
        return;
    }

    // Disable current selection if active
    if (m_isActive) {
        disableSelection();
    }

    cvGenericSelectionTool::setVisualizer(viewer);

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
    // NOTE: In the new architecture, cvRenderViewSelectionReaction handles
    // all selection logic. This method only updates internal state.

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

    // Update internal state
    m_currentMode = mode;
    m_isActive = true;

    emit modeChanged(mode);
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::disableSelection() {
    // NOTE: In the new architecture, cvRenderViewSelectionReaction handles
    // all selection logic. This method only updates internal state.

    if (!m_isActive) {
        return;
    }

    m_isActive = false;
    m_currentMode = static_cast<SelectionMode>(-1);

    emit modeChanged(m_currentMode);
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::isSelectionActive() const { return m_isActive; }

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setSelectionModifier(SelectionModifier modifier) {
    if (m_currentModifier == modifier) {
        return;
    }

    m_currentModifier = modifier;
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

    if (!m_algebra) {
        CVLog::Warning("[cvViewSelectionManager] No algebra module available");
        return;
    }

    if (!hasSelection()) {
        CVLog::Warning(
                "[cvViewSelectionManager] No selection to expand (grow/shrink "
                "requires an existing selection)");
        return;
    }

    // Note: We can't reliably detect if m_sourceObject has been deleted in C++
    // without causing undefined behavior. We rely on the application to clear
    // m_sourceObject when objects are deleted (e.g., via
    // clearCurrentSelection()).

    // Get polyData with enhanced fallback
    // Reference: ParaView's selection expansion uses the source data
    vtkPolyData* polyData = getPolyData();

    if (!polyData) {
        CVLog::Warning(
                "[cvViewSelectionManager] No polyData available for expand "
                "operation. Make sure you have data loaded in the viewer.");
        return;
    }

    // Get current selection as cvSelectionData
    cvSelectionData current = currentSelection();
    CVLog::Print(QString("[cvViewSelectionManager] Expand selection: %1 "
                         "%2, layers=%3")
                         .arg(current.count())
                         .arg(current.fieldTypeString())
                         .arg(layers));

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

        // Update highlight (ParaView-style: visually reflect expanded
        // selection)
        if (m_highlighter) {
            if (!result.isEmpty()) {
                m_highlighter->highlightSelection(
                        polyData, result.vtkArray(),
                        static_cast<int>(result.fieldAssociation()),
                        cvSelectionHighlighter::SELECTED);
            } else {
                m_highlighter->clearHighlights();
            }
        }

        CVLog::Print(QString("[cvViewSelectionManager] %1: %2 -> %3 %4")
                             .arg(operationName)
                             .arg(current.count())
                             .arg(result.count())
                             .arg(result.fieldTypeString()));
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

    // Clear source object reference
    m_sourceObject = nullptr;

    // Clear pipeline cache
    if (m_pipeline) {
        m_pipeline->invalidateCachedSelection();
    }

    // Emit signal to notify listeners
    cvSelectionData emptySelection;
    emit selectionChanged(emptySelection);
    emit selectionChanged();  // Legacy signal
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setSourceObject(ccHObject* obj) {
    m_sourceObject = obj;
    if (obj) {
        CVLog::PrintDebug(QString("[cvViewSelectionManager] Source object set: "
                                  "'%1' (type=%2)")
                                  .arg(obj->getName())
                                  .arg(obj->getClassID()));
    } else {
        CVLog::PrintDebug("[cvViewSelectionManager] Source object cleared");
    }
}

//-----------------------------------------------------------------------------
ccHObject* cvViewSelectionManager::getSourceObject() const {
    return m_sourceObject;
}

//-----------------------------------------------------------------------------
ccPointCloud* cvViewSelectionManager::getSourcePointCloud() const {
    ccHObject* obj = getSourceObject();
    if (!obj) return nullptr;

    // Note: We can't reliably detect if obj has been deleted in C++ without
    // causing undefined behavior. The caller must ensure the object is still
    // valid. We rely on the application to clear m_sourceObject when objects
    // are deleted (e.g., via clearCurrentSelection()).

    // Check if it's a point cloud
    if (obj->isA(CV_TYPES::POINT_CLOUD)) {
        return static_cast<ccPointCloud*>(obj);
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
ccMesh* cvViewSelectionManager::getSourceMesh() const {
    ccHObject* obj = getSourceObject();
    if (!obj) return nullptr;

    // Note: We can't reliably detect if obj has been deleted in C++ without
    // causing undefined behavior. The caller must ensure the object is still
    // valid. We rely on the application to clear m_sourceObject when objects
    // are deleted (e.g., via clearCurrentSelection()).

    // Check if it's a mesh
    if (obj->isKindOf(CV_TYPES::MESH)) {
        return static_cast<ccMesh*>(obj);
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::isSourceObjectValid() const {
    // Note: In C++, there's no reliable way to detect if a raw pointer points
    // to a deleted object without causing undefined behavior. We rely on the
    // application to clear m_sourceObject when objects are deleted.
    // The actual safety check happens in getSourcePointCloud/getSourceMesh
    // where we verify the object is still accessible before using it.
    return m_sourceObject != nullptr;
}
