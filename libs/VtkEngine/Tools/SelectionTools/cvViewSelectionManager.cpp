// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvViewSelectionManager.h"

#include "cvSelectionData.h"

// Selection utility modules
#include "cvSelectionAlgebra.h"
#include "cvSelectionAnnotation.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionPipeline.h"

// LOCAL
#include "Visualization/VtkVis.h"

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include <vector>

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkActor.h>
#include <vtkHardwareSelector.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkPolyData.h>

// QT
#include <QSet>
#include <algorithm>

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
      m_algebra(nullptr),
      m_pipeline(nullptr),
      m_filter(nullptr),
      m_annotations(nullptr),
      m_highlighter(nullptr),
      m_sourceObject(nullptr) {
    // Initialize utility modules (ParaView-style architecture)
    // Note: m_history and m_bookmarks removed - UI not implemented
    m_algebra = new cvSelectionAlgebra(this);
    m_pipeline = new cvSelectionPipeline(this);
    m_filter = new cvSelectionFilter(this);
    m_annotations = new cvSelectionAnnotationManager(this);
    m_highlighter =
            new cvSelectionHighlighter();  // Shared highlighter for all tools

    CVLog::PrintVerbose(
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
    if (m_highlighter) {
        m_highlighter->prepareForShutdown();
        delete m_highlighter;
        m_highlighter = nullptr;
    }
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
    if (m_annotations) {
        m_annotations->setVisualizer(getVtkVis());
    }
    if (m_pipeline) {
        m_pipeline->setVisualizer(getVtkVis());

        // ParaView-style: Invalidate cached selection when visualizer changes
        // Reference: pqRenderViewSelectionReaction connects to dataUpdated
        // signal
        m_pipeline->invalidateCachedSelection();
    }

    // Update visualizer for shared highlighter
    if (m_highlighter) {
        m_highlighter->setVisualizer(getVtkVis());
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
    if (m_pipeline) m_pipeline->invalidateCachedSelection();
    setCurrentSelection(cvSelectionData());
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
    // Reference: vtkSMSelectionHelper::ExpandSelection()
    // ParaView ALWAYS expands/shrinks from the ORIGINAL selection, not the
    // current one This is the key difference that makes grow/shrink reversible

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

    // ParaView-style: Calculate the NEW total number of layers
    int newNumberOfLayers = m_numberOfLayers + layers;

    // Ensure it doesn't go negative
    if (newNumberOfLayers < 0) {
        newNumberOfLayers = 0;
    }

    // ParaView-style: ALWAYS expand from the original selection
    // Use the stored original selection as the base
    cvSelectionData baseSelection = m_originalSelection.isEmpty()
                                            ? currentSelection()
                                            : m_originalSelection;

    // Expand from the original selection with the NEW total number of layers
    cvSelectionData result;
    if (newNumberOfLayers == 0) {
        // No layers = return to original selection
        result = baseSelection;
        // CVLog::Print("[cvViewSelectionManager] Returning to original
        // selection");
    } else {
        const auto displayed = cvSelectionPipeline::resolveSelectionForView(
                baseSelection, getVtkVis());
        for (auto info : displayed.actorInfos()) {
            cvSelectionData local(info.selectedIds,
                                  baseSelection.fieldAssociation());
            const auto expanded = m_algebra->expandSelection(
                    info.polyData, local, newNumberOfLayers, removeSeed,
                    removeIntermediateLayers);
            if (expanded.isEmpty()) continue;
            info.selectedIds = expanded.ids();
            const auto sourceIds = cvSelectionPipeline::remapToSourceIds(
                    expanded.vtkArray(), info.polyData,
                    baseSelection.fieldAssociation() == cvSelectionData::CELLS);
            info.sourceIds.clear();
            for (vtkIdType i = 0; i < sourceIds->GetNumberOfTuples(); ++i)
                info.sourceIds.append(sourceIds->GetValue(i));
            if (result.isEmpty())
                result = cvSelectionData(info.sourceIds,
                                         baseSelection.fieldAssociation());
            result.addActorInfo(info);
        }
        // A global selection can contain sources absent from this view.
        // Growing visible sources must not silently delete those records.
        for (const auto& original : baseSelection.actorInfos()) {
            const auto records = displayed.actorInfos();
            const bool visible = std::any_of(
                    records.cbegin(), records.cend(),
                    [&](const cvActorSelectionInfo& info) {
                        return !original.sourceId.isEmpty()
                                       ? info.sourceId == original.sourceId
                                       : info.actor == original.actor &&
                                                 info.polyData ==
                                                         original.polyData &&
                                                 info.blockIndex ==
                                                         original.blockIndex;
                    });
            if (visible) continue;
            if (result.isEmpty())
                result = cvSelectionData(original.sourceIds,
                                         baseSelection.fieldAssociation());
            result.addActorInfo(original);
        }
    }

    if (!result.isEmpty() || newNumberOfLayers == 0) {
        // Don't reset layers since this is a grow/shrink operation
        setCurrentSelection(result, false);

        // Update number of layers
        m_numberOfLayers = newNumberOfLayers;

        CVLog::PrintVerbose(
                QString("[cvViewSelectionManager] Updated NumberOfLayers: %1")
                        .arg(m_numberOfLayers));

        // History removed - UI not implemented

        // Update highlight
        if (m_highlighter) {
            if (!result.isEmpty()) {
                m_highlighter->highlightSelection(
                        result, cvSelectionHighlighter::SELECTED);
            } else {
                m_highlighter->clearHighlights();
            }
        }

        QString operationName =
                layers > 0 ? "Grow Selection" : "Shrink Selection";
        CVLog::Print(
                QString("[cvViewSelectionManager] %1: %2 -> %3 %4 (layers=%5)")
                        .arg(operationName)
                        .arg(baseSelection.count())
                        .arg(result.count())
                        .arg(result.fieldTypeString())
                        .arg(m_numberOfLayers));
        emit selectionChanged();
    } else {
        CVLog::Warning(
                "[cvViewSelectionManager] Expand resulted in empty selection");
    }
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setGrowSelectionRemoveSeed(bool remove) {
    m_growRemoveSeed = remove;
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setGrowSelectionRemoveIntermediateLayers(
        bool remove) {
    m_growRemoveIntermediateLayers = remove;
    CVLog::PrintVerbose(QString("[cvViewSelectionManager] "
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
    return m_storedSelectionData;
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setCurrentSelection(
        const cvSelectionData& selectionData, bool resetLayers) {
    // IDs only have meaning together with their source/block and field.
    // Another object or view can carry the same numeric IDs and still needs
    // a complete notification.
    const auto previousInfos = m_storedSelectionData.actorInfos();
    const auto incomingInfos = selectionData.actorInfos();
    bool changed = m_storedSelectionData.fieldAssociation() !=
                           selectionData.fieldAssociation() ||
                   m_storedSelectionData.ids() != selectionData.ids() ||
                   previousInfos.size() != incomingInfos.size();
    for (int i = 0; !changed && i < previousInfos.size(); ++i) {
        const auto& previous = previousInfos[i];
        const auto& incoming = incomingInfos[i];
        changed = previous.sourceId != incoming.sourceId ||
                  previous.sourceIds != incoming.sourceIds ||
                  previous.actor != incoming.actor ||
                  previous.polyData != incoming.polyData ||
                  previous.blockIndex != incoming.blockIndex ||
                  previous.selectedIds != incoming.selectedIds;
    }

    // Commit metadata, IDs and expansion state before synchronous consumers
    // run. Never publish an array-only temporary and repair it after emit.
    m_storedSelectionData = selectionData;
    m_currentSelection = m_storedSelectionData.vtkArray();
    m_currentSelectionFieldAssociation =
            static_cast<int>(selectionData.fieldAssociation());
    if (resetLayers || selectionData.isEmpty()) {
        m_originalSelection = selectionData;
        m_numberOfLayers = 0;
    }
    if (selectionData.isEmpty()) {
        m_sourceObject = nullptr;
    }
    if (changed) {
        // A direct slot may change/clear the manager again. Publish a stable
        // value so later subscribers still receive this committed transition.
        const cvSelectionData published = m_storedSelectionData;
        emit selectionChanged(published);
        emit selectionChanged();
    }
}

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setCurrentSelection(
        const vtkSmartPointer<vtkIdTypeArray>& selection,
        int fieldAssociation,
        bool resetLayers) {
    setCurrentSelection(cvSelectionData(selection, fieldAssociation),
                        resetLayers);
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::hasSelection() const {
    return m_currentSelection && m_currentSelection->GetNumberOfTuples() > 0;
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

    // History removed - UI not implemented

    return result;
}

//-----------------------------------------------------------------------------
vtkPolyData* cvViewSelectionManager::getPolyData() const {
    // Hover/capture results may belong to another object than the committed
    // selection. Operations on selection IDs must prefer their own dataset.
    if (!m_storedSelectionData.isEmpty()) {
        return m_storedSelectionData.primaryPolyData();
    }
    // ParaView-style: Try to get from last selection result first (from
    // pipeline)
    if (m_pipeline && m_pipeline->getLastSelection()) {
        vtkDataSet* data = cvSelectionPipeline::getPrimaryDataFromSelection(
                m_pipeline->getLastSelection());
        if (data) {
            vtkPolyData* polyData = vtkPolyData::SafeDownCast(data);
            if (polyData) {
                CVLog::PrintVerbose(
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
                CVLog::PrintVerbose(
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
void cvViewSelectionManager::clearCurrentSelection() { clearSelection(); }

//-----------------------------------------------------------------------------
void cvViewSelectionManager::setSourceObject(ccHObject* obj) {
    m_sourceObject = obj;
}

//-----------------------------------------------------------------------------
ccHObject* cvViewSelectionManager::getSourceObject() const {
    auto* root = ecvDisplayTools::GetSceneDB();
    if (!root) return nullptr;
    const auto sourceId = m_storedSelectionData.actorInfo().sourceId;
    // Resolve through live DB membership, never dereference the retained raw
    // hint after a programmatic removal or undo-history cleanup.
    std::vector<ccHObject*> pending{root};
    while (!pending.empty()) {
        auto* object = pending.back();
        pending.pop_back();
        if (sourceId.isEmpty() ? object == m_sourceObject
                               : object->getViewId() == sourceId)
            return object;
        for (unsigned i = 0; i < object->getChildrenNumber(); ++i)
            pending.push_back(object->getChild(i));
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
ccPointCloud* cvViewSelectionManager::getSourcePointCloud() const {
    ccHObject* obj = getSourceObject();
    if (!obj) {
        CVLog::Print(
                "[cvViewSelectionManager::getSourcePointCloud] No source "
                "object");
        return nullptr;
    }

    // Check if it's a point cloud
    if (obj->isA(CV_TYPES::POINT_CLOUD)) {
        ccPointCloud* cloud = static_cast<ccPointCloud*>(obj);
        CVLog::PrintVerbose(
                QString("[cvViewSelectionManager::getSourcePointCloud] "
                        "Returning point cloud '%1' with %2 points")
                        .arg(cloud->getName())
                        .arg(cloud->size()));
        return cloud;
    }

    CVLog::PrintVerbose(
            QString("[cvViewSelectionManager::getSourcePointCloud] "
                    "Source object '%1' is not a point cloud (type=%2)")
                    .arg(obj->getName())
                    .arg(obj->getClassID()));
    return nullptr;
}

//-----------------------------------------------------------------------------
ccMesh* cvViewSelectionManager::getSourceMesh() const {
    ccHObject* obj = getSourceObject();
    if (!obj) {
        CVLog::Warning(
                "[cvViewSelectionManager::getSourceMesh] No source object");
        return nullptr;
    }

    // Check if it's a mesh
    if (obj->isKindOf(CV_TYPES::MESH)) {
        ccMesh* mesh = static_cast<ccMesh*>(obj);
        CVLog::PrintVerbose(QString("[cvViewSelectionManager::getSourceMesh] "
                                    "Returning mesh '%1' with %2 triangles")
                                    .arg(mesh->getName())
                                    .arg(mesh->size()));
        return mesh;
    }

    CVLog::Warning(QString("[cvViewSelectionManager::getSourceMesh] "
                           "Source object '%1' is not a mesh (type=%2)")
                           .arg(obj->getName())
                           .arg(obj->getClassID()));
    return nullptr;
}

//-----------------------------------------------------------------------------
bool cvViewSelectionManager::isSourceObjectValid() const {
    return getSourceObject() != nullptr;
}
