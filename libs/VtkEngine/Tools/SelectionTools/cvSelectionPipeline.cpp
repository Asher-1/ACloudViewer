// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file cvSelectionPipeline.cpp
 * @brief Implementation of selection pipeline and VTK selection handling.
 */

#include "cvSelectionPipeline.h"

#include "cvHardwareSelector.h"

// LOCAL
#include "Visualization/VtkVis.h"
#include "cvSelectionData.h"

// CV_CORE_LIB
#include <CVLog.h>

// STL
#include <algorithm>
#include <cmath>

// Qt
#include <QApplication>
#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QSet>
#include <QSettings>
#include <QStyle>
#include <QVBoxLayout>

// VTK
#include <vtkActor.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkExtractSelectedFrustum.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkIntArray.h>
#include <vtkMapper.h>
#include <vtkPlanes.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>

// Qt
#include <QMap>

//-----------------------------------------------------------------------------
namespace {

// Expand thin/single-pixel rubber bands so HW selection captures sub-pixel
// points.
void normalizeSelectionRegion(int region[4],
                              bool forPoints,
                              unsigned int pointPickingRadius) {
    int x0 = std::min(region[0], region[2]);
    int x1 = std::max(region[0], region[2]);
    int y0 = std::min(region[1], region[3]);
    int y1 = std::max(region[1], region[3]);

    constexpr int kMinSpan = 2;
    if (x1 - x0 < kMinSpan) {
        const int cx = (x0 + x1) / 2;
        x0 = cx - kMinSpan / 2;
        x1 = cx + kMinSpan / 2;
    }
    if (y1 - y0 < kMinSpan) {
        const int cy = (y0 + y1) / 2;
        y0 = cy - kMinSpan / 2;
        y1 = cy + kMinSpan / 2;
    }

    if (forPoints && pointPickingRadius > 0) {
        int pad = static_cast<int>(pointPickingRadius / 8);
        pad = std::max(1, std::min(pad, 8));
        x0 -= pad;
        y0 -= pad;
        x1 += pad;
        y1 += pad;
    }

    region[0] = x0;
    region[1] = y0;
    region[2] = x1;
    region[3] = y1;
}

}  // namespace

cvSelectionPipeline::cvSelectionPipeline(QObject* parent)
    : QObject(parent),
      m_viewer(nullptr),
      m_renderer(nullptr),
      m_cachingEnabled(false),
      m_cacheHits(0),
      m_cacheMisses(0) {
    // CVLog::PrintVerbose("[cvSelectionPipeline] Created");
}

//-----------------------------------------------------------------------------
cvSelectionPipeline::~cvSelectionPipeline() {
    clearCache();
    CVLog::PrintVerbose(
            QString("[cvSelectionPipeline] Destroyed - Cache stats: %1 "
                    "hits, %2 misses")
                    .arg(m_cacheHits)
                    .arg(m_cacheMisses));
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::setVisualizer(Visualization::VtkVis* viewer) {
    if (m_viewer == viewer) {
        return;
    }

    m_viewer = viewer;
    m_renderer = nullptr;

    if (m_viewer) {
        m_renderer = m_viewer->getCurrentRenderer();
    }

    // Clear cache when visualizer changes
    clearCache();

    CVLog::PrintVerbose(QString("[cvSelectionPipeline] Visualizer set: %1")
                                .arg((quintptr)viewer, 0, 16));
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelection> cvSelectionPipeline::executeRectangleSelection(
        int region[4], SelectionType type) {
    if (!m_viewer || !m_renderer) {
        CVLog::Warning("[cvSelectionPipeline] Invalid viewer or renderer");
        emit errorOccurred("Invalid viewer or renderer");
        return {};
    }

    // Check cache first
    if (m_cachingEnabled) {
        QString cacheKey = generateCacheKey(region, type);
        vtkSmartPointer<vtkSelection> cached = getCachedSelection(cacheKey);
        if (cached) {
            m_cacheHits++;
            CVLog::Print(
                    QString("[cvSelectionPipeline] Cache hit! Total: %1/%2")
                            .arg(m_cacheHits)
                            .arg(m_cacheHits + m_cacheMisses));

            // Return a copy
            vtkSmartPointer<vtkSelection> copy =
                    vtkSmartPointer<vtkSelection>::New();
            copy->DeepCopy(cached);
            return copy;
        }
        m_cacheMisses++;
    }

    // Perform hardware selection
    FieldAssociation fieldAssoc = getFieldAssociation(type);
    vtkSmartPointer<vtkSelection> selection =
            performHardwareSelection(region, fieldAssoc);

    if (!selection) {
        CVLog::Warning("[cvSelectionPipeline] Hardware selection failed");
        emit errorOccurred("Hardware selection failed");
        return {};
    }

    // Cache the result
    if (m_cachingEnabled) {
        QString cacheKey = generateCacheKey(region, type);
        cacheSelection(cacheKey, selection);
    }

    emit selectionCompleted(selection);
    return selection;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelection> cvSelectionPipeline::executePolygonSelection(
        vtkIntArray* polygon, SelectionType type) {
    if (!m_viewer || !m_renderer || !polygon) {
        CVLog::Warning(
                "[cvSelectionPipeline] Invalid viewer, renderer, or polygon");
        emit errorOccurred("Invalid parameters");
        return {};
    }

    // The polygon array has 2 components (x, y) per tuple
    // Each tuple represents one vertex, so GetNumberOfTuples() == number of
    // vertices NOTE: Don't divide by 2! The array is structured as 2-component
    // tuples.
    vtkIdType numPoints = polygon->GetNumberOfTuples();
    // Validate polygon
    if (numPoints < 3) {
        CVLog::Warning("[cvSelectionPipeline] Polygon needs at least 3 points");
        emit errorOccurred("Invalid polygon: needs at least 3 points");
        return {};
    }

    // ParaView approach: Use vtkHardwareSelector::GeneratePolygonSelection
    // for pixel-precise polygon selection
    // Reference: vtkPVRenderView::SelectPolygon

    // Step 1: Find bounding box of polygon
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

    // Validate bounding box
    if (minX >= maxX || minY >= maxY) {
        CVLog::Warning("[cvSelectionPipeline] Invalid polygon bounding box");
        emit errorOccurred("Invalid polygon geometry");
        return {};
    }

    FieldAssociation fieldAssoc = getFieldAssociation(type);
    int bbox[4] = {minX, minY, maxX, maxY};
    normalizeSelectionRegion(bbox, fieldAssoc == FIELD_ASSOCIATION_POINTS,
                             m_pointPickingRadius);
    minX = bbox[0];
    minY = bbox[1];
    maxX = bbox[2];
    maxY = bbox[3];

    // Step 2: Get render window
    vtkRenderWindow* renderWindow = m_viewer->getRenderWindow();
    if (!renderWindow) {
        CVLog::Warning("[cvSelectionPipeline] Invalid render window");
        emit errorOccurred("Invalid render window");
        return {};
    }

    // Step 3: Create or reuse cvHardwareSelector (ParaView-style)
    if (!m_hardwareSelector) {
        m_hardwareSelector = vtkSmartPointer<cvHardwareSelector>::New();
        m_hardwareSelector->SetPointPickingRadius(m_pointPickingRadius);
    }

    m_hardwareSelector->SetRenderer(m_renderer);
    m_hardwareSelector->SetArea(minX, minY, maxX, maxY);

    // Set field association
    if (fieldAssoc == FIELD_ASSOCIATION_CELLS) {
        m_hardwareSelector->SetFieldAssociation(
                vtkDataObject::FIELD_ASSOCIATION_CELLS);
    } else {
        m_hardwareSelector->SetFieldAssociation(
                vtkDataObject::FIELD_ASSOCIATION_POINTS);
    }

    // Step 4: Capture pixel buffers (ParaView-style)
    // This renders the scene with special color encoding for selection
    bool captureSuccess = m_hardwareSelector->CaptureBuffers();
    if (!captureSuccess) {
        CVLog::Warning(
                "[cvSelectionPipeline] Failed to capture buffers for polygon");
        // Try once more after forcing a render
        renderWindow->Render();
        captureSuccess = m_hardwareSelector->CaptureBuffers();
        if (!captureSuccess) {
            emit errorOccurred("Buffer capture failed");
            return {};
        }
    }

    // Step 5: Generate polygon selection with pixel-level testing
    // Reference: vtkHardwareSelector::GeneratePolygonSelection
    // This tests each pixel in the bounding box to see if it's inside the
    // polygon
    std::vector<int> polygonArray(numPoints * 2);
    for (vtkIdType i = 0; i < numPoints * 2; ++i) {
        polygonArray[i] = polygon->GetValue(i);
    }

    vtkSelection* selection = m_hardwareSelector->GeneratePolygonSelection(
            polygonArray.data(), static_cast<vtkIdType>(numPoints * 2));

    if (!selection) {
        CVLog::Print(
                "[cvSelectionPipeline] Polygon selection returned no results");
        // This is not an error - just no items selected
        vtkSmartPointer<vtkSelection> emptySelection =
                vtkSmartPointer<vtkSelection>::New();
        m_lastSelection = emptySelection;
        emit selectionCompleted(emptySelection);
        return emptySelection;
    }

    // Wrap in smart pointer for automatic cleanup
    vtkSmartPointer<vtkSelection> smartSelection;
    smartSelection.TakeReference(selection);

    // Cache last selection
    m_lastSelection = smartSelection;

    // Count total selected items
    int totalItems = 0;
    for (unsigned int i = 0; i < smartSelection->GetNumberOfNodes(); ++i) {
        vtkSelectionNode* node = smartSelection->GetNode(i);
        if (node && node->GetSelectionList()) {
            totalItems += node->GetSelectionList()->GetNumberOfTuples();
        }
    }

    emit selectionCompleted(smartSelection);
    return smartSelection;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkIdTypeArray> cvSelectionPipeline::extractSelectionIds(
        vtkSelection* selection, FieldAssociation fieldAssociation) {
    if (!selection) {
        return {};
    }

    unsigned int numNodes = selection->GetNumberOfNodes();
    if (numNodes == 0) {
        CVLog::Print("[cvSelectionPipeline] Selection has no nodes");
        return {};
    }

    // ParaView merges IDs from all matching selection nodes via
    // vtkSelection::Union. We collect IDs from all nodes whose field type
    // matches the requested association.
    vtkSmartPointer<vtkIdTypeArray> merged =
            vtkSmartPointer<vtkIdTypeArray>::New();

    int targetFieldType = (fieldAssociation == SURFACE_CELLS ||
                           fieldAssociation == FRUSTUM_CELLS ||
                           fieldAssociation == POLYGON_CELLS)
                                  ? vtkSelectionNode::CELL
                                  : vtkSelectionNode::POINT;

    for (unsigned int i = 0; i < numNodes; ++i) {
        vtkSelectionNode* node = selection->GetNode(i);
        if (!node) continue;

        int nodeFieldType = node->GetFieldType();
        if (nodeFieldType != targetFieldType &&
            nodeFieldType != vtkSelectionNode::POINT &&
            nodeFieldType != vtkSelectionNode::CELL) {
            nodeFieldType = targetFieldType;
        }
        if (numNodes > 1 && nodeFieldType != targetFieldType) {
            continue;
        }

        vtkIdTypeArray* selectionList =
                vtkIdTypeArray::SafeDownCast(node->GetSelectionList());
        if (!selectionList) continue;

        for (vtkIdType j = 0; j < selectionList->GetNumberOfTuples(); ++j) {
            merged->InsertNextValue(selectionList->GetValue(j));
        }
    }

    if (merged->GetNumberOfTuples() == 0) {
        CVLog::Print(
                QString("[cvSelectionPipeline] %1 selection nodes but 0 IDs "
                        "for field=%2 (check field association)")
                        .arg(numNodes)
                        .arg(targetFieldType == vtkSelectionNode::CELL
                                     ? "CELL"
                                     : "POINT"));
        return {};
    }

    return merged;
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::setEnableCaching(bool enable) {
    if (m_cachingEnabled != enable) {
        m_cachingEnabled = enable;
        CVLog::Print(QString("[cvSelectionPipeline] Caching %1")
                             .arg(enable ? "enabled" : "disabled"));

        if (!enable) {
            clearCache();
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::clearCache() {
    // Smart pointers handle cleanup automatically
    m_selectionCache.clear();

    // CVLog::PrintVerbose("[cvSelectionPipeline] Cache cleared");
}

//-----------------------------------------------------------------------------
int cvSelectionPipeline::getCacheSize() const {
    return m_selectionCache.size();
}

//-----------------------------------------------------------------------------
int cvSelectionPipeline::getCacheHits() const { return m_cacheHits; }

//-----------------------------------------------------------------------------
int cvSelectionPipeline::getCacheMisses() const { return m_cacheMisses; }

//-----------------------------------------------------------------------------
bool cvSelectionPipeline::hasCachedBuffers() const {
    // Check if hardware selector has cached buffers
    // VTK doesn't expose this directly, so we track it ourselves
    return m_inSelectionMode && m_hardwareSelector != nullptr;
}

//-----------------------------------------------------------------------------
bool cvSelectionPipeline::captureBuffersForFastPreSelection() {
    if (!m_viewer || !m_renderer) {
        CVLog::Warning(
                "[cvSelectionPipeline] Cannot capture buffers - no "
                "viewer/renderer");
        return false;
    }

    vtkRenderWindow* renderWindow = m_viewer->getRenderWindow();
    if (!renderWindow) {
        CVLog::Warning(
                "[cvSelectionPipeline] Cannot capture buffers - no render "
                "window");
        return false;
    }

    // Create cvHardwareSelector if needed (ParaView-style)
    if (!m_hardwareSelector) {
        m_hardwareSelector = vtkSmartPointer<cvHardwareSelector>::New();
        m_hardwareSelector->SetPointPickingRadius(m_pointPickingRadius);
    }

    m_hardwareSelector->SetRenderer(m_renderer);

    // Set area to full viewport
    int* size = m_renderer->GetSize();
    int* origin = m_renderer->GetOrigin();
    m_hardwareSelector->SetArea(origin[0], origin[1], origin[0] + size[0] - 1,
                                origin[1] + size[1] - 1);

    // Capture the buffers
    bool success = m_hardwareSelector->CaptureBuffers();

    if (success) {
        m_inSelectionMode = true;
        CVLog::Print(
                "[cvSelectionPipeline] Captured buffers for fast "
                "pre-selection");
    } else {
        CVLog::Warning("[cvSelectionPipeline] Failed to capture buffers");
    }

    return success;
}

//-----------------------------------------------------------------------------
cvSelectionPipeline::PixelSelectionInfo
cvSelectionPipeline::getPixelSelectionInfo(int x, int y, bool selectCells) {
    PixelSelectionInfo result;

    // CRITICAL: Check if invalidation is in progress
    // This prevents crashes when mouse events arrive during cache invalidation
    if (m_invalidating) {
        return result;  // Return empty result - safe to skip this hover update
    }

    if (!m_viewer || !m_renderer) {
        CVLog::Warning(
                "[cvSelectionPipeline::getPixelSelectionInfo] Invalid viewer "
                "or renderer");
        return result;
    }

    // PARAVIEW STYLE: Always do fresh hardware selection, NO CACHING
    // Caching causes stale actor problems and incorrect IDs
    // Reference: ParaView never caches for hover/tooltip - always fresh render

    // Do a single-pixel hardware selection
    int region[4] = {x, y, x, y};
    vtkSmartPointer<vtkSelection> selection = performHardwareSelection(
            region,
            selectCells ? FIELD_ASSOCIATION_CELLS : FIELD_ASSOCIATION_POINTS);

    if (selection && selection->GetNumberOfNodes() > 0) {
        vtkSelectionNode* node = selection->GetNode(0);
        if (node && node->GetSelectionList() &&
            node->GetSelectionList()->GetNumberOfTuples() > 0) {
            vtkIdTypeArray* ids =
                    vtkIdTypeArray::SafeDownCast(node->GetSelectionList());
            if (ids && ids->GetNumberOfTuples() > 0) {
                result.valid = true;
                result.attributeID = ids->GetValue(0);

                // Get prop from selection node
                vtkInformation* properties = node->GetProperties();
                if (properties && properties->Has(vtkSelectionNode::PROP())) {
                    result.prop = vtkProp::SafeDownCast(
                            properties->Get(vtkSelectionNode::PROP()));

                    // Get polyData from prop
                    vtkActor* actor = vtkActor::SafeDownCast(result.prop);
                    if (actor && actor->GetMapper()) {
                        vtkDataSet* data = actor->GetMapper()->GetInput();
                        result.polyData = vtkPolyData::SafeDownCast(data);
                    }
                }
            }
        }
    }

    return result;
}

//-----------------------------------------------------------------------------
vtkIdType cvSelectionPipeline::fastPreSelectAt(int x, int y, bool selectCells) {
    // Use the new comprehensive method and return only the ID for backward
    // compatibility
    PixelSelectionInfo info = getPixelSelectionInfo(x, y, selectCells);
    return info.valid ? info.attributeID : -1;
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::enterSelectionMode() {
    if (m_inSelectionMode) {
        return;
    }

    m_inSelectionMode = true;

    // ParaView-style: When entering selection mode, the render view
    // switches to INTERACTION_MODE_SELECTION which tells it to cache
    // selection render buffers for faster repeated selections.
    // Reference: vtkPVRenderView::SetInteractionMode
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::exitSelectionMode() {
    if (!m_inSelectionMode) {
        return;
    }

    m_inSelectionMode = false;

    // ParaView-style: Reset hardware selector to release cached buffers
    // We create a fresh selector on next selection rather than trying to
    // call protected ReleasePixBuffers()
    // Reference: vtkPVRenderView doesn't explicitly release buffers,
    // it just lets them be overwritten on next selection
    // Note: vtkSmartPointer uses = nullptr instead of Reset() (unlike
    // std::shared_ptr)
    m_hardwareSelector = nullptr;
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::invalidateCachedSelection() {
    // Set invalidating flag to prevent concurrent access
    // This prevents crashes when mouse events arrive during invalidation
    m_invalidating = true;

    // Clear the selection cache
    clearCache();

    // ParaView-style: Reset hardware selector to invalidate cached buffers
    // Reference: vtkPVRenderView::InvalidateCachedSelection() clears internal
    // state Note: vtkSmartPointer uses = nullptr instead of Reset() (unlike
    // std::shared_ptr)
    m_hardwareSelector = nullptr;

    // Clear last selection
    m_lastSelection = nullptr;

    // Clear invalidating flag
    m_invalidating = false;

    // CVLog::PrintVerbose("[cvSelectionPipeline] Invalidated cached
    // selection");
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::setPointPickingRadius(unsigned int radius) {
    m_pointPickingRadius = radius;

    // Update cvHardwareSelector if it exists
    if (m_hardwareSelector) {
        m_hardwareSelector->SetPointPickingRadius(radius);
    }

    CVLog::Print(QString("[cvSelectionPipeline] Point picking radius set to %1 "
                         "pixels")
                         .arg(radius));
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelection> cvSelectionPipeline::performHardwareSelection(
        int region[4], FieldAssociation fieldAssociation) {
    // Reference: ParaView's vtkPVRenderView::Select() and
    // vtkPVHardwareSelector::Select() This method now uses cvHardwareSelector
    // which is adapted from ParaView's vtkPVHardwareSelector for consistent
    // behavior.

    // CRITICAL: Check if invalidation is in progress
    // This prevents crashes when selection is attempted during cache
    // invalidation
    if (m_invalidating) {
        return {};  // Safe to return - selection will be retried later
    }

    if (!m_viewer || !m_renderer) {
        CVLog::Print(
                "[cvSelectionPipeline] performHardwareSelection: no "
                "viewer/renderer");
        return {};
    }

    vtkRenderWindow* renderWindow = m_viewer->getRenderWindow();
    if (!renderWindow) {
        CVLog::Warning("[cvSelectionPipeline] Invalid render window");
        return {};
    }

    // IMPORTANT: region coordinates come from VTK's interactor events
    // (GetEventPosition), which already use VTK coordinate system:
    // origin at bottom-left, Y increases upward.
    // NO coordinate conversion needed here!

    int vtk_region[4] = {region[0], region[1], region[2], region[3]};
    normalizeSelectionRegion(vtk_region,
                             fieldAssociation == FIELD_ASSOCIATION_POINTS,
                             m_pointPickingRadius);

    CVLog::PrintVerbose(QString("[cvSelectionPipeline] Selection region: "
                                "Input[%1,%2,%3,%4] -> Normalized[%5,%6,%7,%8]")
                                .arg(region[0])
                                .arg(region[1])
                                .arg(region[2])
                                .arg(region[3])
                                .arg(vtk_region[0])
                                .arg(vtk_region[1])
                                .arg(vtk_region[2])
                                .arg(vtk_region[3]));

    // ParaView-style: Disable buffer swapping during selection to avoid
    // clobbering the user's view (BUG #16042 in ParaView)
    // Reference: vtkPVRenderView::PrepareSelect() lines 966-967
    int previousSwapBuffers = renderWindow->GetSwapBuffers();
    renderWindow->SwapBuffersOff();

    // Create or reuse cvHardwareSelector (ParaView-style)
    // Reference: vtkPVRenderView uses vtkPVHardwareSelector
    if (!m_hardwareSelector) {
        m_hardwareSelector = vtkSmartPointer<cvHardwareSelector>::New();
        m_hardwareSelector->SetPointPickingRadius(m_pointPickingRadius);
        CVLog::PrintVerbose(
                QString("[cvSelectionPipeline] Created cvHardwareSelector "
                        "(ParaView-style) with PointPickingRadius=%1")
                        .arg(m_pointPickingRadius));
    }

    // Configure selector
    m_hardwareSelector->SetRenderer(m_renderer);

    // ParaView vtkPVRenderView::PrepareSelect: render full scene before HW pick
    m_hardwareSelector->Modified();
    m_viewer->UpdateScreen();

    // Set field association
    if (fieldAssociation == FIELD_ASSOCIATION_CELLS) {
        m_hardwareSelector->SetFieldAssociation(
                vtkDataObject::FIELD_ASSOCIATION_CELLS);
    } else {
        m_hardwareSelector->SetFieldAssociation(
                vtkDataObject::FIELD_ASSOCIATION_POINTS);
    }

    // Log current state for debugging
    CVLog::PrintVerbose(
            QString("[cvSelectionPipeline] cvHardwareSelector config: "
                    "FieldAssociation=%1, PointPickingRadius=%2")
                    .arg(fieldAssociation == FIELD_ASSOCIATION_CELLS ? "CELLS"
                                                                     : "POINTS")
                    .arg(m_pointPickingRadius));

    // Perform selection using cvHardwareSelector::Select()
    // This method handles:
    // - Buffer caching (NeedToRenderForSelection check)
    // - Point picking radius (automatic radius search if no direct hit)
    // ParaView-style: First try exact selection, only use radius if nothing
    // found Reference: vtkPVHardwareSelector::Select() lines 105-131
    vtkSmartPointer<vtkSelection> selection;
    selection.TakeReference(m_hardwareSelector->Select(vtk_region));

    // ParaView-style: Restore swap buffers setting
    renderWindow->SetSwapBuffers(previousSwapBuffers);

    if (!selection) {
        CVLog::Print("[cvSelectionPipeline] cvHardwareSelector returned null");
        return {};
    }

    if (selection->GetNumberOfNodes() == 0) {
        int pickableActors = 0;
        if (m_renderer) {
            vtkPropCollection* props = m_renderer->GetViewProps();
            if (props) {
                props->InitTraversal();
                while (vtkProp* prop = props->GetNextProp()) {
                    auto* actor = vtkActor::SafeDownCast(prop);
                    if (actor && actor->GetVisibility() && actor->GetPickable())
                        ++pickableActors;
                }
            }
        }
        int* renSize = m_renderer ? m_renderer->GetSize() : nullptr;
        int* renOrigin = m_renderer ? m_renderer->GetOrigin() : nullptr;
        CVLog::Print(
                QString("[cvSelectionPipeline] HW selection empty: "
                        "region=[%1,%2,"
                        "%3,%4] renSize=%5x%6 origin=(%7,%8) pickableActors=%9 "
                        "field=%10")
                        .arg(vtk_region[0])
                        .arg(vtk_region[1])
                        .arg(vtk_region[2])
                        .arg(vtk_region[3])
                        .arg(renSize ? renSize[0] : -1)
                        .arg(renSize ? renSize[1] : -1)
                        .arg(renOrigin ? renOrigin[0] : -1)
                        .arg(renOrigin ? renOrigin[1] : -1)
                        .arg(pickableActors)
                        .arg(fieldAssociation == FIELD_ASSOCIATION_CELLS
                                     ? "CELLS"
                                     : "POINTS"));
    }

    // Log result (debug level - this is called frequently during hover)
    if (selection->GetNumberOfNodes() > 0) {
        vtkSelectionNode* node = selection->GetNode(0);
        if (node && node->GetSelectionList()) {
            vtkIdType numIds = node->GetSelectionList()->GetNumberOfTuples();
            CVLog::Print(
                    QString("[cvSelectionPipeline] HW selection OK: %1 IDs "
                            "field=%2")
                            .arg(numIds)
                            .arg(fieldAssociation == FIELD_ASSOCIATION_CELLS
                                         ? "CELLS"
                                         : "POINTS"));
        }
    }

    // Cache last selection for getPolyData() operations
    m_lastSelection = selection;

    return selection;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelection> cvSelectionPipeline::getCachedSelection(
        const QString& key) {
    auto it = m_selectionCache.find(key);
    if (it != m_selectionCache.end()) {
        return it.value();
    }
    return {};
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::cacheSelection(const QString& key,
                                         vtkSelection* selection) {
    if (!selection) {
        return;
    }

    // Check cache size limit
    if (m_selectionCache.size() >= MAX_CACHE_SIZE) {
        // Remove oldest entry (first in hash)
        // Smart pointer handles cleanup automatically
        auto it = m_selectionCache.begin();
        m_selectionCache.erase(it);
        CVLog::Print("[cvSelectionPipeline] Cache full, removed oldest entry");
    }

    // Store a copy (use smart pointer from the start)
    vtkSmartPointer<vtkSelection> copy = vtkSmartPointer<vtkSelection>::New();
    copy->DeepCopy(selection);
    m_selectionCache.insert(key, copy);
}

//-----------------------------------------------------------------------------
QString cvSelectionPipeline::generateCacheKey(int region[4],
                                              SelectionType type) const {
    // Generate a unique key based on selection parameters
    return QString("%1_%2_%3_%4_%5")
            .arg(region[0])
            .arg(region[1])
            .arg(region[2])
            .arg(region[3])
            .arg(type);
}

//-----------------------------------------------------------------------------
cvSelectionPipeline::FieldAssociation cvSelectionPipeline::getFieldAssociation(
        SelectionType type) const {
    switch (type) {
        case SURFACE_CELLS:
        case FRUSTUM_CELLS:
        case POLYGON_CELLS:
            return FIELD_ASSOCIATION_CELLS;

        case SURFACE_POINTS:
        case FRUSTUM_POINTS:
        case POLYGON_POINTS:
            return FIELD_ASSOCIATION_POINTS;

        default:
            return FIELD_ASSOCIATION_CELLS;
    }
}

//-----------------------------------------------------------------------------
// ParaView-style selection data extraction
//-----------------------------------------------------------------------------

QMap<vtkProp*, vtkDataSet*> cvSelectionPipeline::extractDataFromSelection(
        vtkSelection* selection) {
    QMap<vtkProp*, vtkDataSet*> result;

    if (!selection) {
        CVLog::Warning(
                "[cvSelectionPipeline::extractDataFromSelection] selection is "
                "nullptr");
        return result;
    }

    for (unsigned int i = 0; i < selection->GetNumberOfNodes(); ++i) {
        vtkSelectionNode* node = selection->GetNode(i);
        if (!node) continue;

        // Get the prop (actor) from selection node properties
        vtkInformation* properties = node->GetProperties();
        if (!properties || !properties->Has(vtkSelectionNode::PROP())) {
            continue;
        }

        vtkProp* prop = vtkProp::SafeDownCast(
                properties->Get(vtkSelectionNode::PROP()));
        if (!prop) continue;

        // Get data from actor's mapper
        vtkActor* actor = vtkActor::SafeDownCast(prop);
        if (actor) {
            vtkMapper* mapper = actor->GetMapper();
            if (mapper) {
                vtkDataSet* data = mapper->GetInput();
                if (data) {
                    result[prop] = data;
                    CVLog::PrintVerbose(
                            QString("[cvSelectionPipeline] Extracted data from "
                                    "actor: %1 points, %2 cells, type=%3")
                                    .arg(data->GetNumberOfPoints())
                                    .arg(data->GetNumberOfCells())
                                    .arg(data->GetClassName()));
                }
            }
        } else {
            CVLog::Warning(
                    QString("[cvSelectionPipeline] prop is not vtkActor: %1")
                            .arg(prop ? prop->GetClassName() : "null"));
        }
    }

    return result;
}

//-----------------------------------------------------------------------------
vtkDataSet* cvSelectionPipeline::getPrimaryDataFromSelection(
        vtkSelection* selection) {
    QMap<vtkProp*, vtkDataSet*> dataMap = extractDataFromSelection(selection);

    if (dataMap.isEmpty()) {
        CVLog::PrintVerbose(
                "[cvSelectionPipeline::getPrimaryDataFromSelection] No data "
                "found in selection");
        return nullptr;
    }

    // Return the data with most elements (points + cells)
    vtkDataSet* primaryData = nullptr;
    vtkIdType maxCount = 0;

    for (auto it = dataMap.begin(); it != dataMap.end(); ++it) {
        vtkDataSet* data = it.value();
        vtkIdType count = data->GetNumberOfPoints() + data->GetNumberOfCells();
        if (count > maxCount) {
            maxCount = count;
            primaryData = data;
        }
    }

    if (primaryData) {
        CVLog::PrintVerbose(
                QString("[cvSelectionPipeline::getPrimaryDataFromSelection] "
                        "Primary data: %1 points, %2 cells")
                        .arg(primaryData->GetNumberOfPoints())
                        .arg(primaryData->GetNumberOfCells()));
    }

    return primaryData;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::convertToCvSelectionData(
        vtkSelection* selection, FieldAssociation fieldAssociation) {
    if (!selection) {
        CVLog::Warning(
                "[cvSelectionPipeline::convertToCvSelectionData] selection is "
                "nullptr");
        return cvSelectionData();
    }

    // Extract IDs
    vtkSmartPointer<vtkIdTypeArray> ids =
            extractSelectionIds(selection, fieldAssociation);
    if (!ids || ids->GetNumberOfTuples() == 0) {
        CVLog::Print(
                "[cvSelectionPipeline::convertToCvSelectionData] No IDs in "
                "selection");
        return cvSelectionData();
    }

    // Create selection data
    cvSelectionData result(ids, static_cast<cvSelectionData::FieldAssociation>(
                                        fieldAssociation));

    // Extract and populate actor information (ParaView-style)
    QMap<vtkProp*, vtkDataSet*> dataMap = extractDataFromSelection(selection);

    for (auto it = dataMap.begin(); it != dataMap.end(); ++it) {
        vtkProp* prop = it.key();
        vtkDataSet* data = it.value();

        vtkActor* actor = vtkActor::SafeDownCast(prop);

        // Handle both vtkPolyData and other data types (e.g.,
        // vtkUnstructuredGrid) The mapper's input might be vtkPolyData or
        // another data type
        vtkPolyData* polyData = vtkPolyData::SafeDownCast(data);

        // If data is not vtkPolyData, try to get it from the actor's mapper
        if (!polyData && actor) {
            vtkMapper* mapper = actor->GetMapper();
            if (mapper) {
                polyData = vtkPolyData::SafeDownCast(mapper->GetInput());
            }
        }

        if (actor && polyData) {
            // Get Z-value from selection node if available
            // Z-value represents depth (closer to camera = smaller value)
            double zValue = 1.0;  // Default: far plane

            for (unsigned int i = 0; i < selection->GetNumberOfNodes(); ++i) {
                vtkSelectionNode* node = selection->GetNode(i);
                if (node &&
                    node->GetProperties()->Has(vtkSelectionNode::PROP())) {
                    vtkProp* nodeProp =
                            vtkProp::SafeDownCast(node->GetProperties()->Get(
                                    vtkSelectionNode::PROP()));
                    if (nodeProp == prop) {
                        // Extract Z-value if available
                        // Note: VTK's hardware selector doesn't typically store
                        // Z in properties Z-buffering is handled internally
                        // during rendering For multi-actor selection, we use
                        // the order of nodes as priority
                        zValue = 1.0 - (static_cast<double>(i) /
                                        selection->GetNumberOfNodes());
                        break;
                    }
                }
            }

            // Add actor info to selection data
            cvActorSelectionInfo info;
            info.actor = actor;
            info.polyData = polyData;
            info.zValue = zValue;
            result.addActorInfo(info);
        } else {
            CVLog::Warning(
                    QString("[cvSelectionPipeline] Failed to add actor info: "
                            "actor=%1, polyData=%2")
                            .arg(actor ? "valid" : "null")
                            .arg(polyData ? "valid" : "null"));
        }
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionPipeline::convertToCvSelectionData] "
                    "Created selection: %1 IDs, %2 actors")
                    .arg(result.count())
                    .arg(result.actorCount()));

    return result;
}

//-----------------------------------------------------------------------------
// Frustum-based through-selection (ParaView-style)
// Reference: ParaView's vtkSMRenderViewProxy::SelectFrustumInternal()
//-----------------------------------------------------------------------------

cvSelectionData cvSelectionPipeline::performFrustumSelection(
        const int region[4], FieldAssociation fieldAssoc) {
    if (!m_viewer || !m_renderer) {
        CVLog::Warning(
                "[cvSelectionPipeline::performFrustumSelection] "
                "Invalid viewer or renderer");
        return cvSelectionData();
    }

    int x0 = std::min(region[0], region[2]);
    int y0 = std::min(region[1], region[3]);
    int x1 = std::max(region[0], region[2]);
    int y1 = std::max(region[1], region[3]);

    if (x0 == x1 || y0 == y1) {
        CVLog::Warning(
                "[cvSelectionPipeline::performFrustumSelection] "
                "Zero-area selection region");
        return cvSelectionData();
    }

    // Convert screen rectangle to 8 world-space frustum corners
    // (4 near-plane corners at z=0, 4 far-plane corners at z=1)
    double frustum[32];
    int idx = 0;
    int corners[4][2] = {{x0, y0}, {x0, y1}, {x1, y0}, {x1, y1}};
    for (auto& corner : corners) {
        for (int z = 0; z <= 1; ++z) {
            m_renderer->SetDisplayPoint(corner[0], corner[1], z);
            m_renderer->DisplayToWorld();
            double* wp = m_renderer->GetWorldPoint();
            if (wp[3] != 0.0) {
                frustum[idx * 4 + 0] = wp[0] / wp[3];
                frustum[idx * 4 + 1] = wp[1] / wp[3];
                frustum[idx * 4 + 2] = wp[2] / wp[3];
            } else {
                frustum[idx * 4 + 0] = wp[0];
                frustum[idx * 4 + 1] = wp[1];
                frustum[idx * 4 + 2] = wp[2];
            }
            frustum[idx * 4 + 3] = 1.0;
            ++idx;
        }
    }

    auto extractor = vtkSmartPointer<vtkExtractSelectedFrustum>::New();
    extractor->CreateFrustum(frustum);

    struct ActorSelection {
        vtkActor* actor;
        vtkPolyData* polyData;
        QVector<qint64> ids;
    };
    QVector<ActorSelection> actorSelections;

    auto props = m_renderer->GetViewProps();
    props->InitTraversal();
    while (auto* prop = props->GetNextProp()) {
        auto* actor = vtkActor::SafeDownCast(prop);
        if (!actor || !actor->GetPickable() || !actor->GetVisibility() ||
            !actor->GetMapper())
            continue;

        auto* input = actor->GetMapper()->GetInput();
        if (!input) continue;
        auto* ds = vtkDataSet::SafeDownCast(input);
        if (!ds) continue;

        double bounds[6];
        ds->GetBounds(bounds);
        if (!extractor->OverallBoundsTest(bounds)) continue;

        QVector<qint64> ids;
        if (fieldAssoc == FIELD_ASSOCIATION_POINTS) {
            for (vtkIdType i = 0; i < ds->GetNumberOfPoints(); ++i) {
                double pt[3];
                ds->GetPoint(i, pt);
                if (extractor->GetFrustum()->EvaluateFunction(pt) <= 0.0) {
                    ids.append(static_cast<qint64>(i));
                }
            }
        } else {
            for (vtkIdType i = 0; i < ds->GetNumberOfCells(); ++i) {
                double bounds_cell[6];
                ds->GetCell(i)->GetBounds(bounds_cell);
                double center[3] = {(bounds_cell[0] + bounds_cell[1]) * 0.5,
                                    (bounds_cell[2] + bounds_cell[3]) * 0.5,
                                    (bounds_cell[4] + bounds_cell[5]) * 0.5};
                if (extractor->GetFrustum()->EvaluateFunction(center) <= 0.0) {
                    ids.append(static_cast<qint64>(i));
                }
            }
        }

        if (!ids.isEmpty()) {
            actorSelections.append(
                    {actor, vtkPolyData::SafeDownCast(ds), std::move(ids)});
        }
    }

    if (actorSelections.isEmpty()) {
        CVLog::PrintVerbose(
                "[cvSelectionPipeline::performFrustumSelection] "
                "No items selected in frustum");
        return cvSelectionData();
    }

    // Use the actor with the most selected items as primary
    // (consistent with hardware selection behavior)
    int bestIdx = 0;
    for (int i = 1; i < actorSelections.size(); ++i) {
        if (actorSelections[i].ids.size() > actorSelections[bestIdx].ids.size())
            bestIdx = i;
    }

    const auto& primary = actorSelections[bestIdx];
    cvSelectionData result(primary.ids, fieldAssoc == FIELD_ASSOCIATION_CELLS
                                                ? cvSelectionData::CELLS
                                                : cvSelectionData::POINTS);

    result.setActorInfo(primary.actor, primary.polyData);
    for (int i = 0; i < actorSelections.size(); ++i) {
        if (i != bestIdx) {
            cvActorSelectionInfo info;
            info.actor = actorSelections[i].actor;
            info.polyData = actorSelections[i].polyData;
            result.addActorInfo(info);
        }
    }

    int totalSelected = 0;
    for (const auto& as : actorSelections) totalSelected += as.ids.size();
    CVLog::PrintVerbose(
            "[cvSelectionPipeline] Frustum selection: %d %s from %d actors "
            "(%d primary)",
            totalSelected,
            fieldAssoc == FIELD_ASSOCIATION_CELLS ? "cells" : "points",
            actorSelections.size(), primary.ids.size());

    return result;
}

//-----------------------------------------------------------------------------
// Helper method to reduce code duplication
//-----------------------------------------------------------------------------

cvSelectionData cvSelectionPipeline::convertSelectionToData(
        vtkSelection* vtkSel,
        FieldAssociation fieldAssoc,
        const QString& errorContext) {
    if (!vtkSel) {
        CVLog::Warning(
                QString("[cvSelectionPipeline] %1 failed").arg(errorContext));
        return cvSelectionData();
    }

    // Use convertToCvSelectionData to properly extract actor info
    // This ensures source object lookup will work for direct extraction
    // Note: convertToCvSelectionData expects
    // cvSelectionPipeline::FieldAssociation
    return convertToCvSelectionData(vtkSel, fieldAssoc);
}

//-----------------------------------------------------------------------------
// High-level selection API (ParaView-style)
//-----------------------------------------------------------------------------

cvSelectionData cvSelectionPipeline::selectCellsOnSurface(const int region[4]) {
    CVLog::Print(
            QString("[cvSelectionPipeline] selectCellsOnSurface region=[%1,%2,"
                    "%3,%4]")
                    .arg(region[0])
                    .arg(region[1])
                    .arg(region[2])
                    .arg(region[3]));
    vtkSmartPointer<vtkSelection> vtkSel =
            executeRectangleSelection(const_cast<int*>(region), SURFACE_CELLS);

    cvSelectionData result = convertSelectionToData(
            vtkSel, FIELD_ASSOCIATION_CELLS, "selectCellsOnSurface");
    CVLog::PrintVerbose(
            QString("[cvSelectionPipeline] selectCellsOnSurface -> %1 ids")
                    .arg(result.count()));
    return result;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectPointsOnSurface(
        const int region[4]) {
    CVLog::Print(
            QString("[cvSelectionPipeline] selectPointsOnSurface region=[%1,%2,"
                    "%3,%4]")
                    .arg(region[0])
                    .arg(region[1])
                    .arg(region[2])
                    .arg(region[3]));
    vtkSmartPointer<vtkSelection> vtkSel =
            executeRectangleSelection(const_cast<int*>(region), SURFACE_POINTS);

    cvSelectionData result = convertSelectionToData(
            vtkSel, FIELD_ASSOCIATION_POINTS, "selectPointsOnSurface");
    CVLog::Print(
            QString("[cvSelectionPipeline] selectPointsOnSurface -> %1 ids")
                    .arg(result.count()));
    return result;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectCellsInFrustum(const int region[4]) {
    return performFrustumSelection(region, FIELD_ASSOCIATION_CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectPointsInFrustum(
        const int region[4]) {
    return performFrustumSelection(region, FIELD_ASSOCIATION_POINTS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectBlocksOnSurface(
        const int region[4]) {
    cvSelectionData cellSel = selectCellsOnSurface(region);
    return expandToBlockSelection(cellSel);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectBlocksInFrustum(
        const int region[4]) {
    cvSelectionData cellSel = selectCellsInFrustum(region);
    return expandToBlockSelection(cellSel);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::expandToBlockSelection(
        const cvSelectionData& partialSelection) {
    if (partialSelection.isEmpty() || !partialSelection.hasActorInfo()) {
        return partialSelection;
    }

    // Block selection: select ALL cells of the primary actor
    // (the actor that was partially selected). This matches ParaView's
    // behavior where block selection selects entire blocks/actors.
    vtkActor* primaryActor = partialSelection.primaryActor();
    if (!primaryActor || !primaryActor->GetMapper()) {
        return partialSelection;
    }

    auto* ds = vtkDataSet::SafeDownCast(primaryActor->GetMapper()->GetInput());
    if (!ds) return partialSelection;

    vtkIdType numCells = ds->GetNumberOfCells();
    QVector<qint64> allCellIds;
    allCellIds.reserve(numCells);
    for (vtkIdType i = 0; i < numCells; ++i) {
        allCellIds.append(static_cast<qint64>(i));
    }

    vtkPolyData* polyData = vtkPolyData::SafeDownCast(ds);
    cvSelectionData result(allCellIds, cvSelectionData::CELLS);
    result.setActorInfo(primaryActor, polyData);

    for (const auto& info : partialSelection.actorInfos()) {
        if (info.actor != primaryActor) {
            result.addActorInfo(info);
        }
    }

    CVLog::PrintVerbose(
            "[cvSelectionPipeline] Block selection: %d actors touched, "
            "primary has %d cells",
            partialSelection.actorCount(), numCells);

    return result;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectCellsInPolygon(
        vtkIntArray* polygon) {
    CVLog::PrintVerbose(
            QString("[cvSelectionPipeline] selectCellsInPolygon: %1 vertices")
                    .arg(polygon ? polygon->GetNumberOfTuples() : 0));

    if (!polygon) {
        CVLog::Warning("[cvSelectionPipeline] Invalid polygon");
        return cvSelectionData();
    }

    vtkSmartPointer<vtkSelection> vtkSel =
            executePolygonSelection(polygon, POLYGON_CELLS);

    return convertSelectionToData(vtkSel, FIELD_ASSOCIATION_CELLS,
                                  "selectCellsInPolygon");
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectPointsInPolygon(
        vtkIntArray* polygon) {
    CVLog::PrintVerbose(
            QString("[cvSelectionPipeline] selectPointsInPolygon: %1 vertices")
                    .arg(polygon ? polygon->GetNumberOfTuples() : 0));

    if (!polygon) {
        CVLog::Warning("[cvSelectionPipeline] Invalid polygon");
        return cvSelectionData();
    }

    vtkSmartPointer<vtkSelection> vtkSel =
            executePolygonSelection(polygon, POLYGON_POINTS);

    return convertSelectionToData(vtkSel, FIELD_ASSOCIATION_POINTS,
                                  "selectPointsInPolygon");
}

//-----------------------------------------------------------------------------
// Selection combination (ParaView-style)
//-----------------------------------------------------------------------------

cvSelectionData cvSelectionPipeline::combineSelections(
        const cvSelectionData& sel1,
        const cvSelectionData& sel2,
        CombineOperation operation) {
    CVLog::PrintVerbose(
            QString("[cvSelectionPipeline] combineSelections: operation=%1")
                    .arg(operation));

    // Handle empty selections first (before field association check!)
    // ParaView behavior (vtkSMSelectionHelper::CombineSelection line 156-159):
    // - If sel1 is empty and deepCopy=false, return true (no-op, result is
    // sel2)
    // - If sel2 is empty, return false (cannot combine)
    if (sel2.isEmpty()) {
        return sel1;  // Return sel1 unchanged
    }

    if (sel1.isEmpty()) {
        // ParaView vtkSMSelectionHelper::CombineSelection: empty sel1 → sel2
        return sel2;
    }

    // Handle DEFAULT (replace with sel2) BEFORE checking field association
    // ParaView behavior: When switching selection types (e.g., cells to
    // points), the new selection should replace the old one, not fail
    if (operation == OPERATION_DEFAULT) {
        return sel2;
    }

    // Check compatibility (only when both are non-empty AND not replacing)
    // This is only for ADDITION/SUBTRACTION/TOGGLE operations
    if (sel1.fieldAssociation() != sel2.fieldAssociation()) {
        // ParaView behavior: When field associations differ and trying to
        // add/subtract, just use the new selection
        return sel2;
    }

    // Get ID sets
    QSet<vtkIdType> set1, set2, resultSet;

    if (!sel1.isEmpty()) {
        vtkSmartPointer<vtkIdTypeArray> arr1 = sel1.vtkArray();
        for (vtkIdType i = 0; i < arr1->GetNumberOfTuples(); ++i) {
            set1.insert(arr1->GetValue(i));
        }
    }

    if (!sel2.isEmpty()) {
        vtkSmartPointer<vtkIdTypeArray> arr2 = sel2.vtkArray();
        for (vtkIdType i = 0; i < arr2->GetNumberOfTuples(); ++i) {
            set2.insert(arr2->GetValue(i));
        }
    }

    // Perform operation
    switch (operation) {
        case OPERATION_ADDITION:
            // Union: sel1 | sel2
            resultSet = set1 + set2;
            CVLog::PrintVerbose(
                    QString("[cvSelectionPipeline] ADDITION: %1 + %2 = %3")
                            .arg(set1.size())
                            .arg(set2.size())
                            .arg(resultSet.size()));
            break;

        case OPERATION_SUBTRACTION:
            // Difference: sel1 & !sel2
            resultSet = set1 - set2;
            CVLog::PrintVerbose(
                    QString("[cvSelectionPipeline] SUBTRACTION: %1 - %2 = %3")
                            .arg(set1.size())
                            .arg(set2.size())
                            .arg(resultSet.size()));
            break;

        case OPERATION_TOGGLE:
            // XOR: sel1 ^ sel2
            resultSet = (set1 - set2) + (set2 - set1);
            CVLog::PrintVerbose(
                    QString("[cvSelectionPipeline] TOGGLE: %1 ^ %2 = %3")
                            .arg(set1.size())
                            .arg(set2.size())
                            .arg(resultSet.size()));
            break;

        default:
            CVLog::Warning(
                    QString("[cvSelectionPipeline] Unknown operation: %1")
                            .arg(operation));
            return cvSelectionData();
    }

    // Convert result set to array
    if (resultSet.isEmpty()) {
        return cvSelectionData();
    }

    vtkSmartPointer<vtkIdTypeArray> resultArray =
            vtkSmartPointer<vtkIdTypeArray>::New();
    for (vtkIdType id : resultSet) {
        resultArray->InsertNextValue(id);
    }

    cvSelectionData result(resultArray, sel1.fieldAssociation());

    // CRITICAL: Preserve actor info from sel2 (the new selection)
    // This allows source object lookup to work correctly for extraction
    // Reference: ParaView maintains representation info across selection
    // operations
    for (int i = 0; i < sel2.actorCount(); ++i) {
        result.addActorInfo(sel2.actorInfo(i));
    }
    // Also add actor info from sel1 if not already present (for
    // ADDITION/TOGGLE)
    if (operation == OPERATION_ADDITION || operation == OPERATION_TOGGLE) {
        for (int i = 0; i < sel1.actorCount(); ++i) {
            // Check if actor already exists in result
            bool found = false;
            for (int j = 0; j < result.actorCount(); ++j) {
                if (result.actorInfo(j).actor == sel1.actorInfo(i).actor) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                result.addActorInfo(sel1.actorInfo(i));
            }
        }
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionPipeline] combineSelections result: %1 IDs, "
                    "%2 actors")
                    .arg(result.count())
                    .arg(result.actorCount()));

    return result;
}

//-----------------------------------------------------------------------------
bool cvSelectionPipeline::pointInPolygon(const int point[2],
                                         vtkIntArray* polygon,
                                         vtkIdType numPoints) {
    // Ray casting algorithm for point-in-polygon test
    // Reference: ParaView's vtkHardwareSelector::Internals::PixelInsidePolygon
    // http://en.wikipedia.org/wiki/Point_in_polygon

    float px = static_cast<float>(point[0]);
    float py = static_cast<float>(point[1]);
    bool inside = false;

    vtkIdType count = numPoints * 2;  // Total values (x,y pairs)

    for (vtkIdType i = 0; i < count; i += 2) {
        float p1X = static_cast<float>(polygon->GetValue(i));
        float p1Y = static_cast<float>(polygon->GetValue(i + 1));
        float p2X = static_cast<float>(polygon->GetValue((i + 2) % count));
        float p2Y = static_cast<float>(polygon->GetValue((i + 3) % count));

        // Check if ray from point crosses edge (p1X,p1Y)-(p2X,p2Y)
        if (py > std::min(p1Y, p2Y) && py <= std::max(p1Y, p2Y) && p1Y != p2Y) {
            if (px <= std::max(p1X, p2X)) {
                float xintersection =
                        (py - p1Y) * (p2X - p1X) / (p2Y - p1Y) + p1X;
                if (p1X == p2X || px <= xintersection) {
                    // Each time intersect, toggle inside
                    inside = !inside;
                }
            }
        }
    }

    return inside;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelection> cvSelectionPipeline::refinePolygonSelection(
        vtkSelection* selection, vtkIntArray* polygon, vtkIdType numPoints) {
    // ParaView-aligned: Refine selection by testing each point against polygon
    // This is a fallback for when vtkHardwareSelector::GeneratePolygonSelection
    // is not available or when additional filtering is needed

    if (!selection || !polygon || numPoints < 3) {
        CVLog::Warning(
                "[cvSelectionPipeline::refinePolygonSelection] Invalid "
                "parameters");
        return {};
    }

    vtkSmartPointer<vtkSelection> refinedSelection =
            vtkSmartPointer<vtkSelection>::New();

    for (unsigned int nodeIdx = 0; nodeIdx < selection->GetNumberOfNodes();
         ++nodeIdx) {
        vtkSelectionNode* node = selection->GetNode(nodeIdx);
        if (!node) continue;

        vtkIdTypeArray* selectionList =
                vtkIdTypeArray::SafeDownCast(node->GetSelectionList());
        if (!selectionList) continue;

        // Get the prop (actor) from selection node for coordinate conversion
        vtkInformation* properties = node->GetProperties();
        if (!properties || !properties->Has(vtkSelectionNode::PROP())) {
            // No prop info, copy the node as-is
            vtkSmartPointer<vtkSelectionNode> newNode =
                    vtkSmartPointer<vtkSelectionNode>::New();
            newNode->DeepCopy(node);
            refinedSelection->AddNode(newNode);
            continue;
        }

        vtkProp* prop = vtkProp::SafeDownCast(
                properties->Get(vtkSelectionNode::PROP()));
        vtkActor* actor = vtkActor::SafeDownCast(prop);

        if (!actor) {
            // Not an actor, copy the node as-is
            vtkSmartPointer<vtkSelectionNode> newNode =
                    vtkSmartPointer<vtkSelectionNode>::New();
            newNode->DeepCopy(node);
            refinedSelection->AddNode(newNode);
            continue;
        }

        // For each selected ID, check if it's inside the polygon
        // This requires world-to-screen coordinate conversion
        vtkSmartPointer<vtkIdTypeArray> filteredList =
                vtkSmartPointer<vtkIdTypeArray>::New();
        filteredList->SetName(selectionList->GetName());

        vtkMapper* mapper = actor->GetMapper();
        if (!mapper) continue;

        vtkDataSet* data = mapper->GetInput();
        if (!data) continue;

        int fieldAssociation = node->GetFieldType();

        for (vtkIdType i = 0; i < selectionList->GetNumberOfTuples(); ++i) {
            vtkIdType id = selectionList->GetValue(i);

            // Get world coordinates
            double worldPos[3] = {0, 0, 0};
            if (fieldAssociation == vtkDataObject::FIELD_ASSOCIATION_POINTS) {
                if (id >= 0 && id < data->GetNumberOfPoints()) {
                    data->GetPoint(id, worldPos);
                } else {
                    continue;
                }
            } else {
                // For cells, use cell center
                if (id >= 0 && id < data->GetNumberOfCells()) {
                    double bounds[6];
                    data->GetCellBounds(id, bounds);
                    worldPos[0] = (bounds[0] + bounds[1]) / 2.0;
                    worldPos[1] = (bounds[2] + bounds[3]) / 2.0;
                    worldPos[2] = (bounds[4] + bounds[5]) / 2.0;
                } else {
                    continue;
                }
            }

            // Convert world to display coordinates
            if (m_renderer) {
                double displayPos[3];
                m_renderer->SetWorldPoint(worldPos[0], worldPos[1], worldPos[2],
                                          1.0);
                m_renderer->WorldToDisplay();
                m_renderer->GetDisplayPoint(displayPos);

                int screenPoint[2] = {static_cast<int>(displayPos[0]),
                                      static_cast<int>(displayPos[1])};

                // Test if point is inside polygon
                if (pointInPolygon(screenPoint, polygon, numPoints)) {
                    filteredList->InsertNextValue(id);
                }
            } else {
                // No renderer, keep all points
                filteredList->InsertNextValue(id);
            }
        }

        // Create new node with filtered list
        if (filteredList->GetNumberOfTuples() > 0) {
            vtkSmartPointer<vtkSelectionNode> newNode =
                    vtkSmartPointer<vtkSelectionNode>::New();
            newNode->DeepCopy(node);
            newNode->SetSelectionList(filteredList);
            refinedSelection->AddNode(newNode);
        }
    }

    CVLog::Print(QString("[cvSelectionPipeline::refinePolygonSelection] "
                         "Refined from %1 to %2 nodes")
                         .arg(selection->GetNumberOfNodes())
                         .arg(refinedSelection->GetNumberOfNodes()));

    return refinedSelection;
}

//-----------------------------------------------------------------------------
// Static Utility Methods (merged from cvSelectionToolHelper)
//-----------------------------------------------------------------------------
bool cvSelectionPipeline::promptUser(const QString& settingsKey,
                                     const QString& title,
                                     const QString& message,
                                     QWidget* parent) {
    // Check if user has disabled this instruction
    QSettings settings;
    QString key = QString("SelectionTools/DontShowAgain/%1").arg(settingsKey);
    bool dontShow = settings.value(key, false).toBool();

    if (dontShow) {
        return false;  // Don't show dialog
    }

    // Create custom dialog (ParaView-style)
    QDialog dialog(parent);
    dialog.setWindowTitle(title);
    dialog.setModal(true);  // Modal - blocks until user responds

    QVBoxLayout* mainLayout = new QVBoxLayout(&dialog);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    mainLayout->setSpacing(15);

    // Icon + message text
    QHBoxLayout* contentLayout = new QHBoxLayout();

    // Information icon
    QLabel* iconLabel = new QLabel(&dialog);
    QIcon infoIcon =
            dialog.style()->standardIcon(QStyle::SP_MessageBoxInformation);
    iconLabel->setPixmap(infoIcon.pixmap(32, 32));
    iconLabel->setAlignment(Qt::AlignTop);
    contentLayout->addWidget(iconLabel);

    // Message text
    QLabel* textLabel = new QLabel(message, &dialog);
    textLabel->setWordWrap(true);
    textLabel->setTextFormat(Qt::RichText);
    textLabel->setMinimumWidth(400);
    contentLayout->addWidget(textLabel, 1);

    mainLayout->addLayout(contentLayout);

    // "Don't show this message again" checkbox (ParaView-style)
    QCheckBox* dontShowAgainCheckBox = new QCheckBox(
            QObject::tr("Do not show this message again"), &dialog);
    mainLayout->addWidget(dontShowAgainCheckBox);

    // OK button
    QDialogButtonBox* buttonBox =
            new QDialogButtonBox(QDialogButtonBox::Ok, &dialog);
    QObject::connect(buttonBox, &QDialogButtonBox::accepted, &dialog,
                     &QDialog::accept);
    mainLayout->addWidget(buttonBox);

    // Show dialog (modal - exec() blocks until user responds)
    dialog.exec();

    // Save preference if user checked "don't show again"
    if (dontShowAgainCheckBox->isChecked()) {
        settings.setValue(key, true);
        CVLog::Print(QString("[cvSelectionPipeline::promptUser] User checked "
                             "'don't show again' for key: %1")
                             .arg(settingsKey));
    }

    return true;
}
