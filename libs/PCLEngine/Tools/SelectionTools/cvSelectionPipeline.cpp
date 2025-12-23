// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionPipeline.h"

// LOCAL
#include "PclUtils/PCLVis.h"
#include "cvSelectionData.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QSet>

// VTK
#include <vtkActor.h>
#include <vtkCellData.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkHardwareSelector.h>
#include <vtkIdTypeArray.h>
#include <vtkInformation.h>
#include <vtkIntArray.h>
#include <vtkMapper.h>
#include <vtkPointData.h>
#include <vtkProp.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>

// Qt
#include <QMap>

//-----------------------------------------------------------------------------
cvSelectionPipeline::cvSelectionPipeline(QObject* parent)
    : QObject(parent),
      m_viewer(nullptr),
      m_renderer(nullptr),
      m_cachingEnabled(true),
      m_cacheHits(0),
      m_cacheMisses(0) {
    CVLog::Print("[cvSelectionPipeline] Created");
}

//-----------------------------------------------------------------------------
cvSelectionPipeline::~cvSelectionPipeline() {
    clearCache();
    CVLog::Print(QString("[cvSelectionPipeline] Destroyed - Cache stats: %1 "
                         "hits, %2 misses")
                         .arg(m_cacheHits)
                         .arg(m_cacheMisses));
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::setVisualizer(PclUtils::PCLVis* viewer) {
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

    CVLog::Print(QString("[cvSelectionPipeline] Visualizer set: %1")
                         .arg((quintptr)viewer, 0, 16));
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelection> cvSelectionPipeline::executeRectangleSelection(
        int region[4], SelectionType type) {
    if (!m_viewer || !m_renderer) {
        CVLog::Warning("[cvSelectionPipeline] Invalid viewer or renderer");
        emit errorOccurred("Invalid viewer or renderer");
        return nullptr;
    }

    CVLog::Print(QString("[cvSelectionPipeline] Execute rectangle selection: "
                         "[%1, %2, %3, %4], type=%5")
                         .arg(region[0])
                         .arg(region[1])
                         .arg(region[2])
                         .arg(region[3])
                         .arg(type));

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
        return nullptr;
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
        return nullptr;
    }

    vtkIdType numPoints = polygon->GetNumberOfTuples() / 2;
    CVLog::Print(QString("[cvSelectionPipeline] Execute polygon selection: %1 "
                         "vertices, type=%2")
                         .arg(numPoints)
                         .arg(type));

    // Validate polygon
    if (numPoints < 3) {
        CVLog::Warning("[cvSelectionPipeline] Polygon needs at least 3 points");
        emit errorOccurred("Invalid polygon: needs at least 3 points");
        return nullptr;
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
        return nullptr;
    }

    CVLog::Print(QString("[cvSelectionPipeline] Polygon bounding box: [%1, %2, %3, %4]")
                         .arg(minX).arg(minY).arg(maxX).arg(maxY));

    // Step 2: Get render window
    vtkRenderWindow* renderWindow = m_viewer->getRenderWindow();
    if (!renderWindow) {
        CVLog::Warning("[cvSelectionPipeline] Invalid render window");
        emit errorOccurred("Invalid render window");
        return nullptr;
    }

    // Step 3: Create or reuse hardware selector
    if (!m_hardwareSelector) {
        m_hardwareSelector = vtkSmartPointer<vtkHardwareSelector>::New();
        CVLog::Print("[cvSelectionPipeline] Created hardware selector for polygon");
    }

    m_hardwareSelector->SetRenderer(m_renderer);
    m_hardwareSelector->SetArea(minX, minY, maxX, maxY);

    // Set field association
    FieldAssociation fieldAssoc = getFieldAssociation(type);
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
        CVLog::Warning("[cvSelectionPipeline] Failed to capture buffers for polygon");
        // Try once more after forcing a render
        renderWindow->Render();
        captureSuccess = m_hardwareSelector->CaptureBuffers();
        if (!captureSuccess) {
            emit errorOccurred("Buffer capture failed");
            return nullptr;
        }
    }

    // Step 5: Generate polygon selection with pixel-level testing
    // Reference: vtkHardwareSelector::GeneratePolygonSelection
    // This tests each pixel in the bounding box to see if it's inside the polygon
    std::vector<int> polygonArray(numPoints * 2);
    for (vtkIdType i = 0; i < numPoints * 2; ++i) {
        polygonArray[i] = polygon->GetValue(i);
    }

    vtkSelection* selection = m_hardwareSelector->GeneratePolygonSelection(
            polygonArray.data(), static_cast<vtkIdType>(numPoints * 2));

    if (!selection) {
        CVLog::Print("[cvSelectionPipeline] Polygon selection returned no results");
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

    CVLog::Print(QString("[cvSelectionPipeline] Polygon selection completed "
                         "(pixel-precise): %1 nodes, %2 total items")
                         .arg(smartSelection->GetNumberOfNodes())
                         .arg(totalItems));

    emit selectionCompleted(smartSelection);
    return smartSelection;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkIdTypeArray> cvSelectionPipeline::extractSelectionIds(
        vtkSelection* selection, FieldAssociation fieldAssociation) {
    if (!selection) {
        return nullptr;
    }

    // Get the first selection node
    if (selection->GetNumberOfNodes() == 0) {
        CVLog::Print("[cvSelectionPipeline] Selection has no nodes");
        return nullptr;
    }

    vtkSelectionNode* node = selection->GetNode(0);
    if (!node) {
        CVLog::Warning("[cvSelectionPipeline] Invalid selection node");
        return nullptr;
    }

    // Get the selection list
    vtkIdTypeArray* selectionList =
            vtkIdTypeArray::SafeDownCast(node->GetSelectionList());
    if (!selectionList) {
        CVLog::Print("[cvSelectionPipeline] No selection list in node");
        return nullptr;
    }

    CVLog::Print(QString("[cvSelectionPipeline] Extracted %1 IDs")
                         .arg(selectionList->GetNumberOfTuples()));

    // Return a copy (automatic memory management)
    vtkSmartPointer<vtkIdTypeArray> copy =
            vtkSmartPointer<vtkIdTypeArray>::New();
    copy->DeepCopy(selectionList);
    return copy;
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

    CVLog::Print("[cvSelectionPipeline] Cache cleared");
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
        CVLog::Warning("[cvSelectionPipeline] Cannot capture buffers - no viewer/renderer");
        return false;
    }

    vtkRenderWindow* renderWindow = m_viewer->getRenderWindow();
    if (!renderWindow) {
        CVLog::Warning("[cvSelectionPipeline] Cannot capture buffers - no render window");
        return false;
    }

    // Create hardware selector if needed
    if (!m_hardwareSelector) {
        m_hardwareSelector = vtkSmartPointer<vtkHardwareSelector>::New();
    }

    m_hardwareSelector->SetRenderer(m_renderer);
    
    // Set area to full viewport
    int* size = m_renderer->GetSize();
    int* origin = m_renderer->GetOrigin();
    m_hardwareSelector->SetArea(origin[0], origin[1], 
                                origin[0] + size[0] - 1, 
                                origin[1] + size[1] - 1);

    // Capture the buffers
    bool success = m_hardwareSelector->CaptureBuffers();
    
    if (success) {
        m_inSelectionMode = true;
        CVLog::Print("[cvSelectionPipeline] Captured buffers for fast pre-selection");
    } else {
        CVLog::Warning("[cvSelectionPipeline] Failed to capture buffers");
    }
    
    return success;
}

//-----------------------------------------------------------------------------
cvSelectionPipeline::PixelSelectionInfo cvSelectionPipeline::getPixelSelectionInfo(
        int x, int y, bool selectCells) {
    PixelSelectionInfo result;
    
    if (!m_viewer || !m_renderer) {
        CVLog::Warning("[cvSelectionPipeline::getPixelSelectionInfo] Invalid viewer or renderer");
        return result;
    }

    // PARAVIEW STYLE: Always do fresh hardware selection, NO CACHING
    // Caching causes stale actor problems and incorrect IDs
    // Reference: ParaView never caches for hover/tooltip - always fresh render
    
    CVLog::PrintDebug("[cvSelectionPipeline::getPixelSelectionInfo] Performing fresh hardware selection (ParaView style)");
    
    // Do a single-pixel hardware selection
    int region[4] = { x, y, x, y };
    vtkSmartPointer<vtkSelection> selection = performHardwareSelection(
            region, selectCells ? FIELD_ASSOCIATION_CELLS : FIELD_ASSOCIATION_POINTS);
    
    if (selection && selection->GetNumberOfNodes() > 0) {
        vtkSelectionNode* node = selection->GetNode(0);
        if (node && node->GetSelectionList() && 
            node->GetSelectionList()->GetNumberOfTuples() > 0) {
            vtkIdTypeArray* ids = vtkIdTypeArray::SafeDownCast(
                    node->GetSelectionList());
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
    // Use the new comprehensive method and return only the ID for backward compatibility
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
    
    CVLog::Print("[cvSelectionPipeline] Entered selection mode (caching enabled)");
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
    // Note: vtkSmartPointer uses = nullptr instead of Reset() (unlike std::shared_ptr)
    m_hardwareSelector = nullptr;
    
    CVLog::Print("[cvSelectionPipeline] Exited selection mode (cache released)");
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::invalidateCachedSelection() {
    // Clear the selection cache
    clearCache();
    
    // ParaView-style: Reset hardware selector to invalidate cached buffers
    // Reference: vtkPVRenderView::InvalidateCachedSelection() clears internal state
    // Note: vtkSmartPointer uses = nullptr instead of Reset() (unlike std::shared_ptr)
    m_hardwareSelector = nullptr;
    
    // Clear last selection
    m_lastSelection = nullptr;
    
    CVLog::Print("[cvSelectionPipeline] Invalidated cached selection");
}

//-----------------------------------------------------------------------------
void cvSelectionPipeline::setPointPickingRadius(unsigned int radius) {
    m_pointPickingRadius = radius;
    CVLog::Print(QString("[cvSelectionPipeline] Point picking radius set to %1 pixels")
                         .arg(radius));
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelection> cvSelectionPipeline::performHardwareSelection(
        int region[4], FieldAssociation fieldAssociation) {
    if (!m_viewer || !m_renderer) {
        return nullptr;
    }

    vtkRenderWindow* renderWindow = m_viewer->getRenderWindow();
    if (!renderWindow) {
        CVLog::Warning("[cvSelectionPipeline] Invalid render window");
        return nullptr;
    }

    // CRITICAL FIX: Convert screen coordinates to VTK/OpenGL coordinates
    // VTK uses OpenGL coordinate system: origin at bottom-left, Y increases upward
    // Mouse events use screen coordinates: origin at top-left, Y increases downward
    // ParaView reference: vtkPVRenderView::ConvertDisplayToRenderCoordinate
    
    int* renderWindowSize = renderWindow->GetSize();
    int windowHeight = renderWindowSize[1];
    
    // Convert Y coordinates (flip vertically)
    int vtk_region[4];
    vtk_region[0] = region[0];                       // X1 (no change)
    vtk_region[1] = windowHeight - region[3] - 1;    // Y1 (flip from Y2)
    vtk_region[2] = region[2];                       // X2 (no change)
    vtk_region[3] = windowHeight - region[1] - 1;    // Y2 (flip from Y1)
    
    CVLog::PrintDebug(QString("[cvSelectionPipeline] Coordinate conversion: "
                              "Screen[%1,%2,%3,%4] -> VTK[%5,%6,%7,%8] (windowHeight=%9)")
                             .arg(region[0]).arg(region[1]).arg(region[2]).arg(region[3])
                             .arg(vtk_region[0]).arg(vtk_region[1]).arg(vtk_region[2]).arg(vtk_region[3])
                             .arg(windowHeight));

    // Create or reuse hardware selector
    if (!m_hardwareSelector) {
        m_hardwareSelector = vtkSmartPointer<vtkHardwareSelector>::New();
        CVLog::Print("[cvSelectionPipeline] Created hardware selector");
    }

    m_hardwareSelector->SetRenderer(m_renderer);
    m_hardwareSelector->SetArea(vtk_region[0], vtk_region[1], vtk_region[2], vtk_region[3]);

    // Set field association
    if (fieldAssociation == FIELD_ASSOCIATION_CELLS) {
        m_hardwareSelector->SetFieldAssociation(
                vtkDataObject::FIELD_ASSOCIATION_CELLS);
    } else {
        m_hardwareSelector->SetFieldAssociation(
                vtkDataObject::FIELD_ASSOCIATION_POINTS);
    }

    // Perform selection (returns smart pointer automatically)
    vtkSmartPointer<vtkSelection> selection = m_hardwareSelector->Select();

    // ParaView-style Point Picking Radius support
    // Reference: vtkPVHardwareSelector::Select() lines 105-130
    // If selecting points with a single-pixel region and no hit found,
    // search in a radius around the click point
    if (fieldAssociation == FIELD_ASSOCIATION_POINTS && 
        m_pointPickingRadius > 0 &&
        region[0] == region[2] && region[1] == region[3])  // Single point click
    {
        bool hasSelection = (selection && selection->GetNumberOfNodes() > 0);
        if (!hasSelection) {
            CVLog::Print(QString("[cvSelectionPipeline] No direct hit, searching "
                                 "in radius %1 pixels...")
                                 .arg(m_pointPickingRadius));
            
            // Use GetPixelInformation to find nearest point in radius
            unsigned int pos[2] = { 
                static_cast<unsigned int>(region[0]), 
                static_cast<unsigned int>(region[1]) 
            };
            unsigned int out_pos[2];
            
            vtkHardwareSelector::PixelInformation info = 
                    m_hardwareSelector->GetPixelInformation(
                            pos, m_pointPickingRadius, out_pos);
            
            if (info.Valid) {
                CVLog::Print(QString("[cvSelectionPipeline] Found point at "
                                     "nearby pixel (%1, %2)")
                                     .arg(out_pos[0]).arg(out_pos[1]));
                
                // Re-generate selection at the found position
                selection.TakeReference(m_hardwareSelector->GenerateSelection(
                        out_pos[0], out_pos[1], out_pos[0], out_pos[1]));
            }
        }
    }

    if (!selection) {
        CVLog::Warning("[cvSelectionPipeline] Hardware selector returned null");
        return nullptr;
    }

    // Cache last selection for getPolyData() operations
    m_lastSelection = selection;

    CVLog::Print(QString("[cvSelectionPipeline] Hardware selection completed, "
                         "nodes: %1")
                         .arg(selection->GetNumberOfNodes()));

    return selection;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkSelection> cvSelectionPipeline::getCachedSelection(
        const QString& key) {
    auto it = m_selectionCache.find(key);
    if (it != m_selectionCache.end()) {
        return it.value();
    }
    return nullptr;
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

    CVLog::Print(QString("[cvSelectionPipeline] Cached selection, size: %1/%2")
                         .arg(m_selectionCache.size())
                         .arg(MAX_CACHE_SIZE));
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
                    CVLog::PrintDebug(
                            QString("[cvSelectionPipeline] Extracted data from "
                                    "actor: %1 points, %2 cells")
                                    .arg(data->GetNumberOfPoints())
                                    .arg(data->GetNumberOfCells()));
                }
            }
        }
    }

    CVLog::Print(QString("[cvSelectionPipeline::extractDataFromSelection] "
                         "Extracted %1 data objects")
                         .arg(result.size()));

    return result;
}

//-----------------------------------------------------------------------------
vtkDataSet* cvSelectionPipeline::getPrimaryDataFromSelection(
        vtkSelection* selection) {
    QMap<vtkProp*, vtkDataSet*> dataMap = extractDataFromSelection(selection);

    if (dataMap.isEmpty()) {
        CVLog::Warning(
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
        CVLog::Print(
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
        vtkPolyData* polyData = vtkPolyData::SafeDownCast(data);

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
                        // Note: VTK's hardware selector doesn't typically store Z in properties
                        // Z-buffering is handled internally during rendering
                        // For multi-actor selection, we use the order of nodes as priority
                        zValue = 1.0 - (static_cast<double>(i) / selection->GetNumberOfNodes());
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

            CVLog::PrintDebug(QString("[cvSelectionPipeline] Added actor info: "
                                      "%1 points, %2 cells")
                                      .arg(polyData->GetNumberOfPoints())
                                      .arg(polyData->GetNumberOfCells()));
        }
    }

    CVLog::Print(QString("[cvSelectionPipeline::convertToCvSelectionData] "
                         "Created selection: %1 IDs, %2 actors")
                         .arg(result.count())
                         .arg(result.actorCount()));

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
        CVLog::Warning(QString("[cvSelectionPipeline] %1 failed").arg(errorContext));
        return cvSelectionData();
    }

    // Extract IDs
    vtkSmartPointer<vtkIdTypeArray> ids = extractSelectionIds(vtkSel, fieldAssoc);

    if (!ids || ids->GetNumberOfTuples() == 0) {
        CVLog::Print(QString("[cvSelectionPipeline] No %1 selected in %2")
                             .arg(fieldAssoc == FIELD_ASSOCIATION_CELLS ? "cells" : "points")
                             .arg(errorContext));
        return cvSelectionData();
    }

    // Convert to cvSelectionData
    cvSelectionData::FieldAssociation cvFieldAssoc =
            (fieldAssoc == FIELD_ASSOCIATION_CELLS) ? cvSelectionData::CELLS
                                                     : cvSelectionData::POINTS;
    
    return cvSelectionData(ids, cvFieldAssoc);
}

//-----------------------------------------------------------------------------
// High-level selection API (ParaView-style)
//-----------------------------------------------------------------------------

cvSelectionData cvSelectionPipeline::selectCellsOnSurface(const int region[4]) {
    CVLog::Print(QString("[cvSelectionPipeline] selectCellsOnSurface: [%1, %2, "
                         "%3, %4]")
                         .arg(region[0])
                         .arg(region[1])
                         .arg(region[2])
                         .arg(region[3]));

    vtkSmartPointer<vtkSelection> vtkSel =
            executeRectangleSelection(const_cast<int*>(region), SURFACE_CELLS);

    return convertSelectionToData(vtkSel, FIELD_ASSOCIATION_CELLS, "selectCellsOnSurface");
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectPointsOnSurface(
        const int region[4]) {
    CVLog::Print(QString("[cvSelectionPipeline] selectPointsOnSurface: [%1, "
                         "%2, %3, %4]")
                         .arg(region[0])
                         .arg(region[1])
                         .arg(region[2])
                         .arg(region[3]));

    vtkSmartPointer<vtkSelection> vtkSel =
            executeRectangleSelection(const_cast<int*>(region), SURFACE_POINTS);

    return convertSelectionToData(vtkSel, FIELD_ASSOCIATION_POINTS, "selectPointsOnSurface");
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectCellsInPolygon(
        vtkIntArray* polygon) {
    CVLog::Print(
            QString("[cvSelectionPipeline] selectCellsInPolygon: %1 vertices")
                    .arg(polygon ? polygon->GetNumberOfTuples() : 0));

    if (!polygon) {
        CVLog::Warning("[cvSelectionPipeline] Invalid polygon");
        return cvSelectionData();
    }

    vtkSmartPointer<vtkSelection> vtkSel =
            executePolygonSelection(polygon, POLYGON_CELLS);

    return convertSelectionToData(vtkSel, FIELD_ASSOCIATION_CELLS, "selectCellsInPolygon");
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPipeline::selectPointsInPolygon(
        vtkIntArray* polygon) {
    CVLog::Print(
            QString("[cvSelectionPipeline] selectPointsInPolygon: %1 vertices")
                    .arg(polygon ? polygon->GetNumberOfTuples() : 0));

    if (!polygon) {
        CVLog::Warning("[cvSelectionPipeline] Invalid polygon");
        return cvSelectionData();
    }

    vtkSmartPointer<vtkSelection> vtkSel =
            executePolygonSelection(polygon, POLYGON_POINTS);

    return convertSelectionToData(vtkSel, FIELD_ASSOCIATION_POINTS, "selectPointsInPolygon");
}

//-----------------------------------------------------------------------------
// Selection combination (ParaView-style)
//-----------------------------------------------------------------------------

cvSelectionData cvSelectionPipeline::combineSelections(
        const cvSelectionData& sel1,
        const cvSelectionData& sel2,
        CombineOperation operation) {
    CVLog::Print(
            QString("[cvSelectionPipeline] combineSelections: operation=%1")
                    .arg(operation));

    // Check compatibility
    if (sel1.fieldAssociation() != sel2.fieldAssociation()) {
        CVLog::Warning(
                "[cvSelectionPipeline] Cannot combine selections "
                "with different field associations");
        return cvSelectionData();
    }

    // Handle DEFAULT (replace with sel2)
    if (operation == OPERATION_DEFAULT) {
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
            CVLog::Print(QString("[cvSelectionPipeline] ADDITION: %1 + %2 = %3")
                                 .arg(set1.size())
                                 .arg(set2.size())
                                 .arg(resultSet.size()));
            break;

        case OPERATION_SUBTRACTION:
            // Difference: sel1 & !sel2
            resultSet = set1 - set2;
            CVLog::Print(
                    QString("[cvSelectionPipeline] SUBTRACTION: %1 - %2 = %3")
                            .arg(set1.size())
                            .arg(set2.size())
                            .arg(resultSet.size()));
            break;

        case OPERATION_TOGGLE:
            // XOR: sel1 ^ sel2
            resultSet = (set1 - set2) + (set2 - set1);
            CVLog::Print(QString("[cvSelectionPipeline] TOGGLE: %1 ^ %2 = %3")
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

    return cvSelectionData(resultArray, sel1.fieldAssociation());
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
                float xintersection = (py - p1Y) * (p2X - p1X) / (p2Y - p1Y) + p1X;
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
        vtkSelection* selection,
        vtkIntArray* polygon,
        vtkIdType numPoints) {
    // ParaView-aligned: Refine selection by testing each point against polygon
    // This is a fallback for when vtkHardwareSelector::GeneratePolygonSelection
    // is not available or when additional filtering is needed
    
    if (!selection || !polygon || numPoints < 3) {
        CVLog::Warning("[cvSelectionPipeline::refinePolygonSelection] Invalid parameters");
        return nullptr;
    }

    vtkSmartPointer<vtkSelection> refinedSelection =
            vtkSmartPointer<vtkSelection>::New();

    for (unsigned int nodeIdx = 0; nodeIdx < selection->GetNumberOfNodes(); ++nodeIdx) {
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
                m_renderer->SetWorldPoint(worldPos[0], worldPos[1], worldPos[2], 1.0);
                m_renderer->WorldToDisplay();
                m_renderer->GetDisplayPoint(displayPos);

                int screenPoint[2] = {
                        static_cast<int>(displayPos[0]),
                        static_cast<int>(displayPos[1])
                };

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

