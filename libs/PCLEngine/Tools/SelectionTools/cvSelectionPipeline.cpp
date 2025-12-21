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

    CVLog::Print(QString("[cvSelectionPipeline] Execute polygon selection: %1 "
                         "vertices, type=%2")
                         .arg(polygon->GetNumberOfTuples())
                         .arg(type));

    // For polygon selections, we don't cache (too many variations)
    // This would require a more sophisticated cache key

    // ParaView approach: Use bounding box for hardware selection, 
    // then filter results with polygon
    // Reference: pqRenderView.cxx, selectPolygonPoints/selectPolygonCells
    
    // Step 1: Find bounding box of polygon
    vtkIdType numPoints = polygon->GetNumberOfTuples() / 2;
    if (numPoints < 3) {
        CVLog::Warning("[cvSelectionPipeline] Polygon needs at least 3 points");
        emit errorOccurred("Invalid polygon");
        return nullptr;
    }

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

    CVLog::Print(QString("[cvSelectionPipeline] Polygon bounding box: [%1, %2, %3, %4]")
                         .arg(minX).arg(minY).arg(maxX).arg(maxY));

    // Step 2: Setup hardware selector (ParaView-style)
    // Reference: vtkPVRenderView::SelectPolygon
    vtkRenderWindow* renderWindow = m_viewer->getRenderWindow();
    if (!renderWindow) {
        CVLog::Warning("[cvSelectionPipeline] Invalid render window");
        emit errorOccurred("Invalid render window");
        return nullptr;
    }

    // Create or reuse hardware selector
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

    // Step 3: Capture pixel buffers (ParaView-style)
    // This renders the scene with special color encoding
    if (!m_hardwareSelector->CaptureBuffers()) {
        CVLog::Warning("[cvSelectionPipeline] Failed to capture buffers for polygon");
        emit errorOccurred("Buffer capture failed");
        return nullptr;
    }

    // Step 4: Generate polygon selection with pixel-level testing
    // Reference: vtkHardwareSelector::GeneratePolygonSelection
    // This tests each pixel in the bounding box to see if it's inside the polygon
    std::vector<int> polygonArray(numPoints * 2);
    for (vtkIdType i = 0; i < numPoints * 2; ++i) {
        polygonArray[i] = polygon->GetValue(i);
    }

    vtkSelection* selection = m_hardwareSelector->GeneratePolygonSelection(
            polygonArray.data(), numPoints * 2);

    if (!selection) {
        CVLog::Warning("[cvSelectionPipeline] Polygon selection failed");
        emit errorOccurred("Polygon selection failed");
        return nullptr;
    }

    // Wrap in smart pointer for automatic cleanup
    vtkSmartPointer<vtkSelection> smartSelection;
    smartSelection.TakeReference(selection);

    // Cache last selection
    m_lastSelection = smartSelection;

    CVLog::Print(QString("[cvSelectionPipeline] Polygon selection completed "
                         "(pixel-precise), nodes: %1")
                         .arg(smartSelection->GetNumberOfNodes()));

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

    // Create or reuse hardware selector
    if (!m_hardwareSelector) {
        m_hardwareSelector = vtkSmartPointer<vtkHardwareSelector>::New();
        CVLog::Print("[cvSelectionPipeline] Created hardware selector");
    }

    m_hardwareSelector->SetRenderer(m_renderer);
    m_hardwareSelector->SetArea(region[0], region[1], region[2], region[3]);

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

