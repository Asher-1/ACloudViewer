// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvSelectionData.h"
#include "qPCL.h"

// VTK
#include <vtkSmartPointer.h>

// Qt
#include <QHash>
#include <QMap>
#include <QObject>

// Forward declarations
class vtkSelection;
class vtkIntArray;
class vtkIdTypeArray;
class vtkDataObject;
class vtkDataSet;
class vtkRenderer;
class vtkHardwareSelector;
class vtkProp;

namespace PclUtils {
class PCLVis;
}

/**
 * @brief Selection pipeline abstraction layer
 *
 * This class provides a clean abstraction for all selection operations,
 * similar to ParaView's vtkSMSelectionHelper.
 *
 * Responsibilities:
 * - Execute hardware-accelerated selections
 * - Convert selection regions to vtkSelection objects
 * - Cache selection results for performance
 * - Provide unified API for all selection types
 *
 * Reference: ParaView/Remoting/Core/vtkSMSelectionHelper.cxx
 */
class QPCL_ENGINE_LIB_API cvSelectionPipeline : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Selection type
     */
    enum SelectionType {
        SURFACE_CELLS,   ///< Surface cells (rectangle)
        SURFACE_POINTS,  ///< Surface points (rectangle)
        FRUSTUM_CELLS,   ///< Frustum cells
        FRUSTUM_POINTS,  ///< Frustum points
        POLYGON_CELLS,   ///< Polygon cells
        POLYGON_POINTS   ///< Polygon points
    };

    /**
     * @brief Field association type
     */
    enum FieldAssociation {
        FIELD_ASSOCIATION_CELLS = 0,
        FIELD_ASSOCIATION_POINTS = 1
    };

    explicit cvSelectionPipeline(QObject* parent = nullptr);
    ~cvSelectionPipeline() override;

    /**
     * @brief Set the visualizer for selection operations
     */
    void setVisualizer(PclUtils::PCLVis* viewer);

    /**
     * @brief Execute a rectangular selection
     * @param region Screen-space rectangle [x1, y1, x2, y2]
     * @param type Selection type
     * @return Smart pointer to vtkSelection (automatic memory management)
     */
    vtkSmartPointer<vtkSelection> executeRectangleSelection(int region[4],
                                                            SelectionType type);

    /**
     * @brief Execute a polygon selection
     * @param polygon Polygon vertices (screen coordinates)
     * @param type Selection type
     * @return Smart pointer to vtkSelection (automatic memory management)
     */
    vtkSmartPointer<vtkSelection> executePolygonSelection(vtkIntArray* polygon,
                                                          SelectionType type);

    /**
     * @brief Extract selected IDs from vtkSelection
     * @param selection The selection object
     * @param fieldAssociation Field association (cells or points)
     * @return Smart pointer to selected IDs array (automatic memory management)
     */
    static vtkSmartPointer<vtkIdTypeArray> extractSelectionIds(
            vtkSelection* selection, FieldAssociation fieldAssociation);

    /**
     * @brief Get data objects from selection (ParaView-style)
     * @param selection The vtkSelection result from vtkHardwareSelector
     * @return Map of prop -> data object
     *
     * This method extracts the data objects associated with each selected
     * prop (actor) from the selection result. This is the correct ParaView
     * way to handle multi-actor selections.
     */
    static QMap<vtkProp*, vtkDataSet*> extractDataFromSelection(
            vtkSelection* selection);

    /**
     * @brief Get the primary data object from selection
     * @param selection The vtkSelection result
     * @return Primary data object (with most selected elements), or nullptr
     *
     * When multiple actors are selected, this returns the data object
     * with the most selected elements (points + cells).
     */
    static vtkDataSet* getPrimaryDataFromSelection(vtkSelection* selection);

    /**
     * @brief Convert vtkSelection to cvSelectionData with actor info
     * (ParaView-style)
     * @param selection The vtkSelection result from vtkHardwareSelector
     * @param fieldAssociation Field association (cells or points)
     * @return cvSelectionData with actor information populated
     *
     * This is the correct ParaView way: extract IDs AND actor information
     * from the selection result, so downstream code knows which actor was
     * selected.
     */
    static cvSelectionData convertToCvSelectionData(
            vtkSelection* selection, FieldAssociation fieldAssociation);
    
    /**
     * @brief Test if a 2D point is inside a polygon (ParaView-aligned)
     * Uses the ray casting algorithm for robust point-in-polygon testing
     * @param point 2D point coordinates [x, y]
     * @param polygon Array of polygon vertices (x1, y1, x2, y2, ...)
     * @param numPoints Number of polygon vertices
     * @return True if point is inside polygon
     */
    static bool pointInPolygon(const int point[2],
                                vtkIntArray* polygon,
                                vtkIdType numPoints);
    
    /**
     * @brief Refine polygon selection with point-in-polygon testing
     * @param selection Initial selection from hardware selector
     * @param polygon Polygon vertices in screen coordinates
     * @param numPoints Number of polygon vertices
     * @return Refined selection with only points inside polygon
     */
    vtkSmartPointer<vtkSelection> refinePolygonSelection(
            vtkSelection* selection,
            vtkIntArray* polygon,
            vtkIdType numPoints);

    /**
     * @brief Get the last selection result
     * @return Smart pointer to last vtkSelection, or nullptr if no selection
     * was made
     */
    vtkSmartPointer<vtkSelection> getLastSelection() const {
        return m_lastSelection;
    }

    /**
     * @brief Enable/disable selection caching
     * @param enable True to enable caching
     */
    void setEnableCaching(bool enable);

    /**
     * @brief Clear the selection cache
     */
    void clearCache();

    /**
     * @brief Get cache statistics
     */
    int getCacheSize() const;
    int getCacheHits() const;
    int getCacheMisses() const;

    ///@{
    /**
     * @brief High-level selection API (ParaView-style)
     *
     * These methods provide a simplified interface that hides VTK details
     * and returns cvSelectionData directly.
     */

    /**
     * @brief Select cells on surface in a rectangular region
     * @param region Screen-space rectangle [x1, y1, x2, y2]
     * @return Selection data with selected cell IDs
     */
    cvSelectionData selectCellsOnSurface(const int region[4]);

    /**
     * @brief Select points on surface in a rectangular region
     * @param region Screen-space rectangle [x1, y1, x2, y2]
     * @return Selection data with selected point IDs
     */
    cvSelectionData selectPointsOnSurface(const int region[4]);

    /**
     * @brief Select cells in polygon region
     * @param polygon Polygon vertices (screen coordinates)
     * @return Selection data with selected cell IDs
     */
    cvSelectionData selectCellsInPolygon(vtkIntArray* polygon);

    /**
     * @brief Select points in polygon region
     * @param polygon Polygon vertices (screen coordinates)
     * @return Selection data with selected point IDs
     */
    cvSelectionData selectPointsInPolygon(vtkIntArray* polygon);
    ///@}

    ///@{
    /**
     * @brief Selection combination methods (ParaView-style)
     *
     * These static methods combine two selections using various operations.
     * Similar to vtkSMSelectionHelper::CombineSelection()
     */

    enum CombineOperation {
        OPERATION_DEFAULT = 0,      ///< Replace (sel2 only)
        OPERATION_ADDITION = 1,     ///< Union (sel1 | sel2)
        OPERATION_SUBTRACTION = 2,  ///< Difference (sel1 & !sel2)
        OPERATION_TOGGLE = 3        ///< XOR (sel1 ^ sel2)
    };

    /**
     * @brief Combine two selections
     * @param sel1 First selection
     * @param sel2 Second selection
     * @param operation Combine operation
     * @return Combined selection
     */
    static cvSelectionData combineSelections(const cvSelectionData& sel1,
                                             const cvSelectionData& sel2,
                                             CombineOperation operation);
    ///@}

signals:
    /**
     * @brief Emitted when selection is completed
     */
    void selectionCompleted(vtkSelection* selection);

    /**
     * @brief Emitted when an error occurs
     */
    void errorOccurred(const QString& message);

private:
    /**
     * @brief Perform hardware selection using vtkHardwareSelector
     * @return Smart pointer to vtkSelection (automatic memory management)
     */
    vtkSmartPointer<vtkSelection> performHardwareSelection(
            int region[4], FieldAssociation fieldAssociation);

    /**
     * @brief Check if selection result is in cache
     * @return Smart pointer to cached selection, or nullptr if not found
     */
    vtkSmartPointer<vtkSelection> getCachedSelection(const QString& key);

    /**
     * @brief Add selection result to cache
     */
    void cacheSelection(const QString& key, vtkSelection* selection);

    /**
     * @brief Generate cache key from selection parameters
     */
    QString generateCacheKey(int region[4], SelectionType type) const;

    /**
     * @brief Convert SelectionType to FieldAssociation
     */
    FieldAssociation getFieldAssociation(SelectionType type) const;

    /**
     * @brief Helper: Convert vtkSelection to cvSelectionData (reduces code duplication)
     * @param vtkSel VTK selection object
     * @param fieldAssoc Field association (CELLS or POINTS)
     * @param errorContext Context string for error messages
     * @return cvSelectionData with extracted IDs
     */
    cvSelectionData convertSelectionToData(vtkSelection* vtkSel,
                                           FieldAssociation fieldAssoc,
                                           const QString& errorContext);

private:
    PclUtils::PCLVis* m_viewer;
    vtkRenderer* m_renderer;

    // Hardware selector (reused for performance)
    vtkSmartPointer<vtkHardwareSelector> m_hardwareSelector;

    // Last selection result (for getPolyData() operations)
    vtkSmartPointer<vtkSelection> m_lastSelection;

    // Selection cache
    bool m_cachingEnabled;
    QHash<QString, vtkSmartPointer<vtkSelection>> m_selectionCache;

    // Cache statistics
    int m_cacheHits;
    int m_cacheMisses;

    // Maximum cache size
    static const int MAX_CACHE_SIZE = 50;
};
