// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
// Qt - must be included before qPCL.h for MOC to work correctly
#include <QHash>
#include <QMap>
#include <QObject>
// clang-format on

#include "cvSelectionData.h"
#include "qPCL.h"

// VTK
#include <vtkHardwareSelector.h>  // Full definition needed for vtkSmartPointer<vtkHardwareSelector>
#include <vtkSmartPointer.h>

// Forward declarations
class vtkSelection;
class vtkIntArray;
class vtkIdTypeArray;
class vtkDataObject;
class vtkDataSet;
class vtkRenderer;
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
            vtkSelection* selection, vtkIntArray* polygon, vtkIdType numPoints);

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

    /**
     * @brief Enter selection mode (ParaView-style cache optimization)
     *
     * Call this before starting a selection operation to enable
     * caching of selection render buffers. This prevents unnecessary
     * re-renders during interactive selection.
     * Reference: vtkPVRenderView::INTERACTION_MODE_SELECTION
     */
    void enterSelectionMode();

    /**
     * @brief Exit selection mode and release cached buffers
     */
    void exitSelectionMode();

    /**
     * @brief Check if currently in selection mode
     */
    bool isInSelectionMode() const { return m_inSelectionMode; }

    /**
     * @brief Clear selection cache and invalidate cached buffers
     *
     * Call this when the scene changes (e.g., data update, camera change)
     * to ensure stale selection data is not used.
     * Reference: vtkPVRenderView::InvalidateCachedSelection
     */
    void invalidateCachedSelection();

    ///@{
    /**
     * @brief Point Picking Radius support (ParaView-aligned)
     *
     * When selecting a single point and no hit is found at the exact
     * pixel location, the selector will search in a radius around the
     * click point to find nearby points. This improves usability for
     * point cloud selection.
     *
     * Reference: vtkPVRenderViewSettings::GetPointPickingRadius()
     *            vtkPVHardwareSelector::Select()
     */

    /**
     * @brief Set the point picking radius (in pixels)
     * @param radius Radius in pixels (0 = disabled)
     *
     * Default is 5 pixels. Set to 0 to disable radius-based picking.
     */
    void setPointPickingRadius(unsigned int radius);

    /**
     * @brief Get the current point picking radius
     * @return Radius in pixels
     */
    unsigned int getPointPickingRadius() const { return m_pointPickingRadius; }
    ///@}

    ///@{
    /**
     * @brief Fast Pre-Selection API (ParaView-aligned)
     *
     * These methods provide fast hover/preview selection using cached
     * hardware selection buffers. Much faster than software picking
     * for interactive selection modes.
     *
     * Reference: pqRenderViewSelectionReaction::fastPreSelection()
     */

    /**
     * @brief Structure to hold complete pixel selection information
     * Including actor and polyData for proper tooltip display
     */
    struct PixelSelectionInfo {
        bool valid;             // Whether selection is valid
        vtkIdType attributeID;  // Element ID (point or cell) within the actor
        vtkProp* prop;          // The selected actor/prop
        vtkPolyData* polyData;  // PolyData from the selected actor

        PixelSelectionInfo()
            : valid(false), attributeID(-1), prop(nullptr), polyData(nullptr) {}
    };

    /**
     * @brief Get complete pixel selection information at a screen position
     * @param x Screen X coordinate
     * @param y Screen Y coordinate
     * @param selectCells True for cell selection, false for points
     * @return Complete selection info including actor and polyData
     *
     * This method returns full selection context needed for tooltips,
     * including the specific actor and its polyData that was selected.
     * This fixes the "Invalid cell ID" issue when multiple actors are present.
     */
    PixelSelectionInfo getPixelSelectionInfo(int x, int y, bool selectCells);

    /**
     * @brief Perform fast pre-selection at a screen position
     * @param x Screen X coordinate
     * @param y Screen Y coordinate
     * @param selectCells True for cell selection, false for points
     * @return Selected element ID, or -1 if nothing found
     *
     * This method uses cached hardware selection buffers when available,
     * falling back to a single-pixel selection if no cache exists.
     * Much faster than software picking for interactive hover.
     *
     * NOTE: This only returns the ID. For tooltip display with multiple actors,
     * use getPixelSelectionInfo() instead to get the correct polyData.
     */
    vtkIdType fastPreSelectAt(int x, int y, bool selectCells);

    /**
     * @brief Check if fast pre-selection buffers are available
     * @return True if buffers are cached and can be used for fast picking
     */
    bool hasCachedBuffers() const;

    /**
     * @brief Capture buffers for fast pre-selection
     *
     * Call this at the start of interactive selection mode to pre-cache
     * the hardware selection buffers. Subsequent fastPreSelectAt() calls
     * will be very fast until invalidateCachedSelection() is called.
     */
    bool captureBuffersForFastPreSelection();
    ///@}

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
     * @brief Helper: Convert vtkSelection to cvSelectionData (reduces code
     * duplication)
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

    // Selection mode state (for cache optimization)
    bool m_inSelectionMode = false;

    // Point picking radius (ParaView-style)
    // When selecting a single point, search in this radius if no direct hit
    unsigned int m_pointPickingRadius = 5;
};
