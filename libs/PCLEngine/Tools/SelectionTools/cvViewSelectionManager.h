// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
// Qt - must be included before qPCL.h for MOC to work correctly
#include <QtCore/QMap>
#include <QtCore/QObject>
#include <QtCore/QPointer>
// clang-format on

// LOCAL
#include "cvGenericSelectionTool.h"
#include "cvSelectionTypes.h"  // For SelectionMode and SelectionModifier enums
#include "qPCL.h"

// Include cvRenderViewSelectionTool.h for QPointer template instantiation
// Note: cvRenderViewSelectionTool.h no longer needs cvViewSelectionManager's
// full definition since enums are now in cvSelectionTypes.h
#include "cvRenderViewSelectionTool.h"

// Forward declarations
// Note: cvRenderViewSelectionTool must be fully defined for QPointer template
// instantiation
class cvSelectionData;
class cvSelectionHistory;
class cvSelectionAlgebra;
class cvSelectionPipeline;
class cvSelectionFilter;
class cvSelectionBookmarks;
class cvSelectionAnnotationManager;
class cvSelectionHighlighter;
class vtkIntArray;
class vtkIdTypeArray;
class vtkPolyData;

namespace PclUtils {
class PCLVis;
}

/**
 * @brief Central manager for all selection operations in the view
 *
 * This class manages the lifecycle of selection tools and coordinates
 * between different selection modes. It provides a unified API for
 * all selection operations, similar to ParaView's selection system.
 *
 * Based on ParaView's pqRenderViewSelectionReaction and
 * pqStandardViewFrameActionsImplementation.
 */
class QPCL_ENGINE_LIB_API cvViewSelectionManager
    : public QObject,
      public cvGenericSelectionTool {
    Q_OBJECT

public:
    // SelectionMode and SelectionModifier are now defined in cvSelectionTypes.h
    // to avoid circular dependency with cvRenderViewSelectionTool
    // Use type aliases for backward compatibility
    using SelectionMode = ::SelectionMode;
    using SelectionModifier = ::SelectionModifier;

    /**
     * @brief Get the singleton instance
     * @return The singleton instance of the manager
     */
    static cvViewSelectionManager* instance();

    /**
     * @brief Set the visualizer for all selection operations
     * @param viewer The generic 3D visualizer
     *
     * Overrides cvGenericSelectionTool::setVisualizer to also update
     * all cached selection tools.
     */
    void setVisualizer(ecvGenericVisualizer3D* viewer) override;

    // Inherited from cvGenericSelectionTool:
    // ecvGenericVisualizer3D* getVisualizer() const;

    /**
     * @brief Enable a selection mode
     * @param mode The selection mode to enable
     *
     * If another mode is active, it will be disabled first.
     * This ensures only one selection mode is active at a time.
     */
    void enableSelection(SelectionMode mode);

    /**
     * @brief Disable the current selection mode
     *
     * Restores the previous interactor style and cleans up resources.
     */
    void disableSelection();

    /**
     * @brief Check if any selection mode is currently active
     */
    bool isSelectionActive() const;

    /**
     * @brief Get the current active selection mode
     * @return Current mode, or -1 if no mode is active
     */
    SelectionMode getCurrentMode() const { return m_currentMode; }

    /**
     * @brief Set the selection modifier
     * @param modifier The modifier to use for the next selection
     *
     * Reference: pqStandardViewFrameActionsImplementation,
     * addSelectionModifierActions
     */
    void setSelectionModifier(SelectionModifier modifier);

    /**
     * @brief Get the current selection modifier
     */
    SelectionModifier getCurrentModifier() const { return m_currentModifier; }

    /**
     * @brief Clear all selections
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 415-420
     */
    void clearSelection();

    /**
     * @brief Clear the current selection data (prevents crashes from stale
     * references)
     * @note This is called when objects might have been deleted to avoid
     * dangling pointers
     */
    void clearCurrentSelection();

    /**
     * @brief Grow the current selection by one layer
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 422-431
     */
    void growSelection();

    /**
     * @brief Shrink the current selection by one layer
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 422-431
     */
    void shrinkSelection();

    /**
     * @brief Check if a selection mode is compatible with the current mode
     *
     * Used to determine if selection modifier buttons should remain active
     * when switching between related modes (e.g., surface cells vs surface
     * points)
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 1037-1060
     */
    bool isCompatible(SelectionMode mode1, SelectionMode mode2) const;

    /**
     * @brief Get the current selection data (VTK-independent)
     * @return Const reference to selection data
     */
    const cvSelectionData& currentSelection() const;

    /**
     * @brief Set the current selection data
     * @param selectionData The selection data (will be copied)
     */
    void setCurrentSelection(const cvSelectionData& selectionData);

    /**
     * @brief Set the current selection data (from VTK array - internal use)
     * @param selection The VTK selection array (smart pointer)
     * @param fieldAssociation 0 for cells, 1 for points
     */
    void setCurrentSelection(const vtkSmartPointer<vtkIdTypeArray>& selection,
                             int fieldAssociation);

    /**
     * @brief Check if there is a current selection
     * @return True if selection exists and is not empty
     */
    bool hasSelection() const;

    //-------------------------------------------------------------------------
    // Utility module accessors (ParaView-style integration)
    //-------------------------------------------------------------------------

    /**
     * @brief Get the selection history manager
     * @return Pointer to history manager (for undo/redo operations)
     */
    cvSelectionHistory* getHistory() { return m_history; }

    /**
     * @brief Get the selection algebra utility
     * @return Pointer to algebra utility (for set operations)
     */
    cvSelectionAlgebra* getAlgebra() { return m_algebra; }

    /**
     * @brief Get the selection pipeline
     * @return Pointer to selection pipeline (for hardware selection)
     */
    cvSelectionPipeline* getPipeline() { return m_pipeline; }

    /**
     * @brief Get the selection filter
     * @return Pointer to filter (for advanced filtering)
     */
    cvSelectionFilter* getFilter() { return m_filter; }

    /**
     * @brief Get the bookmarks manager
     * @return Pointer to bookmarks manager
     */
    cvSelectionBookmarks* getBookmarks() { return m_bookmarks; }

    /**
     * @brief Get the annotation manager
     * @return Pointer to annotation manager
     */
    cvSelectionAnnotationManager* getAnnotations() { return m_annotations; }

    /**
     * @brief Get the shared highlighter
     * @return Pointer to the shared highlighter used by all tools
     *
     * This is the single source of truth for highlight colors/opacity.
     * Both cvSelectionPropertiesWidget and all tooltip tools share this
     * instance.
     */
    cvSelectionHighlighter* getHighlighter() { return m_highlighter; }

    //-------------------------------------------------------------------------
    // Advanced selection operations (using utility modules)
    //-------------------------------------------------------------------------

    /**
     * @brief Undo to previous selection
     * @return True if undo was successful
     */
    bool undo();

    /**
     * @brief Redo to next selection
     * @return True if redo was successful
     */
    bool redo();

    /**
     * @brief Check if undo is available
     */
    bool canUndo() const;

    /**
     * @brief Check if redo is available
     */
    bool canRedo() const;

    /**
     * @brief Perform algebra operation on selections
     * @param op Operation type
     * @param selectionA First selection
     * @param selectionB Second selection (for binary operations)
     * @return Result selection
     */
    cvSelectionData performAlgebraOperation(
            int op,
            const cvSelectionData& selectionA,
            const cvSelectionData& selectionB = cvSelectionData());

    /**
     * @brief Get the underlying polyData for selection operations
     * @return PolyData pointer or nullptr
     */
    vtkPolyData* getPolyData() const;

    /**
     * @brief Notify that scene data has been updated
     *
     * Call this method when the 3D scene data changes (e.g., point cloud
     * updated, mesh modified, actor added/removed). This invalidates
     * cached selection buffers to ensure correct selection results.
     *
     * Reference: ParaView connects this to pqActiveObjects::dataUpdated signal
     */
    void notifyDataUpdated();

    /**
     * @brief Set the point picking radius for single-point selection
     * @param radius Radius in pixels (0 = disabled)
     *
     * When selecting a single point and no exact hit is found,
     * the selector will search in this radius around the click point.
     * Default is 5 pixels.
     */
    void setPointPickingRadius(unsigned int radius);

    /**
     * @brief Get the current point picking radius
     */
    unsigned int getPointPickingRadius() const;

    ///@{
    /**
     * @brief Grow/Shrink Selection Settings (ParaView-aligned)
     *
     * Reference: vtkPVRenderViewSettings::GrowSelectionRemoveSeed
     *            vtkPVRenderViewSettings::GrowSelectionRemoveIntermediateLayers
     */

    /**
     * @brief Set whether to remove seed elements when growing selection
     */
    void setGrowSelectionRemoveSeed(bool remove);
    bool getGrowSelectionRemoveSeed() const { return m_growRemoveSeed; }

    /**
     * @brief Set whether to remove intermediate layers when growing selection
     */
    void setGrowSelectionRemoveIntermediateLayers(bool remove);
    bool getGrowSelectionRemoveIntermediateLayers() const {
        return m_growRemoveIntermediateLayers;
    }

    /**
     * @brief Expand selection by given number of layers
     * @param layers Positive for grow, negative for shrink
     * @param removeSeed Override setting to remove seed
     * @param removeIntermediateLayers Override setting to remove intermediate
     * layers
     *
     * This is the ParaView-compatible expand selection API.
     */
    void expandSelection(int layers,
                         bool removeSeed = false,
                         bool removeIntermediateLayers = false);
    ///@}

signals:
    /**
     * @brief Emitted when a selection operation is completed
     */
    void selectionCompleted();

    /**
     * @brief Emitted when the selection has changed (VTK-independent)
     * @param selectionData The new selection data
     */
    void selectionChanged(const cvSelectionData& selectionData);

    /**
     * @brief Emitted when the selection has changed (legacy - for backward
     * compatibility)
     */
    void selectionChanged();

    /**
     * @brief Emitted when the selection mode changes
     * @param mode The new selection mode
     */
    void modeChanged(SelectionMode mode);

    /**
     * @brief Emitted when selection modifier changes
     * @param modifier The new modifier
     */
    void modifierChanged(SelectionModifier modifier);

    /**
     * @brief Emitted for custom box selection
     * @param region Array of [x1, y1, x2, y2] in screen coordinates
     *
     * Reference: pqRenderViewSelectionReaction.h, line 78-79
     */
    void customBoxSelected(int region[4]);
    void customBoxSelected(int xmin, int ymin, int xmax, int ymax);

    /**
     * @brief Emitted for custom polygon selection
     * @param polygon Array of polygon vertices
     *
     * Reference: pqRenderViewSelectionReaction.h, line 80
     */
    void customPolygonSelected(vtkIntArray* polygon);

    /**
     * @brief Emitted for zoom to box
     * @param region Box region [x1, y1, x2, y2] in screen coordinates
     *
     * Reference: pqRenderViewSelectionReaction ZOOM_TO_BOX case
     */
    void zoomToBoxRequested(int region[4]);
    void zoomToBoxRequested(int xmin, int ymin, int xmax, int ymax);

public:
    /**
     * @brief Get or create a selection tool for the given mode
     * @param mode The selection mode
     * @return The selection tool (cached or newly created)
     *
     * This is public to allow cvSelectionToolController to access it.
     */
    cvRenderViewSelectionTool* getOrCreateTool(SelectionMode mode);

protected:
    /**
     * @brief Clean up inactive tools to free resources
     */
    void cleanupInactiveTools();

private slots:
    /**
     * @brief Handle selection completion from a tool
     */
    void onToolSelectionCompleted();

    /**
     * @brief Handle selection change from a tool
     */
    void onToolSelectionChanged();

private:
    // Singleton constructor/destructor
    explicit cvViewSelectionManager(QObject* parent = nullptr);
    ~cvViewSelectionManager() override;

    // Disable copy
    Q_DISABLE_COPY(cvViewSelectionManager)

    // m_viewer is inherited from cvGenericSelectionTool

    // Current state
    SelectionMode m_currentMode;
    SelectionModifier m_currentModifier;
    bool m_isActive;

    // Current tool
    QPointer<cvRenderViewSelectionTool> m_currentTool;

    // Tool cache (lazy creation)
    QMap<SelectionMode, QPointer<cvRenderViewSelectionTool>> m_toolCache;

    // Maximum number of cached tools
    static const int MAX_CACHED_TOOLS = 5;

    // Selection state (encapsulated - no VTK exposure in MainWindow)
    // Note: Stored as VTK array + field association for backward compatibility
    // Consider migrating to cvSelectionData in future major version
    vtkSmartPointer<vtkIdTypeArray> m_currentSelection;
    int m_currentSelectionFieldAssociation;  // 0=cells, 1=points

    //-------------------------------------------------------------------------
    // Utility modules (ParaView-style architecture)
    //-------------------------------------------------------------------------
    cvSelectionHistory* m_history;  ///< Undo/redo history
    cvSelectionAlgebra*
            m_algebra;  ///< Set operations (union, intersection, etc.)
    cvSelectionPipeline* m_pipeline;    ///< Hardware selection pipeline
    cvSelectionFilter* m_filter;        ///< Advanced filtering
    cvSelectionBookmarks* m_bookmarks;  ///< Save/load selections
    cvSelectionAnnotationManager* m_annotations;  ///< Annotation system
    cvSelectionHighlighter*
            m_highlighter;  ///< Shared highlighter for all tools

    // Grow selection settings (ParaView-style)
    bool m_growRemoveSeed = false;
    bool m_growRemoveIntermediateLayers = false;
};
