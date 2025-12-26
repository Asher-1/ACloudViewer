// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"

// LOCAL
#include "cvGenericSelectionTool.h"
#include "cvSelectionData.h"

// Qt
#include <QString>

// VTK
#include <vtkActor.h>        // Full definition needed for vtkSmartPointer<vtkActor>
#include <vtkSmartPointer.h>
class vtkIdTypeArray;
class vtkPolyData;
class vtkDataSetMapper;
class vtkExtractSelection;
class vtkSelection;
class vtkSelectionNode;
class vtkRenderer;

namespace PclUtils {
class PCLVis;
}

/**
 * @brief Helper class for highlighting selected elements in the visualizer
 *
 * Inherits from cvGenericSelectionTool (lightweight) since it only needs
 * visualizer access for rendering highlights. Does not need picking
 * functionality.
 *
 * Based on ParaView's selection highlighting system
 * Supports different visualization modes:
 * - Selected elements (solid color)
 * - Boundary elements (different color)
 * - Preview/hover (semi-transparent)
 *
 * Reference: ParaView's vtkSMPVRepresentationProxy selection highlighting
 */
class QPCL_ENGINE_LIB_API cvSelectionHighlighter
    : public cvGenericSelectionTool {
public:
    /**
     * @brief Highlight mode (Enhanced multi-level highlighting)
     *
     * Enhanced for maximum visibility with bright colors and high opacity:
     * 1. HOVER - Immediate feedback on mouse movement (bright cyan, high
     * opacity)
     * 2. PRESELECTED - Preview before confirming (bright yellow, high opacity)
     * 3. SELECTED - Final selection (bright green, opaque)
     */
    enum HighlightMode {
        HOVER,        ///< Hover highlight (bright cyan, 0.9 opacity) - enhanced
                      ///< visibility
        PRESELECTED,  ///< Pre-selected (bright yellow, 0.8 opacity) - enhanced
                      ///< visibility
        SELECTED,  ///< Final selection (bright green, 1.0 opacity) - enhanced
                   ///< visibility
        BOUNDARY   ///< Boundary elements (bright orange, 0.85 opacity) -
                   ///< enhanced visibility
    };

    /**
     * @brief Constructor
     */
    cvSelectionHighlighter();

    /**
     * @brief Destructor
     */
    ~cvSelectionHighlighter();

    /**
     * @brief Highlight selected elements (automatically gets polyData from
     * visualizer)
     * @param selection Array of selected IDs (smart pointer)
     * @param fieldAssociation 0 for cells, 1 for points
     * @param mode Highlight mode
     * @return true on success
     *
     * Note: This method gets polyData internally from visualizer,
     *       following SelectionTools architecture pattern
     */
    bool highlightSelection(const vtkSmartPointer<vtkIdTypeArray>& selection,
                            int fieldAssociation,
                            HighlightMode mode = SELECTED);

    /**
     * @brief Highlight selected elements (with explicit polyData)
     * @param polyData The mesh data
     * @param selection Array of selected IDs (smart pointer)
     * @param fieldAssociation 0 for cells, 1 for points
     * @param mode Highlight mode
     * @return true on success
     *
     * Note: Use this overload when polyData is already available
     */
    bool highlightSelection(vtkPolyData* polyData,
                            const vtkSmartPointer<vtkIdTypeArray>& selection,
                            int fieldAssociation,
                            HighlightMode mode = SELECTED);

    /**
     * @brief Highlight a selection (high-level interface - no VTK types)
     * @param selectionData Selection data wrapper
     * @param mode Highlight mode
     * @return True on success
     * @note This is the recommended interface for UI code to avoid VTK
     * dependencies
     */
    bool highlightSelection(const cvSelectionData& selectionData,
                            HighlightMode mode = SELECTED);

    /**
     * @brief Highlight a single element (for hover preview)
     * @param polyData The mesh data
     * @param elementId ID of the element to highlight
     * @param fieldAssociation 0 for cells, 1 for points
     * @return true on success
     */
    bool highlightElement(vtkPolyData* polyData,
                          vtkIdType elementId,
                          int fieldAssociation);

    /**
     * @brief Clear all highlights
     */
    void clearHighlights();

    /**
     * @brief Clear only hover highlight (keep selected/preselected)
     * @note Used during hover to avoid clearing persistent selections
     */
    void clearHoverHighlight();

    /**
     * @brief Set highlight color
     * @param r Red component (0-1)
     * @param g Green component (0-1)
     * @param b Blue component (0-1)
     * @param mode Which mode to set color for
     */
    void setHighlightColor(double r,
                           double g,
                           double b,
                           HighlightMode mode = SELECTED);

    /**
     * @brief Set highlight opacity
     * @param opacity Opacity value (0-1)
     * @param mode Which mode to set opacity for
     */
    void setHighlightOpacity(double opacity, HighlightMode mode = SELECTED);

    /**
     * @brief Get highlight color for a specific mode
     * @param mode Which mode to get color for
     * @return Pointer to color array [r, g, b] or nullptr if invalid mode
     */
    const double* getHighlightColor(HighlightMode mode) const;

    /**
     * @brief Get highlight opacity for a specific mode
     * @param mode Which mode to get opacity for
     * @return Opacity value (0-1)
     */
    double getHighlightOpacity(HighlightMode mode) const;

    /**
     * @brief Set point size for highlight rendering
     * @param size Point size in pixels
     * @param mode Which mode to set point size for
     */
    void setPointSize(int size, HighlightMode mode = SELECTED);

    /**
     * @brief Get point size for a specific mode
     * @param mode Which mode to get point size for
     * @return Point size in pixels
     */
    int getPointSize(HighlightMode mode) const;

    /**
     * @brief Set line width for highlight rendering
     * @param width Line width in pixels
     * @param mode Which mode to set line width for
     */
    void setLineWidth(int width, HighlightMode mode = SELECTED);

    /**
     * @brief Get line width for a specific mode
     * @param mode Which mode to get line width for
     * @return Line width in pixels
     */
    int getLineWidth(HighlightMode mode) const;

    /**
     * @brief Enable/disable highlight
     * @param enabled True to enable, false to disable
     */
    void setEnabled(bool enabled);

private:
    /**
     * @brief Create selection highlight actor
     * @return Smart pointer to vtkActor (automatic memory management)
     */
    vtkSmartPointer<vtkActor> createHighlightActor(vtkPolyData* polyData,
                                                   vtkIdTypeArray* selection,
                                                   int fieldAssociation,
                                                   HighlightMode mode);

    /**
     * @brief Create VTK selection node
     * @return Smart pointer to vtkSelectionNode (automatic memory management)
     */
    vtkSmartPointer<vtkSelectionNode> createSelectionNode(
            vtkIdTypeArray* selection, int fieldAssociation);

    /**
     * @brief Add actor to visualizer
     */
    void addActorToVisualizer(vtkActor* actor, const QString& id);

    /**
     * @brief Remove actor from visualizer
     */
    void removeActorFromVisualizer(const QString& id);

    // m_viewer is inherited from cvGenericSelectionTool

    // Highlight actors (ParaView-style multi-level)
    vtkSmartPointer<vtkActor> m_hoverActor;        ///< Hover highlight actor
    vtkSmartPointer<vtkActor> m_preselectedActor;  ///< Pre-selected actor
    vtkSmartPointer<vtkActor> m_selectedActor;     ///< Final selection actor
    vtkSmartPointer<vtkActor> m_boundaryActor;     ///< Boundary actor

    // Colors for different modes (Enhanced for maximum visibility)
    // Using bright, highly saturated colors that stand out
    double m_hoverColor[3];  ///< Bright Cyan (0.0, 1.0, 1.0) for hover - highly
                             ///< visible
    double m_preselectedColor[3];  ///< Bright Yellow (1.0, 1.0, 0.0) for
                                   ///< preselect - highly visible
    double m_selectedColor[3];  ///< Bright Green (0.0, 1.0, 0.0) for selected -
                                ///< highly visible
    double m_boundaryColor[3];  ///< Bright Orange (1.0, 0.65, 0.0) for boundary
                                ///< - highly visible

    // Opacities (Enhanced for maximum visibility)
    double m_hoverOpacity;        ///< 0.9 for hover (highly visible)
    double m_preselectedOpacity;  ///< 0.8 for preselect (highly visible)
    double m_selectedOpacity;     ///< 1.0 for selected (opaque)
    double m_boundaryOpacity;     ///< 0.85 for boundary (highly visible)

    // Point sizes for different modes
    int m_hoverPointSize;
    int m_preselectedPointSize;
    int m_selectedPointSize;
    int m_boundaryPointSize;

    // Line widths for different modes
    int m_hoverLineWidth;
    int m_preselectedLineWidth;
    int m_selectedLineWidth;
    int m_boundaryLineWidth;

    bool m_enabled;
    QString m_hoverActorId;
    QString m_preselectedActorId;
    QString m_selectedActorId;
    QString m_boundaryActorId;
};
