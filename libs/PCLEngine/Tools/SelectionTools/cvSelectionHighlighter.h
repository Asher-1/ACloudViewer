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
#include <QColor>
#include <QObject>
#include <QString>

// VTK
#include <vtkActor.h>  // Full definition needed for vtkSmartPointer<vtkActor>
#include <vtkSmartPointer.h>
class vtkIdTypeArray;
class vtkPolyData;
class vtkDataArray;
class vtkFieldData;
class vtkDataSetMapper;
class vtkExtractSelection;
class vtkSelection;
class vtkSelectionNode;
class vtkRenderer;
class vtkActor2D;
class vtkLabeledDataMapper;

namespace PclUtils {
class PCLVis;
}

/**
 * @brief Label properties for selection annotations
 *
 * Unified structure for both cell and point labels.
 * This is the single source of truth for label appearance settings.
 */
struct QPCL_ENGINE_LIB_API SelectionLabelProperties {
    // General (ParaView defaults from VTK/Rendering/Core/vtkProperty.h)
    double opacity = 1.0;
    int pointSize = 5;  // ParaView default
    int lineWidth = 2;

    // Cell Label Font (ParaView defaults from
    // representations_remotingviews.xml) Line 4468-4521: CellLabelColor
    // default_values="0.0 1.0 0.0" (Green)
    QString cellLabelFontFamily = "Arial";  // ParaView: FontFamily = 0 (Arial)
    int cellLabelFontSize = 18;             // ParaView: CellLabelFontSize = 18
    QColor cellLabelColor = QColor(0, 255, 0);  // ParaView: Green (0, 1, 0)
    double cellLabelOpacity = 1.0;
    bool cellLabelBold = false;    // ParaView: default_values="0"
    bool cellLabelItalic = false;  // ParaView: default_values="0"
    bool cellLabelShadow = false;  // ParaView: default_values="0" (NOT true!)
    int cellLabelHorizontalJustification = 0;  // ParaView: 0=Left
    int cellLabelVerticalJustification = 0;    // ParaView: 0=Bottom
    QString cellLabelFormat = "";  // ParaView: empty (auto-select format)

    // Point Label Font (ParaView defaults from
    // representations_remotingviews.xml) Line 4326-4401: PointLabelColor
    // default_values="1 1 0" (Yellow)
    QString pointLabelFontFamily = "Arial";  // ParaView: FontFamily = 0 (Arial)
    int pointLabelFontSize = 18;  // ParaView: PointLabelFontSize = 18
    QColor pointLabelColor = QColor(255, 255, 0);  // ParaView: Yellow (1, 1, 0)
    double pointLabelOpacity = 1.0;
    bool pointLabelBold = false;    // ParaView: default_values="0"
    bool pointLabelItalic = false;  // ParaView: default_values="0"
    bool pointLabelShadow = false;  // ParaView: default_values="0" (NOT true!)
    int pointLabelHorizontalJustification = 0;  // ParaView: 0=Left
    int pointLabelVerticalJustification = 0;    // ParaView: 0=Bottom
    QString pointLabelFormat = "";  // ParaView: empty (auto-select format)

    // Tooltip
    bool showTooltips = true;
    int maxTooltipAttributes = 15;
};

/**
 * @brief Helper class for highlighting selected elements in the visualizer
 *
 * This is the SINGLE SOURCE OF TRUTH for all selection visualization
 * properties:
 * - Colors for each highlight mode (HOVER, PRESELECTED, SELECTED, BOUNDARY)
 * - Opacities for each mode
 * - Point sizes and line widths
 * - Label properties for annotations
 *
 * Inherits from QObject to provide property change notifications.
 * UI components should connect to signals and read properties from this class.
 *
 * Based on ParaView's selection highlighting system
 * Reference: ParaView's vtkSMPVRepresentationProxy selection highlighting
 */
class QPCL_ENGINE_LIB_API cvSelectionHighlighter
    : public QObject,
      public cvGenericSelectionTool {
    Q_OBJECT
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
     * @brief Set visibility of all highlight actors
     * @param visible True to show highlights, false to hide
     * @note Used to temporarily hide highlights during hardware selection
     *       to prevent depth buffer occlusion issues with subtract selection
     */
    void setHighlightsVisible(bool visible);

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

    // =========================================================================
    // Label Properties (Single Source of Truth)
    // =========================================================================

    /**
     * @brief Get label properties for selection mode
     * @param interactive If true, return interactive (hover) label properties
     * @return Reference to label properties
     */
    const SelectionLabelProperties& getLabelProperties(
            bool interactive = false) const;

    /**
     * @brief Set label properties for selection mode
     * @param props Label properties to set
     * @param interactive If true, set interactive (hover) label properties
     */
    void setLabelProperties(const SelectionLabelProperties& props,
                            bool interactive = false);

    // =========================================================================
    // Selection Label Array Methods (ParaView-style)
    // =========================================================================

    /**
     * @brief Set the point label array name
     * @param arrayName Name of array to use for point labels (empty to disable)
     * @param visible Whether labels should be visible
     */
    void setPointLabelArray(const QString& arrayName, bool visible = true);

    /**
     * @brief Set the cell label array name
     * @param arrayName Name of array to use for cell labels (empty to disable)
     * @param visible Whether labels should be visible
     */
    void setCellLabelArray(const QString& arrayName, bool visible = true);

    /**
     * @brief Get current point label array name
     */
    QString getPointLabelArrayName() const { return m_pointLabelArrayName; }

    /**
     * @brief Get current cell label array name
     */
    QString getCellLabelArrayName() const { return m_cellLabelArrayName; }

    /**
     * @brief Check if point labels are visible
     */
    bool isPointLabelVisible() const { return m_pointLabelVisible; }

    /**
     * @brief Check if cell labels are visible
     */
    bool isCellLabelVisible() const { return m_cellLabelVisible; }

    // =========================================================================
    // QColor convenience methods (for Qt UI components)
    // =========================================================================

    /**
     * @brief Get highlight color as QColor
     */
    QColor getHighlightQColor(HighlightMode mode) const;

    /**
     * @brief Set highlight color from QColor
     */
    void setHighlightQColor(const QColor& color, HighlightMode mode = SELECTED);

signals:
    /**
     * @brief Emitted when any highlight color changes
     * @param mode The mode that changed
     */
    void colorChanged(int mode);

    /**
     * @brief Emitted when any opacity changes
     * @param mode The mode that changed
     */
    void opacityChanged(int mode);

    /**
     * @brief Emitted when point size changes
     * @param mode The mode that changed
     */
    void pointSizeChanged(int mode);

    /**
     * @brief Emitted when line width changes
     * @param mode The mode that changed
     */
    void lineWidthChanged(int mode);

    /**
     * @brief Emitted when label properties change
     * @param interactive Whether interactive properties changed
     */
    void labelPropertiesChanged(bool interactive);

    /**
     * @brief Emitted when any property changes (general notification)
     */
    void propertiesChanged();

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

    /**
     * @brief Update label actor for point or cell labels
     * @param isPointLabels True for point labels, false for cell labels
     */
    void updateLabelActor(bool isPointLabels);

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

    // Label properties (single source of truth)
    SelectionLabelProperties m_labelProperties;  ///< For SELECTED mode
    SelectionLabelProperties m_interactiveLabelProperties;  ///< For HOVER mode

    // Label array names for displaying on selected elements
    QString m_pointLabelArrayName;  ///< Array name for point labels (empty =
                                    ///< disabled)
    QString m_cellLabelArrayName;   ///< Array name for cell labels (empty =
                                    ///< disabled)
    bool m_pointLabelVisible;       ///< Point label visibility
    bool m_cellLabelVisible;        ///< Cell label visibility

    // Label actors for rendering labels on selected elements
    vtkSmartPointer<vtkActor2D> m_pointLabelActor;  ///< Actor for point labels
    vtkSmartPointer<vtkActor2D> m_cellLabelActor;   ///< Actor for cell labels
    vtkSmartPointer<vtkPolyData>
            m_lastSelectionPolyData;  ///< Cached selection data for labels
};

//=============================================================================
// Tooltip Formatter (merged from cvTooltipFormatter.h)
//=============================================================================

/**
 * @brief Formatter class for generating tooltip text from VTK data
 *
 * This class formats tooltip information for selected elements (points/cells)
 * from VTK PolyData into HTML-formatted text. It extracts and formats:
 * - Element ID
 * - Coordinates (for points)
 * - Type (for cells)
 * - Data array values (Intensity, RGB, normals, etc.)
 *
 * Based on ParaView's vtkSMTooltipSelectionPipeline
 * Reference: ParaView/Remoting/Misc/vtkSMTooltipSelectionPipeline.cxx
 */
class QPCL_ENGINE_LIB_API cvTooltipFormatter {
public:
    /**
     * @brief Element association type
     */
    enum AssociationType { POINTS = 1, CELLS = 0 };

    /**
     * @brief Constructor
     */
    cvTooltipFormatter();

    /**
     * @brief Destructor
     */
    ~cvTooltipFormatter();

    /**
     * @brief Generate tooltip information for a selected element
     * @param polyData The mesh data
     * @param elementId The ID of the selected element (point or cell)
     * @param association Type of element (POINTS or CELLS)
     * @param datasetName Optional name of the dataset
     * @return HTML-formatted tooltip string
     */
    QString getTooltipInfo(vtkPolyData* polyData,
                           vtkIdType elementId,
                           AssociationType association,
                           const QString& datasetName = QString());

    /**
     * @brief Generate plain text tooltip (no HTML formatting)
     * @param polyData The mesh data
     * @param elementId The ID of the selected element
     * @param association Type of element (POINTS or CELLS)
     * @param datasetName Optional name of the dataset
     * @return Plain text tooltip string
     */
    QString getPlainTooltipInfo(vtkPolyData* polyData,
                                vtkIdType elementId,
                                AssociationType association,
                                const QString& datasetName = QString());

    /**
     * @brief Set maximum number of attributes to display
     * @param maxAttribs Maximum number of data arrays to show (default: 15)
     */
    void setMaxAttributes(int maxAttribs);

private:
    /**
     * @brief Format point tooltip information
     */
    QString formatPointTooltip(vtkPolyData* polyData,
                               vtkIdType pointId,
                               const QString& datasetName);

    /**
     * @brief Format cell tooltip information
     */
    QString formatCellTooltip(vtkPolyData* polyData,
                              vtkIdType cellId,
                              const QString& datasetName);

    /**
     * @brief Add data array values to tooltip
     */
    void addArrayValues(QString& tooltip,
                        vtkFieldData* fieldData,
                        vtkIdType tupleIndex);

    /**
     * @brief Format a single data array value
     */
    QString formatArrayValue(vtkDataArray* array, vtkIdType tupleIndex);

    /**
     * @brief Format a number with ParaView-style intelligent formatting
     * Uses scientific notation for very large/small numbers, otherwise 'g'
     * format
     */
    QString formatNumber(double value);

    int m_maxAttributes;  ///< Maximum number of attributes to display
};
