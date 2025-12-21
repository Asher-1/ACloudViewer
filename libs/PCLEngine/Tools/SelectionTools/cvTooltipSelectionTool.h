// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_TOOLTIP_SELECTION_TOOL_H
#define CV_TOOLTIP_SELECTION_TOOL_H

#include "cvRenderViewSelectionTool.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionTooltipHelper.h"

// VTK
#include <vtkSmartPointer.h>

// Qt
#include <QShortcut>
#include <QTimer>

// Forward declarations
class vtkIdTypeArray;
class vtkActor;
class vtkPolyData;

/**
 * @brief Tool for tooltip and interactive selection modes
 *
 * This tool implements ParaView's tooltip and interactive selection
 * functionality:
 * - Mouse hover displays detailed information about points/cells
 * - Real-time hover highlighting (immediate, no delay)
 * - Ctrl-C/Cmd-C copies tooltip content to clipboard
 * - Optional click-to-select functionality (interactive mode)
 * - ESC key to exit/disable the tool
 *
 * Supported modes:
 * 1. Tooltip mode (read-only):
 *    - SELECT_SURFACE_CELLS_TOOLTIP
 *    - SELECT_SURFACE_POINTS_TOOLTIP
 *    - No selection on click, only information display
 *
 * 2. Interactive mode (hover + select):
 *    - SELECT_SURFACE_CELLS_INTERACTIVELY
 *    - SELECT_SURFACE_POINTS_INTERACTIVELY
 *    - Click to select/deselect elements
 *    - Supports selection modifiers (Ctrl, Shift)
 *
 * Reference: pqRenderViewSelectionReaction.cxx (ParaView)
 * - SELECT_SURFACE_CELLS_TOOLTIP / SELECT_SURFACE_POINTS_TOOLTIP
 * - SELECT_SURFACE_CELLS_INTERACTIVELY / SELECT_SURFACE_POINTS_INTERACTIVELY
 *
 * Architecture note:
 * This class replaces the previous cvInteractiveSelectionTool by supporting
 * both modes in a single class, following ParaView's unified design pattern.
 */
class QPCL_ENGINE_LIB_API cvTooltipSelectionTool
    : public cvRenderViewSelectionTool {
    Q_OBJECT

public:
    explicit cvTooltipSelectionTool(SelectionMode mode,
                                    QObject* parent = nullptr);
    ~cvTooltipSelectionTool() override;

    /**
     * @brief Enable or disable tooltip display
     * @param enabled True to show tooltips, false to hide
     */
    void setTooltipEnabled(bool enabled);

    /**
     * @brief Set maximum number of attributes to display in tooltip
     * @param maxAttributes Maximum number of attributes
     */
    void setMaxTooltipAttributes(int maxAttributes);

    /**
     * @brief Get the current tooltip text (HTML format)
     * @return The formatted tooltip text
     */
    QString getTooltipText() const { return m_currentTooltipText; }

    /**
     * @brief Get the current tooltip text (plain text format)
     * @return The plain text tooltip (for clipboard copy)
     */
    QString getPlainTooltipText() const { return m_currentPlainText; }

    /**
     * @brief Disable the tooltip tool and clear highlights
     * Overrides base class to ensure highlights are cleared when switching
     * tools
     */
    void disable() override;

signals:
    /**
     * @brief Emitted when tooltip text changes
     * @param htmlText HTML formatted tooltip text
     * @param plainText Plain text tooltip (for clipboard)
     */
    void tooltipChanged(const QString& htmlText, const QString& plainText);

    /**
     * @brief Emitted when hover element changes (VTK-independent)
     * @param hoveredId The ID of the hovered element (-1 if none)
     * @param fieldAssociation 0 for cells, 1 for points
     */
    void hoverChanged(qint64 hoveredId, int fieldAssociation);

    /**
     * @brief Emitted when user presses ESC to request disabling the tool
     */
    void requestDisable();

    /**
     * @brief Emitted when selection is finished (interactive mode only)
     * @param selectionData The selected elements
     */
    void selectionFinished(const cvSelectionData& selectionData);

protected:
    void setupInteractorStyle() override;
    void setupObservers() override;
    void onSelectionChanged(vtkObject* caller,
                            unsigned long eventId,
                            void* callData) override;

private slots:
    /**
     * @brief Called when mouse stops moving (for tooltip display)
     */
    void onMouseStop();

    /**
     * @brief Called when right button is pressed
     */
    void onRightButtonPress();

    /**
     * @brief Called when right button is released
     */
    void onRightButtonRelease();

private:
    void onMouseMove();
    void onLeftButtonPress();  // Interactive mode: handle click selection
    void updateHighlight();    // ParaView-style: immediate highlight update
    // Note: pickAtCursor(), isSelectingCells() are
    // now in base class
    void updateTooltip(vtkIdType id);
    void hideTooltip();
    // Check if in interactive (click-to-select) mode
    bool isInteractiveMode() const;

    // Interactive mode methods
    void toggleSelection(vtkIdType id);  // Add/remove element from selection

private:
    // Note: m_cellPicker and m_pointPicker are now in base class

    // Hover state
    vtkIdType m_hoveredId;
    int m_fieldAssociation;
    bool m_rightButtonPressed;
    bool m_mouseMoving;

    // Tooltip helpers
    cvSelectionTooltipHelper* m_tooltipHelper;
    cvSelectionHighlighter* m_hoverHighlighter;

    // Tooltip text cache
    QString m_currentTooltipText;  // HTML format
    QString m_currentPlainText;    // Plain text for clipboard

    // Tooltip settings
    bool m_tooltipEnabled;  // Whether to show tooltips

    // Timer to avoid excessive tooltip updates (ParaView-style)
    QTimer m_mouseStopTimer;
    // milliseconds (400ms for tooltip display)
    static const int TOOLTIP_WAITING_TIME = 400;

    // Shortcuts
    QShortcut* m_copyShortcut;    // Ctrl-C/Cmd-C for copying tooltip
    QShortcut* m_escapeShortcut;  // ESC to exit/disable tool

    // Interactive mode selection state
    // Current selected elements
    vtkSmartPointer<vtkIdTypeArray> m_currentSelection;
    // true: interactive mode (click to select), false: tooltip only
    bool m_enableSelection;
};

#endif  // CV_TOOLTIP_SELECTION_TOOL_H
