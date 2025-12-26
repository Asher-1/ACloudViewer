// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt - needed for qHash function
#include <QtCore/QHashFunctions>

/**
 * @brief Selection types and enums
 *
 * This file contains shared selection types to avoid circular dependencies
 * between cvViewSelectionManager and cvRenderViewSelectionTool.
 */

/**
 * @brief Selection modes matching ParaView's SelectionMode enum
 *
 * Reference: pqRenderViewSelectionReaction.h, lines 36-58
 */
enum class SelectionMode {
    // Surface selection
    SELECT_SURFACE_CELLS,          ///< Select cells on surface (rectangle)
    SELECT_SURFACE_POINTS,         ///< Select points on surface (rectangle)
    SELECT_SURFACE_CELLS_POLYGON,  ///< Select cells on surface (polygon)
    SELECT_SURFACE_POINTS_POLYGON,  ///< Select points on surface (polygon)

    // Frustum selection
    SELECT_FRUSTUM_CELLS,   ///< Select cells in frustum
    SELECT_FRUSTUM_POINTS,  ///< Select points in frustum

    // Block selection
    SELECT_BLOCKS,          ///< Select blocks (rectangle)
    SELECT_FRUSTUM_BLOCKS,  ///< Select blocks in frustum

    // Interactive selection
    SELECT_SURFACE_CELLS_INTERACTIVELY,  ///< Hover highlight + click select
                                         ///< (cells)
    SELECT_SURFACE_POINTS_INTERACTIVELY,     ///< Hover highlight + click
                                             ///< select (points)
    SELECT_SURFACE_CELLDATA_INTERACTIVELY,   ///< Hover highlight cell data
    SELECT_SURFACE_POINTDATA_INTERACTIVELY,  ///< Hover highlight point data

    // Tooltip selection (Hovering mode)
    HOVER_CELLS_TOOLTIP,   ///< Show cell data tooltip on hover (read-only)
    HOVER_POINTS_TOOLTIP,  ///< Show point data tooltip on hover (read-only)

    // Custom selection (emit signals with selection region)
    SELECT_CUSTOM_POLYGON,  ///< Custom polygon selection (signal only)

    // Zoom mode (ParaView-aligned)
    ZOOM_TO_BOX,  ///< Zoom to box region

    // Selection management
    CLEAR_SELECTION,  ///< Clear current selection
    GROW_SELECTION,   ///< Expand selection by one layer
    SHRINK_SELECTION  ///< Shrink selection by one layer
};

/**
 * @brief Selection modifiers for combining selections
 *
 * Reference: pqView.h, PV_SELECTION_* enums
 */
enum class SelectionModifier {
    SELECTION_DEFAULT,      ///< Replace selection (default)
    SELECTION_ADDITION,     ///< Add to selection (Ctrl)
    SELECTION_SUBTRACTION,  ///< Subtract from selection (Shift)
    SELECTION_TOGGLE        ///< Toggle selection (Ctrl+Shift)
};

// Provide qHash function for SelectionMode to support QSet<SelectionMode>
inline uint qHash(SelectionMode key, uint seed = 0) noexcept {
    return qHash(static_cast<int>(key), seed);
}

// Provide qHash function for SelectionModifier to support QSet<SelectionModifier>
inline uint qHash(SelectionModifier key, uint seed = 0) noexcept {
    return qHash(static_cast<int>(key), seed);
}

