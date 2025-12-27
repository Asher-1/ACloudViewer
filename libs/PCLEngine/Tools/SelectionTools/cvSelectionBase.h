// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "qPCL.h"

// Forward declarations
class ecvGenericVisualizer3D;
class vtkPolyData;
class vtkActor;
class vtkDataSet;
class cvSelectionData;
class cvViewSelectionManager;

namespace PclUtils {
class PCLVis;
}

/**
 * @brief Lightweight base class for all selection-related components
 *
 * Provides only visualizer access without adding unnecessary functionality.
 * This base class is designed to be minimal (~16 bytes) to avoid memory
 * overhead for components that only need visualizer access.
 *
 * Usage guidelines:
 * - Use cvSelectionBase when you only need visualizer access
 *   (e.g., cvSelectionHighlighter, cvSelectionPropertiesWidget)
 * - Use cvGenericSelectionTool when you need picking/selection functionality
 *   (e.g., cvSurfaceSelectionTool, cvTooltipSelectionTool)
 *
 * Design rationale:
 * - Follows Interface Segregation Principle (ISP)
 * - Avoids forcing clients to depend on methods they don't use
 * - Reduces memory footprint for non-picking components by ~60-70%
 */
class QPCL_ENGINE_LIB_API cvSelectionBase {
public:
    cvSelectionBase() : m_viewer(nullptr) {}
    virtual ~cvSelectionBase() = default;

    /**
     * @brief Set the visualizer instance
     * @param viewer Pointer to the generic visualizer
     *
     * This uses the abstract ecvGenericVisualizer3D interface to avoid
     * exposing PCLVis to upper layers (MainWindow, PropertiesDelegate, etc.)
     */
    virtual void setVisualizer(ecvGenericVisualizer3D* viewer) {
        m_viewer = viewer;
    }

    /**
     * @brief Get the visualizer instance
     * @return Pointer to the generic visualizer
     */
    ecvGenericVisualizer3D* getVisualizer() const { return m_viewer; }

protected:
    /**
     * @brief Get PCLVis instance (for VTK-specific operations)
     * @return Pointer to PCLVis, or nullptr if cast fails
     *
     * Use this method when you need to access VTK-specific functionality
     * like getCurrentRenderer(), UpdateScreen(), etc.
     *
     * Always check for nullptr before using!
     */
    PclUtils::PCLVis* getPCLVis() const;

    /**
     * @brief Check if visualizer is valid and is PCLVis
     * @return true if visualizer is set and is PCLVis instance
     */
    bool hasValidPCLVis() const;

    /**
     * @brief Get data object from a specific actor (ParaView-style)
     * @param actor The target actor
     * @return Data object from actor's mapper, or nullptr
     */
    vtkDataSet* getDataFromActor(vtkActor* actor);

    /**
     * @brief Get all visible data actors from visualizer
     * @return List of pickable actors with data
     */
    QList<vtkActor*> getDataActors() const;

    /**
     * @brief Get all polyData from the visualizer
     * @return Vector of all valid vtkPolyData pointers (may be empty)
     *
     * Returns all valid polyData from visible and pickable actors in the
     * renderer. Useful when you need to process multiple actors or when
     * the default selection strategy (largest actor) is not appropriate.
     */
    std::vector<vtkPolyData*> getAllPolyDataFromVisualizer();

    /**
     * @brief Get polyData using ParaView-style priority (centralized method)
     * @param selectionData Optional selection data to extract from (highest
     * priority)
     * @return PolyData pointer, or nullptr if not available
     *
     * This method encapsulates the ParaView-style priority logic:
     * Priority 1: From selectionData's actor info (if provided and has actor
     * info) Priority 2: From selection manager singleton's getPolyData()
     * Priority 3: From first data actor (fallback for non-selection operations)
     *
     * This centralizes the logic and avoids duplication across all tools.
     * Uses cvViewSelectionManager::instance() internally (singleton pattern).
     *
     * Note: Subclasses (like cvGenericSelectionTool) can provide enhanced
     * versions that also check their m_currentSelection.
     */
    virtual vtkPolyData* getPolyDataForSelection(
            const cvSelectionData* selectionData = nullptr);

protected:
    //! Visualizer instance (abstract interface)
    ecvGenericVisualizer3D* m_viewer;
};
