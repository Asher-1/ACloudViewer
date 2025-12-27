// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvRenderViewSelectionTool.h"

// VTK
#include <vtkSmartPointer.h>

// Forward declarations
class vtkInteractorStyleRubberBandPick;

/**
 * @brief Tool for block selection (surface and frustum)
 *
 * This tool provides block-based selection modes:
 * - SELECT_BLOCKS: Select entire data blocks on surface
 * - SELECT_FRUSTUM_BLOCKS: Select blocks within frustum
 *
 * Blocks are logical groupings of data, useful for multi-block datasets
 * or hierarchical data structures.
 *
 * Based on ParaView's block selection implementation.
 *
 * Reference: pqRenderViewSelectionReaction.cxx, SELECT_BLOCKS and
 * SELECT_FRUSTUM_BLOCKS
 */
class QPCL_ENGINE_LIB_API cvBlockSelectionTool
    : public cvRenderViewSelectionTool {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param mode Selection mode (SELECT_BLOCKS or SELECT_FRUSTUM_BLOCKS)
     * @param parent Parent QObject
     */
    explicit cvBlockSelectionTool(SelectionMode mode,
                                  QObject* parent = nullptr);
    ~cvBlockSelectionTool() override;

signals:
    /**
     * @brief Emitted when block selection is completed
     * @param blockIds Array of selected block IDs
     */
    void blockSelectionFinished(const QVector<int>& blockIds);

protected:
    void setupInteractorStyle() override;
    void setupObservers() override;
    void onSelectionChanged(vtkObject* caller,
                            unsigned long eventId,
                            void* callData) override;

private:
    /**
     * @brief Extract block IDs from selection region
     * @param x1, y1, x2, y2 Selection rectangle in screen coordinates
     * @return Vector of block IDs
     */
    QVector<int> extractBlockIds(int x1, int y1, int x2, int y2);

    /**
     * @brief Check if a block intersects with selection region
     * @param blockId Block ID to check
     * @param x1, y1, x2, y2 Selection rectangle
     * @return true if block is within selection
     */
    bool isBlockInSelection(int blockId, int x1, int y1, int x2, int y2);

    vtkSmartPointer<vtkInteractorStyleRubberBandPick> m_interactorStyle;
};
