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
class vtkCamera;
class vtkInteractorStyleRubberBandZoom;

/**
 * @brief Tool for zoom-to-box functionality in the selection framework
 *
 * This tool allows the user to draw a rectangular region on the screen
 * and zooms the camera to fit that region. Uses
 * vtkInteractorStyleRubberBandZoom for rubber band interaction.
 *
 * This class integrates with the selection tool framework
 * (cvRenderViewSelectionTool) while providing zoom functionality similar to
 * ParaView.
 *
 * Note: This is different from CameraTools/cvZoomToBoxTool which is a
 * standalone tool. This class is designed to work within the selection system
 * architecture.
 *
 * Based on ParaView's ZOOM_TO_BOX implementation.
 *
 * Reference: pqRenderViewSelectionReaction.cxx, ZOOM_TO_BOX case
 *            vtkInteractorStyleRubberBandZoom.{h,cxx}
 */
// Forward declare callback class
class cvZoomBoxCallback;

class QPCL_ENGINE_LIB_API cvZoomBoxSelectionTool
    : public cvRenderViewSelectionTool {
    Q_OBJECT

    friend class cvZoomBoxCallback;  // Allow callback to access protected
                                     // methods

public:
    /**
     * @brief Constructor
     * @param parent Parent QObject
     */
    explicit cvZoomBoxSelectionTool(QObject* parent = nullptr);
    ~cvZoomBoxSelectionTool() override;

    /**
     * @brief Get the cursor for zoom mode
     * @return Zoom cursor
     */
    QCursor getCursor() const override;

signals:
    /**
     * @brief Emitted when zoom to box is completed
     * @param xmin, ymin, xmax, ymax The box region in screen coordinates
     */
    void zoomToBoxCompleted(int xmin, int ymin, int xmax, int ymax);

protected:
    /**
     * @brief Set up the interactor style for zoom mode
     *
     * Uses vtkInteractorStyleRubberBandZoom for rubber band interaction.
     * Reference: pqRenderViewSelectionReaction.cxx, line 403-406
     */
    void setupInteractorStyle() override;

    /**
     * @brief Set up event observers for zoom mode
     *
     * Observes LeftButtonReleaseEvent from the interactor.
     * Reference: pqRenderViewSelectionReaction.cxx, line 443-447
     */
    void setupObservers() override;

    /**
     * @brief Show instruction and set cursor for zoom mode
     */
    void showInstructionAndSetCursor() override;

    /**
     * @brief Handle selection changed event
     *
     * For zoom mode, this performs the actual camera zoom.
     * Reference: pqRenderViewSelectionReaction.cxx, line 593-594
     */
    void onSelectionChanged(vtkObject* caller,
                            unsigned long eventId,
                            void* callData) override;

    /**
     * @brief Perform the zoom operation
     * @param region The screen region [x1, y1, x2, y2]
     * @return true if zoom was successful
     */
    bool performSelection(int region[4]) override;

private:
    /**
     * @brief Perform traditional zoom (parallel projection or dolly)
     *
     * Moves camera and adjusts zoom based on box size.
     * Reference: vtkInteractorStyleRubberBandZoom::ZoomTraditional()
     */
    void zoomTraditional(int region[4]);

    /**
     * @brief Perform perspective zoom using focal point
     *
     * Adjusts focal point and view angle for perspective projection.
     * Reference: vtkInteractorStyleRubberBandZoom::Zoom() perspective case
     */
    void zoomPerspective(int region[4]);

    /**
     * @brief Store the start/end positions for the zoom box
     */
    int m_startPosition[2];
    int m_endPosition[2];

    /**
     * @brief Whether to use dolly for perspective projection
     */
    bool m_useDollyForPerspective;
};
