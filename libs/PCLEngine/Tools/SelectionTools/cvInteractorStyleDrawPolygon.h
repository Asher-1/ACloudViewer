// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file cvInteractorStyleDrawPolygon.h
 * @brief Custom interactor style for polygon selection with improved rendering
 *
 * This class fixes the flickering issue in VTK's vtkInteractorStyleDrawPolygon
 * where the polygon would disappear when moving slowly or when the mouse stops.
 *
 * Key improvements over VTK's implementation:
 * - Redraws polygon on every mouse move (not just when adding new points)
 * - Uses timer-based redraw to ensure polygon stays visible even when stopped
 * - Handles VTK render events that might overwrite the pixel buffer
 *
 * Reference: ParaView vtkPVRenderView polygon selection
 */

#ifndef CV_INTERACTOR_STYLE_DRAW_POLYGON_H
#define CV_INTERACTOR_STYLE_DRAW_POLYGON_H

#include <vtkInteractorStyle.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVector.h>

#include <vector>

#include "qPCL.h"

class vtkRenderWindowInteractor;
class vtkRenderer;

/**
 * @class cvInteractorStyleDrawPolygon
 * @brief Interactor style for drawing polygons with mouse during selection
 *
 * This interactor style allows the user to draw a polygon in the render window
 * using the left mouse button while the mouse is moving. Unlike the base VTK
 * class, this implementation ensures the polygon is visible at all times by
 * redrawing on every mouse move event.
 *
 * When the mouse button is released, a SelectionChangedEvent is fired.
 */
class QPCL_ENGINE_LIB_API cvInteractorStyleDrawPolygon
    : public vtkInteractorStyle {
public:
    static cvInteractorStyleDrawPolygon* New();
    vtkTypeMacro(cvInteractorStyleDrawPolygon, vtkInteractorStyle);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    ///@{
    /**
     * Event bindings
     */
    void OnMouseMove() override;
    void OnLeftButtonDown() override;
    void OnLeftButtonUp() override;
    ///@}

    ///@{
    /**
     * Whether to draw polygon in screen pixels. Default is ON.
     */
    vtkSetMacro(DrawPolygonPixels, bool);
    vtkGetMacro(DrawPolygonPixels, bool);
    vtkBooleanMacro(DrawPolygonPixels, bool);
    ///@}

    ///@{
    /**
     * Set the minimum distance (squared) required between consecutive points.
     * Default is 100 (10 pixels).
     * Set to 0 to add a point on every mouse move.
     */
    vtkSetMacro(MinimumPointDistanceSquared, int);
    vtkGetMacro(MinimumPointDistanceSquared, int);
    ///@}

    /**
     * Get the current polygon points in display units.
     * Returns a vector of 2D points (x, y in screen coordinates).
     */
    std::vector<vtkVector2i> GetPolygonPoints();

    /**
     * Clear all polygon points and reset state.
     */
    void ClearPolygonPoints();

protected:
    cvInteractorStyleDrawPolygon();
    ~cvInteractorStyleDrawPolygon() override;

    /**
     * Draw the polygon on the render window.
     * This is called on every mouse move to ensure the polygon stays visible.
     */
    virtual void DrawPolygon();

    /**
     * Restore the original pixels (erase the polygon).
     */
    virtual void RestorePixels();

    /**
     * Draw a line between two points using XOR pixel manipulation.
     * @param start Start point in screen coordinates
     * @param end End point in screen coordinates
     * @param pixels Pixel buffer to draw into
     * @param size Render window size [width, height]
     */
    void DrawLine(const vtkVector2i& start,
                  const vtkVector2i& end,
                  unsigned char* pixels,
                  const int* size);

    int StartPosition[2];
    int EndPosition[2];
    int Moving;
    bool DrawPolygonPixels;
    int MinimumPointDistanceSquared;

    vtkSmartPointer<vtkUnsignedCharArray> PixelArray;

private:
    cvInteractorStyleDrawPolygon(const cvInteractorStyleDrawPolygon&) = delete;
    void operator=(const cvInteractorStyleDrawPolygon&) = delete;

    class vtkInternal;
    vtkInternal* Internal;
};

#endif  // CV_INTERACTOR_STYLE_DRAW_POLYGON_H
