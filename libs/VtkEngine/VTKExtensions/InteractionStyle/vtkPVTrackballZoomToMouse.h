// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVTrackballZoomToMouse.h
 * @brief Zoom manipulator that zooms toward mouse cursor position.
 */

#include "qVTK.h"  // needed for export macro
#include "vtkPVTrackballZoom.h"

/**
 * @class vtkPVTrackballZoomToMouse
 * @brief Zoom toward the mouse cursor instead of center of rotation.
 */
class QVTK_ENGINE_LIB_API vtkPVTrackballZoomToMouse
    : public vtkPVTrackballZoom {
public:
    static vtkPVTrackballZoomToMouse* New();
    vtkTypeMacro(vtkPVTrackballZoomToMouse, vtkPVTrackballZoom);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Event bindings controlling the effects of pressing mouse buttons
     * or moving the mouse.
     */
    /// @param x Mouse X coordinate
    /// @param y Mouse Y coordinate
    /// @param ren Renderer
    /// @param rwi Render window interactor
    void OnMouseMove(int x,
                     int y,
                     vtkRenderer* ren,
                     vtkRenderWindowInteractor* rwi) override;
    void OnButtonDown(int x,
                      int y,
                      vtkRenderer* ren,
                      vtkRenderWindowInteractor* rwi) override;
    //@}

protected:
    vtkPVTrackballZoomToMouse();
    ~vtkPVTrackballZoomToMouse() override;

    int ZoomPosition[2];

    vtkPVTrackballZoomToMouse(const vtkPVTrackballZoomToMouse&) = delete;
    void operator=(const vtkPVTrackballZoomToMouse&) = delete;
};
