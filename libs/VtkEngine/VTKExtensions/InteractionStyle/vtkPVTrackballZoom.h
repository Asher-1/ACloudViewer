// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVTrackballZoom.h
 * @brief Camera manipulator for zoom (dolly or view angle change).
 */

#include "qVTK.h"  // needed for export macro
#include "vtkCameraManipulator.h"

/**
 * @class vtkPVTrackballZoom
 * @brief ParaView-style trackball zoom manipulator (dolly or zoom).
 */
class QVTK_ENGINE_LIB_API vtkPVTrackballZoom : public vtkCameraManipulator {
public:
    static vtkPVTrackballZoom* New();
    vtkTypeMacro(vtkPVTrackballZoom, vtkCameraManipulator);
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
    void OnButtonUp(int x,
                    int y,
                    vtkRenderer* ren,
                    vtkRenderWindowInteractor* rwi) override;
    //@}

    /**
     * Set this to false (true by default) to not dolly in case of perspective
     * projection and use zoom i.e. change view angle, instead.
     */
    vtkSetMacro(UseDollyForPerspectiveProjection, bool);
    vtkGetMacro(UseDollyForPerspectiveProjection, bool);
    vtkBooleanMacro(UseDollyForPerspectiveProjection, bool);

protected:
    vtkPVTrackballZoom();
    ~vtkPVTrackballZoom() override;

    bool UseDollyForPerspectiveProjection;
    double ZoomScale;

    vtkPVTrackballZoom(const vtkPVTrackballZoom&) = delete;
    void operator=(const vtkPVTrackballZoom&) = delete;
};
