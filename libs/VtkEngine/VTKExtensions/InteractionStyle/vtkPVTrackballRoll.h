// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVTrackballRoll.h
 * @brief Camera manipulator for roll (rotation around view axis).
 */

#include "qVTK.h"  // needed for export macro
#include "vtkCameraManipulator.h"

/**
 * @class vtkPVTrackballRoll
 * @brief ParaView-style trackball roll manipulator (rotate around view
 * direction).
 */
class QVTK_ENGINE_LIB_API vtkPVTrackballRoll : public vtkCameraManipulator {
public:
    static vtkPVTrackballRoll* New();
    vtkTypeMacro(vtkPVTrackballRoll, vtkCameraManipulator);
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

protected:
    vtkPVTrackballRoll();
    ~vtkPVTrackballRoll() override;

    vtkPVTrackballRoll(const vtkPVTrackballRoll&) = delete;
    void operator=(const vtkPVTrackballRoll&) = delete;
};
