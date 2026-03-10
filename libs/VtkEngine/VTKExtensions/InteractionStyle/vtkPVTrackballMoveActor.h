// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVTrackballMoveActor.h
 * @brief Camera manipulator for translating (moving) actors in the scene.
 */

#include "qVTK.h"  // needed for export macro
#include "vtkCameraManipulator.h"

/**
 * @class vtkPVTrackballMoveActor
 * @brief ParaView-style trackball manipulator for moving/translating actors.
 */
class QVTK_ENGINE_LIB_API vtkPVTrackballMoveActor
    : public vtkCameraManipulator {
public:
    static vtkPVTrackballMoveActor* New();
    vtkTypeMacro(vtkPVTrackballMoveActor, vtkCameraManipulator);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Event bindings controlling the effects of pressing mouse buttons
     * or moving the mouse.
     */
    /// @param x Mouse X coordinate
    /// @param y Mouse Y coordinate
    /// @param ren Renderer
    /// @param iren Render window interactor
    void OnMouseMove(int x,
                     int y,
                     vtkRenderer* ren,
                     vtkRenderWindowInteractor* iren) override;
    void OnButtonDown(int x,
                      int y,
                      vtkRenderer* ren,
                      vtkRenderWindowInteractor* iren) override;
    void OnButtonUp(int x,
                    int y,
                    vtkRenderer* ren,
                    vtkRenderWindowInteractor* iren) override;
    //@}

protected:
    vtkPVTrackballMoveActor();
    ~vtkPVTrackballMoveActor() override;

    vtkPVTrackballMoveActor(const vtkPVTrackballMoveActor&) = delete;
    void operator=(const vtkPVTrackballMoveActor&) = delete;
};
