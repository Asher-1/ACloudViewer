// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVTrackballMultiRotate.h
 * @brief Camera manipulator combining rotate and roll for multi-axis rotation.
 */

#include "qVTK.h"  // needed for export macro
#include "vtkCameraManipulator.h"

class vtkCameraManipulator;
class vtkPVTrackballRoll;
class vtkPVTrackballRotate;

/**
 * @class vtkPVTrackballMultiRotate
 * @brief Combines rotate and roll manipulators for ParaView-style multi-axis
 * trackball rotation.
 */
class QVTK_ENGINE_LIB_API vtkPVTrackballMultiRotate
    : public vtkCameraManipulator {
public:
    vtkTypeMacro(vtkPVTrackballMultiRotate, vtkCameraManipulator);
    static vtkPVTrackballMultiRotate* New();
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
    vtkPVTrackballMultiRotate();
    ~vtkPVTrackballMultiRotate() override;

    vtkPVTrackballRotate* RotateManipulator;
    vtkPVTrackballRoll* RollManipulator;

    vtkCameraManipulator* CurrentManipulator;

private:
    vtkPVTrackballMultiRotate(const vtkPVTrackballMultiRotate&) = delete;
    void operator=(const vtkPVTrackballMultiRotate&) = delete;
};
