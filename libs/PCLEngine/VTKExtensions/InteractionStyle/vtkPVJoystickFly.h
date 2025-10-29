// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  // needed for export macro
#include "vtkCameraManipulator.h"

class vtkRenderer;

class QPCL_ENGINE_LIB_API vtkPVJoystickFly : public vtkCameraManipulator {
public:
    vtkTypeMacro(vtkPVJoystickFly, vtkCameraManipulator);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Event bindings controlling the effects of pressing mouse buttons
     * or moving the mouse.
     */
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

    //@{
    /**
     * Set and get the speed of flying.
     */
    vtkSetClampMacro(FlySpeed, double, 1, 30);
    vtkGetMacro(FlySpeed, double);
    //@}

protected:
    vtkPVJoystickFly();
    ~vtkPVJoystickFly() override;

    int In;
    int FlyFlag;

    double FlySpeed;
    double Scale;
    double LastRenderTime;
    double CameraXAxis[3];
    double CameraYAxis[3];
    double CameraZAxis[3];

    void Fly(vtkRenderer* ren,
             vtkRenderWindowInteractor* rwi,
             double scale,
             double speed);
    void ComputeCameraAxes(vtkRenderer*);

    vtkPVJoystickFly(const vtkPVJoystickFly&) = delete;
    void operator=(const vtkPVJoystickFly&) = delete;
};
