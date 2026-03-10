// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVJoystickFlyOut.h
 * @brief Joystick fly manipulator for flying out of the scene.
 */

#include "qVTK.h"  // needed for export macro
#include "vtkPVJoystickFly.h"

/**
 * @class vtkPVJoystickFlyOut
 * @brief Fly camera backward out of the scene (joystick-style).
 */
class QVTK_ENGINE_LIB_API vtkPVJoystickFlyOut : public vtkPVJoystickFly {
public:
    static vtkPVJoystickFlyOut* New();
    vtkTypeMacro(vtkPVJoystickFlyOut, vtkPVJoystickFly);
    void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
    vtkPVJoystickFlyOut();
    ~vtkPVJoystickFlyOut() override;

private:
    vtkPVJoystickFlyOut(const vtkPVJoystickFlyOut&) = delete;
    void operator=(const vtkPVJoystickFlyOut&) = delete;
};
