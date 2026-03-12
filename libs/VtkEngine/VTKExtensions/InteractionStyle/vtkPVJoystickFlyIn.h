// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVJoystickFlyIn.h
 * @brief Joystick-style fly-in camera manipulator (moves toward focal point).
 */

#include "qVTK.h"  // needed for export macro
#include "vtkPVJoystickFly.h"

/**
 * @class vtkPVJoystickFlyIn
 * @brief Camera manipulator that flies the camera toward the focal point
 *        in joystick interaction mode.
 */
class QVTK_ENGINE_LIB_API vtkPVJoystickFlyIn : public vtkPVJoystickFly {
public:
    static vtkPVJoystickFlyIn* New();
    vtkTypeMacro(vtkPVJoystickFlyIn, vtkPVJoystickFly);
    void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
    vtkPVJoystickFlyIn();
    ~vtkPVJoystickFlyIn() override;

private:
    vtkPVJoystickFlyIn(const vtkPVJoystickFlyIn&) = delete;
    void operator=(const vtkPVJoystickFlyIn&) = delete;
};
