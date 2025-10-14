// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  // needed for export macro
#include "vtkPVJoystickFly.h"

class QPCL_ENGINE_LIB_API vtkPVJoystickFlyIn : public vtkPVJoystickFly {
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
