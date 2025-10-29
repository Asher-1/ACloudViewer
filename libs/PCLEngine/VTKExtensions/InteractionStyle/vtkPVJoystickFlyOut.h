// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  // needed for export macro
#include "vtkPVJoystickFly.h"

class QPCL_ENGINE_LIB_API vtkPVJoystickFlyOut : public vtkPVJoystickFly {
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
