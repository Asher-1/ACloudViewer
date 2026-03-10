// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkBoxWidgetRestricted.h
 * @brief Box widget with rotation restricted to Z axis.
 */

#include <vtkBoxWidget.h>

#include "qVTK.h"

/**
 * @class vtkBoxWidgetRestricted
 * @brief Box widget that restricts rotation to the Z axis only.
 */
class QVTK_ENGINE_LIB_API vtkBoxWidgetRestricted : public vtkBoxWidget {
public:
    static vtkBoxWidgetRestricted *New();

    vtkTypeMacro(vtkBoxWidgetRestricted, vtkBoxWidget);

    /// @param X Mouse X
    /// @param Y Mouse Y
    /// @param p1 First point
    /// @param p2 Second point
    /// @param vpn View plane normal
    virtual void Rotate(
            int X, int Y, double *p1, double *p2, double *vpn) override;
};
