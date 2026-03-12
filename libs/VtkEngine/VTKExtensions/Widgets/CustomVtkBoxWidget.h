// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qVTK.h"

/**
 * @brief The CustomVtkBoxWidget class
 * CustomVtkBoxWidget restricts the transformation
 */

#include <vtkBoxWidget.h>

class QVTK_ENGINE_LIB_API CustomVtkBoxWidget : public vtkBoxWidget {
public:
    static CustomVtkBoxWidget *New();

    vtkTypeMacro(CustomVtkBoxWidget, vtkBoxWidget);

    virtual void Translate(double *p1, double *p2) override;
    virtual void Scale(double *p1, double *p2, int X, int Y) override;
    virtual void Rotate(
            int X, int Y, double *p1, double *p2, double *vpn) override;

    /// @param state Enable/disable X translation
    void SetTranslateXEnabled(bool state) { m_translateX = state; }
    /// @param state Enable/disable Y translation
    void SetTranslateYEnabled(bool state) { m_translateY = state; }
    /// @param state Enable/disable Z translation
    void SetTranslateZEnabled(bool state) { m_translateZ = state; }
    /// @param state Enable/disable X rotation
    void SetRotateXEnabled(bool state) { m_rotateX = state; }
    /// @param state Enable/disable Y rotation
    void SetRotateYEnabled(bool state) { m_rotateY = state; }
    /// @param state Enable/disable Z rotation
    void SetRotateZEnabled(bool state) { m_rotateZ = state; }
    /// @param state Enable/disable scaling
    void SetScaleEnabled(bool state) { m_scale = state; }

private:
    bool m_translateX = true;
    bool m_translateY = true;
    bool m_translateZ = true;
    bool m_rotateX = true;
    bool m_rotateY = true;
    bool m_rotateZ = true;
    bool m_scale = true;
};
