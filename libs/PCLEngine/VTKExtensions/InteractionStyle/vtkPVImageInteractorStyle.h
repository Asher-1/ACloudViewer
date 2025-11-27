// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkInteractorStyleImage.h>

#include "qPCL.h"  // needed for export macro

/**
 * @class vtkPVImageInteractorStyle
 * @brief ParaView-style interactor for 2D image viewing
 *
 * Provides ParaView-like interaction for images:
 * - Left button: Pan (translate)
 * - Middle button: Rotate around Z-axis (perpendicular to image plane)
 * - Wheel: Zoom
 */
class QPCL_ENGINE_LIB_API vtkPVImageInteractorStyle
    : public vtkInteractorStyleImage {
public:
    static vtkPVImageInteractorStyle* New();
    vtkTypeMacro(vtkPVImageInteractorStyle, vtkInteractorStyleImage);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Event bindings controlling the effects of pressing mouse buttons
     * or moving the mouse.
     */
    void OnLeftButtonDown() override;
    void OnLeftButtonUp() override;
    void OnMiddleButtonDown() override;
    void OnMiddleButtonUp() override;
    void OnMouseMove() override;
    //@}

    //@{
    /**
     * Override Pan and Rotate methods for custom behavior
     */
    void Pan() override;
    void Rotate() override;
    //@}

protected:
    vtkPVImageInteractorStyle();
    ~vtkPVImageInteractorStyle() override;

    double RotationFactor;

private:
    vtkPVImageInteractorStyle(const vtkPVImageInteractorStyle&) = delete;
    void operator=(const vtkPVImageInteractorStyle&) = delete;
};
