// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVCenterAxesActor.h
 * @brief OpenGL axes actor centered at origin with symmetric option.
 */

#include "qVTK.h"  // needed for export macro
#include "vtkOpenGLActor.h"

class vtkAxes;
class vtkPolyDataMapper;

namespace VTKExtensions {

/**
 * @class vtkPVCenterAxesActor
 * @brief OpenGL actor for XYZ axes at center with symmetric and normal options.
 */
class QVTK_ENGINE_LIB_API vtkPVCenterAxesActor : public vtkOpenGLActor {
public:
    static vtkPVCenterAxesActor* New();
    vtkTypeMacro(vtkPVCenterAxesActor, vtkOpenGLActor);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * If Symmetric is on, the axis continues to negative values.
     * @param val Non-zero to enable symmetric axes
     */
    void SetSymmetric(int);

    /**
     * Option for computing normals. By default they are computed.
     * @param val Non-zero to enable normal computation
     */
    void SetComputeNormals(int);

protected:
    vtkPVCenterAxesActor();
    ~vtkPVCenterAxesActor() override;

    vtkAxes* Axes;
    vtkPolyDataMapper* Mapper;

private:
    vtkPVCenterAxesActor(const vtkPVCenterAxesActor&) = delete;
    void operator=(const vtkPVCenterAxesActor&) = delete;
};

}  // namespace VTKExtensions
