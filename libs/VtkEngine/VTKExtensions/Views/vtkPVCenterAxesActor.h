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
#include "vtkLookupTable.h"
#include "vtkNew.h"
#include "vtkOpenGLActor.h"

class vtkAxes;
class vtkPolyDataMapper;

namespace VTKExtensions {

/**
 * @class vtkPVCenterAxesActor
 * @brief OpenGL actor for XYZ axes at center with symmetric and normal options.
 * Uses a LUT to color axes: X=red, Y=yellow, Z=blue (matching ParaView).
 */
class QVTK_ENGINE_LIB_API vtkPVCenterAxesActor : public vtkOpenGLActor {
public:
    static vtkPVCenterAxesActor* New();
    vtkTypeMacro(vtkPVCenterAxesActor, vtkOpenGLActor);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    void SetSymmetric(int);
    void SetComputeNormals(int);

    void SetXAxisColor(double r, double g, double b);
    void SetYAxisColor(double r, double g, double b);
    void SetZAxisColor(double r, double g, double b);

protected:
    vtkPVCenterAxesActor();
    ~vtkPVCenterAxesActor() override;

    vtkAxes* Axes;
    vtkPolyDataMapper* Mapper;
    vtkNew<vtkLookupTable> LUT;

private:
    vtkPVCenterAxesActor(const vtkPVCenterAxesActor&) = delete;
    void operator=(const vtkPVCenterAxesActor&) = delete;

    void SetAxisColor(int axis, double r, double g, double b);
};

}  // namespace VTKExtensions
