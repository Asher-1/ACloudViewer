// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_VTK_CENTER_AXES_ACTOR_H
#define ECV_VTK_CENTER_AXES_ACTOR_H

#include "qPCL.h"  // needed for export macro
#include "vtkOpenGLActor.h"

class vtkAxes;
class vtkPolyDataMapper;

namespace VTKExtensions {
class QPCL_ENGINE_LIB_API vtkPVCenterAxesActor : public vtkOpenGLActor {
public:
    static vtkPVCenterAxesActor* New();
    vtkTypeMacro(vtkPVCenterAxesActor, vtkOpenGLActor);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * If Symmetric is on, the the axis continue to negative values.
     */
    void SetSymmetric(int);

    /**
     * Option for computing normals.  By default they are computed.
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

#endif  // ECV_VTK_CENTER_AXES_ACTOR_H
