// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkRenderWindowInteractorFix.h"

#include <CVLog.h>
#include <vtkVersion.h>

#if __unix__ && VTK_MAJOR_VERSION == 9 &&                        \
        ((VTK_MINOR_VERSION == 0 &&                              \
          (VTK_BUILD_VERSION == 2 || VTK_BUILD_VERSION == 3)) || \
         (VTK_MINOR_VERSION == 1 && VTK_BUILD_VERSION == 0))
#include "vtkFixedXRenderWindowInteractor.h"
#endif

#ifndef __APPLE__
vtkRenderWindowInteractor* vtkRenderWindowInteractorFixNew() {
#if __unix__ && VTK_MAJOR_VERSION == 9 &&                        \
        ((VTK_MINOR_VERSION == 0 &&                              \
          (VTK_BUILD_VERSION == 2 || VTK_BUILD_VERSION == 3)) || \
         (VTK_MINOR_VERSION == 1 && VTK_BUILD_VERSION == 0))
    // VTK versions 9.0.2, 9.0.3, 9.1.0
    vtkRenderWindowInteractor* interactor = vtkRenderWindowInteractor::New();
    if (interactor->IsA("vtkXRenderWindowInteractor")) {
        CVLog::Print(
                "Using a fixed version of the vtkXRenderWindowInteractor!");
        interactor->Delete();
        interactor = VtkRendering::vtkXRenderWindowInteractor::New();
    }
    return (interactor);
#else
    return (vtkRenderWindowInteractor::New());
#endif
}
#endif