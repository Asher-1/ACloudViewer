// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pcl/console/print.h>  // for PCL_DEBUG
#include <pcl/visualization/vtk/vtkRenderWindowInteractorFix.h>
#include <vtkVersion.h>
#if __unix__ && VTK_MAJOR_VERSION == 9 &&                        \
        ((VTK_MINOR_VERSION == 0 &&                              \
          (VTK_BUILD_VERSION == 2 || VTK_BUILD_VERSION == 3)) || \
         (VTK_MINOR_VERSION == 1 && VTK_BUILD_VERSION == 0))
#include <pcl/visualization/vtk/vtkFixedXRenderWindowInteractor.h>
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
        PCL_DEBUG("Using a fixed version of the vtkXRenderWindowInteractor!\n");
        interactor->Delete();
        interactor = pcl::vtkXRenderWindowInteractor::New();
    }
    return (interactor);
#else
    return (vtkRenderWindowInteractor::New());
#endif
}
#endif