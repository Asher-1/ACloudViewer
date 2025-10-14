// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pcl/visualization/vtk/pcl_image_canvas_source_2d.h>
#include <vtkImageData.h>
#include <vtkObjectFactory.h>

namespace pcl {
namespace visualization {
// Standard VTK macro for *New ()
vtkStandardNewMacro(PCLImageCanvasSource2D)
}  // namespace visualization
}  // namespace pcl

//////////////////////////////////////////////////////////////////////////////
void pcl::visualization::PCLImageCanvasSource2D::DrawImage(
        vtkImageData* image) {
    this->ImageData->DeepCopy(image);
    this->Modified();
}
