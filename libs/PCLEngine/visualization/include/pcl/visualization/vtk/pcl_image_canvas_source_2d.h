// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <pcl/pcl_macros.h>
#include <vtkImageCanvasSource2D.h>
class vtkImageData;

namespace pcl {
namespace visualization {
/** \brief PCLImageCanvasSource2D represents our own custom version of
 * vtkImageCanvasSource2D, used by the ImageViewer class.
 */
class PCL_EXPORTS PCLImageCanvasSource2D : public vtkImageCanvasSource2D {
public:
    static PCLImageCanvasSource2D* New();

    void DrawImage(vtkImageData* image);
};
}  // namespace visualization
}  // namespace pcl
