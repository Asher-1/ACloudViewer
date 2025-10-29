// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "../qPCL.h"

// ECV_DB_LIB
#include <ecvColorTypes.h>

class ccHObject;
class ccMesh;
class ccPointCloud;
class ccPolyline;
class vtkPolyData;

//! CC to PCL cloud converter
class QPCL_ENGINE_LIB_API vtk2cc {
public:
    static ccPointCloud* ConvertToPointCloud(vtkPolyData* polydata,
                                             bool silent = false);
    static ccMesh* ConvertToMesh(vtkPolyData* polydata, bool silent = false);
    static ccPolyline* ConvertToPolyline(vtkPolyData* polydata,
                                         bool silent = false);

    static ccPolyline* ConvertToPolyline(ccPointCloud* vertices);
    static std::vector<ccHObject*> ConvertToMultiPolylines(
            vtkPolyData* polydata,
            QString baseName = "Slice",
            const ecvColor::Rgb& color = ecvColor::green);
};
