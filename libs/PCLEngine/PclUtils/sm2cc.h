// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "../qPCL.h"
#include "PCLCloud.h"

// ECV_DB_LIB
#include <ecvMaterial.h>

// system
#include <list>

class ccMesh;
class ccPointCloud;

//! PCL to CC cloud converter
class QPCL_ENGINE_LIB_API pcl2cc {
public:
    //! Converts a PCL point cloud to a ccPointCloud
    static ccMesh* Convert(PCLTextureMesh::ConstPtr textureMesh);
    static ccPointCloud* Convert(const PCLCloud& pclCloud,
                                 bool ignoreScalars = false,
                                 bool ignoreRgb = false);
    static ccMesh* Convert(const PCLCloud& pclCloud,
                           const std::vector<pcl::Vertices>& polygons,
                           bool ignoreScalars = false,
                           bool ignoreRgb = false);

public:  // other related utility functions
    static bool CopyXYZ(const PCLCloud& pclCloud,
                        ccPointCloud& ccCloud,
                        uint8_t coordinateType);
    static bool CopyNormals(const PCLCloud& pclCloud, ccPointCloud& ccCloud);
    static bool CopyRGB(const PCLCloud& pclCloud, ccPointCloud& ccCloud);
    static bool CopyScalarField(const PCLCloud& pclCloud,
                                const std::string& sfName,
                                ccPointCloud& ccCloud,
                                bool overwriteIfExist = true);

    static void FromPCLMaterial(const PCLMaterial& inMaterial,
                                ccMaterial::Shared& outMaterial);
};
