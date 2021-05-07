//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER BACKEND : qPCL                       #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                        COPYRIGHT: DAHAI LU                             #
//#                                                                        #
//##########################################################################

#ifndef Q_PCL_PLUGIN_SM2CC_H
#define Q_PCL_PLUGIN_SM2CC_H

//Local
#include "../qPCL.h"
#include "PCLCloud.h"

//system
#include <list>

class ccMesh;
class ccPointCloud;

//! PCL to CC cloud converter
class QPCL_ENGINE_LIB_API pcl2cc
{
public:

    //! Converts a PCL point cloud to a ccPointCloud
    static ccMesh* Convert(PCLTextureMesh::ConstPtr textureMesh);
    static ccPointCloud* Convert(const PCLCloud& pclCloud, bool ignoreScalars = false, bool ignoreRgb = false);
    static ccMesh* Convert(const PCLCloud& pclCloud, const std::vector<pcl::Vertices>& polygons,
                           bool ignoreScalars = false, bool ignoreRgb = false);

public: // other related utility functions

    static bool CopyXYZ(const PCLCloud& pclCloud, ccPointCloud& ccCloud, uint8_t coordinateType);
    static bool CopyNormals(const PCLCloud& pclCloud, ccPointCloud& ccCloud);
    static bool CopyRGB(const PCLCloud& pclCloud, ccPointCloud& ccCloud);
    static bool CopyScalarField(const PCLCloud& pclCloud, const std::string& sfName,
                                ccPointCloud& ccCloud, bool overwriteIfExist = true);

};

#endif // Q_PCL_PLUGIN_SM2CC_H
