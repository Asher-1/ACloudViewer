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
//
#ifndef PCL_CLOUD_H
#define PCL_CLOUD_H

#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <pcl/TextureMesh.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/geometry/planar_polygon.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointXYZI PointIntensity;
typedef pcl::PointCloud<PointIntensity> PointCloudI;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointRGB> PointCloudRGB;
typedef pcl::PointXYZRGBA PointRGBA;
typedef pcl::PointCloud<PointRGBA> PointCloudRGBA;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNormal;
typedef pcl::Normal NormalT;
typedef pcl::PointCloud<NormalT> CloudNormal;
typedef pcl::PointXYZRGBNormal PointRGBNormal;
typedef pcl::PointCloud<PointRGBNormal> PointCloudRGBNormal;

typedef pcl::PolygonMesh PCLMesh;
typedef pcl::TexMaterial PCLMaterial;
typedef pcl::TextureMesh PCLTextureMesh;
typedef pcl::PCLPointCloud2 PCLCloud;
typedef pcl::PointCloud<PointT> PCLPolyLine;
typedef pcl::PlanarPolygon<PointT> PCLPolygon;

#endif // PCL_CLOUD_H
