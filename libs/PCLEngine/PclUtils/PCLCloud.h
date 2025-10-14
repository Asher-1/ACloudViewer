// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PCL_CLOUD_H
#define PCL_CLOUD_H

#include <pcl/PCLPointCloud2.h>
#include <pcl/PolygonMesh.h>
#include <pcl/TextureMesh.h>
#include <pcl/geometry/planar_polygon.h>
#include <pcl/point_types.h>

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

#endif  // PCL_CLOUD_H
