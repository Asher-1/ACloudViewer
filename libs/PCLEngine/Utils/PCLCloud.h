// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file PCLCloud.h
 * @brief Common PCL type definitions for CloudViewer
 * 
 * Provides convenient type aliases for frequently used PCL point cloud types,
 * reducing verbosity and improving code readability throughout the codebase.
 */

#pragma once

#include <pcl/PCLPointCloud2.h>
#include <pcl/PolygonMesh.h>
#include <pcl/TextureMesh.h>
#include <pcl/geometry/planar_polygon.h>
#include <pcl/point_types.h>

// =====================================================================
// Point Types
// =====================================================================

/// Basic XYZ point type
typedef pcl::PointXYZ PointT;
/// XYZ point cloud
typedef pcl::PointCloud<PointT> PointCloudT;

/// Point with intensity field
typedef pcl::PointXYZI PointIntensity;
/// Intensity point cloud
typedef pcl::PointCloud<PointIntensity> PointCloudI;

/// Point with RGB color
typedef pcl::PointXYZRGB PointRGB;
/// RGB point cloud
typedef pcl::PointCloud<PointRGB> PointCloudRGB;

/// Point with RGBA color (includes alpha channel)
typedef pcl::PointXYZRGBA PointRGBA;
/// RGBA point cloud
typedef pcl::PointCloud<PointRGBA> PointCloudRGBA;

/// Point with normal vector
typedef pcl::PointNormal PointNT;
/// Point+normal cloud
typedef pcl::PointCloud<PointNT> PointCloudNormal;

/// Normal vector only (no position)
typedef pcl::Normal NormalT;
/// Normal-only cloud
typedef pcl::PointCloud<NormalT> CloudNormal;

/// Point with RGB color and normal
typedef pcl::PointXYZRGBNormal PointRGBNormal;
/// RGB+normal point cloud
typedef pcl::PointCloud<PointRGBNormal> PointCloudRGBNormal;

// =====================================================================
// Mesh and Geometry Types
// =====================================================================

/// Polygon mesh with vertices and faces
typedef pcl::PolygonMesh PCLMesh;

/// Texture material definition
typedef pcl::TexMaterial PCLMaterial;

/// Textured polygon mesh
typedef pcl::TextureMesh PCLTextureMesh;

/// Generic point cloud format (for I/O)
typedef pcl::PCLPointCloud2 PCLCloud;

/// Polyline as point cloud
typedef pcl::PointCloud<PointT> PCLPolyLine;

/// Planar polygon
typedef pcl::PlanarPolygon<PointT> PCLPolygon;
