// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file PCLConv.h
 * @brief PCL point cloud conversion macros
 * 
 * Provides convenience macros for converting between PCL point cloud formats.
 * These macros wrap PCL's native conversion functions for easier use.
 */

#pragma once

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>

/**
 * @def FROM_PCL_CLOUD
 * @brief Convert from pcl::PCLPointCloud2 to typed point cloud
 * 
 * Wraps pcl::fromPCLPointCloud2 for converting generic PCL cloud format
 * to typed point clouds (e.g., pcl::PointCloud<pcl::PointXYZ>).
 * 
 * Usage: FROM_PCL_CLOUD(cloud2, typed_cloud);
 */
#define FROM_PCL_CLOUD pcl::fromPCLPointCloud2

/**
 * @def TO_PCL_CLOUD
 * @brief Convert from typed point cloud to pcl::PCLPointCloud2
 * 
 * Wraps pcl::toPCLPointCloud2 for converting typed point clouds
 * to generic PCL cloud format for I/O and filter operations.
 * 
 * Usage: TO_PCL_CLOUD(typed_cloud, cloud2);
 */
#define TO_PCL_CLOUD pcl::toPCLPointCloud2
