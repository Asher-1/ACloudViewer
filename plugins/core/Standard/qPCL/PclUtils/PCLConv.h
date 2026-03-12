// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file PCLConv.h
 *  @brief PCL point cloud conversion macros (fromPCLPointCloud2 /
 * toPCLPointCloud2)
 */

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#define FROM_PCL_CLOUD pcl::fromPCLPointCloud2
#define TO_PCL_CLOUD pcl::toPCLPointCloud2
