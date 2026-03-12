// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file copy.h
 *  @brief Utilities for copying scalar fields and RGB colors between
 * ccPointCloud instances
 */

// Local
#include <PclUtils/PCLCloud.h>

// PCL
#include <pcl/PointIndices.h>

class ccPointCloud;

//! Makes a copy of all scalar fields from one cloud to another
/**	This algorithm simply copy the scalar fields from a cloud
        to another using the the mapping contained in a pcl::PointIndicesPtr.
        \param inCloud the input cloud from which to copy scalars
        \param outCloud the output cloud in which to copy the scalar fields
        \param in2outMapping indices of the input cloud for each point in the
output \param overwrite you can chose to not overwrite existing fields
**/
void copyScalarFields(const ccPointCloud *inCloud,
                      ccPointCloud *outCloud,
                      pcl::PointIndicesPtr &in2outMapping,
                      bool overwrite = true);

//! Makes a copy of RGB colors from one cloud to another
/** @param inCloud the input cloud from which to copy RGB colors
 *  @param outCloud the output cloud in which to copy the RGB colors
 *  @param in2outMapping indices of the input cloud for each point in the output
 *  @param overwrite whether to overwrite existing colors (default: true)
 */
void copyRGBColors(const ccPointCloud *inCloud,
                   ccPointCloud *outCloud,
                   pcl::PointIndicesPtr &in2outMapping,
                   bool overwrite = true);

// #endif // LP_PCL_PATCH_ENABLED
