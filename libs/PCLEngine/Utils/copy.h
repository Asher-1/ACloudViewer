// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include <Utils/PCLCloud.h>

#include "qPCL.h"

// PCL
#include <pcl/PointIndices.h>

class ccPointCloud;

/**
 * @brief Copy all scalar fields from one cloud to another with index mapping
 * @param inCloud Source cloud from which to copy scalar fields
 * @param outCloud Target cloud to receive scalar fields
 * @param in2outMapping Indices mapping: for each output point, index of input point
 * @param overwrite Whether to overwrite existing fields with same names (default: true)
 * 
 * This function copies all scalar fields from the input cloud to the output cloud
 * using the provided index mapping. The mapping allows for point reordering or
 * subsampling during the copy operation.
 * 
 * Example: If in2outMapping->indices = [5, 2, 8], then:
 * - outCloud point 0 gets scalar values from inCloud point 5
 * - outCloud point 1 gets scalar values from inCloud point 2
 * - outCloud point 2 gets scalar values from inCloud point 8
 * 
 * @note The output cloud must be pre-allocated with correct size.
 * @note Use after filtering operations that modify point count or order.
 */
void QPCL_ENGINE_LIB_API copyScalarFields(const ccPointCloud *inCloud,
                                          ccPointCloud *outCloud,
                                          pcl::PointIndicesPtr &in2outMapping,
                                          bool overwrite = true);

/**
 * @brief Copy RGB colors from one cloud to another with index mapping
 * @param inCloud Source cloud from which to copy colors
 * @param outCloud Target cloud to receive colors
 * @param in2outMapping Indices mapping: for each output point, index of input point
 * @param overwrite Whether to overwrite existing colors (default: true)
 * 
 * Copies RGB colors using the same index mapping strategy as copyScalarFields.
 * If input cloud has no colors, the operation completes successfully without effect.
 * 
 * @note The output cloud must be pre-allocated with correct size.
 * @see copyScalarFields for mapping details
 */
void QPCL_ENGINE_LIB_API copyRGBColors(const ccPointCloud *inCloud,
                                       ccPointCloud *outCloud,
                                       pcl::PointIndicesPtr &in2outMapping,
                                       bool overwrite = true);

// #endif // LP_PCL_PATCH_ENABLED
