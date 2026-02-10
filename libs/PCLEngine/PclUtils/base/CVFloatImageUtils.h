// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: BSD-3-Clause
// ----------------------------------------------------------------------------
// CVFloatImageUtils.h - Standalone float-to-RGB color mapping utilities.
// Replaces pcl::visualization::FloatImageUtils with no PCL dependency.

#pragma once

#include <limits>

namespace PclUtils {

/** @b Float-to-RGB image conversion utilities for 2D visualization.
 *  Originally from PCL (Bastian Steder), re-implemented standalone. */
class FloatImageUtils {
public:
    static void getColorForFloat(float value,
                                 unsigned char& r,
                                 unsigned char& g,
                                 unsigned char& b);

    static void getColorForAngle(float value,
                                 unsigned char& r,
                                 unsigned char& g,
                                 unsigned char& b);

    static void getColorForHalfAngle(float value,
                                     unsigned char& r,
                                     unsigned char& g,
                                     unsigned char& b);

    /** Returns a new[]-allocated RGB byte array (3*width*height). Caller owns
     * the memory. */
    static unsigned char* getVisualImage(
            const float* float_image,
            int width,
            int height,
            float min_value = -std::numeric_limits<float>::infinity(),
            float max_value = std::numeric_limits<float>::infinity(),
            bool gray_scale = false);

    static unsigned char* getVisualImage(
            const unsigned short* short_image,
            int width,
            int height,
            unsigned short min_value = 0,
            unsigned short max_value =
                    std::numeric_limits<unsigned short>::max(),
            bool gray_scale = false);

    static unsigned char* getVisualAngleImage(const float* angle_image,
                                              int width,
                                              int height);

    static unsigned char* getVisualHalfAngleImage(const float* angle_image,
                                                  int width,
                                                  int height);
};

}  // namespace PclUtils

