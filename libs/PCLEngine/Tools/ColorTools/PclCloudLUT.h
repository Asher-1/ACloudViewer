// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QPCL_CloudLUT_H
#define QPCL_CloudLUT_H

#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>

#include <pcl/impl/point_types.hpp>

class PclCloudLUT {
public:
    /** Get a color from the lookup table with a given id.
     *
     * The id should be less than the size of the LUT (see size()). */
    static pcl::RGB at(int color_id);

    /** Get the number of colors in the lookup table.
     *
     * Note: the number of colors is different from the number of elements
     * in the lookup table (each color is defined by three bytes). */
    static size_t size();

    /** Get a raw pointer to the lookup table. */
    static const unsigned char* data();
};

#endif /* QPCL_CloudLUT_H */
