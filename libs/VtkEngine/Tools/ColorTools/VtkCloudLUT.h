// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file VtkCloudLUT.h
 * @brief Lookup table for point cloud RGB colors (Glasbey-style palette).
 */

#include <cstddef>
#include <cstdint>

/// RGB color value for LUT entries.
struct CloudRGB {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

/**
 * @class VtkCloudLUT
 * @brief Static lookup table mapping color IDs to RGB values.
 */
class VtkCloudLUT {
public:
    /// @param color_id Index into the LUT.
    /// @return RGB color for the given index.
    static CloudRGB at(int color_id);
    /// @return Number of colors in the LUT.
    static size_t size();
    /// @return Raw pointer to LUT data (3 bytes per color).
    static const unsigned char* data();
};
