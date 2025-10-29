// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #            CLOUDVIEWER PLUGIN: ColorimetricSegmenter                   #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 of the License.               #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #    COPYRIGHT:	Tri-Thien TRUONG, Ronan COLLIER, Mathieu LETRONE       #
// #                                                                        #
// ##########################################################################

// qCC_db
#include <ecvColorTypes.h>

//! HSV color
struct Hsv {
    //! Default constrctor
    Hsv() : h(0), s(0), v(0) {}

    //! Constrctor from a RGB color
    Hsv(const ecvColor::Rgb& rgb) {
        float r = rgb.r / 255.0f;
        float g = rgb.g / 255.0f;
        float b = rgb.b / 255.0f;
        float maxComp = std::max(std::max(r, g), b);
        float minComp = std::min(std::min(r, g), b);
        float deltaComp = maxComp - minComp;

        float hue = 0;
        if (deltaComp != 0) {
            if (r == maxComp) {
                hue = (g - b) / deltaComp;
            } else {
                if (g == maxComp) {
                    hue = 2 + (b - r) / deltaComp;
                } else {
                    hue = 4 + (r - g) / deltaComp;
                }
            }
            hue *= 60;
            if (hue < 0) hue += 360;
        }

        h = (static_cast<uint16_t>(hue) % 360);
        s = static_cast<uint16_t>(maxComp == 0 ? 0
                                               : (deltaComp / maxComp) * 100);
        v = static_cast<uint16_t>(maxComp * 100);
    }

    // HSV components
    uint16_t h, s, v;
};
