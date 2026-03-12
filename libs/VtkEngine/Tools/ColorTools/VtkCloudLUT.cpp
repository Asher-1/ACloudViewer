// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VtkCloudLUT.h"

#include <ecvColorTypes.h>

static const unsigned char CloudLUT[] = {
        255, 255, 255,  // default (label 0)
        255, 0,   0,    // highlighted (label -1 maps to 1)
};

static const unsigned int CloudLUT_SIZE =
        sizeof(CloudLUT) / (sizeof(CloudLUT[0]) * 3);

CloudRGB VtkCloudLUT::at(int color_id) {
    CloudRGB color;
    if (color_id == -1) {
        color_id = 1;
        color.r = CloudLUT[color_id * 3 + 0];
        color.g = CloudLUT[color_id * 3 + 1];
        color.b = CloudLUT[color_id * 3 + 2];
    } else if (color_id == 0) {
        color.r = CloudLUT[0];
        color.g = CloudLUT[1];
        color.b = CloudLUT[2];
    } else if (color_id > 0) {
        ecvColor::Rgb col = ecvColor::LookUpTable::at(color_id);
        color.r = col.r;
        color.g = col.g;
        color.b = col.b;
    }
    return color;
}

size_t VtkCloudLUT::size() { return CloudLUT_SIZE; }

const unsigned char* VtkCloudLUT::data() { return CloudLUT; }
