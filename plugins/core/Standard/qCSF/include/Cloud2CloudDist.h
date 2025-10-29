// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "Cloth.h"
#include "wlPointCloud.h"

// computing distance between clouds
class Cloud2CloudDist {
public:
    static bool Compute(const Cloth& cloth,
                        const wl::PointCloud& pc,
                        double class_threshold,
                        std::vector<int>& groundIndexes,
                        std::vector<int>& offGroundIndexes,
                        unsigned N = 3);
};
